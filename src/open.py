# vim: set expandtab shiftwidth=4 softtabstop=4:

import os
import posixpath
from typing import List, Optional, Tuple

import fsspec
import zarr
from chimerax.core.models import Model
from chimerax.core.session import Session
from chimerax.map.volume import Volume, show_volume_dialog
from fsspec import AbstractFileSystem

from .map_data.labels import build_label_bridge
from .map_data.ome_metadata import (
    OMEZarrFormatError,
    bioformats2raw_series_paths,
    ome_zarr_group_kind,
    parse_labels_metadata,
    parse_ome_zarr_metadata,
)
from .map_data.store_cache import cached_group
from .map_data.zarr_grid import ZarrModel


def _warning(session, template: str, *args) -> None:
    message = template.format(*args)
    session.logger.warning(message)


def _store_from_filesystem(fs: AbstractFileSystem, path: str):
    """Create a read-only Zarr-Python 3 store from an existing fsspec filesystem."""

    mapper = fs.get_mapper(path)
    return zarr.storage.FsspecStore.from_mapper(mapper, read_only=True)


def _open_group(session, root):
    return cached_group(session, root)


def _wrap_time_and_channels(name: str, volumes, session):
    """Build the ChimeraX model appropriate for one resolution level."""

    time_values = [volume.data.time for volume in volumes if volume.data.time is not None]
    channel_values = [volume.data.channel for volume in volumes if volume.data.channel is not None]
    is_time_series = len(time_values) == len(volumes) and len(set(time_values)) > 1
    is_multichannel = len(channel_values) == len(volumes) and len(set(channel_values)) > 1

    if is_time_series and is_multichannel:
        from chimerax.map.volume import MultiChannelSeries
        from chimerax.map_series import MapSeries

        original_display = {volume: volume.display for volume in volumes}
        channel_groups = {}
        for volume in volumes:
            channel_groups.setdefault(volume.data.channel, []).append(volume)

        map_series = []
        for channel in sorted(channel_groups):
            channel_volumes = sorted(channel_groups[channel], key=lambda volume: volume.data.time)
            series = MapSeries(f"{name} channel {channel}", channel_volumes, session)
            for volume in channel_volumes:
                volume.display = original_display[volume]
            series.set_maps(channel_volumes)
            map_series.append(series)
        return MultiChannelSeries(name, map_series, session)

    if is_time_series:
        from chimerax.map_series import MapSeries

        return MapSeries(name, sorted(volumes, key=lambda volume: volume.data.time), session)

    if is_multichannel:
        from chimerax.map.volume import MapChannelsModel

        return MapChannelsModel(name, sorted(volumes, key=lambda volume: volume.data.channel), session)

    return volumes[0] if len(volumes) == 1 else None


def _remove_from_volume_update_manager(session, volumes) -> None:
    vm = getattr(session, "_volume_update_manager", None)
    if vm is not None:
        for volume in volumes:
            vm._volumes_to_update.discard(volume)
            vm._displayed_volumes_to_update.discard(volume)


def _wrap_scale_hierarchies(name: str, volumes, scales, session):
    scale_groups = {}
    for volume in volumes:
        scale_groups.setdefault(getattr(volume.data, "scale_path", None), []).append(volume)
    if len(scale_groups) == 1:
        return _wrap_time_and_channels(name, volumes, session)

    container = Model(name, session)
    wrappers = []
    ordered_scales = scales or list(scale_groups)
    for scale in ordered_scales:
        scale_volumes = scale_groups.get(scale, [])
        if scale_volumes:
            wrappers.append(_wrap_time_and_channels(f"{name} - {scale}", scale_volumes, session))
    container.add(wrappers)
    return container


def _prepare_image_model(
    session: Session,
    group: zarr.Group,
    scales: Optional[List[str]],
    name: str,
    initial_step: Tuple[int, int, int],
):
    if scales is not None:
        initial_step = (1, 1, 1)

    model = ZarrModel(name, session, group, scales, initial_step)
    volumes = list(model.child_models())
    time_indices = [volume.data.time for volume in volumes if volume.data.time is not None]
    channel_indices = [volume.data.channel for volume in volumes if volume.data.channel is not None]
    needs_wrapper = (
        len(time_indices) == len(volumes)
        and len(set(time_indices)) > 1
        or len(channel_indices) == len(volumes)
        and len(set(channel_indices)) > 1
    )

    if needs_wrapper:
        for volume in volumes:
            if volume.display:
                volume.update_drawings()
        _remove_from_volume_update_manager(session, volumes)
        model.remove_drawings(volumes, delete=False)
        model = _wrap_scale_hierarchies(name, volumes, scales, session)

    return model, volumes


def _hide_volumes(model) -> None:
    for child in model.all_models():
        if isinstance(child, Volume):
            child.display = False


def _standalone_label_model(session, label_group, scales, name, initial_step):
    model, _ = _prepare_image_model(session, label_group, scales, name, initial_step)
    _hide_volumes(model)
    return model


def _label_model(
    session,
    label_group,
    label_path,
    label_metadata,
    source_group,
    source_metadata,
    source_volumes,
    scales,
    initial_step,
):
    bridge = build_label_bridge(
        session,
        label_group,
        label_path,
        label_metadata,
        source_group,
        source_metadata,
        source_volumes,
        (1, 1, 1) if scales is not None else initial_step,
    )
    label_name = label_metadata.multiscales.name or label_path.rsplit("/", 1)[-1]
    model = Model(label_name, session)
    index_model = _wrap_scale_hierarchies(f"{label_name} index maps", bridge.index_volumes, scales, session)
    if index_model is not None:
        _hide_volumes(index_model)
        model.add([index_model])
    if bridge.segmentations:
        editable = Model(f"{label_name} editable segmentations", session)
        editable.add(bridge.segmentations)
        model.add([editable])
    else:
        _warning(
            session,
            "OME-Zarr label image '{}' declares no nonzero label IDs in colors or properties; "
            "opened its index map without editable ChimeraX segmentations.",
            label_path,
        )
    return model


def _associated_label_groups(source_group: zarr.Group):
    try:
        labels_group = source_group["labels"]
    except KeyError:
        return []
    if not isinstance(labels_group, zarr.Group):
        raise OMEZarrFormatError("The OME-Zarr 'labels' child must be a group.")
    labels_metadata = parse_labels_metadata(labels_group)
    groups = []
    for relative_path in labels_metadata.paths:
        try:
            label_group = labels_group[relative_path]
        except KeyError as error:
            raise OMEZarrFormatError(f"Registered label path '{relative_path}' does not exist.") from error
        if not isinstance(label_group, zarr.Group):
            raise OMEZarrFormatError(f"Registered label path '{relative_path}' is not a group.")
        groups.append((f"labels/{relative_path}", label_group))
    return groups


def _attach_labels(
    session,
    name,
    source_model,
    source_group,
    source_volumes,
    label_groups,
    scales,
    initial_step,
):
    source_metadata = parse_ome_zarr_metadata(source_group)
    label_models = []
    for label_path, label_group in label_groups:
        try:
            label_metadata = parse_ome_zarr_metadata(label_group)
            if label_metadata.image_label is None:
                raise OMEZarrFormatError(f"Registered label path '{label_path}' has no image-label metadata.")
        except OMEZarrFormatError as error:
            _warning(session, "Skipping OME-Zarr label image '{}': {}", label_path, error)
            continue

        try:
            label_model = _label_model(
                session,
                label_group,
                label_path,
                label_metadata,
                source_group,
                source_metadata,
                source_volumes,
                scales,
                initial_step,
            )
        except OMEZarrFormatError as error:
            _warning(
                session,
                "Could not associate OME-Zarr label image '{}' with its source: {} "
                "Opening a standalone, read-only index map instead.",
                label_path,
                error,
            )
            label_model = _standalone_label_model(session, label_group, scales, label_path, initial_step)
        label_models.append(label_model)

    if not label_models:
        return source_model
    dataset = Model(name, session)
    labels_container = Model("labels", session)
    labels_container.add(label_models)
    dataset.add([source_model, labels_container])
    return dataset


def _resolve_relative_group_path(group_path: str, relative_path: str) -> str:
    return posixpath.normpath(posixpath.join(group_path, relative_path))


def _open_source_with_label_groups(
    session,
    source_group,
    source_name,
    label_groups,
    scales,
    initial_step,
):
    source_model, source_volumes = _prepare_image_model(
        session,
        source_group,
        scales,
        source_name,
        initial_step,
    )
    return _attach_labels(
        session,
        source_name,
        source_model,
        source_group,
        source_volumes,
        label_groups,
        scales,
        initial_step,
    )


def _open_image_group(session, group, name, scales, initial_step, labels):
    source_model, source_volumes = _prepare_image_model(session, group, scales, name, initial_step)
    if not labels:
        return source_model
    try:
        label_groups = _associated_label_groups(group)
    except OMEZarrFormatError as error:
        _warning(session, "Could not read associated OME-Zarr labels: {}", error)
        label_groups = []
    return _attach_labels(
        session,
        name,
        source_model,
        group,
        source_volumes,
        label_groups,
        scales,
        initial_step,
    )


def _open_bioformats2raw_collection(session, group, name, scales, initial_step, labels):
    series_paths = bioformats2raw_series_paths(group)
    series_models = []
    multiple_series = len(series_paths) > 1
    for series_path in series_paths:
        series_name = f"{name} - {series_path}" if multiple_series else name
        series_models.append(
            _open_image_group(
                session,
                group[series_path],
                series_name,
                scales,
                initial_step,
                labels,
            ),
        )
    if not multiple_series:
        return series_models[0]
    collection = Model(name, session)
    collection.add(series_models)
    return collection


def _open_direct_label(
    session,
    label_group,
    label_path,
    filesystem,
    scales,
    initial_step,
):
    label_metadata = parse_ome_zarr_metadata(label_group)
    label_name = os.path.basename(label_path.rstrip("/")) or "label"
    if filesystem is None or label_metadata.image_label is None:
        _warning(
            session,
            "Could not resolve a source image for OME-Zarr label '{}'; opened a standalone index map.",
            label_path,
        )
        return _standalone_label_model(session, label_group, scales, label_name, initial_step)

    source_path = _resolve_relative_group_path(label_path, label_metadata.image_label.source_image)
    if source_path == posixpath.normpath(label_path):
        raise OMEZarrFormatError(f"OME-Zarr label '{label_path}' refers to itself as its source image.")
    try:
        source_group = _open_group(session, _store_from_filesystem(filesystem, source_path))
        if ome_zarr_group_kind(source_group) != "image":
            raise OMEZarrFormatError(f"Referenced source '{source_path}' is not an OME-Zarr image group.")
    except Exception as error:
        _warning(
            session,
            "Could not open source image '{}' for OME-Zarr label '{}': {}. Opened a standalone index map instead.",
            source_path,
            label_path,
            error,
        )
        return _standalone_label_model(session, label_group, scales, label_name, initial_step)

    source_name = os.path.basename(source_path.rstrip("/")) or "image"
    return _open_source_with_label_groups(
        session,
        source_group,
        source_name,
        [(label_path, label_group)],
        scales,
        initial_step,
    )


def _open_labels_collection(
    session,
    labels_group,
    labels_path,
    filesystem,
    scales,
    initial_step,
):
    labels_metadata = parse_labels_metadata(labels_group)
    models = []
    for relative_path in labels_metadata.paths:
        label_group = labels_group[relative_path]
        label_path = posixpath.join(labels_path, relative_path)
        models.append(
            _open_direct_label(
                session,
                label_group,
                label_path,
                filesystem,
                scales,
                initial_step,
            ),
        )
    if len(models) == 1:
        return models[0]
    container = Model(os.path.basename(labels_path.rstrip("/")) or "labels", session)
    container.add(models)
    return container


def _open(
    session: Session,
    root,
    scales: Optional[List[str]],
    full_name: str = "",
    name: str = "",
    initial_step: Tuple[int, int, int] = (4, 4, 4),
    labels: bool = True,
    filesystem: Optional[AbstractFileSystem] = None,
) -> Tuple[List[Model], str]:
    group = _open_group(session, root)
    kind = ome_zarr_group_kind(group)
    if kind == "image":
        model = _open_image_group(session, group, name, scales, initial_step, labels)
    elif kind == "bioformats2raw":
        model = _open_bioformats2raw_collection(
            session,
            group,
            name,
            scales,
            initial_step,
            labels,
        )
    elif kind == "image-label":
        model = _open_direct_label(session, group, full_name, filesystem, scales, initial_step)
    elif kind == "labels":
        model = _open_labels_collection(session, group, full_name, filesystem, scales, initial_step)
    else:
        raise OMEZarrFormatError("Opening this OME-Zarr non-image group is not supported.")

    show_volume_dialog(session)
    return [model], f"Opened {full_name}."


def open_ome_zarr(
    session,
    data: List[str],
    scales: List[str] = None,
    labels: bool = True,
) -> Tuple[List[Model], str]:
    """Open local or remote OME-Zarr images and their associated labels."""

    retm = []
    rets = []
    for location in data:
        filesystem, path = fsspec.core.url_to_fs(location)
        root = _store_from_filesystem(filesystem, path)
        name = os.path.basename(path.rstrip("/"))
        models, message = _open(
            session,
            root,
            scales,
            full_name=path,
            name=name,
            labels=labels,
            filesystem=filesystem,
        )
        retm.extend(models)
        rets.append(message)
    return retm, "\n".join(rets)


def open_ome_zarr_from_fs(
    session,
    fs: AbstractFileSystem,
    path: str,
    scales: List[str] = None,
    initial_step: Tuple[int, int, int] = (4, 4, 4),
    log: bool = True,
    labels: bool = True,
) -> Tuple[List[Model], str]:
    root = _store_from_filesystem(fs, path)
    if log:
        from chimerax.core.commands import log_equivalent_command

        proto = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
        label_option = "" if labels else " labels false"
        log_equivalent_command(session, f"open ngff:{proto}://{path}{label_option}")
    return _open(
        session,
        root,
        scales,
        full_name=path,
        name=os.path.basename(path.rstrip("/")),
        initial_step=initial_step,
        labels=labels,
        filesystem=fs,
    )


def open_ome_zarr_from_store(
    session,
    root: zarr.abc.store.Store,
    name: str,
    scales: List[str] = None,
    initial_step: Tuple[int, int, int] = (4, 4, 4),
    labels: bool = True,
) -> Tuple[List[Model], str]:
    return _open(
        session,
        root,
        scales,
        full_name=name,
        name=name,
        initial_step=initial_step,
        labels=labels,
    )

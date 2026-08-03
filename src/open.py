# vim: set expandtab shiftwidth=4 softtabstop=4:

import os
from typing import List, Tuple

import fsspec
import zarr
from chimerax.core.models import Model
from chimerax.core.session import Session
from chimerax.map.volume import show_volume_dialog
from fsspec import AbstractFileSystem

from .map_data.zarr_grid import ZarrModel


def _store_from_filesystem(fs: AbstractFileSystem, path: str):
    """Create a read-only Zarr-Python 3 store from an existing fsspec filesystem."""

    mapper = fs.get_mapper(path)
    return zarr.storage.FsspecStore.from_mapper(mapper, read_only=True)


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


def _open(
    session: Session,
    root: zarr.storage,
    scales: List[str],
    full_name: str = "",
    name: str = "",
    initial_step: Tuple[int, int, int] = (4, 4, 4),
) -> Tuple[List[Model], str]:
    if scales is not None:
        initial_step = (1, 1, 1)

    model = ZarrModel(name, session, root, scales, initial_step)

    # Check if we have time/channel volumes that need to be wrapped.
    volumes = list(model.child_models())
    if volumes and hasattr(volumes[0], "data"):
        time_indices = [volume.data.time for volume in volumes if volume.data.time is not None]
        channel_indices = [volume.data.channel for volume in volumes if volume.data.channel is not None]
        needs_wrapper = (
            len(time_indices) == len(volumes)
            and len(set(time_indices)) > 1
            or len(channel_indices) == len(volumes)
            and len(set(channel_indices)) > 1
        )

        if needs_wrapper:
            # Force initial rendering before reparenting to prevent KeyError issues
            # This follows ChimeraX's pattern in volume.py:3724-3727 where update_drawings()
            # is called to ensure surfaces/images are created before final model setup
            for volume in volumes:
                if volume.display:
                    volume.update_drawings()

            # Clean up volumes from VolumeUpdateManager after rendering completes
            # This prevents KeyError when volumes are reparented and their display state changes
            vm = getattr(session, "_volume_update_manager", None)
            if vm is not None:
                for volume in volumes:
                    # Use discard() to safely remove from tracking sets
                    vm._volumes_to_update.discard(volume)
                    vm._displayed_volumes_to_update.discard(volume)

            # Remove volumes from ZarrModel without deleting them
            # (The model was never added to session, so just detach the volumes)
            model.remove_drawings(volumes, delete=False)

            scale_groups = {}
            for volume in volumes:
                scale_groups.setdefault(getattr(volume.data, "scale_path", None), []).append(volume)

            if len(scale_groups) == 1:
                model = _wrap_time_and_channels(name, volumes, session)
            else:
                # Preserve one independent time/channel hierarchy per explicitly opened scale.
                wrappers = []
                ordered_scales = scales or list(scale_groups)
                for scale in ordered_scales:
                    scale_volumes = scale_groups.get(scale, [])
                    if scale_volumes:
                        wrappers.append(_wrap_time_and_channels(f"{name} - {scale}", scale_volumes, session))
                model.add(wrappers)

    show_volume_dialog(session)
    return [model], f"Opened {full_name}."


def open_ome_zarr(
    session,
    data: List[str],
    scales: List[str] = None,
) -> Tuple[List[Model], str]:
    """
    Open OME-Zarr files from a list of URLs. Will return one ZarrModel per URL, which has one or more Volumes as
    children.

    :param session: ChimeraX session
    :param data: the list of URLs to open
    :param scales: if provided, each scale will be opened as a separate child volume. If not provided, the multiscales
    will be opened as a single volume, accessible through the step setting in the Volume Viewer or the volume command.
    :return: List of opened models and a string message describing the operation
    """
    retm = []
    rets = []

    for d in data:
        fs, d = fsspec.core.url_to_fs(d)

        # The initial store to get sizes and units
        root = _store_from_filesystem(fs, d)
        name = os.path.basename(d)

        m, s = _open(session, root, scales, full_name=d, name=name)

        retm += m
        rets.append(s)

    return retm, "\n".join(rets)


def open_ome_zarr_from_fs(
    session,
    fs: AbstractFileSystem,
    path: str,
    scales: List[str] = None,
    initial_step: Tuple[int, int, int] = (4, 4, 4),
    log: bool = True,
) -> Tuple[List[Model], str]:
    root = _store_from_filesystem(fs, path)

    if log:
        from chimerax.core.commands import log_equivalent_command

        proto = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
        log_equivalent_command(session, f"open ngff:{proto}://{path}")

    return _open(
        session,
        root,
        scales,
        full_name=path,
        name=os.path.basename(path),
        initial_step=initial_step,
    )


def open_ome_zarr_from_store(
    session,
    root: zarr.abc.store.Store,
    name: str,
    scales: List[str] = None,
    initial_step: Tuple[int, int, int] = (4, 4, 4),
) -> Tuple[List[Model], str]:
    return _open(
        session,
        root,
        scales,
        full_name=name,
        name=name,
        initial_step=initial_step,
    )

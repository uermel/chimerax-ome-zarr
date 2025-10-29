# vim: set expandtab shiftwidth=4 softtabstop=4:

import os
from typing import List, Tuple

import fsspec
import zarr
import zarr.attrs
from chimerax.core.models import Model
from chimerax.core.session import Session
from chimerax.map.volume import show_volume_dialog
from fsspec import AbstractFileSystem

from .map_data.zarr_grid import ZarrModel


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

    # Check if we have time/channel volumes that need to be wrapped
    volumes = list(model.child_models())
    if len(volumes) > 0 and hasattr(volumes[0], "data"):
        # Check for time series and/or multichannel data
        time_indices = [v.data.time for v in volumes if hasattr(v.data, "time") and v.data.time is not None]
        channel_indices = [v.data.channel for v in volumes if hasattr(v.data, "channel") and v.data.channel is not None]

        is_time_series = len(time_indices) == len(volumes) and len(set(time_indices)) > 1
        is_multichannel = len(channel_indices) == len(volumes) and len(set(channel_indices)) > 1

        if is_time_series or is_multichannel:
            # Remove volumes from ZarrModel
            session.models.close([model])

            # Create appropriate wrapper model
            if is_time_series and is_multichannel:
                # Both time and channel dimensions
                from chimerax.map.volume import MultiChannelSeries
                from chimerax.map_series import MapSeries

                # Group volumes by channel
                channel_groups = {}
                for v in volumes:
                    c = v.data.channel
                    if c not in channel_groups:
                        channel_groups[c] = []
                    channel_groups[c].append(v)

                # Create MapSeries for each channel
                map_series = []
                for c in sorted(channel_groups.keys()):
                    ms = MapSeries(f"{name} channel {c}", channel_groups[c], session)
                    map_series.append(ms)

                # Create MultiChannelSeries
                model = MultiChannelSeries(name, map_series, session)

            elif is_time_series:
                # Time dimension only
                from chimerax.map_series import MapSeries

                model = MapSeries(name, volumes, session)

            elif is_multichannel:
                # Channel dimension only
                from chimerax.map.volume import MapChannelsModel

                model = MapChannelsModel(name, volumes, session)
                # Show first 3 channels by default (user requested all channels, but let's be conservative)
                model.show_n_channels(len(volumes))  # Show all channels as requested

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
        root = zarr.storage.FSStore(d, key_separator="/", mode="r", dimension_separator="/", fs=fs)
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
    root = zarr.storage.FSStore(path, key_separator="/", mode="r", dimension_separator="/", fs=fs)

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
    root: zarr.storage,
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

# vim: set expandtab shiftwidth=4 softtabstop=4:

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import zarr
from chimerax.core.models import Model
from chimerax.core.session import Session
from chimerax.map.volume import Volume
from chimerax.map_data import GridData
from zarr.core import Array

from .constants import UNITFACTOR


@dataclass
class Axis:
    """OME-Zarr axis metadata."""

    name: str
    unit: Optional[str] = "angstrom"
    type: Optional[Union[Literal["space"], Literal["time"], Literal["channel"]]] = "space"


@dataclass
class VectorScaleTransform:
    """OME-Zarr scale or translation transformation metadata."""

    scale: Optional[List[float]] = None
    translation: Optional[List[float]] = None
    type: Union[Literal["scale"], Literal["translation"], Literal["identity"]] = "scale"


@dataclass
class MultiscaleDataset:
    """OME-Zarr dataset metadata."""

    path: str
    coordinateTransformations: List[VectorScaleTransform]


@dataclass
class Multiscales:
    """OME-Zarr multiscales metadata."""

    axes: List[Axis]
    datasets: List[MultiscaleDataset]


@dataclass
class OmeroChannelWindow:
    """OMERO channel window (display range) metadata."""

    start: float
    end: float
    min: float
    max: float


@dataclass
class OmeroChannel:
    """OMERO channel metadata."""

    label: str
    color: str  # Hex color string like "FFFFFF"
    active: bool = True
    coefficient: float = 1.0
    family: str = "linear"
    inverted: bool = False
    window: Optional[OmeroChannelWindow] = None


@dataclass
class OmeroMetadata:
    """OMERO metadata from OME-Zarr."""

    channels: List[OmeroChannel]
    version: Optional[str] = None
    id: Optional[int] = None
    name: Optional[str] = None


def get_unit_factor(ms: Multiscales) -> Tuple[float, float, float]:
    """Get a multiplication factor that converts scaling information from OME-Zarr header to angstrom."""
    zunit = UNITFACTOR.get(ms.axes[0].unit, "angstrom")
    yunit = UNITFACTOR.get(ms.axes[1].unit, "angstrom")
    xunit = UNITFACTOR.get(ms.axes[2].unit, "angstrom")

    return (zunit, yunit, xunit)


def get_pixelsize(ms: Multiscales) -> List[Tuple[float, float, float]]:
    """Get the pixel sizes in the OME-Zarr header in units specified by the axes metadata."""
    sizes = []

    datasets = ms.datasets
    for ds in datasets:
        zs = ds.coordinateTransformations[0].scale[0]
        ys = ds.coordinateTransformations[0].scale[1]
        xs = ds.coordinateTransformations[0].scale[2]

        sizes.append((zs, ys, xs))

    return sizes


def get_spatial_axes_indices(axes: List[Axis]) -> List[int]:
    """Get the indices of spatial axes in the axis list."""
    return [i for i, a in enumerate(axes) if a.type == "space"]


def get_unit_factor_spatial(ms: Multiscales) -> Tuple[float, float, float]:
    """Get multiplication factor for spatial axes only, converting to angstrom."""
    spatial_axes = [a for a in ms.axes if a.type == "space"]
    if len(spatial_axes) != 3:
        raise ValueError(f"Expected 3 spatial axes, got {len(spatial_axes)}")

    zunit = UNITFACTOR.get(spatial_axes[0].unit, 1.0)
    yunit = UNITFACTOR.get(spatial_axes[1].unit, 1.0)
    xunit = UNITFACTOR.get(spatial_axes[2].unit, 1.0)

    return (zunit, yunit, xunit)


def get_pixelsize_spatial(ms: Multiscales) -> List[Tuple[float, float, float]]:
    """Get pixel sizes for spatial dimensions only."""
    spatial_indices = get_spatial_axes_indices(ms.axes)
    if len(spatial_indices) != 3:
        raise ValueError(f"Expected 3 spatial axes, got {len(spatial_indices)}")

    sizes = []
    for ds in ms.datasets:
        spatial_scale = tuple(ds.coordinateTransformations[0].scale[i] for i in spatial_indices)
        sizes.append(spatial_scale)

    return sizes


def parse_multiscales(zattrs: zarr.attrs.Attributes) -> Union[Multiscales, None]:
    """Parse multiscales metadata from OME-Zarr header."""
    if "multiscales" not in zattrs:
        return None

    ms = zattrs["multiscales"][0]

    axes = []
    for a in ms["axes"]:
        axes.append(Axis(**a))

    datasets = []
    for ds in ms["datasets"]:
        cts = []
        for ct in ds["coordinateTransformations"]:
            cts.append(VectorScaleTransform(**ct))
        datasets.append(MultiscaleDataset(ds["path"], cts))

    return Multiscales(axes, datasets)


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Convert hex color string to RGBA tuple with values 0-1.

    Args:
        hex_color: Hex color string like "FFFFFF" or "#FFFFFF"
        alpha: Alpha value (0-1), default 1.0

    Returns:
        Tuple of (r, g, b, a) with values 0-1
    """
    # Remove # if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0-255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Normalize to 0-1
    return (r / 255.0, g / 255.0, b / 255.0, alpha)


def parse_omero(zattrs: zarr.attrs.Attributes) -> Optional[OmeroMetadata]:
    """
    Parse OMERO metadata from OME-Zarr header.

    Returns None if OMERO metadata is not present.
    """
    if "omero" not in zattrs:
        return None

    omero = zattrs["omero"]

    # Parse channels
    channels = []
    if "channels" in omero:
        for ch in omero["channels"]:
            # Parse window if present
            window = None
            if "window" in ch:
                window = OmeroChannelWindow(**ch["window"])

            # Create channel with required and optional fields
            channel = OmeroChannel(
                label=ch.get("label", ""),
                color=ch.get("color", "FFFFFF"),
                active=ch.get("active", True),
                coefficient=ch.get("coefficient", 1.0),
                family=ch.get("family", "linear"),
                inverted=ch.get("inverted", False),
                window=window,
            )
            channels.append(channel)

    return OmeroMetadata(
        channels=channels,
        version=omero.get("version"),
        id=omero.get("id"),
        name=omero.get("name"),
    )


def parse_labels(zattrs: zarr.attrs.Attributes, session: Session) -> None:
    """Parse labels metadata from OME-Zarr header."""
    if "labels" not in zattrs:
        return None

    session.logger.warning("Labels not implemented yet.")
    return None


class ZarrGrid3DSlice(GridData):
    """
    A GridData object that represents a 3D slice of a 4D or 5D Zarr array.
    Used for time series and/or multi-channel data where each time point and channel
    is represented as a separate 3D volume.

    The parent array is assumed to be in OME-Zarr order: (T, C, Z, Y, X) where T and C
    are optional. This class fixes the T and/or C indices and provides a 3D view of the
    spatial dimensions (Z, Y, X).
    """

    def __init__(
        self,
        array: Array,
        time_index: Optional[int] = None,
        channel_index: Optional[int] = None,
        origin: Tuple[float, float, float] = (0, 0, 0),
        step: Tuple[float, float, float] = (1, 1, 1),
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ):
        self.data = array
        self.time_index = time_index
        self.channel_index = channel_index

        # Determine how many leading dimensions are time/channel
        n_leading_dims = 0
        if time_index is not None:
            n_leading_dims += 1
        if channel_index is not None:
            n_leading_dims += 1

        # Extract spatial shape from the last 3 dimensions (ZYX in OME-Zarr)
        # and reverse to XYZ for ChimeraX
        spatial_shape = array.shape[-3:][::-1]
        origin = origin[::-1]
        step = step[::-1]

        GridData.__init__(
            self,
            spatial_shape,
            self.data.dtype,
            origin,
            step,
            path=path,
            file_type=file_type,
            name=name,
        )

    def read_matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Tuple[int, ...] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
    ):
        # Maximum spatial size
        sz = self.size[::-1]  # XYZ to ZYX

        # Limit origin to an index inside the grid
        ijk_origin = ijk_origin[::-1]  # XYZ to ZYX
        ijk_origin = [min(sz[i] - 1, ijk_origin[i]) for i in range(3)]

        # Invert step
        ijk_step = ijk_step[::-1]  # XYZ to ZYX

        if ijk_size is None:
            ijk_size = sz
        else:
            ijk_size = ijk_size[::-1]  # XYZ to ZYX
            # Limit the max coord to the grid size
            ijk_size = [min(sz[i], ijk_origin[i] + ijk_size[i]) for i in range(3)]

        # Build the slice indices: (time, channel, z, y, x)
        slices = []

        # Add fixed time index if present
        if self.time_index is not None:
            slices.append(self.time_index)

        # Add fixed channel index if present
        if self.channel_index is not None:
            slices.append(self.channel_index)

        # Add spatial slices (ZYX)
        slices.extend(
            [
                slice(ijk_origin[0], ijk_size[0], ijk_step[0]),
                slice(ijk_origin[1], ijk_size[1], ijk_step[1]),
                slice(ijk_origin[2], ijk_size[2], ijk_step[2]),
            ],
        )

        m = self.data[tuple(slices)]

        # Handle type conversions
        from numpy import float16, float32, uint64

        if m.dtype == float16:
            m = m.astype(float32)

        if m.dtype == uint64:
            m = m.astype(float32)

        return m


class ZarrModel(Model):
    """
    ZarrModel encapsulates an OME-Zarr file. There are two modes of loading the multiscale data:
    1. Load all scales in a single Volume powered by WrappedZarrGrid. This is the default behavior when the `scales`
       argument is `None`. The user can switch between scales using the typical ChimeraX "step" behavior (e.g. in the
       Volume Viewer widget or with `volume #X step N`).
    2. Load selected scales as separate Volumes powered by ZarrGrid that will be children of this Model object.
       This is the behavior when the `scales` argument is a list of strings. The strings should be the paths to the
       scale array roots in the OME-Zarr file (typically ['0', '1', '2', ...].

    The images are loaded lazily, i.e. only the chunks around the visible region are loaded into memory. All data is
    cached in memory using Zarr's LRUStoreCache. The cache size is unlimited.

    :param name: The name of the model.
    :param session: The ChimeraX session.
    :param root: A ZarrStore of any kind.
    :param scales: A list of scales to load. If `None`, all scales will be loaded.
    :param initial_step: The initial step size displayed. Default is (1, 1, 1).
    """

    def __init__(
        self,
        name: str,
        session,
        root: zarr.storage,
        scales: Optional[List[str]] = None,
        initial_step: Tuple[int, ...] = (1, 1, 1),
    ) -> None:
        Model.__init__(self, name, session)

        group = zarr.open(root, mode="r")
        attrs = group.attrs

        # Multiscales
        mlt = parse_multiscales(attrs)

        # Detect time and channel axes
        axes_types = [a.type for a in mlt.axes]
        has_time = "time" in axes_types
        has_channel = "channel" in axes_types

        # Validate that we have exactly 3 spatial axes
        spatial_axes = [a for a in mlt.axes if a.type == "space"]
        if len(spatial_axes) != 3:
            raise ValueError(f"Expected 3 spatial axes, got {len(spatial_axes)}")

        # Check for unknown axis types
        known_types = {"space", "time", "channel"}
        unknown_types = set(axes_types) - known_types
        if unknown_types:
            raise ValueError(f"Unknown axis types: {unknown_types}")

        self.avail_scales = [d.path for d in mlt.datasets]

        if scales is not None:
            for s in scales:
                if s not in self.avail_scales:
                    raise ValueError(f"Scale {s} not available in file.")

        # Labels (only to warn about ignoring them for now)
        _ = parse_labels(attrs, session)

        # OMERO metadata (optional)
        omero = parse_omero(attrs)
        self.omero = omero

        # No multiscales, return
        if mlt is None:
            return

        # Get pixelsizes in Angstrom from unit and scale transformations
        # Use spatial-only functions if we have time/channel axes
        if has_time or has_channel:
            ufacs = get_unit_factor_spatial(mlt)
            sizes = get_pixelsize_spatial(mlt)
        else:
            ufacs = get_unit_factor(mlt)
            sizes = get_pixelsize(mlt)
        sizes = [(ufacs[0] * s[0], ufacs[1] * s[1], ufacs[2] * s[2]) for s in sizes]

        # The cached store, group and arrays
        root_cached = zarr.LRUStoreCache(
            root,
            max_size=None,
        )
        group_cached = zarr.open(root_cached, mode="r")
        arrays_cached = list(group_cached.arrays())
        arrays_cached = [a for _, a in arrays_cached]

        arrays_datasets_sizes = list(zip(arrays_cached, mlt.datasets, sizes, strict=True))

        # Sort arrays by size for quicker loading
        self.arrays_datasets_sizes = sorted(arrays_datasets_sizes, key=lambda x: x[0].nbytes, reverse=False)

        # If no scales requested, load all scales async
        if not scales:
            if initial_step is None:
                initial_step = (4, 4, 4)

            # Handle time/channel dimensions
            if has_time or has_channel:
                # Get time and channel axis indices (OME-Zarr order: TCZYX)
                time_axis_idx = axes_types.index("time") if has_time else None
                channel_axis_idx = axes_types.index("channel") if has_channel else None

                # Get array with most detail (finest resolution)
                first_array = self.arrays_datasets_sizes[-1][0]

                # Determine dimensions
                n_time = first_array.shape[time_axis_idx] if has_time else 1
                n_channel = first_array.shape[channel_axis_idx] if has_channel else 1

                # Create one WrappedZarrGrid per time/channel combination
                volumes = []
                for t in range(n_time):
                    for c in range(n_channel):
                        # Create grids for this time/channel across all scales
                        tc_grids = []
                        tc_sizes = []
                        for array, _, size in self.arrays_datasets_sizes:
                            grid = ZarrGrid3DSlice(
                                array,
                                time_index=t if has_time else None,
                                channel_index=c if has_channel else None,
                                step=size,
                                name=f"{name} t={t} c={c}",
                            )
                            tc_grids.append(grid)
                            tc_sizes.append(size)

                        # Create WrappedZarrGrid for this time/channel
                        dgd = WrappedZarrGrid(grids=tc_grids, name=f"{name} t={t} c={c}")

                        # Set time and channel metadata
                        if has_time:
                            dgd.time = t
                        if has_channel:
                            dgd.channel = c

                        # Start slice in the middle of the volume
                        ijk_min = (0, 0, dgd.size[2] // 2)
                        ijk_max = (dgd.size[0], dgd.size[1], dgd.size[2] // 2)
                        ijk_step = initial_step

                        vol = Volume(session, dgd, region=(ijk_min, ijk_max, ijk_step))
                        vol.set_display_style("image")

                        # Adjust rendering limit (see comment below about 16 MVoxel limit)
                        vol.new_region(vol.region[0], vol.region[1], vol.region[2], adjust_step=False)

                        # Apply OMERO metadata if available
                        if omero and has_channel and c < len(omero.channels):
                            ch_meta = omero.channels[c]

                            # Apply channel label to volume name
                            if ch_meta.label:
                                vol_name = f"{name} - {ch_meta.label}"
                                if has_time:
                                    vol_name += f" t={t}"
                                vol.name = vol_name

                            # Apply channel color
                            try:
                                rgba = hex_to_rgba(ch_meta.color)
                                vol.set_parameters(default_rgba=rgba)
                            except (ValueError, IndexError):
                                # If color parsing fails, use default
                                pass

                            # Set initial display based on OMERO active flag
                            vol.display = t == 0 and ch_meta.active
                        else:
                            # Set initial display: show all channels at t=0, hide others
                            vol.display = t == 0

                        volumes.append(vol)

                self.add(volumes)

            else:
                # Original 3D-only behavior
                arrays = [a for a, _, _ in self.arrays_datasets_sizes]
                sizes = [sz for _, _, sz in self.arrays_datasets_sizes]
                dgd = WrappedZarrGrid(arrays, steps=sizes, name=f"{name}")

                # Start slice in the middle of the volume
                ijk_min = (0, 0, dgd.size[2] // 2)
                ijk_max = (
                    dgd.size[0],
                    dgd.size[1],
                    dgd.size[2] // 2,
                )
                ijk_step = initial_step
                vol = Volume(session, dgd, region=(ijk_min, ijk_max, ijk_step))
                vol.set_display_style("image")

                # ChimeraX has an upper limit of 16 MVoxel for rendered voxels. This limit is set in the rendering_options
                # of the Volume. If this is too high, ChimeraX will automatically show the volume at full resolution, i.e.
                # step = (1,1,1). This is not ideal when we're streaming the data from a remote source on demand.
                #
                # To avoid that, we need to make sure that the limit is adjusted according to the current region. This will
                # prevent moving the slider in the volume viewer to change the step size upon first move.
                # This is how to do it:
                vol.new_region(vol.region[0], vol.region[1], vol.region[2], adjust_step=False)
                self.add([vol])

        else:
            # Load only requested scales
            if initial_step is None:
                initial_step = (1, 1, 1)

            self.arrays_datasets_sizes = [a for a in self.arrays_datasets_sizes if a[1].path in scales]

            # Handle time/channel dimensions
            if has_time or has_channel:
                # Get time and channel axis indices (OME-Zarr order: TCZYX)
                time_axis_idx = axes_types.index("time") if has_time else None
                channel_axis_idx = axes_types.index("channel") if has_channel else None

                # Get array with most detail (finest resolution in requested scales)
                first_array = self.arrays_datasets_sizes[-1][0]

                # Determine dimensions
                n_time = first_array.shape[time_axis_idx] if has_time else 1
                n_channel = first_array.shape[channel_axis_idx] if has_channel else 1

                # Create grids for each time/channel/scale combination
                volumes = []
                for t in range(n_time):
                    for c in range(n_channel):
                        for array, dataset, size in self.arrays_datasets_sizes:
                            dgd = ZarrGrid3DSlice(
                                array,
                                time_index=t if has_time else None,
                                channel_index=c if has_channel else None,
                                step=size,
                                name=f"{name} - {dataset.path} t={t} c={c}",
                            )

                            # Set time and channel metadata
                            if has_time:
                                dgd.time = t
                            if has_channel:
                                dgd.channel = c

                            # Start slice in the middle of the volume
                            ijk_min = (0, 0, dgd.size[2] // 2)
                            ijk_max = (dgd.size[0], dgd.size[1], dgd.size[2] // 2)
                            ijk_step = initial_step

                            vol = Volume(session, dgd, (ijk_min, ijk_max, ijk_step))
                            vol.set_display_style("image")

                            # See explanation above about rendering limit
                            vol.new_region(vol.region[0], vol.region[1], vol.region[2], adjust_step=False)

                            # Apply OMERO metadata if available
                            if omero and has_channel and c < len(omero.channels):
                                ch_meta = omero.channels[c]

                                # Apply channel label to volume name
                                if ch_meta.label:
                                    vol_name = f"{name} - {dataset.path} - {ch_meta.label}"
                                    if has_time:
                                        vol_name += f" t={t}"
                                    vol.name = vol_name

                                # Apply channel color
                                try:
                                    rgba = hex_to_rgba(ch_meta.color)
                                    vol.set_parameters(default_rgba=rgba)
                                except (ValueError, IndexError):
                                    # If color parsing fails, use default
                                    pass

                                # Set initial display based on OMERO active flag
                                vol.display = t == 0 and ch_meta.active
                            else:
                                # Set initial display: show all channels at t=0, hide others
                                vol.display = t == 0

                            volumes.append(vol)

                self.add(volumes)

            else:
                # Original 3D-only behavior
                for array, dataset, size in self.arrays_datasets_sizes:
                    dgd = ZarrGrid(array, step=size, name=f"{name} - {dataset.path}")

                    # Start slice in the middle of the volume
                    ijk_min = (0, 0, dgd.size[2] // 2)
                    ijk_max = (
                        dgd.size[0],
                        dgd.size[1],
                        dgd.size[2] // 2,
                    )
                    ijk_step = initial_step
                    vol = Volume(session, dgd, (ijk_min, ijk_max, ijk_step))
                    vol.set_display_style("image")
                    # See explanation above
                    vol.new_region(vol.region[0], vol.region[1], vol.region[2], adjust_step=False)
                    self.add([vol])

    @property
    def scales(self):
        return self.avail_scales

    def open_scales(self, scales: List[str]):
        """Load additional scales."""
        raise NotImplementedError("Not implemented yet.")


class ZarrGrid(GridData):
    """
    A GridData object that wraps a Zarr array. Assumes ZYX axis ordering, as defined in the OME-Zarr specification.
    """

    def __init__(
        self,
        array: Array,
        origin: Tuple[float, float, float] = (0, 0, 0),
        step: Tuple[float, float, float] = (1, 1, 1),
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ):
        self.data = array

        shape = self.data.shape[::-1]
        origin = origin[::-1]
        step = step[::-1]

        GridData.__init__(
            self,
            shape,
            self.data.dtype,
            origin,
            step,
            path=path,
            file_type=file_type,
            name=name,
        )

    def read_matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Tuple[int, ...] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
    ):
        # Maximum size
        sz = self.size[::-1]

        # Limit origin to an index inside the grid
        ijk_origin = ijk_origin[::-1]
        ijk_origin = [min(sz[i] - 1, ijk_origin[i]) for i in range(3)]

        # Invert step
        ijk_step = ijk_step[::-1]

        if ijk_size is None:
            ijk_size = sz
        else:
            ijk_size = ijk_size[::-1]
            # Limit the max coord to the grid size
            ijk_size = [min(sz[i], ijk_origin[i] + ijk_size[i]) for i in range(3)]

        m = self.data[
            ijk_origin[0] : ijk_size[0] : ijk_step[0],
            ijk_origin[1] : ijk_size[1] : ijk_step[1],
            ijk_origin[2] : ijk_size[2] : ijk_step[2],
        ]

        from numpy import float16, float32, uint64

        if m.dtype == float16:
            m = m.astype(float32)

        if m.dtype == uint64:
            m = m.astype(float32)

        return m


class WrappedZarrGrid(GridData):
    """
    A GridData object that wraps multiple ZarrGrids at different resolutions and automatically redirects any read_matrix
    calls to the lowest resolution grid that can support the requested step size. This is useful for streaming data from
    remote multiscale OME-Zarr files.
    """

    def __init__(
        self,
        arrays: List[Array] = None,
        origins: List[Tuple[float, float, float]] = None,
        steps: List[Tuple[float, float, float]] = None,
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
        grids: List[GridData] = None,
    ) -> None:
        # If grids are provided, use them directly instead of creating from arrays
        if grids is not None:
            # Use the finest resolution grid for initialization
            finest_grid = grids[-1]

            GridData.__init__(
                self,
                finest_grid.size,
                finest_grid.value_type,
                finest_grid.origin,
                finest_grid.step,
                path=finest_grid.path,
                file_type=finest_grid.file_type,
                name=name,
            )

            self.arrays = None
            self.grids = grids

            # Calculate relative step sizes from the provided grids
            self._rel_step_sizes: List[Tuple[int, ...]] = []
            base_step = finest_grid.step
            for g in grids:
                relstep = (g.step[0] / base_step[0], g.step[1] / base_step[1], g.step[2] / base_step[2])

                if not np.allclose(relstep, [int(s) for s in relstep]):
                    raise NotImplementedError(
                        f"Non-integer scaling levels are not supported. Relative steps determined: {relstep}",
                    )

                self._rel_step_sizes.append((int(relstep[0]), int(relstep[1]), int(relstep[2])))

            # Precompute sampling strategies for isotropic steps (1, 1, 1) - (16, 16, 16)
            self._strats: Dict[Tuple[int, ...], Tuple[GridData, Tuple[int, ...], Tuple[int, ...]]] = {
                (s, s, s): self.get_sampling_strategy((s, s, s)) for s in range(1, 17)
            }

            return

        # Original behavior: create grids from arrays
        # Default origins and steps
        if origins is None:
            origins = [(0, 0, 0) for _ in range(len(arrays))]
        if steps is None:
            steps = [(1, 1, 1) for _ in range(len(arrays))]

        # Relative transformation between grids
        self._rel_step_sizes: List[Tuple[int, ...]] = []
        base_step = steps[-1]
        for s in steps:
            relstep = (s[0] / base_step[0], s[1] / base_step[1], s[2] / base_step[2])

            # if not np.allclose(relstep, relstep[0]):
            #     raise NotImplementedError(
            #         f"""Anisotropically scaled input data is not supported. Finest step: {base_step}, current step: {s},
            #         relative step: {relstep}""",
            #     )

            if not np.allclose(relstep, [int(s) for s in relstep]):
                raise NotImplementedError(
                    f"Non-integer scaling levels are not supported. Relative steps determined: {relstep}",
                )

            self._rel_step_sizes.append((int(relstep[0]), int(relstep[1]), int(relstep[2])))

        # Init as GridData at highest resolution
        shape = arrays[-1].shape[::-1]
        origin = origins[-1][::-1]
        step = steps[-1][::-1]

        GridData.__init__(
            self,
            shape,
            arrays[-1].dtype,
            origin,
            step,
            path=path,
            file_type=file_type,
            name=name,
        )

        # Init subgrids
        self.arrays = arrays
        self.grids: List[ZarrGrid] = []
        """Storage for the ZarrGrids at different resolutions."""

        for i, array in enumerate(arrays):
            origin = origins[i][::-1]
            step = steps[i][::-1]
            self.grids.append(
                ZarrGrid(array=array, origin=origin, step=step, file_type=file_type, path=path, name=name),
            )

        # Precompute sampling strategies for isotropic steps (1, 1, 1) - (16, 16, 16) (defaults in volume viewer)
        self._strats: Dict[Tuple[int, ...], Tuple[ZarrGrid, Tuple[int, ...], Tuple[int, ...]]] = {
            (s, s, s): self.get_sampling_strategy((s, s, s)) for s in range(1, 17)
        }

    def get_sampling_strategy(
        self,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
    ) -> Tuple[ZarrGrid, Tuple[int, ...], Tuple[int, ...]]:
        """Return the grid and step size to use for the given step size."""

        # Grid needs to be fine enough to support the finest requested step
        minstep = min(ijk_step)

        if not all(s % minstep == 0 for s in ijk_step):
            raise ValueError(
                f"When step sizes are anisotropic, they must be multiples of the smallest step (steps: {ijk_step}.",
            )

        # Find the closest available step size and adjust the step size to that grid
        # Start with the coarsest grid
        finest_step = self._rel_step_sizes[0]
        grid_idx = 0
        for i, step in enumerate(self._rel_step_sizes):
            # The grid is fine enough for minstep, but as coarse as possible, and minstep is evenly divisible by the
            # step
            if all(minstep >= s for s in step) and all(minstep % s == 0 for s in step):
                finest_step = step
                grid_idx = i
                break

        # scale_factors = tuple(int(ijks / fs) for ijks, fs in zip(ijk_step, finest_step, strict=True))
        ijk_step_out = tuple(int(ijks / fs) for ijks, fs in zip(ijk_step, finest_step, strict=True))

        # Return the grid, the adjusted step size and the factors to divide size/origin by
        return self.grids[grid_idx], ijk_step_out, finest_step

    def read_matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Tuple[int, ...] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
    ):
        ijk_size = (
            ijk_step[0] if ijk_size[0] < ijk_step[0] else ijk_size[0],
            ijk_step[1] if ijk_size[1] < ijk_step[1] else ijk_size[1],
            ijk_step[2] if ijk_size[2] < ijk_step[2] else ijk_size[2],
        )

        # Precomputed strats for isotropic steps
        grid, ijk_step, facts = self._strats.get(tuple(ijk_step), self.get_sampling_strategy(ijk_step))
        ijk_origin = tuple(o // f for o, f in zip(ijk_origin, facts, strict=True))
        if ijk_size:
            ijk_size = tuple(s // f for s, f in zip(ijk_size, facts, strict=True))

        return grid.read_matrix(ijk_origin, ijk_size, ijk_step, progress)

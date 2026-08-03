# vim: set expandtab shiftwidth=4 softtabstop=4:

from contextlib import suppress
from numbers import Integral
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from chimerax.core.models import Model
from chimerax.map.volume import Volume, set_data_cache
from chimerax.map_data import GridData

from .ome_metadata import (
    Axis,
    OmeroMetadata,
    OMEZarrFormatError,
    parse_ome_zarr_metadata,
    spatial_transform_angstrom,
)
from .store_cache import cached_group
from .temporal_cache import decoded_matrix_bytes, session_temporal_read_ahead

MAX_DECODED_SLAB_BYTES = 4 * 1024**3


def get_spatial_axes_indices(axes: Sequence[Axis]) -> List[int]:
    """Return indices of spatial axes (kept for compatibility with earlier releases)."""

    return [i for i, axis in enumerate(axes) if axis.type == "space"]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert a six-digit RGB hex string to normalized RGBA."""

    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit RGB color, got {hex_color!r}.")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return red / 255.0, green / 255.0, blue / 255.0, alpha


def _supported_matrix_type(matrix):
    """Convert Zarr dtypes that ChimeraX cannot render directly."""

    from numpy import float16, float32, uint64

    if matrix.dtype in (float16, uint64):
        return matrix.astype(float32)
    return matrix


class ZarrGridSlice(GridData):
    """A lazy ChimeraX 3D grid view of a 2D or 3D OME-Zarr image slice."""

    def __init__(
        self,
        array: zarr.Array,
        fixed_indices: Sequence[int] = (),
        spatial_ndim: int = 3,
        origin: Optional[Tuple[float, ...]] = None,
        step: Optional[Tuple[float, ...]] = None,
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
        time_index: Optional[int] = None,
        channel_index: Optional[int] = None,
        scale_path: Optional[str] = None,
        scale_index: Optional[int] = None,
    ) -> None:
        if spatial_ndim not in (2, 3):
            raise ValueError(f"Expected two or three spatial dimensions, got {spatial_ndim}.")
        if array.ndim != len(fixed_indices) + spatial_ndim:
            raise ValueError(
                f"Array has {array.ndim} dimensions, but {len(fixed_indices)} fixed and {spatial_ndim} spatial "
                "dimensions were specified.",
            )

        self.data = array
        self.fixed_indices = tuple(fixed_indices)
        self.spatial_ndim = spatial_ndim
        self.scale_path = scale_path
        self.scale_index = scale_index
        self._temporal_read_ahead = None

        spatial_shape = tuple(array.shape[-spatial_ndim:])
        spatial_origin = tuple(origin if origin is not None else (0.0,) * spatial_ndim)
        spatial_step = tuple(step if step is not None else (1.0,) * spatial_ndim)
        if len(spatial_origin) != spatial_ndim or len(spatial_step) != spatial_ndim:
            raise ValueError("Origin and step dimensionality must match the spatial dimensionality.")

        if spatial_ndim == 3:
            size_xyz = spatial_shape[::-1]
            origin_xyz = spatial_origin[::-1]
            step_xyz = spatial_step[::-1]
        else:
            size_y, size_x = spatial_shape
            origin_y, origin_x = spatial_origin
            step_y, step_x = spatial_step
            size_xyz = (size_x, size_y, 1)
            origin_xyz = (origin_x, origin_y, 0.0)
            step_xyz = (step_x, step_y, 1.0)

        spatial_chunks = tuple(int(value) for value in array.chunks[-spatial_ndim:])
        self.chunk_size = spatial_chunks[::-1] if spatial_ndim == 3 else (spatial_chunks[1], spatial_chunks[0], 1)

        GridData.__init__(
            self,
            size_xyz,
            array.dtype,
            origin_xyz,
            step_xyz,
            path=path,
            file_type=file_type,
            name=name,
            time=time_index,
            channel=channel_index,
        )

    def _decoded_slab_limit(self) -> int:
        cache_size = getattr(self.data_cache, "size", 0)
        return min(MAX_DECODED_SLAB_BYTES, int(cache_size // 4))

    def set_temporal_read_ahead(self, sequence, index: int) -> None:
        """Associate this resolution grid with its time-adjacent grids."""

        self._temporal_read_ahead = (sequence, index)

    def _is_temporal_volume_request(self, ijk_size, ijk_step) -> bool:
        if self.spatial_ndim != 3 or self._temporal_read_ahead is None:
            return False
        sampled_size = tuple((size + step - 1) // step for size, step in zip(ijk_size, ijk_step, strict=True))
        return all(size > 1 for size in sampled_size)

    def _consume_temporal_read_ahead(self, ijk_origin, ijk_size, ijk_step):
        if not self._is_temporal_volume_request(ijk_size, ijk_step):
            return None
        sequence, index = self._temporal_read_ahead
        return sequence.consume(index, ijk_origin, ijk_size, ijk_step)

    def _observe_temporal_request(self, matrix, ijk_origin, ijk_size, ijk_step) -> None:
        if matrix is None or not self._is_temporal_volume_request(ijk_size, ijk_step):
            return
        sequence, index = self._temporal_read_ahead
        sequence.observe(
            index,
            ijk_origin,
            ijk_size,
            ijk_step,
            decoded_matrix_bytes(ijk_size, ijk_step, matrix.dtype.itemsize),
        )

    def _bounded_request(
        self,
        ijk_origin: Tuple[int, ...],
        ijk_size: Optional[Tuple[int, ...]],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Clamp a ChimeraX request to the grid before using it as a cache region."""

        if len(ijk_origin) != 3 or (ijk_size is not None and len(ijk_size) != 3):
            raise ValueError("Grid origins and sizes must contain exactly three values.")
        bounded_origin = tuple(
            max(0, min(axis_size - 1, origin)) for origin, axis_size in zip(ijk_origin, self.size, strict=True)
        )
        requested_size = self.size if ijk_size is None else ijk_size
        bounded_size = tuple(
            max(0, min(size, axis_size - origin))
            for origin, size, axis_size in zip(bounded_origin, requested_size, self.size, strict=True)
        )
        return bounded_origin, bounded_size

    def _chunk_aligned_plane_region(
        self,
        ijk_origin: Tuple[int, ...],
        ijk_size: Tuple[int, ...],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        expanded_origin = list(ijk_origin)
        expanded_size = list(ijk_size)
        for axis, (origin, size, chunk_size) in enumerate(zip(ijk_origin, ijk_size, self.chunk_size, strict=True)):
            if size != 1 or chunk_size <= 1:
                continue
            chunk_origin = (origin // chunk_size) * chunk_size
            chunk_end = min(self.size[axis], chunk_origin + chunk_size)
            expanded_origin[axis] = chunk_origin
            expanded_size[axis] = chunk_end - chunk_origin
        return tuple(expanded_origin), tuple(expanded_size)

    def matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Optional[Tuple[int, ...]] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
        from_cache_only: bool = False,
    ):
        """Read through ChimeraX's cache, retaining decoded chunks behind planes."""

        if any(step <= 0 for step in ijk_step):
            raise ValueError(f"Grid steps must be positive, got {ijk_step}.")
        ijk_origin, ijk_size = self._bounded_request(ijk_origin, ijk_size)

        matrix = self.cached_data(ijk_origin, ijk_size, ijk_step)
        if matrix is not None:
            if not from_cache_only:
                self._observe_temporal_request(matrix, ijk_origin, ijk_size, ijk_step)
            return matrix
        if from_cache_only:
            return None

        matrix = self._consume_temporal_read_ahead(ijk_origin, ijk_size, ijk_step)
        if matrix is not None:
            self.cache_data(matrix, ijk_origin, ijk_size, ijk_step)
            self._observe_temporal_request(matrix, ijk_origin, ijk_size, ijk_step)
            return matrix

        slab_origin, slab_size = self._chunk_aligned_plane_region(ijk_origin, ijk_size)
        expands_request = slab_origin != tuple(ijk_origin) or slab_size != tuple(ijk_size)
        output_itemsize = (
            4 if self.value_type in (np.dtype(np.float16), np.dtype(np.uint64)) else self.value_type.itemsize
        )
        slab_bytes = int(np.prod(slab_size, dtype=np.int64)) * output_itemsize
        if expands_request and slab_bytes <= self._decoded_slab_limit():
            slab = self.read_matrix(slab_origin, slab_size, (1, 1, 1), progress)
            self.cache_data(slab, slab_origin, slab_size, (1, 1, 1))
            relative_origin = tuple(
                origin - slab_start for origin, slab_start in zip(ijk_origin, slab_origin, strict=True)
            )
            return self.matrix_slice(slab, relative_origin, ijk_size, ijk_step)

        matrix = super().matrix(ijk_origin, ijk_size, ijk_step, progress)
        self._observe_temporal_request(matrix, ijk_origin, ijk_size, ijk_step)
        return matrix

    def read_matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Optional[Tuple[int, ...]] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
    ):
        del progress
        ijk_origin, ijk_size = self._bounded_request(ijk_origin, ijk_size)
        origin_zyx = ijk_origin[::-1]
        step_zyx = ijk_step[::-1]
        if any(value <= 0 for value in step_zyx):
            raise ValueError(f"Grid steps must be positive, got {ijk_step}.")

        requested_zyx = ijk_size[::-1]
        stop_zyx = tuple(origin_zyx[index] + requested_zyx[index] for index in range(3))

        slices_zyx = tuple(slice(origin_zyx[index], stop_zyx[index], step_zyx[index]) for index in range(3))
        spatial_slices = slices_zyx if self.spatial_ndim == 3 else slices_zyx[1:]

        matrix = self.data[self.fixed_indices + spatial_slices]
        if self.spatial_ndim == 2:
            matrix = np.expand_dims(matrix, axis=0)
        return _supported_matrix_type(matrix)


class ZarrGrid3DSlice(ZarrGridSlice):
    """Backward-compatible constructor for the previous time/channel slice class."""

    def __init__(
        self,
        array: zarr.Array,
        time_index: Optional[int] = None,
        channel_index: Optional[int] = None,
        origin: Tuple[float, float, float] = (0, 0, 0),
        step: Tuple[float, float, float] = (1, 1, 1),
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ) -> None:
        fixed_indices = tuple(index for index in (time_index, channel_index) if index is not None)
        super().__init__(
            array,
            fixed_indices=fixed_indices,
            spatial_ndim=3,
            origin=origin,
            step=step,
            file_type=file_type,
            path=path,
            name=name,
            time_index=time_index,
            channel_index=channel_index,
        )


class ZarrGrid(ZarrGridSlice):
    """Backward-compatible lazy grid for a three-dimensional Zarr array."""

    def __init__(
        self,
        array: zarr.Array,
        origin: Tuple[float, float, float] = (0, 0, 0),
        step: Tuple[float, float, float] = (1, 1, 1),
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ) -> None:
        super().__init__(
            array,
            spatial_ndim=3,
            origin=origin,
            step=step,
            file_type=file_type,
            path=path,
            name=name,
        )


class WrappedZarrGrid(GridData):
    """A grid that selects an aligned OME-Zarr resolution level for each read."""

    def __init__(
        self,
        arrays: Optional[List[zarr.Array]] = None,
        origins: Optional[List[Tuple[float, float, float]]] = None,
        steps: Optional[List[Tuple[float, float, float]]] = None,
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
        grids: Optional[List[GridData]] = None,
    ) -> None:
        if grids is None:
            if not arrays:
                raise ValueError("At least one Zarr array or grid is required.")
            origins = origins or [(0, 0, 0) for _ in arrays]
            steps = steps or [(1, 1, 1) for _ in arrays]
            grids = [
                ZarrGrid(array, origin=origins[index], step=steps[index], file_type=file_type, path=path, name=name)
                for index, array in enumerate(arrays)
            ]
        if not grids:
            raise ValueError("At least one grid is required.")

        self.grids = grids
        self.arrays = arrays
        finest_grid = grids[-1]
        self.scale_index = None
        GridData.__init__(
            self,
            finest_grid.size,
            finest_grid.value_type,
            finest_grid.origin,
            finest_grid.step,
            path=finest_grid.path,
            file_type=finest_grid.file_type,
            name=name,
            time=finest_grid.time,
            channel=finest_grid.channel,
        )

        self._rel_step_sizes: List[Tuple[int, int, int]] = []
        self._grid_offsets: List[Tuple[int, int, int]] = []
        base_step = np.asarray(finest_grid.step, dtype=np.float64)
        base_origin = np.asarray(finest_grid.origin, dtype=np.float64)
        for grid in grids:
            relative_step = np.asarray(grid.step, dtype=np.float64) / base_step
            rounded_step = np.rint(relative_step).astype(int)
            if np.any(rounded_step < 1) or not np.allclose(relative_step, rounded_step):
                raise OMEZarrFormatError(
                    f"Non-integer scaling levels are not supported. Relative steps: {tuple(relative_step)}.",
                )
            relative_origin = (np.asarray(grid.origin, dtype=np.float64) - base_origin) / base_step
            rounded_origin = np.rint(relative_origin).astype(int)
            if not np.allclose(relative_origin, rounded_origin):
                raise OMEZarrFormatError(
                    "Translated multiscale levels must align with the finest grid; open scales separately to "
                    f"display arbitrary origins. Finest origin {tuple(base_origin)}, level origin {grid.origin}.",
                )
            self._rel_step_sizes.append(tuple(int(value) for value in rounded_step))
            self._grid_offsets.append(tuple(int(value) for value in rounded_origin))

    def _get_data_cache(self):
        return self.__dict__["data_cache"]

    def _set_data_cache(self, cache) -> None:
        self.__dict__["data_cache"] = cache
        for grid in self.grids:
            grid.data_cache = cache

    data_cache = property(_get_data_cache, _set_data_cache)

    def get_sampling_strategy(
        self,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
    ) -> Tuple[GridData, Tuple[int, ...], Tuple[int, ...]]:
        """Return the coarsest aligned grid able to satisfy a requested sampling lattice."""

        if any(step <= 0 for step in ijk_step):
            raise ValueError(f"Grid steps must be positive, got {ijk_step}.")
        for grid, relative_step, offset in zip(
            self.grids,
            self._rel_step_sizes,
            self._grid_offsets,
            strict=True,
        ):
            if all(
                requested % available == 0 for requested, available in zip(ijk_step, relative_step, strict=True)
            ) and all(
                (origin - shift) % available == 0
                for origin, shift, available in zip(ijk_origin, offset, relative_step, strict=True)
            ):
                adjusted_step = tuple(
                    requested // available for requested, available in zip(ijk_step, relative_step, strict=True)
                )
                return grid, adjusted_step, relative_step

        # The finest grid is always aligned with itself.
        return self.grids[-1], ijk_step, (1, 1, 1)

    def _adjusted_request(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Optional[Tuple[int, ...]] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
    ):
        grid, adjusted_step, factors = self.get_sampling_strategy(ijk_step, ijk_origin)
        grid_index = self.grids.index(grid)
        offset = self._grid_offsets[grid_index]
        adjusted_origin = tuple(
            (origin - shift) // factor for origin, shift, factor in zip(ijk_origin, offset, factors, strict=True)
        )
        adjusted_size = None
        if ijk_size is not None:
            adjusted_size = tuple(
                max(1, (size + factor - 1) // factor) for size, factor in zip(ijk_size, factors, strict=True)
            )
        return grid, adjusted_origin, adjusted_size, adjusted_step

    def matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Optional[Tuple[int, ...]] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
        from_cache_only: bool = False,
    ):
        grid, adjusted_origin, adjusted_size, adjusted_step = self._adjusted_request(
            ijk_origin,
            ijk_size,
            ijk_step,
        )
        return grid.matrix(adjusted_origin, adjusted_size, adjusted_step, progress, from_cache_only)

    def read_matrix(
        self,
        ijk_origin: Tuple[int, ...] = (0, 0, 0),
        ijk_size: Optional[Tuple[int, ...]] = None,
        ijk_step: Tuple[int, ...] = (1, 1, 1),
        progress: Any = None,
    ):
        return self.matrix(ijk_origin, ijk_size, ijk_step, progress)

    def cached_data(self, ijk_origin, ijk_size, ijk_step):
        return self.matrix(ijk_origin, ijk_size, ijk_step, from_cache_only=True)

    def clear_cache(self) -> None:
        for grid in self.grids:
            grid.clear_cache()


def _volume_region(grid: GridData, initial_step: Tuple[int, int, int]):
    z_index = grid.size[2] // 2
    ijk_min = (0, 0, z_index)
    ijk_max = (max(0, grid.size[0] - 1), max(0, grid.size[1] - 1), z_index)
    if grid.size[2] == 1:
        initial_step = (initial_step[0], initial_step[1], 1)
    return ijk_min, ijk_max, initial_step


def _apply_omero_display(
    volume: Volume,
    omero: Optional[OmeroMetadata],
    channel_index: int,
    time_index: int,
    has_channel: bool,
    has_time: bool,
    base_name: str,
    scale_path: Optional[str] = None,
) -> None:
    if omero and has_channel and channel_index < len(omero.channels):
        channel = omero.channels[channel_index]
        if channel.label:
            name_parts = [base_name]
            if scale_path is not None:
                name_parts.append(scale_path)
            name_parts.append(channel.label)
            if has_time:
                name_parts.append(f"t={time_index}")
            volume.name = " - ".join(name_parts)
        with suppress(TypeError, ValueError):
            volume.set_parameters(default_rgba=hex_to_rgba(channel.color))
        volume.display = time_index == 0 and channel.active
    else:
        volume.display = time_index == 0


class ZarrModel(Model):
    """A lazily loaded OME-Zarr 0.4 or 0.5 image model."""

    def __init__(
        self,
        name: str,
        session,
        root,
        scales: Optional[List[str]] = None,
        initial_step: Tuple[int, ...] = (1, 1, 1),
        read_ahead: Optional[int] = None,
    ) -> None:
        Model.__init__(self, name, session)

        if read_ahead is not None and (not isinstance(read_ahead, Integral) or read_ahead < 0):
            raise OMEZarrFormatError(f"readAhead must be a nonnegative integer, got {read_ahead!r}.")
        read_ahead = None if read_ahead is None else int(read_ahead)

        self.group = cached_group(session, root)
        self._source_store = self.group.store
        metadata = parse_ome_zarr_metadata(self.group)
        self.ome_zarr_metadata = metadata
        multiscales = metadata.multiscales
        self.omero = metadata.omero
        self.avail_scales = [dataset.path for dataset in multiscales.datasets]

        if scales is not None:
            unavailable = [scale for scale in scales if scale not in self.avail_scales]
            if unavailable:
                raise OMEZarrFormatError(
                    f"Scale(s) {', '.join(unavailable)} are not available; choose from {', '.join(self.avail_scales)}.",
                )

        entries = []
        for scale_index, dataset in enumerate(multiscales.datasets):
            if scales is not None and dataset.path not in scales:
                continue
            array = self.group[dataset.path]
            step, origin = spatial_transform_angstrom(multiscales, dataset)
            entries.append((array, dataset, step, origin, scale_index))
        if not entries:
            raise OMEZarrFormatError("No multiscale resolution level was selected.")
        self.arrays_datasets_sizes = entries

        axes_types = [axis.type for axis in multiscales.axes]
        has_time = "time" in axes_types
        has_channel = "channel" in axes_types
        time_axis = axes_types.index("time") if has_time else None
        channel_axis = axes_types.index("channel") if has_channel else None
        finest_array = entries[0][0]
        time_count = finest_array.shape[time_axis] if time_axis is not None else 1
        channel_count = finest_array.shape[channel_axis] if channel_axis is not None else 1
        spatial_ndim = multiscales.spatial_ndim
        initial_step = tuple(initial_step or ((4, 4, 4) if scales is None else (1, 1, 1)))

        volumes = []
        temporal_levels = {}
        for time_index in range(time_count):
            for channel_index in range(channel_count):
                fixed_indices = []
                if has_time:
                    fixed_indices.append(time_index)
                if has_channel:
                    fixed_indices.append(channel_index)

                if scales is None:
                    grids = []
                    # WrappedZarrGrid expects coarsest-to-finest ordering.
                    for array, dataset, step, origin, scale_index in reversed(entries):
                        grids.append(
                            ZarrGridSlice(
                                array,
                                fixed_indices=fixed_indices,
                                spatial_ndim=spatial_ndim,
                                origin=origin,
                                step=step,
                                name=f"{name} t={time_index} c={channel_index}",
                                time_index=time_index if has_time else None,
                                channel_index=channel_index if has_channel else None,
                                scale_path=dataset.path,
                                scale_index=scale_index,
                            ),
                        )
                        temporal_levels.setdefault((channel_index, scale_index), []).append(
                            (time_index, grids[-1]),
                        )
                    grid = WrappedZarrGrid(grids=grids, name=f"{name} t={time_index} c={channel_index}")
                    grid.scale_path = None
                    set_data_cache(grid, session)
                    volume = Volume(session, grid, region=_volume_region(grid, initial_step))
                    volume.set_display_style("image")
                    volume.new_region(volume.region[0], volume.region[1], volume.region[2], adjust_step=False)
                    _apply_omero_display(
                        volume,
                        self.omero,
                        channel_index,
                        time_index,
                        has_channel,
                        has_time,
                        name,
                    )
                    volumes.append(volume)
                else:
                    for array, dataset, step, origin, scale_index in entries:
                        grid = ZarrGridSlice(
                            array,
                            fixed_indices=fixed_indices,
                            spatial_ndim=spatial_ndim,
                            origin=origin,
                            step=step,
                            name=f"{name} - {dataset.path} t={time_index} c={channel_index}",
                            time_index=time_index if has_time else None,
                            channel_index=channel_index if has_channel else None,
                            scale_path=dataset.path,
                            scale_index=scale_index,
                        )
                        temporal_levels.setdefault((channel_index, scale_index), []).append((time_index, grid))
                        set_data_cache(grid, session)
                        volume = Volume(session, grid, region=_volume_region(grid, initial_step))
                        volume.set_display_style("image")
                        volume.new_region(volume.region[0], volume.region[1], volume.region[2], adjust_step=False)
                        _apply_omero_display(
                            volume,
                            self.omero,
                            channel_index,
                            time_index,
                            has_channel,
                            has_time,
                            name,
                            scale_path=dataset.path,
                        )
                        volumes.append(volume)

        if has_time and time_count > 1 and read_ahead != 0:
            manager = session_temporal_read_ahead(session)
            for indexed_grids in temporal_levels.values():
                grids = [grid for _, grid in sorted(indexed_grids)]
                sequence = manager.create_sequence(grids, read_ahead)
                for index, grid in enumerate(grids):
                    grid.set_temporal_read_ahead(sequence, index)

        self.add(volumes)

    @property
    def scales(self):
        return self.avail_scales

    def open_scales(self, scales: List[str]):
        """Load additional scales."""

        raise NotImplementedError("Not implemented yet.")

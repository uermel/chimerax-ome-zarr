"""Bridge OME-Zarr label images to ChimeraX index maps and segmentations."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import zarr
from chimerax.map.volume import Volume, set_data_cache
from chimerax.map_data import GridData
from chimerax.segmentations import Segmentation

from .ome_metadata import LabelValueMetadata, OMEZarrFormatError, OMEZarrMetadata, spatial_transform_angstrom
from .zarr_grid import WrappedZarrGrid, ZarrGridSlice


def _sample_count(size: int, step: int) -> int:
    return (size + step - 1) // step


class BroadcastLabelGrid(GridData):
    """Broadcast singleton label dimensions onto a compatible source grid."""

    def __init__(self, label_grid: GridData, target_grid: GridData, name: str) -> None:
        self.label_grid = label_grid
        self.scale_index = getattr(label_grid, "scale_index", None)
        self.scale_path = getattr(label_grid, "scale_path", None)
        GridData.__init__(
            self,
            target_grid.size,
            label_grid.value_type,
            target_grid.origin,
            target_grid.step,
            cell_angles=target_grid.cell_angles,
            rotation=target_grid.rotation,
            symmetries=target_grid.symmetries,
            path=label_grid.path,
            file_type=label_grid.file_type,
            name=name,
            time=target_grid.time,
            channel=target_grid.channel,
        )

    def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
        del progress
        if ijk_size is None:
            ijk_size = self.size
        source_origin = []
        source_size = []
        source_step = []
        output_size = []
        for axis, (origin, size, step) in enumerate(zip(ijk_origin, ijk_size, ijk_step, strict=True)):
            if step <= 0:
                raise ValueError(f"Grid steps must be positive, got {ijk_step}.")
            output_size.append(_sample_count(size, step))
            if self.label_grid.size[axis] == 1:
                source_origin.append(0)
                source_size.append(1)
                source_step.append(1)
            else:
                source_origin.append(origin)
                source_size.append(size)
                source_step.append(step)
        matrix = self.label_grid.read_matrix(
            tuple(source_origin),
            tuple(source_size),
            tuple(source_step),
        )
        return np.broadcast_to(matrix, tuple(reversed(output_size)))


class LabelSliceState:
    """Shared lazy/read-only state that becomes an editable integer map on demand."""

    def __init__(self, read_grid: GridData, finest_grid: GridData) -> None:
        self.read_grid = read_grid
        self.finest_grid = finest_grid
        self._array = None
        self._index_grids = []
        self._mask_grids = []

    @property
    def materialized(self) -> bool:
        return self._array is not None

    def register_index_grid(self, grid) -> None:
        self._index_grids.append(grid)

    def register_mask_grid(self, grid) -> None:
        self._mask_grids.append(grid)

    def read_matrix(self, ijk_origin, ijk_size, ijk_step):
        if self._array is None:
            return self.read_grid.read_matrix(ijk_origin, ijk_size, ijk_step)
        return GridData.matrix_slice(self.finest_grid, self._array, ijk_origin, ijk_size, ijk_step)

    def materialize(self):
        if self._array is None:
            self._array = np.array(
                self.finest_grid.read_matrix((0, 0, 0), self.finest_grid.size, (1, 1, 1)),
                copy=True,
            )
        return self._array

    def commit_mask(self, active_grid) -> None:
        index_array = self.materialize()
        mask = active_grid._array != 0
        value = active_grid.label_value
        index_array[np.logical_and(mask, index_array != value)] = value
        index_array[np.logical_and(~mask, index_array == value)] = 0

        for grid in self._mask_grids:
            if grid is not active_grid:
                grid._array = None
        for grid in (*self._index_grids, *self._mask_grids):
            grid.clear_cache()
            grid.values_changed()


class OMELabelIndexGrid(GridData):
    """A ChimeraX integer index map backed by a shared label slice."""

    def __init__(self, state: LabelSliceState, target_grid: GridData, name: str) -> None:
        self.state = state
        self.scale_path = getattr(target_grid, "scale_path", None)
        self.scale_index = getattr(target_grid, "scale_index", None)
        GridData.__init__(
            self,
            target_grid.size,
            state.finest_grid.value_type,
            target_grid.origin,
            target_grid.step,
            cell_angles=target_grid.cell_angles,
            rotation=target_grid.rotation,
            symmetries=target_grid.symmetries,
            path=state.finest_grid.path,
            file_type="ome-zarr-label",
            name=name,
            time=target_grid.time,
            channel=target_grid.channel,
        )
        state.register_index_grid(self)

    def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
        del progress
        return self.state.read_matrix(ijk_origin, ijk_size or self.size, ijk_step)


class OMELabelMaskGrid(GridData):
    """A lazy binary view of one value in an :class:`OMELabelIndexGrid`."""

    def __init__(
        self,
        state: LabelSliceState,
        target_grid: GridData,
        label_value: int,
        name: str,
    ) -> None:
        self.state = state
        self.label_value = label_value
        self._array = None
        self.scale_path = getattr(target_grid, "scale_path", None)
        self.scale_index = getattr(target_grid, "scale_index", None)
        GridData.__init__(
            self,
            target_grid.size,
            np.uint8,
            target_grid.origin,
            target_grid.step,
            cell_angles=target_grid.cell_angles,
            rotation=target_grid.rotation,
            symmetries=target_grid.symmetries,
            path=state.finest_grid.path,
            file_type="ome-zarr-label-mask",
            name=name,
            time=target_grid.time,
            channel=target_grid.channel,
        )
        self.writable = True
        state.register_mask_grid(self)

    @property
    def array(self):
        if self._array is None:
            self._array = np.asarray(self.state.materialize() == self.label_value, dtype=np.uint8)
        return self._array

    def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
        del progress
        ijk_size = ijk_size or self.size
        if self._array is not None:
            return self.matrix_slice(self._array, ijk_origin, ijk_size, ijk_step)
        return np.asarray(self.state.read_matrix(ijk_origin, ijk_size, ijk_step) == self.label_value, dtype=np.uint8)

    def commit(self) -> None:
        self.state.commit_mask(self)


class OMEZarrSegmentation(Segmentation):
    """Native ChimeraX segmentation synchronized with an OME integer index map."""

    def __init__(
        self,
        session,
        grid: OMELabelMaskGrid,
        reference_volume: Volume,
        label_path: str,
        label_metadata: LabelValueMetadata,
    ) -> None:
        super().__init__(session, grid)
        self.reference_volume = reference_volume
        self.ome_label_value = label_metadata.value
        self.ome_label_path = label_path
        self.ome_label_properties = dict(label_metadata.properties)
        self.set_parameters(surface_levels=[0.501])
        if label_metadata.rgba is not None:
            self.set_parameters(default_rgba=tuple(component / 255.0 for component in label_metadata.rgba))
        self.set_display_style("surface")
        self.display = False

    def segment(self, strategy) -> None:
        strategy.execute(self.data, self.reference_volume.data)
        self.data.commit()


@dataclass
class LabelBridge:
    """Models produced for one OME label image."""

    index_volumes: List[Volume]
    segmentations: List[OMEZarrSegmentation]


def _axis_signature(metadata: OMEZarrMetadata) -> Tuple[Tuple[str, str], ...]:
    return tuple((axis.name, axis.type) for axis in metadata.multiscales.axes)


def _target_levels(target_grid: GridData) -> List[GridData]:
    if isinstance(target_grid, WrappedZarrGrid):
        return list(target_grid.grids)
    return [target_grid]


def _validate_level_geometry(label_grid: GridData, target_grid: GridData, label_path: str) -> None:
    for axis, (label_size, target_size) in enumerate(zip(label_grid.size, target_grid.size, strict=True)):
        if label_size not in {1, target_size}:
            raise OMEZarrFormatError(
                f"Label image '{label_path}' size {label_grid.size} cannot be broadcast to source size "
                f"{target_grid.size} (axis {axis}).",
            )
        if label_size != 1 and (
            not np.isclose(label_grid.step[axis], target_grid.step[axis])
            or not np.isclose(label_grid.origin[axis], target_grid.origin[axis])
        ):
            raise OMEZarrFormatError(
                f"Label image '{label_path}' does not share the source image's physical grid on axis {axis}.",
            )


def _fixed_indices(metadata: OMEZarrMetadata, finest_array: zarr.Array, reference: Volume) -> Tuple[int, ...]:
    fixed = []
    for axis_index, axis in enumerate(metadata.multiscales.axes):
        if axis.type == "space":
            continue
        reference_index = reference.data.time if axis.type == "time" else reference.data.channel
        reference_index = 0 if reference_index is None else reference_index
        size = finest_array.shape[axis_index]
        if size != 1 and reference_index >= size:
            raise OMEZarrFormatError(
                f"Label {axis.type} dimension has size {size}, but source index {reference_index} was requested.",
            )
        fixed.append(0 if size == 1 else reference_index)
    return tuple(fixed)


def _label_display_name(label_image_name: str, metadata: LabelValueMetadata) -> str:
    properties = metadata.properties
    description = next(
        (str(properties[key]) for key in ("name", "class", "label") if key in properties and str(properties[key])),
        f"label {metadata.value}",
    )
    return f"{label_image_name}: {description} (value {metadata.value})"


def build_label_bridge(
    session,
    label_group: zarr.Group,
    label_path: str,
    label_metadata: OMEZarrMetadata,
    source_group: zarr.Group,
    source_metadata: OMEZarrMetadata,
    references: Sequence[Volume],
    initial_step: Tuple[int, int, int],
) -> LabelBridge:
    """Create lazy index volumes and native segmentations for one label image."""

    if label_metadata.image_label is None:
        raise OMEZarrFormatError(f"Label group '{label_path}' has no image-label metadata.")
    if _axis_signature(label_metadata) != _axis_signature(source_metadata):
        raise OMEZarrFormatError(f"Label image '{label_path}' axes do not match its source image axes.")
    if len(label_metadata.multiscales.datasets) != len(source_metadata.multiscales.datasets):
        raise OMEZarrFormatError(
            f"Label image '{label_path}' has a different number of resolution levels than its source image.",
        )

    label_name = label_metadata.multiscales.name or label_path.rsplit("/", 1)[-1]
    finest_array = label_group[label_metadata.multiscales.datasets[0].path]
    source_finest = source_group[source_metadata.multiscales.datasets[0].path]
    for axis_index, axis in enumerate(label_metadata.multiscales.axes):
        label_size = finest_array.shape[axis_index]
        source_size = source_finest.shape[axis_index]
        if label_size not in {1, source_size}:
            raise OMEZarrFormatError(
                f"Label image '{label_path}' {axis.type} dimension has size {label_size}; expected 1 or {source_size}.",
            )

    states = {}
    index_volumes = []
    segmentations = []
    editable_values = [value for value in label_metadata.image_label.values if value.value != 0]

    for reference in references:
        fixed_indices = _fixed_indices(label_metadata, finest_array, reference)
        target_levels = _target_levels(reference.data)
        scale_indices = tuple(getattr(level, "scale_index", None) for level in target_levels)
        if any(scale_index is None for scale_index in scale_indices):
            raise OMEZarrFormatError(f"Could not match source resolution levels for label image '{label_path}'.")
        state_key = (fixed_indices, scale_indices)
        state = states.get(state_key)
        if state is None:
            broadcast_levels = []
            for target_level, scale_index in zip(target_levels, scale_indices, strict=True):
                dataset = label_metadata.multiscales.datasets[scale_index]
                array = label_group[dataset.path]
                step, origin = spatial_transform_angstrom(label_metadata.multiscales, dataset)
                raw_grid = ZarrGridSlice(
                    array,
                    fixed_indices=fixed_indices,
                    spatial_ndim=label_metadata.multiscales.spatial_ndim,
                    origin=origin,
                    step=step,
                    path=label_path,
                    name=label_name,
                    time_index=reference.data.time,
                    channel_index=reference.data.channel,
                    scale_path=dataset.path,
                    scale_index=scale_index,
                )
                _validate_level_geometry(raw_grid, target_level, label_path)
                broadcast_levels.append(BroadcastLabelGrid(raw_grid, target_level, label_name))
            read_grid = (
                WrappedZarrGrid(grids=broadcast_levels, name=label_name)
                if len(broadcast_levels) > 1
                else broadcast_levels[0]
            )
            finest_grid = min(broadcast_levels, key=lambda grid: grid.scale_index)
            state = states[state_key] = LabelSliceState(read_grid, finest_grid)

        suffix = []
        if reference.data.time is not None:
            suffix.append(f"t={reference.data.time}")
        if reference.data.channel is not None:
            suffix.append(f"c={reference.data.channel}")
        if reference.data.scale_path is not None:
            suffix.append(f"scale={reference.data.scale_path}")
        slice_name = " ".join([label_name, *suffix])

        index_grid = OMELabelIndexGrid(state, reference.data, f"{slice_name} index")
        set_data_cache(index_grid, session)
        index_volume = Volume(session, index_grid)
        index_volume.name = f"{slice_name} index"
        index_volume.ome_image_label = label_metadata.image_label
        index_volume.ome_label_path = label_path
        index_volume._max_segment_id = max((value.value for value in label_metadata.image_label.values), default=0)
        index_volume.set_display_style("image")
        index_volume.display = False
        index_volume.new_region(
            (0, 0, index_grid.size[2] // 2),
            (max(0, index_grid.size[0] - 1), max(0, index_grid.size[1] - 1), index_grid.size[2] // 2),
            initial_step,
            adjust_step=False,
        )
        index_volumes.append(index_volume)

        for value_metadata in editable_values:
            mask_name = _label_display_name(label_name, value_metadata)
            if suffix:
                mask_name = f"{mask_name} {' '.join(suffix)}"
            mask_grid = OMELabelMaskGrid(state, reference.data, value_metadata.value, mask_name)
            set_data_cache(mask_grid, session)
            segmentation = OMEZarrSegmentation(
                session,
                mask_grid,
                reference,
                label_path,
                value_metadata,
            )
            segmentation.name = mask_name
            segmentations.append(segmentation)

    return LabelBridge(index_volumes=index_volumes, segmentations=segmentations)

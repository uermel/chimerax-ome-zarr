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


def parse_labels(zattrs: zarr.attrs.Attributes, session: Session) -> None:
    """Parse labels metadata from OME-Zarr header."""
    if "labels" not in zattrs:
        return None

    session.logger.warning("Labels not implemented yet.")
    return None


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

        if "channel" in [a.type for a in mlt.axes]:
            raise NotImplementedError("Channel axis not supported yet.")

        if "time" in [a.type for a in mlt.axes]:
            raise NotImplementedError("Time axis not supported yet.")

        if not all(a.type == "space" for a in mlt.axes):
            raise ValueError("Unknown space axis.")

        print(mlt)
        print(mlt.datasets)

        self.avail_scales = [d.path for d in mlt.datasets]

        if scales is not None:
            for s in scales:
                if s not in self.avail_scales:
                    raise ValueError(f"Scale {s} not available in file.")

        # Labels (only to warn about ignoring them for now)
        _ = parse_labels(attrs, session)

        # No multiscales, return
        if mlt is None:
            return

        # Get pixelsizes in Angstrom from unit and scale transformations
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
        print(arrays_datasets_sizes)
        print([x[0].nbytes for x in arrays_datasets_sizes])
        self.arrays_datasets_sizes = sorted(arrays_datasets_sizes, key=lambda x: x[0].nbytes, reverse=False)

        # If no scales requested, load all scales async
        if not scales:
            if initial_step is None:
                initial_step = (4, 4, 4)

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

        from numpy import float16, float32

        if m.dtype == float16:
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
        arrays: List[Array],
        origins: List[Tuple[float, float, float]] = None,
        steps: List[Tuple[float, float, float]] = None,
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ) -> None:
        # Default origins and steps
        if origins is None:
            origins = [(0, 0, 0) for _ in range(len(arrays))]
        if steps is None:
            steps = [(1, 1, 1) for _ in range(len(arrays))]

        # Relative transformation between grids
        self._rel_step_sizes: List[Tuple[int, ...]] = []
        base_step = steps[-1]
        print(f"base_step: {base_step}")
        for s in steps:
            relstep = (s[0] / base_step[0], s[1] / base_step[1], s[2] / base_step[2])

            if not np.allclose(relstep, relstep[0]):
                raise NotImplementedError(
                    f"""Anisotropically scaled input data is not supported. Finest step: {base_step}, current step: {s},
                    relative step: {relstep}""",
                )

            if not np.allclose(relstep, int(relstep[0])):
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

from chimerax.map_data import GridData
from chimerax.map_data.datacache import Data_Cache
import numpy as np
from typing import Tuple, Any, List
from zarr.core import Array


class ZarrGrid(GridData):
    # TODO: handle multiscale in here instead of in open.py
    # TODO: async multiscale fetching: translate the ranges to lower scale in here until data is fetched
    def __init__(
        self,
        array: Array,
        origin: Tuple[float, float, float] = (0, 0, 0),
        step: Tuple[float, float, float] = (1, 1, 1),
        file_type: str = "zarr",
        path: str = "",
        name: str = "",
    ):

        self.dask_data = array
        dd = array

        shape = dd.shape[::-1]
        origin = origin[::-1]
        step = step[::-1]

        GridData.__init__(
            self,
            shape,
            dd.dtype,
            origin,
            step,
            path=path,
            file_type=file_type,
            name=name,
        )

    def read_matrix(
        self,
        ijk_origin: Tuple[int, int, int] = (0, 0, 0),
        ijk_size: Tuple[int, int, int] = None,
        ijk_step: Tuple[int, int, int] = (1, 1, 1),
        progress: Any = None,
    ):

        ijk_origin = ijk_origin[::-1]
        ijk_step = ijk_step[::-1]

        if ijk_size is None:
            ijk_size = self.size[::-1]
        else:
            ijk_size = ijk_size[::-1]
            ijk_size = [ijk_origin[i] + ijk_size[i] for i in range(3)]

        m = self.dask_data[
            ijk_origin[0] : ijk_size[0] : ijk_step[0],
            ijk_origin[1] : ijk_size[1] : ijk_step[1],
            ijk_origin[2] : ijk_size[2] : ijk_step[2],
        ]

        from numpy import float16, float32

        if m.dtype == float16:
            m = m.astype(float32)

        return m

import os
import zarr
import zarr.attrs
import dask.array as da


from chimerax.core.models import Model
from chimerax.map.volume import Volume, show_volume_dialog


from typing import List, Tuple

from .map_data.zarr_grid import ZarrGrid

UNITFACTOR = {
    "angstrom": 1.0,
    "nm": 10.0,
    "um": 10000.0,
    "mm": 10000000.0,
}


def get_unit_factor(zattrs: zarr.attrs.Attributes) -> Tuple[float, float, float]:

    ms = zattrs["multiscales"][0]
    zunit = UNITFACTOR[ms["axes"][0].get("unit", "angstrom")]
    yunit = UNITFACTOR[ms["axes"][1].get("unit", "angstrom")]
    xunit = UNITFACTOR[ms["axes"][2].get("unit", "angstrom")]

    return (zunit, yunit, xunit)


def get_pixelsize(zattrs: zarr.attrs.Attributes) -> List[Tuple[float, float, float]]:
    sizes = []

    datasets = zattrs["multiscales"][0]["datasets"]
    for ds in datasets:
        zs = ds["coordinateTransformations"][0]["scale"][0]
        ys = ds["coordinateTransformations"][0]["scale"][1]
        xs = ds["coordinateTransformations"][0]["scale"][2]

        sizes.append((zs, ys, xs))

    return sizes


def open_ome_zarr(session, data: str, scales: List[int] = None, fs: str = ""):

    model = Model(os.path.basename(data), session)

    # Work around ChimeraX
    if fs:
        data = f"{fs}://" + data

    # The initial store to get sizes and units
    root = zarr.storage.FSStore(
        f"{data}", key_separator="/", mode="r", dimension_separator="/"
    )
    group = zarr.open(root, mode="r")
    arrays = list(group.arrays())
    attrs = group.attrs

    # Get pixelsizes in Angstrom from unit and scale transformations
    ufac = get_unit_factor(attrs)
    # print(f"ufac: {ufac}")
    sizes = get_pixelsize(attrs)
    # print(f"sizes: {sizes}")
    sizes = [(ufac[0] * s[0], ufac[1] * s[1], ufac[2] * s[2]) for s in sizes]

    # The cached store, group and arrays
    root_cached = zarr.LRUStoreCache(
        root,
        max_size=None,  # arrays[0][1].size,
    )
    group_cached = zarr.open(root_cached, mode="r")
    arrays_cached = list(group_cached.arrays())

    # If no scales requested, load lowest by default
    # print(f"num arrays: {len(arrays_cached)}")
    if not scales:
        scales = [len(arrays_cached) - 1]

    # print(f"scales: {scales}")
    # print(sizes)

    for scale in scales:
        name, array = arrays_cached[scale]
        # print(type(array))
        # agd = ArrayGridData(array, step=sizes[scale], name=name)#f"{os.path.basename(data)}/{name}")
        dgd = ZarrGrid(array, step=sizes[scale], name=name)
        ijk_min = (0, 0, dgd.size[2] // 2)
        ijk_max = (dgd.size[0], dgd.size[1], dgd.size[2] // 2)
        ijk_step = (4, 4, 4)
        vol = Volume(session, dgd, (ijk_min, ijk_max, ijk_step))
        vol.set_display_style("image")
        model.add([vol])

    show_volume_dialog(session)
    return ([model], f"Opened {data}.")


# def fetch_tomogram(session, identifier: str, ignore_cache: bool = False, **kw):
#     from chimerax.core.errors import UserError
#
#     client = Client()
#     tomo = Tomogram.get_by_id(client, int(identifier))
#     map_url = tomo.https_mrc_scale0
#     vol_name = path.basename(urlsplit(tomo.https_mrc_scale0).path)
#
#     if not tomo:
#         raise UserError(f"Could not find tomogram with id {identifier}")
#
#     from chimerax.core.fetch import fetch_file
#     filename = fetch_file(session, map_url, tomo.name, vol_name, 'cryoet-portal',
#                            uncompress=False, ignore_cache=ignore_cache)
#
#     model_name = tomo.name
#     models, status = session.open_command.open_data(filename, format='mrc',
#                                                     name=model_name, **kw)
#
#     return models, status
#
# def fetch_annotation():
#     pass

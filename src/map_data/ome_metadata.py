"""OME-Zarr 0.4/0.5 metadata normalization and validation."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from .constants import UNITFACTOR


class OMEZarrFormatError(ValueError):
    """Raised when an OME-Zarr hierarchy cannot be represented by this reader."""


@dataclass(frozen=True)
class Axis:
    """Normalized OME-Zarr axis metadata."""

    name: str
    type: str = "space"
    unit: Optional[str] = None


@dataclass(frozen=True)
class MultiscaleDataset:
    """A resolution level with its composed coordinate transform."""

    path: str
    scale: Tuple[float, ...]
    translation: Tuple[float, ...]


@dataclass(frozen=True)
class Multiscales:
    """The single multiscale image supported by this reader."""

    axes: Tuple[Axis, ...]
    datasets: Tuple[MultiscaleDataset, ...]
    version: str
    name: Optional[str] = None

    @property
    def spatial_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, axis in enumerate(self.axes) if axis.type == "space")

    @property
    def spatial_ndim(self) -> int:
        return len(self.spatial_indices)


@dataclass(frozen=True)
class OmeroChannelWindow:
    """OMERO channel display range metadata."""

    start: float
    end: float
    min: float
    max: float


@dataclass(frozen=True)
class OmeroChannel:
    """Normalized OMERO channel display metadata."""

    label: str
    color: str
    active: bool = True
    coefficient: float = 1.0
    family: str = "linear"
    inverted: bool = False
    window: Optional[OmeroChannelWindow] = None


@dataclass(frozen=True)
class OmeroMetadata:
    """Normalized OMERO display metadata."""

    channels: Tuple[OmeroChannel, ...]
    version: Optional[str] = None
    id: Optional[int] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class OMEZarrMetadata:
    """Normalized metadata independent of the underlying Zarr format."""

    zarr_format: int
    ome_version: str
    multiscales: Multiscales
    omero: Optional[OmeroMetadata]


def _version_family(version: Any) -> str:
    text = str(version)
    parts = text.split(".")
    return ".".join(parts[:2])


def _node_zarr_format(group: zarr.Group) -> int:
    try:
        return int(group.metadata.zarr_format)
    except (AttributeError, TypeError, ValueError) as error:
        raise OMEZarrFormatError("Could not determine the underlying Zarr format.") from error


def _metadata_namespace(group: zarr.Group) -> Tuple[int, str, Mapping[str, Any]]:
    zarr_format = _node_zarr_format(group)
    attrs = dict(group.attrs)

    if zarr_format == 3:
        ome = attrs.get("ome")
        if not isinstance(ome, Mapping):
            raise OMEZarrFormatError("Zarr v3 data must contain OME-Zarr metadata under attributes['ome'].")
        version = _version_family(ome.get("version", ""))
        if version != "0.5":
            raise OMEZarrFormatError(
                f"OME-Zarr {ome.get('version', '<missing>')} with Zarr v3 is unsupported; expected OME-Zarr 0.5.",
            )
        return zarr_format, version, ome

    if zarr_format == 2:
        if isinstance(attrs.get("ome"), Mapping) and _version_family(attrs["ome"].get("version", "")) == "0.5":
            raise OMEZarrFormatError("OME-Zarr 0.5 metadata cannot be stored in a Zarr v2 hierarchy.")
        return zarr_format, "0.4", attrs

    raise OMEZarrFormatError(f"Unsupported Zarr format {zarr_format}; expected format 2 or 3.")


def _read_transform_vector(group: zarr.Group, transform: Mapping[str, Any], field: str, ndim: int) -> Tuple[float, ...]:
    vector = transform.get(field)
    if vector is None:
        path = transform.get("path")
        if not isinstance(path, str) or not path:
            raise OMEZarrFormatError(f"A {field} transform must contain either '{field}' or 'path'.")
        try:
            node = group[path]
        except KeyError as error:
            raise OMEZarrFormatError(f"Coordinate-transform vector path '{path}' does not exist.") from error
        if not isinstance(node, zarr.Array):
            raise OMEZarrFormatError(f"Coordinate-transform vector path '{path}' is not a Zarr array.")
        vector = np.asarray(node[:]).reshape(-1).tolist()

    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != ndim:
        raise OMEZarrFormatError(f"The {field} transform must contain exactly {ndim} values.")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise OMEZarrFormatError(f"The {field} transform contains a non-numeric value.") from error
    if not all(np.isfinite(value) for value in values):
        raise OMEZarrFormatError(f"The {field} transform contains a non-finite value.")
    if field == "scale" and any(value <= 0 for value in values):
        raise OMEZarrFormatError("Scale-transform values must be positive.")
    return values


def _parse_transform_sequence(
    group: zarr.Group,
    transforms: Any,
    ndim: int,
    *,
    require_scale: bool,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if transforms is None:
        if require_scale:
            raise OMEZarrFormatError("Each multiscale dataset must define a scale transform.")
        return (1.0,) * ndim, (0.0,) * ndim
    if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise OMEZarrFormatError("coordinateTransformations must be a list.")

    scale_count = 0
    translation_count = 0
    scale = np.ones(ndim, dtype=np.float64)
    translation = np.zeros(ndim, dtype=np.float64)
    seen_translation = False

    for transform in transforms:
        if not isinstance(transform, Mapping):
            raise OMEZarrFormatError("Every coordinate transform must be an object.")
        transform_type = transform.get("type")
        if transform_type == "scale":
            if seen_translation:
                raise OMEZarrFormatError("Scale transforms must precede translation transforms.")
            scale_count += 1
            vector = np.asarray(_read_transform_vector(group, transform, "scale", ndim))
            translation *= vector
            scale *= vector
        elif transform_type == "translation":
            seen_translation = True
            translation_count += 1
            translation += np.asarray(_read_transform_vector(group, transform, "translation", ndim))
        else:
            raise OMEZarrFormatError(
                f"Unsupported multiscale coordinate transform '{transform_type}'; expected scale or translation.",
            )

    if require_scale and scale_count != 1:
        raise OMEZarrFormatError("A dataset coordinate-transform sequence must contain exactly one scale transform.")
    if not require_scale and scale_count > 1:
        raise OMEZarrFormatError("A multiscale coordinate-transform sequence may contain at most one scale transform.")
    if translation_count > 1:
        raise OMEZarrFormatError("A coordinate-transform sequence may contain at most one translation transform.")
    return tuple(scale.tolist()), tuple(translation.tolist())


def _compose_affine(
    first: Tuple[Tuple[float, ...], Tuple[float, ...]],
    second: Tuple[Tuple[float, ...], Tuple[float, ...]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Compose affine transforms in application order: ``second(first(x))``."""

    first_scale, first_translation = (np.asarray(values) for values in first)
    second_scale, second_translation = (np.asarray(values) for values in second)
    scale = second_scale * first_scale
    translation = second_scale * first_translation + second_translation
    return tuple(scale.tolist()), tuple(translation.tolist())


def _parse_axes(raw_axes: Any) -> Tuple[Axis, ...]:
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)):
        raise OMEZarrFormatError("multiscales.axes must be a list.")

    axes = []
    names = set()
    for raw_axis in raw_axes:
        if not isinstance(raw_axis, Mapping) or not isinstance(raw_axis.get("name"), str):
            raise OMEZarrFormatError("Every axis must contain a string name.")
        name = raw_axis["name"]
        if name in names:
            raise OMEZarrFormatError(f"Axis name '{name}' is duplicated.")
        names.add(name)
        axis_type = raw_axis.get("type") or "space"
        if axis_type not in {"space", "time", "channel"}:
            raise OMEZarrFormatError(f"Unsupported axis type '{axis_type}'.")
        axes.append(Axis(name=name, type=axis_type, unit=raw_axis.get("unit")))

    spatial_count = sum(axis.type == "space" for axis in axes)
    if spatial_count not in {2, 3}:
        raise OMEZarrFormatError(f"Expected two or three spatial axes, got {spatial_count}.")
    if sum(axis.type == "time" for axis in axes) > 1 or sum(axis.type == "channel" for axis in axes) > 1:
        raise OMEZarrFormatError("At most one time axis and one channel axis are supported.")

    expected_types = []
    if any(axis.type == "time" for axis in axes):
        expected_types.append("time")
    if any(axis.type == "channel" for axis in axes):
        expected_types.append("channel")
    expected_types.extend(["space"] * spatial_count)
    actual_types = [axis.type for axis in axes]
    if actual_types != expected_types:
        raise OMEZarrFormatError(
            "Axes must use the OME order: optional time, optional channel, then the listed YX or ZYX spatial axes.",
        )
    return tuple(axes)


def _validate_array(
    group: zarr.Group,
    path: str,
    axes: Tuple[Axis, ...],
    zarr_format: int,
) -> zarr.Array:
    try:
        node = group[path]
    except KeyError as error:
        raise OMEZarrFormatError(f"Multiscale dataset path '{path}' does not exist.") from error
    if not isinstance(node, zarr.Array):
        raise OMEZarrFormatError(f"Multiscale dataset path '{path}' is not a Zarr array.")
    if node.ndim != len(axes):
        raise OMEZarrFormatError(
            f"Array '{path}' has {node.ndim} dimensions, but {len(axes)} axes are declared.",
        )
    if zarr_format == 3:
        dimension_names = getattr(node.metadata, "dimension_names", None)
        expected_names = tuple(axis.name for axis in axes)
        if tuple(dimension_names or ()) != expected_names:
            raise OMEZarrFormatError(
                f"Array '{path}' dimension_names {dimension_names!r} do not match OME axes {expected_names!r}.",
            )
    return node


def _parse_omero(raw_omero: Any) -> Optional[OmeroMetadata]:
    if raw_omero is None:
        return None
    if not isinstance(raw_omero, Mapping):
        raise OMEZarrFormatError("omero metadata must be an object.")

    raw_channels = raw_omero.get("channels", [])
    if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes)):
        raise OMEZarrFormatError("omero.channels must be a list.")
    channels = []
    for raw_channel in raw_channels:
        if not isinstance(raw_channel, Mapping):
            raise OMEZarrFormatError("Every OMERO channel must be an object.")
        raw_window = raw_channel.get("window")
        window = None
        if raw_window is not None:
            if not isinstance(raw_window, Mapping):
                raise OMEZarrFormatError("An OMERO channel window must be an object.")
            try:
                window = OmeroChannelWindow(
                    start=float(raw_window["start"]),
                    end=float(raw_window["end"]),
                    min=float(raw_window["min"]),
                    max=float(raw_window["max"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise OMEZarrFormatError(
                    "An OMERO channel window must contain numeric start/end/min/max values.",
                ) from error
        channels.append(
            OmeroChannel(
                label=str(raw_channel.get("label", "")),
                color=str(raw_channel.get("color", "FFFFFF")),
                active=bool(raw_channel.get("active", True)),
                coefficient=float(raw_channel.get("coefficient", 1.0)),
                family=str(raw_channel.get("family", "linear")),
                inverted=bool(raw_channel.get("inverted", False)),
                window=window,
            ),
        )
    return OmeroMetadata(
        channels=tuple(channels),
        version=raw_omero.get("version"),
        id=raw_omero.get("id"),
        name=raw_omero.get("name"),
    )


def parse_ome_zarr_metadata(group: zarr.Group) -> OMEZarrMetadata:
    """Parse one supported OME-Zarr image group into a version-neutral model."""

    zarr_format, ome_version, namespace = _metadata_namespace(group)
    raw_multiscales = namespace.get("multiscales")
    if not isinstance(raw_multiscales, Sequence) or isinstance(raw_multiscales, (str, bytes)):
        if "plate" in namespace:
            detail = "plate"
        elif "bioformats2raw.layout" in namespace or "series" in namespace:
            detail = "multi-image collection"
        else:
            detail = "non-image group"
        raise OMEZarrFormatError(f"Opening an OME-Zarr {detail} is not supported.")
    if len(raw_multiscales) != 1:
        raise OMEZarrFormatError(
            f"Exactly one multiscales entry is supported; found {len(raw_multiscales)}.",
        )
    if "labels" in namespace or "labels" in group:
        raise OMEZarrFormatError("OME-Zarr label images are not supported.")

    raw_multiscale = raw_multiscales[0]
    if not isinstance(raw_multiscale, Mapping):
        raise OMEZarrFormatError("The multiscales entry must be an object.")
    if zarr_format == 2:
        declared_version = raw_multiscale.get("version")
        if declared_version is not None and _version_family(declared_version) != "0.4":
            raise OMEZarrFormatError(
                f"OME-Zarr {declared_version} with Zarr v2 is unsupported; expected OME-Zarr 0.4.",
            )

    axes = _parse_axes(raw_multiscale.get("axes"))
    ndim = len(axes)
    group_transform = _parse_transform_sequence(
        group,
        raw_multiscale.get("coordinateTransformations"),
        ndim,
        require_scale=False,
    )

    raw_datasets = raw_multiscale.get("datasets")
    if not isinstance(raw_datasets, Sequence) or isinstance(raw_datasets, (str, bytes)) or not raw_datasets:
        raise OMEZarrFormatError("multiscales.datasets must be a non-empty list.")

    datasets = []
    nonspatial_indices = tuple(i for i, axis in enumerate(axes) if axis.type != "space")
    previous_spatial_shape = None
    nonspatial_shape = None
    for raw_dataset in raw_datasets:
        if not isinstance(raw_dataset, Mapping) or not isinstance(raw_dataset.get("path"), str):
            raise OMEZarrFormatError("Every multiscale dataset must contain a string path.")
        path = raw_dataset["path"]
        array = _validate_array(group, path, axes, zarr_format)
        current_nonspatial_shape = tuple(array.shape[i] for i in nonspatial_indices)
        if nonspatial_shape is None:
            nonspatial_shape = current_nonspatial_shape
        elif current_nonspatial_shape != nonspatial_shape:
            raise OMEZarrFormatError("Time and channel dimensions must be consistent across all resolution levels.")

        spatial_shape = tuple(array.shape[i] for i, axis in enumerate(axes) if axis.type == "space")
        if previous_spatial_shape is not None and any(
            current > previous for current, previous in zip(spatial_shape, previous_spatial_shape, strict=True)
        ):
            raise OMEZarrFormatError("Multiscale datasets must be ordered from highest to lowest resolution.")
        previous_spatial_shape = spatial_shape

        dataset_transform = _parse_transform_sequence(
            group,
            raw_dataset.get("coordinateTransformations"),
            ndim,
            require_scale=True,
        )
        scale, translation = _compose_affine(dataset_transform, group_transform)
        datasets.append(MultiscaleDataset(path=path, scale=scale, translation=translation))

    multiscales = Multiscales(
        axes=axes,
        datasets=tuple(datasets),
        version=ome_version,
        name=raw_multiscale.get("name"),
    )
    return OMEZarrMetadata(
        zarr_format=zarr_format,
        ome_version=ome_version,
        multiscales=multiscales,
        omero=_parse_omero(namespace.get("omero")),
    )


def spatial_transform_angstrom(
    multiscales: Multiscales,
    dataset: MultiscaleDataset,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return spatial scale and origin in Angstrom, preserving listed axis order."""

    steps = []
    origins = []
    for index in multiscales.spatial_indices:
        axis = multiscales.axes[index]
        unit = axis.unit or "angstrom"
        try:
            factor = UNITFACTOR[unit]
        except KeyError as error:
            raise OMEZarrFormatError(f"Unsupported spatial unit '{unit}' on axis '{axis.name}'.") from error
        steps.append(dataset.scale[index] * factor)
        origins.append(dataset.translation[index] * factor)
    return tuple(steps), tuple(origins)

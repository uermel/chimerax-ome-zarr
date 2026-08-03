# chimerax-ome-zarr
Plugin providing OME-Zarr 0.4 and 0.5 image support for ChimeraX.

**Currently supported:**
- OME-Zarr 0.4 stored as Zarr v2 and OME-Zarr 0.5 stored as Zarr v3.
- 2D and 3D images in YX/ZYX order, with optional leading time and channel axes.
- Time series, multichannel images, and combined multichannel time series.
- Bioformats2raw layout 3 image collections, using declared or consecutively numbered series paths.
- OMERO channel names, colors, and active state.
- OME-Zarr label collections and image-label metadata, including colors, properties, and source-image association.
- Lazy integer index maps for ChimeraX's `segmentation surfaces` / `segmentation colors` commands.
- Native, editable ChimeraX `Segmentation` models for declared nonzero label values.
- Local and remote Zarr stores supported by `fsspec`.
- Identity, scale, and translation transformations, converted to ChimeraX Angstrom coordinates.
- Loading specific resolution levels as separate volumes.
- Loading an integer-scaled, aligned pyramid as one Volume whose resolution follows ChimeraX's `step` setting.

**Currently not supported:**
- Plates and multi-image collection formats other than `bioformats2raw.layout`.
- More than one `multiscales` entry in an image group.
- Custom axes or spatial-axis reordering; spatial axes are interpreted positionally as YX or ZYX for CoPick compatibility.
- Non-integer pyramid scaling.
- Automatic switching between translated levels that do not align with the finest grid. Such levels can still be opened separately with `scales`.

Contributions are welcome.

## Installation

### From the ChimeraX toolshed

Now available on the ChimeraX toolshed! To download and install run this in the ChimeraX command prompt:

```
toolshed reload available
toolshed install ome-zarr
```


### From GitHub release

1. Install [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html)
2. Download the most recent build from the [releases page.](https://github.com/uermel/chimerax-ome-zarr/releases)
3. Run the following command in the ChimeraX command prompt to install the plugin:
```
toolshed install /path/to/ChimeraX_OME_Zarr-0.5.3-py3-none-any.whl
```
4. Restart ChimeraX


## Usage

This plugin integrates with the ChimeraX `open`-command.

**To open a local OME-Zarr file:**
```
open /path/to/file.zarr
```

**To open a remote OME-Zarr file:**
```
open ngff:s3://bucket-name/path/to/file.zarr
```

**To open a remote OME-Zarr file and load specific scales:**
```
open ngff:s3://bucket-name/path/to/file.zarr scales 1,2
```

Associated label images are discovered automatically and opened hidden. Each label image is available as an integer
index map, while label values declared in OME `colors` or `properties` metadata also appear as editable masks in
ChimeraX's **Segmentations** tool. To skip associated labels:

```
open /path/to/file.zarr labels false
```

Label edits are copy-on-write: they update the in-session index map and sibling masks, but never modify the opened
OME-Zarr store. Use ChimeraX's existing segmentation save formats to export an edited mask. If a label image does not
declare its label values, the plugin preserves the complete index map without scanning the finest-resolution array,
but does not create editable masks automatically. Value 0 is treated as background by the ChimeraX segmentation bridge.

**NOTE:** in order to open files from remote locations other than S3, you may have to install additional python
packages (e.g. `smbprotocol` for SAMBA shares).


### Zarr store backend authentication

Authentication to the Zarr storage backend (e.g. S3) may fail if ChimeraX is launched without the appropriate environment
variables present.

#### MacOS
To prevent auth problems, make sure necessary environment variables are set in `~/.zprofile`. This plugin will attempt to
set these variables automatically if they are not present.


Alternatively, launch ChimeraX from a shell that has the necessary environment variables set. Typically the executable
should exist in a location similar to:
```
/Applications/ChimeraX-1.7.1.app/Contents/bin/ChimeraX
```

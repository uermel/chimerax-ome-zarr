# chimerax-ome-zarr
OME-Zarr v0.4 support for ChimeraX.

**Currently supported:**
- xyz-order multiscale volumes.
- Opening local or remote zarr files
- Loading specific multiscales as volumes (streamed on demand and cached in memory)
- Loading all multiscales as a single Volume, accessible using Chimerax's `step` setting

**Currently not supported:**
- Labels data
- Channel axes
- Time axes
- non-integer scaling
- translation transformations
- 2D data
- multi-image files

## Installation

1. Install [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html)
2. Download the most recent build from the [releases page.](https://github.com/czimaginginstitute/chimerax-ome-zarr/releases)
3. Run the following command in the ChimeraX command prompt to install the plugin:
```
toolshed install /path/to/ChimeraX_OME_Zarr-0.4-py3-none-any.whl
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

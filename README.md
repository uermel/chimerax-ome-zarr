# chimerax-ome-zarr
OME-Zarr support for ChimeraX. Currently only supports **opening** local or remote Zarr files, not saving.

## Installation

1. Install [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html)
2. Download the most recent build from the [releases page.](https://github.com/czimaginginstitute/chimerax-ome-zarr/releases)
3. Run the following command in the ChimeraX command prompt to install the plugin:
```
toolshed install /path/to/ChimeraX_OME_Zarr-0.1-py3-none-any.whl
```
4. Restart ChimeraX
5. Optional: Use the `envfile set` command to provide a file that can be sourced to set environment variables necessary for authentication with the storage backend. 


## Usage

This plugin integrates with the ChimeraX `open`-command. Currently, the smallest scale is loaded by default. 
Users can specify the scale to load using the `scales` flag and a comma-separated list of scale indeces. The remote store
type can be specified using the `fs` flag. Currently, only `s3` is supported.

### Examples:

**Open a local zarr file**
```
open /path/to/file.zarr format zarr
```

**Open a zarr file from S3** (will use default AWS profile, set environment variable `AWS_PROFILE` in the shell that runs
ChimeraX to change)
```
open bucket-name/path/to/file.zarr format zarr fs s3
```

**Open a zarr file from S3, load 3 scales** 
```
open bucket-name/path/to/file.zarr format zarr fs s3 scales 0,1,2
```

### Zarr store backend authentication

Authentication to the Zarr storage backend (e.g. S3) may fail if ChimeraX is launched without the appropriate environment variables set. To prevent, launch
ChimeraX from terminal or use the command `envfile set` to provide a file that can be sourced to provide the correct environment. 

For example, on MacOS using zsh:
```
envfile set /Users/example.user/.zprofile
```

The location of this file is stored as a ChimeraX setting and can be printed using `envfile get` or cleared using `envfile clear`.


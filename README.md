# chimerax-ome-zarr
OME-Zarr support for ChimeraX. Currently only supports **opening** local or remote Zarr files, not saving.

## Installation

1. Install [ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html)
2. Download the most recent build from the [releases page.]()
3. Run the following command in the ChimeraX command prompt to install the plugin:
```
toolshed install /path/to/ChimeraX_OME_Zarr-0.1-py3-none-any.whl
```
4. Restart ChimeraX

## Usage

This plugin integrates with the ChimeraX `open`-command. Currently, the smallest scale is loaded by default. 
Users can specify the scale to load using the `scales` flag and a comma-separated list of scale indeces. The remote store
type can be specified using the `fs` flag. Currently, only `s3` is supported.

Examples:

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

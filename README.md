[![PyPI](https://img.shields.io/pypi/v/ridal.svg)](https://pypi.org/project/ridal/)
[![Crates.io](https://img.shields.io/crates/v/ridal.svg)](https://crates.io/crates/ridal)
[![CI](https://github.com/erikmannerfelt/ridal/actions/workflows/rust.yml/badge.svg)](
https://github.com/erikmannerfelt/ridal/actions/workflows/rust.yml
)
[![codecov](https://codecov.io/github/erikmannerfelt/ridal/graph/badge.svg?token=187FAAJB4B)](https://codecov.io/github/erikmannerfelt/ridal)

# ![](https://raw.githubusercontent.com/erikmannerfelt/ridal/v0.5.0/images/logo.svg) Ridal — Speeding up Ground Penetrating Radar (GPR) processing
The aim of `ridal` is to quickly and accurately process GPR data.
In one command, most data can be processed in pre-set profiles or with custom filter settings, and batch modes allow for sequences of datasets to be processed with the same settings.
Built in [rust](https://rust-lang.org/) with a high focus on testing and performance, `ridal` may be for you if large data volumes and strange fileformats are common issues.

The name is a take on the loosely defined "Data Abstraction Library" (DAL) projects like [GDAL](https://gdal.org) and [PDAL](https://pdal.org), but for radar.
A near-term goal of Ridal is to enable easy conversion between formats, such as `ridal convert input.rad output.dzt` (this is not yet implemented).


Much of the functionality has been inspired from the projects [RGPR](https://github.com/emanuelhuber/RGPR) and [ImpDAR](https://github.com/dlilien/ImpDAR); both of which are more mature projects.
For example, Ridal currently only works on Malå (.rd3), GSSI (.dzt) and pulseEKKO (.dt1) radar formats.
For many uses, these will more likely be the tools for you!

![Image of a glacier radargram](https://raw.githubusercontent.com/erikmannerfelt/ridal/v0.5.0/images/kroppbreen_rgm.webp)
*Radargram (100 MHz Malå) of Kroppbreen in Svalbard. Collected 28 Feb. 2023.*

### Notes
- Prior to Feb. 2026, this program was called `rsgpr`.
- The CLI/Python interface changed completely in version 0.5. See [Issue #82](https://github.com/erikmannerfelt/ridal/issues/82) for more info.

### Installation

#### Requirements
- `cargo`(only for the CLI; not the python package). Easiest installed using [rustup](https://rustup.rs).
- `gdal` (optional, for sampling heights from DEMs). For Debian or derivatives, this means `gdal-bin`.
- `proj` (optional, for CRS support other than WGS84 UTM Zones). For Debian or derivatives, this means `proj-bin`.

Using cargo, the `ridal` CLI can be installed (after installing the requirements):
```bash
cargo install ridal
```

with nix, the flake can be used without worrying about the requirements above:
```nix
inputs = {
  ridal.url = "github:erikmannerfelt/ridal";
};
```
or in an ephemeral shell:
```bash
nix shell github:erikmannerfelt/ridal
```

#### Python
There's an early implementation of a Python package:

```bash
pip install ridal
```

```python
>>> import ridal
# Print useful info as a dictionary
>>> ridal.info("path/to/file.rad")
'{...}'
# Process and save to a file
>>> ridal.process("path/to/file.rad", steps=["zero_corr", "auto_gain"], output="processed.nc")
# Process and load into memory
>>> ridal.process("path/to/file.rad", steps=["zero_corr"], return_dataset=True, return_dataset_format="xarray")
<xarray.Dataset> Size: ...
Dimensions:     (y: ..., x: ...)
Coordinates:
...
# Only load into memory with no processing
>>> ridal.read("path/to/file.rad", return_dataset_format="xarray")
<same as above>
```

See [scripts/render_kroppbreen.py](https://github.com/erikmannerfelt/ridal/blob/main/scripts/render_kroppbreen.py) for an example of how it can be used.


### Simple CLI usage
See the help page of `ridal` for info on how to interact with the CLI:
```bash
ridal -h
```

To toggle useful information on a file, the `info` subcommand shows the metadata and a summary of the location data:
```bash
ridal info DAT_001_A1.rd3
```

Processing a file using the default processing profile:

```bash
ridal process DAT_001_A1.rd3 --default
```

**All processing steps** are shown in the [steps.md](https://github.com/erikmannerfelt/ridal/blob/main/steps.md) file. It can also be printed with `ridal steps`.

A processing step pipeline is defined using `ridal process file.rad --steps "zero_corr,dewow,..."` or using a file: `--steps steps.txt`:
```bash
subset(1 100) # Comments are supported!
zero_corr
dewow

correct_topography
```

The output will be a NetCDF file with the same name but an `.nc` suffix.
By default, the output is saved in the same directory as the input.
For more control, the output directory and/or filename can be controlled with `-o` or `--output`.

To process multiple files in "batch mode", provide a ["glob"](https://en.wikipedia.org/wiki/Glob_(programming)) pattern as the filename.
Optionally, for many sequential files, the `--merge` argument allows merging multiple files into one.
```bash
ridal batch-process data/*.rd3 --merge "10 min" --default -o output/
```

A rudimentary profile renderer is available with the `-r` argument.
This will be saved in the same location as the output file as a JPG if another filename is not given.


### Interpreting layers

Reflectors can be picked in the GUI and exported as evenly spaced geographic points.
Picking needs somewhere to save, which means a project — a directory with a `ridal.toml` in it:
```bash
ridal project init my-survey
ridal gui my-survey
```
Pointing the GUI at a plain directory still works exactly as before; it is simply read-only.

Layers are defined once per project, with a name and a colour, and picked lines refer to them rather than carrying their own.
Renaming a layer therefore orphans nothing.
Picks themselves are stored as [gprinterp](https://github.com/erikmannerfelt/gprinterp) documents, one per user per radargram, and those are the source of truth.
The point product is derived from them on demand rather than stored alongside them, so reprocessing a radargram does not invalidate the picks drawn on it.

Points are spaced by arc length along the ground track in metres, not by trace number.
Trace spacing varies with survey speed and GPS noise perturbs it further, so "every 10th trace" and "every 25 m" are different products, and only the second still means something once the points leave the radargram.

The same export is available without a browser:
```bash
ridal interp export line.nc picks.gprinterp.json -o points.geojson --spacing 25
```
Each point carries its layer, trace and sample, distance along the profile, two-way travel time, depth, and both projected and WGS84 coordinates — along with which radargram and which processing revision it came from, since depth depends on how the radargram was processed.


## Papers using Ridal

- [Kleber et al. (2023): Groundwater springs formed during glacial retreat are a large source of methane in the high Arctic](https://doi.org/10.1038/s41561-023-01210-6)
- [Harcourt et al. (2026): Surging glaciers in Svalbard: Observing their distribution, characteristics and evolution](https://doi.org/10.1016/j.earscirev.2026.105410)

... and many others in preparation/review

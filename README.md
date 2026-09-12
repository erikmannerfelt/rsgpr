[![PyPI](https://img.shields.io/pypi/v/ridal.svg)](https://pypi.org/project/ridal/)
[![Crates.io](https://img.shields.io/crates/v/ridal.svg)](https://crates.io/crates/ridal)
[![CI](https://github.com/erikmannerfelt/ridal/actions/workflows/rust.yml/badge.svg)](
https://github.com/erikmannerfelt/ridal/actions/workflows/rust.yml
)
[![codecov](https://codecov.io/github/erikmannerfelt/ridal/graph/badge.svg?token=187FAAJB4B)](https://codecov.io/github/erikmannerfelt/ridal)

# ![](https://raw.githubusercontent.com/erikmannerfelt/ridal/v0.5.0/images/logo.svg) Ridal — Speeding up Ground Penetrating Radar (GPR) processing
The aim of `ridal` is to quickly and accurately process GPR data.
In one command, most data can be processed in pre-set profiles or with custom filter settings, and batch modes allow for sequences of datasets to be processed with the same settings.
Once processed, the results can be browsed and inspected in a [browser GUI](#browser-gui).
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

An image of the result can be written at the same time with the `-r` argument.
This is saved next to the output file as a JPG if another filename is not given.

Already-processed files can be rendered on their own with the `render` subcommand, which takes a `.nc` and writes a PNG or JPG:
```bash
ridal render processed.nc -o out.png --profile positive --width 8000
```
The format follows the extension, and `--width` is capped at one pixel per trace, which is also the default.
`--profile` takes either one of the built-in names or a path to a TOML file of your own.
Both of these draw the same picture the browser GUI does, from the same profiles and the same code.


### Browser GUI

Processed files can be browsed in a local web GUI:
```bash
ridal gui path/to/processed/
```
This opens a browser on every Ridal `.nc` file it can find below that directory, grouped by survey, each group with a map of its tracks.
Opening one gives a pan/zoom view of the radargram where the cursor reports trace number, distance and two-way travel time, alongside a second map showing where on the ground that cursor is.
Radargrams are rendered server-side in tiles as they are needed, so a file larger than memory is no obstacle.

Four rendering profiles are available (`default`, `positive`, `abslog` and `high-contrast`), since the settings that make a bed reflector legible rarely make the internal layers legible too.

For something longer-lived than `ridal gui`, which picks an ephemeral port and opens a browser, `ridal server start` binds a fixed port and stays up:
```bash
ridal server start path/to/processed/ --port 8080
```
That is the mode to put behind a reverse proxy or run as a systemd service.
Out of the box it serves everything to anyone who can reach the port; see [sharing a project](#sharing-a-project-with-other-people) for how to put accounts in front of it.


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


### Sharing a project with other people

A project with no accounts works the way it always has: everyone using it is the user `default`, and `ridal gui` asks for no login.
Accounts start the first time one is made, which has to be done from the command line since there is no administrator yet to allow it:
```bash
ridal project user add erik --role admin
```
That prints a one-time invite link, good for a week.
Send it however you normally would; opening it is what sets the password, so nobody else ever learns it.
A forgotten password is the same command again, `ridal project user reset`.

Each person gets a role and a download scope, and the two are set separately.
The role is what they may do, with each level including the ones below it:

- `viewer` reads the catalog, the radargrams and the layer names, and sets their own preferences.
- `picker` also writes their own interpretation.
- `operator` also edits the layer list and the project defaults.
- `admin` also manages accounts, roles, download scopes and the access settings.

The download scope is what they may take away: `none`, `picks`, `derived` or `all`.
It is kept apart from the role because a picker who may not export the raw data and a viewer who may export everything are both sensible.
`derived` is usually the interesting line, since the level 2 points are often enough to publish with.

Interpretations belong to whoever drew them.
One user cannot change another's picks, and neither can an admin, because that is how the files are laid out rather than a permission anyone can grant.
Removing an account leaves its picks in place.

To serve a project to other machines:
```bash
ridal server start my-survey --host 127.0.0.1 --port 8000
```
Ridal does not do TLS itself, so the supported arrangement is to leave it on loopback behind a reverse proxy that does.
Binding a public address refuses to start unless the project has accounts, and refuses password logins unless `--allow-insecure-login` says you know what is in front of it.
`--read-only` still works, and now means everyone is capped at `viewer` whatever their account says.

Note that the download scope limits downloading rather than seeing.
Anyone who can open a radargram is already looking at the image and at the track on the map, and could save those a piece at a time whatever the scope says.
It is there to keep bulk downloads deliberate, so if somebody should not have the data at all, do not give them access to the project.


The GUI, interpretation, accounts and `ridal render` all arrived after `v0.5.2`, so they need version 0.6.0 or newer.
Until that is on crates.io, a git install has them:
```bash
cargo install --git https://github.com/erikmannerfelt/ridal
```


## Papers using Ridal

- [Kleber et al. (2023): Groundwater springs formed during glacial retreat are a large source of methane in the high Arctic](https://doi.org/10.1038/s41561-023-01210-6)
- [Harcourt et al. (2026): Surging glaciers in Svalbard: Observing their distribution, characteristics and evolution](https://doi.org/10.1016/j.earscirev.2026.105410)
- [Kleber et al. (2026): Subglacial geology and thermal conditions regulate methane emissions from Svalbard glaciers](https://doi.org/10.1038/s41467-026-77190-z)

... and many others in preparation/review

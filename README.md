# rsgpr — Speeding up Ground Penetrating Radar (GPR) processing
The aim of `rsgpr` is to quickly and accurately process GPR data.
In one command, most data can be processed in pre-set profiles or with custom filter settings, and batch modes allow for sequences of datasets to be processed with the same settings.

Thank you to the creators of [RGPR](https://github.com/emanuelhuber/RGPR) from which the name and functionality where both inspired.

This is still an early WIP, and currently only works with Malå (.rd3) radar formats.

### Possible meanings of the name
- Really Simple GPR
- rust GPR
- [rsgpr](./README.md) simplifies GPR


### Installation

#### Requirements
- `cargo` for installing rust projects
- `gdal` (optional, for sampling heights from DEMs). For Debian or derivatives, this means `gdal-bin`.
- `proj` (optional, for CRS support other than WGS84 UTM Zones). For Debian or derivatives, this means `proj-bin`.

Using cargo, `rsgpr` can be installed from the repo (after installing the requirements):
```bash
cargo install --git https://github.com/erikmannerfelt/rsgpr.git
```

with nix, the flake can be used without worrying about the requirements above:
```nix
{
  inputs = {
    rsgpr.url = "github:erikmannerfelt/rsgpr";
  };
}
```
or in an ephemeral shell:
```bash
nix shell github:erikmannerfelt/rsgpr

```


### Simple usage
See the help page of `rsgpr` for info on how to interact with the CLI:
```bash
rsgpr -h
```

To toggle useful information on a file, the `-i` or `--info` argument shows the metadata and a summary of the location data:
```bash
rsgpr -f DAT_001_A1.rd3 -i
```

Processing a file using the default processing profile:

```bash
rsgpr -f DAT_001_A1.rd3 --default
```

The output will be a NetCDF file with the same name but an `.nc` suffix.
By default, the output is saved in the same directory as the input.
For more control, the output directory and/or filename can be controlled with `-o` or `--output`.

To process multiple files in "batch mode", provide a ["glob"](https://en.wikipedia.org/wiki/Glob_(programming)) pattern as the filename.
Optionally, for many sequential files, the `--merge` argument allows merging multiple files into one.
```bash
rsgpr -f "data/*.rd3" --merge "10 min" --default -o output/
```

A rudimentary profile renderer is available with the `-r` argument.
This will be saved in the same location as the output file as a JPG if another filename is not given.

# cets-imod

Converters between [IMOD](https://bio3d.colorado.edu/imod/) files and CETS, the cryo-electron
tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | IMOD files | CETS objects | Directions |
|---|---|---|---|
| Tilt-series | stack (`.mrc`/`.mrcs`), tilt angles (`.rawtlt`, `.tlt`), alignment (`.xf`) | `TiltSeries`, `Alignment`, `Instrument`, `AcquisitionSession` | IMOD → CETS, CETS → IMOD |
| CTF | `.defocus` (ctfplotter / ctfphaseflip format) | `CTFMetadata` (one per tilt-image) | IMOD → CETS, CETS → IMOD |
| Tomogram | `.mrc` (and its even/odd halves) | `Tomogram` | IMOD → CETS, CETS → IMOD |

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `imod` package and the `cets-imod` command.

## Command line

```text
cets-imod <data type> {to-cets,from-cets} [options]
```

`to-cets` writes a CETS yaml file with one entry per kind of object (`tilt_series`, `alignment`,
`instrument`, `acquisition_session`, `ctf`, `tomogram`); `from-cets` reads it. Every
sub-command has `--help`. Errors print a one-line message and exit with code 1; put `--debug`
before the data type to also print the traceback (`cets-imod --debug tilt-series ...`).
Output directories are created if needed.

### Tilt-series

```bash
# IMOD -> CETS: the minimal command
cets-imod tilt-series to-cets --stack TS_03.mrc --tilt-angles TS_03.rawtlt -o TS_03_cets.yaml

# IMOD -> CETS with the alignment, the CTF and the (optional) acquisition values
cets-imod tilt-series to-cets --stack TS_03.mrc \
    --tilt-angles TS_03.rawtlt --refined-tilt-angles TS_03.tlt \
    --xf TS_03.xf --defocus TS_03.defocus \
    --voltage 300 --spherical-aberration 2.7 --amplitude-contrast 0.07 \
    --tilt-axis-angle 85.3 -o TS_03_cets.yaml

# CETS -> IMOD
cets-imod tilt-series from-cets -i TS_03_cets.yaml \
    --tlt out/TS_03.tlt --rawtlt out/TS_03.rawtlt --xf out/TS_03.xf
```

| `to-cets` option | Meaning |
|---|---|
| `--stack` (required) | Tilt-series stack. |
| `--tilt-angles` (required) | Nominal tilt angles (`.rawtlt`, or a `.tlt` holding them), optionally with the dose (2nd column) and acquisition order (3rd column). |
| `--refined-tilt-angles` | Angles refined by the alignment (the `.tlt` written by tiltalign). |
| `--xf` | Alignment. Without it, the identity is used. |
| `--defocus` | CTF of the tilt-images. |
| `--even`, `--odd` | Even/odd stacks. |
| `--ctf-corrected` | The stack is corrected for the CTF. |
| `--pixel-size` | Pixel size (Å). Default: the one of the stack header; an error is raised if neither is known. |
| `--voltage`, `--spherical-aberration`, `--amplitude-contrast`, `--dose-rate`, `--tilt-axis-angle` | Optional acquisition values, which IMOD files do not record. Only what is given is written: the `instrument` needs `--voltage`, and the `acquisition_session` any of them (it links the tilt-series to the instrument). Without them, the yaml has neither, and the tilt-series no `acquisition_session_id`. |

| `from-cets` option | Meaning |
|---|---|
| `-i` (required) | CETS yaml with the tilt-series and, for `--xf`, its alignment. |
| `--tlt` (required) | Tilt angles IMOD reconstructs with: the refined ones if the alignment has one for every tilt-image, else the nominal ones. |
| `--rawtlt` | Nominal tilt angles. |
| `--xf` | Alignment. |
| `--no-dose` | Write only the angles column. |

### CTF

```bash
cets-imod ctf to-cets --stack TS_03.mrc --defocus TS_03.defocus -o TS_03_ctf.yaml
# From a tilt-series yaml (its tilt-images' CTF and nominal tilt angles)...
cets-imod ctf from-cets -i TS_03_cets.yaml -o out/TS_03.defocus
# ...or from the output of 'ctf to-cets', which has no tilt angles
cets-imod ctf from-cets -i TS_03_ctf.yaml --tilt-angles TS_03.rawtlt -o out/TS_03.defocus
```

### Tomogram

```bash
cets-imod tomogram to-cets --tomogram TS_03_rec.mrc [--even E.mrc --odd O.mrc] -o TS_03_tomo.yaml
cets-imod tomogram from-cets -i TS_03_tomo.yaml --output-dir out [--copy]
```

`from-cets` makes the tomogram available as `<id>.mrc` (and `<id>_even.mrc`, `<id>_odd.mrc`):
the files are linked, or copied with `--copy` or when the voxel size of their header differs
from the CETS one (the header of the copy is then fixed).

## Python API

```python
from imod.converters.ctf import ImodCtfSeries
from imod.converters.tilt_series import ImodTiltSeries
from imod.converters.tomogram import ImodTomogram

ctf = ImodCtfSeries("TS_03.mrc", "TS_03.defocus").imod_to_cets()
# instrument / session are None if their acquisition values are not given
ts, alignment, instrument, session = ImodTiltSeries(
    "TS_03.mrc", "TS_03.rawtlt", ctf_md_list=ctf, refined_tilt_angles="TS_03.tlt",
    voltage=300, tilt_axis_angle=85.3,
).imod_to_cets(xf_file="TS_03.xf")

ImodTiltSeries.cets_to_imod(ts, "out/TS_03.tlt", alignment=alignment, xf_file="out/TS_03.xf")
tomogram = ImodTomogram("TS_03_rec.mrc").imod_to_cets()
```

`cets_to_imod` also accepts yaml files (written by the converters or by the command line) in
place of the objects. `cets_data_model.utils.yaml_io` reads and writes the command-line yaml
files.

## Conventions

- **Units.** CETS lengths are in Å: the `.xf` shifts are converted from pixels with the pixel
  size of the tilt-images, and the defoci from nm (underfocus positive in both).
- **Alignment.** Each tilt-image gets a `ProjectionAlignment` whose `sequence` is
  `[Affine, Translation]`: the `.xf` rotation, then its shift (2-D, in Å), from the tilt-image
  physical frame to the tilt-series aligned frame.
- **Tilt angles.** `TiltImage.nominal_tilt_angle` is the angle reported by the microscope; the
  angle refined by an alignment is `ProjectionAlignment.refined_tilt_angle`, as several
  alignments may refine it differently.
- **Defocus files.** Every flag combination of the version 3 format is read (astigmatism,
  angles and phase shifts in degrees or radians, inverted tilt angles, cut-on frequency).
  Written files are version 3 with astigmatism and, if any tilt-image has one, phase shift
  (flag 1 or 5).
- **Sections** are 0-based.

## Limitations

- The rotation centre of the `.xf` alignment is not stored: IMOD rotates about the centre of
  the image, while the CETS tilt-image frame has its origin at the corner. The values round
  trip through this package, but applying them as a CETS transformation needs that offset.
  This waits for a decision on the frame of `ProjectionAlignment`.
- Excluded views (`excludelist`) and the reconstruction geometry of `tilt.com` (thickness,
  shifts, binning) are not converted; CETS has no place for them yet.
- The cut-on frequency of the defocus files is read but not kept: `CTFMetadata` has no field
  for it.
- The acquisition values (voltage, tilt axis angle, ...) are not written back, as no IMOD file
  written here has a place for them.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest imod/tests/tests_*.py      # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy imod
```

`imod/tests/tests_cets_imod.py::CetsImodTsReaderTest::test_ts_imod_to_cets` is a placeholder
that fails until the tilt-series reader test is written. Example scripts are in
`imod/scripts/`.

### Tests

| File | What it tests |
|---|---|
| `imod/tests/tests_cets_imod.py` | The CTF converter on every `.defocus` format of the test data. |
| `imod/tests/tests_cets_to_imod_tomogram.py` | The tomogram converter, both directions. |
| `imod/tests/tests_cli.py` | Every `cets-imod` sub-command, on small synthetic files (built in a temporary directory), including the optional acquisition values. |
| `imod/tests/tests_cli_fixtures.py` | Every `cets-imod` sub-command on the data of a real IMOD processing (`imod/tests/data`): the outputs are compared with the values of the input files and with reference CETS files, plus the round trips, the options and the errors. |

```bash
pytest imod/tests/tests_cli_fixtures.py                 # one file
pytest imod/tests/tests_cli_fixtures.py -k ctf          # some tests
```

The test data (`imod/tests/data`, described by `ImodTestDataFiles` in
`imod/tests/__init__.py`; another location can be given with the `TEST_DATA_ROOT` environment
variable) is the tilt-series TS_03, with 40 tilt-images:

| File | Content |
|---|---|
| `tilt_series/TS_03.mrcs` | The stack (464 x 480 pixels, 10.8 Å/pixel). |
| `tilt_series/TS_03.tlt` | Nominal tilt angles and accumulated dose (2 columns: the acquisition order is inferred from the dose). |
| `tilt_series/03.rawtlt` | Nominal tilt angles only. |
| `alignment_files/TS_03.xf` | Alignment of the tilt-images. |
| `imod_defocus_files/TS_03_plain_estimation.defocus` | CTF: defocus only (ctfplotter flag 0, no header line). |
| `imod_defocus_files/TS_03_only_astigmatism.defocus` | CTF with astigmatism (flag 1). |
| `imod_defocus_files/TS_03_only_phase_shift.defocus` | CTF with phase shift (flag 4). |
| `imod_defocus_files/TS_03_astigmatism_and_phase_shift.defocus` | CTF with astigmatism and phase shift (flag 5). |
| `imod_defocus_files/TS_03_astigmatism_phase_shift_and_cutoff_freq.defocus` | CTF with astigmatism, phase shift and cut-on frequency (flag 37). |
| `tomograms/TS_03.mrc` | The tomogram (480 x 464 x 40 voxels, 10.8 Å/voxel). |
| `TS_03_parameters.txt` | The `tilt-series to-cets` acquisition options of TS_03 (`--voltage`, `--spherical-aberration`, `--amplitude-contrast`, `--dose-rate`, `--tilt-axis-angle`, `--ctf-corrected`), which IMOD files do not record, and the pixel size of the stack (`--pixel-size`, the one of its header). |
| `cets_files/TS_03_cets_ts.yaml` | Reference CETS tilt-series (its paths are relative to `cets_files`). |
| `cets_files/TS_03_cets_ts_alignment.yaml` | Reference CETS alignment. |
| `cets_files/TS_03_cets_ctf.yaml` | Reference CETS CTF (one document per tilt-image). |

The reference CETS files are the output of `tilt-series to-cets` with `TS_03.mrcs`,
`TS_03.tlt`, `TS_03.xf` and `TS_03_only_astigmatism.defocus` and the options of
`TS_03_parameters.txt` (`--ctf-corrected False` means the flag is not given).

If the conversion changes on purpose, regenerate the reference CETS files with that command
and review their diff. Every output of the tests is written to a temporary directory; the
files in `imod/tests/data` are never modified.

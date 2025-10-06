from pathlib import Path
from imod.converters.ctf import ImodCtfSeries
from imod.converters.tilt_series import ImodTiltSeries
from imod.converters.tomogram import ImodTomogram
from imod.tests import ImodTestDataFiles
from imod.utils.utils import parse_tlt_file

### IMOD TO CETS #################################################################
# Files
ts_fn = ImodTestDataFiles.ts_03_mrcs.path
tomo_fn = ImodTestDataFiles.ts_03_tomogram.path
defocus_fn = ImodTestDataFiles.defocus_only_astigmatism.path
tlt_fn = ImodTestDataFiles.ts_03_tlt.path
xf_fn = ImodTestDataFiles.ts_03_xf.path
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
yaml_file_ts = scratch_dir / "TS_03_cets_ts.yaml"
yaml_file_ctf = scratch_dir / "TS_03_cets_ctf.yaml"
yaml_file_tomo = scratch_dir / "TS_03_cets_tomo.yaml"
# CTF metadata
ics = ImodCtfSeries(ts_file_name=ts_fn, defocus_file=defocus_fn)
cets_ctf_md_list = ics.imod_to_cets(out_yaml_file=yaml_file_ctf)
# TS Metadata
its = ImodTiltSeries(
    ts_file_name=ts_fn, tilt_angles=tlt_fn, ctf_md_list=cets_ctf_md_list
)
cets_ts_md = its.imod_to_cets(xf_file=xf_fn, out_yaml_file=yaml_file_ts)
# Tomogram metadata
it = ImodTomogram(tomo_file=tomo_fn)
cets_tomo_md = it.imod_to_cets(out_yaml_file=yaml_file_tomo)

### CETS TO IMOD #################################################################
# Files
out_tlt = scratch_dir / "cets_to_imod_TS_03.tlt"
out_xf = scratch_dir / "cets_to_imod_TS_03.xf"
out_defocus = scratch_dir / "cets_to_imod_TS_03.defocus"
# Write the defocus
tilt_angles: list[float]
doses: list[float]
orders: list[int]
tilt_angles, doses, orders = parse_tlt_file(tlt_fn)
ics.cets_to_imod(yaml_file_ctf, tilt_angles, out_defocus)
# Write the tlt and xf
its.cets_to_imod(
    cets_ts=cets_ts_md, tlt_file=out_tlt, add_dose_to_tlt=True, xf_file=out_xf
)

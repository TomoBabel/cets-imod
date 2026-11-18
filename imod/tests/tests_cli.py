# -*- coding: utf-8 -*-
import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    AcquisitionSession,
    Alignment,
    Instrument,
    TiltSeries,
    Tomogram,
)
from cets_data_model.utils.yaml_io import has_cets_key, load_cets_object
from imod.cli import main

VOXEL_SIZE = 2.0
NOMINAL = [-3.0, 0.0, 3.0]
REFINED = [-2.91, 0.04, 3.12]
XF = [
    [0.99, -0.01, 0.01, 0.99, 1.5, -2.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.98, 0.02, -0.02, 0.98, -3.25, 4.5],
]
# start, end, start angle, end angle, defocus U (nm), defocus V (nm), astigmatism angle
DEFOCUS = "1\t0\t0.0\t0.0\t0.0\t3\n" + "".join(
    f"{i}\t{i}\t{a:.2f}\t{a:.2f}\t{3000 + i}.0\t{2900 + i}.0\t{10 * i}.00\n"
    for i, a in enumerate(NOMINAL, start=1)
)


def run(args: List[str]) -> int:
    """Runs cets-imod with the given arguments, hiding its messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return main(args)


class CetsImodCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_imod_cli"))
        self.stack = self.dir / "TS_01.mrc"
        with mrcfile.new(self.stack) as mrc:
            mrc.set_data(np.zeros((len(NOMINAL), 8, 8), dtype=np.float32))
            mrc.voxel_size = VOXEL_SIZE
        self.rawtlt = self.dir / "TS_01.rawtlt"
        self.rawtlt.write_text("".join(f"{a}\n" for a in NOMINAL))
        self.tlt = self.dir / "TS_01.tlt"
        self.tlt.write_text("".join(f"{a}\n" for a in REFINED))
        self.xf = self.dir / "TS_01.xf"
        self.xf.write_text("".join(" ".join(map(str, row)) + "\n" for row in XF))
        self.defocus = self.dir / "TS_01.defocus"
        self.defocus.write_text(DEFOCUS)
        self.yaml = self.dir / "TS_01_cets.yaml"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ts_to_cets(self) -> None:
        code = run(
            [
                "tilt-series",
                "to-cets",
                "--stack",
                str(self.stack),
                "--tilt-angles",
                str(self.rawtlt),
                "--refined-tilt-angles",
                str(self.tlt),
                "--xf",
                str(self.xf),
                "--defocus",
                str(self.defocus),
                "--voltage",
                "300",
                "--tilt-axis-angle",
                "-85.3",
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)

    def _ts_to_cets_acquisition(self, *acquisition: str) -> None:
        """Converts the tilt-series with only the given acquisition options."""
        code = run(
            [
                "tilt-series",
                "to-cets",
                "--stack",
                str(self.stack),
                "--tilt-angles",
                str(self.rawtlt),
                "-o",
                str(self.yaml),
                *acquisition,
            ]
        )
        self.assertEqual(code, 0)

    def test_tilt_series_to_cets(self) -> None:
        self._ts_to_cets()
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        alignment = load_cets_object(self.yaml, "alignment", Alignment)
        images = ts.images or []
        self.assertEqual([ti.nominal_tilt_angle for ti in images], NOMINAL)
        self.assertEqual(
            [p.refined_tilt_angle for p in alignment.projection_alignments or []],
            REFINED,
        )
        # Defocus in Å (nm * 10)
        self.assertAlmostEqual(images[0].ctf_metadata.defocus_u, 30010.0)
        # Acquisition: tilt-series -> session -> instrument
        instrument = load_cets_object(self.yaml, "instrument", Instrument)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        self.assertEqual(instrument.voltage, 300.0)
        self.assertEqual(session.tilt_axis_angle, -85.3)
        self.assertEqual(session.instrument_id, instrument.id)
        self.assertEqual(ts.acquisition_session_id, session.id)

    def test_tilt_series_without_acquisition(self) -> None:
        """IMOD files do not record the acquisition: without its values, no empty
        instrument or acquisition session is written."""
        self._ts_to_cets_acquisition()
        self.assertFalse(has_cets_key(self.yaml, "instrument"))
        self.assertFalse(has_cets_key(self.yaml, "acquisition_session"))
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        self.assertIsNone(ts.acquisition_session_id)
        self.assertEqual([ti.nominal_tilt_angle for ti in ts.images or []], NOMINAL)

    def test_tilt_series_only_voltage(self) -> None:
        """The tilt-series reaches its instrument through a session, which then only
        links them."""
        self._ts_to_cets_acquisition("--voltage", "200")
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        instrument = load_cets_object(self.yaml, "instrument", Instrument)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        self.assertEqual(instrument.voltage, 200.0)
        self.assertEqual(session.instrument_id, instrument.id)
        self.assertEqual(ts.acquisition_session_id, session.id)
        self.assertIsNone(session.spherical_aberration)
        self.assertIsNone(session.tilt_axis_angle)

    def test_tilt_series_only_session_values(self) -> None:
        """Without --voltage there is no instrument, but the session values are kept."""
        self._ts_to_cets_acquisition(
            "--spherical-aberration", "2.7", "--amplitude-contrast", "0.07"
        )
        self.assertFalse(has_cets_key(self.yaml, "instrument"))
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        self.assertEqual(ts.acquisition_session_id, session.id)
        self.assertIsNone(session.instrument_id)
        self.assertEqual(session.spherical_aberration, 2.7)
        self.assertEqual(session.amplitude_contrast, 0.07)

    def test_tilt_series_round_trip(self) -> None:
        self._ts_to_cets()
        out = self.dir / "out" / "sub"  # created by the command
        code = run(
            [
                "tilt-series",
                "from-cets",
                "-i",
                str(self.yaml),
                "--tlt",
                str(out / "TS_01.tlt"),
                "--rawtlt",
                str(out / "TS_01.rawtlt"),
                "--xf",
                str(out / "TS_01.xf"),
            ]
        )
        self.assertEqual(code, 0)
        tlt = np.loadtxt(out / "TS_01.tlt", ndmin=2)[:, 0]
        rawtlt = np.loadtxt(out / "TS_01.rawtlt", ndmin=2)[:, 0]
        np.testing.assert_allclose(tlt, REFINED, atol=1e-3)
        np.testing.assert_allclose(rawtlt, NOMINAL, atol=1e-3)
        np.testing.assert_allclose(np.loadtxt(out / "TS_01.xf"), XF, atol=1e-6)

    def test_ctf_from_tilt_series(self) -> None:
        self._ts_to_cets()
        out = self.dir / "out.defocus"
        self.assertEqual(
            run(["ctf", "from-cets", "-i", str(self.yaml), "-o", str(out)]), 0
        )
        rows = [line.split() for line in out.read_text().splitlines()[1:]]
        self.assertEqual([float(r[2]) for r in rows], NOMINAL)
        self.assertEqual([float(r[4]) for r in rows], [3001.0, 3002.0, 3003.0])

    def test_ctf_round_trip(self) -> None:
        ctf_yaml = self.dir / "ctf.yaml"
        to_cets = [
            "ctf",
            "to-cets",
            "--stack",
            str(self.stack),
            "--defocus",
            str(self.defocus),
            "-o",
            str(ctf_yaml),
        ]
        self.assertEqual(run(to_cets), 0)
        out = self.dir / "out.defocus"
        # The CTF alone has no tilt angles
        self.assertEqual(
            run(["ctf", "from-cets", "-i", str(ctf_yaml), "-o", str(out)]), 1
        )
        from_cets = [
            "ctf",
            "from-cets",
            "-i",
            str(ctf_yaml),
            "--tilt-angles",
            str(self.rawtlt),
            "-o",
            str(out),
        ]
        self.assertEqual(run(from_cets), 0)
        rows = [line.split() for line in out.read_text().splitlines()[1:]]
        self.assertEqual([float(r[5]) for r in rows], [2901.0, 2902.0, 2903.0])

    def test_tomogram_round_trip(self) -> None:
        tomo_yaml = self.dir / "tomo.yaml"
        code = run(
            ["tomogram", "to-cets", "--tomogram", str(self.stack), "-o", str(tomo_yaml)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(load_cets_object(tomo_yaml, "tomogram", Tomogram).id, "TS_01")
        out_dir = self.dir / "tomos"
        code = run(
            [
                "tomogram",
                "from-cets",
                "-i",
                str(tomo_yaml),
                "--output-dir",
                str(out_dir),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "TS_01.mrc").exists())

    def test_errors_give_exit_code_1(self) -> None:
        missing_stack = [
            "tilt-series",
            "to-cets",
            "--stack",
            str(self.dir / "missing.mrc"),
            "--tilt-angles",
            str(self.rawtlt),
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(missing_stack), 1)
        # A tilt-series without alignment cannot write the xf file
        self.yaml.write_text("tilt_series: {id: TS_01}\n")
        no_alignment = [
            "tilt-series",
            "from-cets",
            "-i",
            str(self.yaml),
            "--tlt",
            str(self.dir / "x.tlt"),
            "--xf",
            str(self.dir / "x.xf"),
        ]
        self.assertEqual(run(no_alignment), 1)

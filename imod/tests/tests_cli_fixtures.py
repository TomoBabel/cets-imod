# -*- coding: utf-8 -*-
"""Tests of the ``cets-imod`` command line with the data of a real IMOD processing
(``imod/tests/data``, described by ``ImodTestDataFiles``): the tilt-series TS_03
(40 tilt-images), its alignment, its CTF estimated in five ways and its tomogram.

Fixtures in ``imod/tests/data``:

- ``tilt_series/TS_03.mrcs``: the stack (464 x 480 pixels, 40 images, 10.8 Å/pixel).
- ``tilt_series/TS_03.tlt``: nominal tilt angles and accumulated dose (2 columns).
- ``tilt_series/03.rawtlt``: the nominal tilt angles only.
- ``alignment_files/TS_03.xf``: alignment of the tilt-images.
- ``imod_defocus_files/TS_03_*.defocus``: the CTF of the tilt-images, written by ctfplotter
  with each of its formats (flags 0, 1, 4, 5 and 37).
- ``tomograms/TS_03.mrc``: the tomogram (480 x 464 x 40 voxels, 10.8 Å/voxel).
- ``TS_03_parameters.txt``: the ``tilt-series to-cets`` options of the acquisition of TS_03
  (voltage, Cs, ...), which IMOD files do not record, and the pixel size of the stack (the
  one of its header).
- ``cets_files/TS_03_cets_ts.yaml``, ``TS_03_cets_ts_alignment.yaml``,
  ``TS_03_cets_ctf.yaml``: reference CETS tilt-series, alignment and CTF, written from
  ``TS_03.mrcs``, ``TS_03.tlt``, ``TS_03.xf`` and ``TS_03_only_astigmatism.defocus`` with the
  options of ``TS_03_parameters.txt``.

The expected values are read from the fixtures with numpy and ``ImodTestDataFiles``, not
with the converters, and the outputs are also compared with the reference CETS files.
"""

import contextlib
import hashlib
import io
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    AcquisitionSession,
    Affine,
    Alignment,
    CTFMetadata,
    Instrument,
    Scale,
    TiltSeries,
    Tomogram,
    Translation,
)
from cets_data_model.utils.yaml_io import (
    has_cets_key,
    load_cets_object,
    load_cets_objects,
)
from imod.cli import main
from imod.tests import TEST_DATA_ROOT, ImodTestDataFiles

F = ImodTestDataFiles
DEFOCUS_FILES = [
    F.defocus_plain_estimation,
    F.defocus_only_astigmatism,
    F.defocus_only_phase_shift,
    F.defocus_astig_and_phase_shift,
    F.defocus_astig_phase_shift_and_cutoff_freq,
]
TS_ID = "TS_03"
N_IMGS = F.ts_03_tlt.n_imgs
NM_TO_A = 10.0  # .defocus files are in nm, CETS in Å
ATOL = 1e-6


def read_tlt() -> Tuple[np.ndarray, np.ndarray]:
    """Nominal tilt angles and accumulated dose of TS_03.tlt."""
    tlt = np.loadtxt(F.ts_03_tlt.path, ndmin=2)
    return tlt[:, 0], tlt[:, 1]


def read_parameters() -> Dict[str, str]:
    """The options of TS_03_parameters.txt ({"--voltage": "300.000000", ...}); the lines
    that are not options (the title) are skipped."""
    parameters = {}
    for line in F.ts_03_parameters.path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("--"):
            parameters[parts[0]] = parts[1]
    return parameters


def acquisition_args(parameters: Dict[str, str]) -> List[str]:
    """The options of the parameters file as command line arguments: the flags
    (``--ctf-corrected False``) are given only if True."""
    args: List[str] = []
    for option, value in parameters.items():
        if value in ("True", "False"):
            args += [option] if value == "True" else []
        else:
            args += [option, value]
    return args


def stack_pixel_size(path: Path) -> float:
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        return float(mrc.voxel_size.x)


def expected_ctfs(defocus: ImodTestDataFiles) -> List[Dict[str, float]]:
    """The CTF of each tilt-image of a .defocus file, in CETS units (Å, degrees): without
    astigmatism, defocus V is defocus U and the angle 0; without phase shift, it is 0."""
    expected = []
    for row in defocus.get_test_dict_list():
        defocus_u = float(row["defocus_u"]) * NM_TO_A
        expected.append(
            {
                "defocus_u": defocus_u,
                "defocus_v": float(row.get("defocus_v", row["defocus_u"])) * NM_TO_A,
                "defocus_angle": float(row.get("defocus_angle", 0.0)),
                "phase_shift": float(row.get("phase_shift", 0.0)),
            }
        )
    return expected


def without_paths(ts: TiltSeries) -> Dict[str, Any]:
    """The tilt-series as a dict, without the paths of its files."""
    data = ts.model_dump(mode="json")
    data["path"] = None
    for image in data["images"] or []:
        image["path"] = None
    return data


def ctf_values(ctfs: List[CTFMetadata]) -> List[Dict[str, Any]]:
    return [
        {
            "defocus_u": c.defocus_u,
            "defocus_v": c.defocus_v,
            "defocus_angle": c.defocus_angle,
            "phase_shift": c.phase_shift,
        }
        for c in ctfs
    ]


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file under a directory, to check that it is not modified."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-imod with the given arguments; returns the exit code and what it printed
    to stdout and stderr. The warnings of the stack header (its statistics are not
    accurate) are hidden."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(args)
    return code, out.getvalue(), err.getvalue()


class CetsImodCliFixturesTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_imod_cli_fixtures"))
        self.yaml = self.dir / "TS_03_cets.yaml"
        self.out = self.dir / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-imod, checks its exit code and returns what it printed to stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return err

    def _ts_args(self, *extra: str) -> List[str]:
        return [
            "tilt-series",
            "to-cets",
            "--stack",
            str(F.ts_03_mrcs.path),
            "--tilt-angles",
            str(F.ts_03_tlt.path),
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _ts_to_cets(self, *extra: str) -> Tuple[TiltSeries, Alignment]:
        self.assert_run(self._ts_args(*extra))
        return (
            load_cets_object(self.yaml, "tilt_series", TiltSeries),
            load_cets_object(self.yaml, "alignment", Alignment),
        )

    def _ctf_to_cets(self, defocus: ImodTestDataFiles, yaml_file: Path) -> None:
        self.assert_run(
            [
                "ctf",
                "to-cets",
                "--stack",
                str(F.ts_03_mrcs.path),
                "--defocus",
                str(defocus.path),
                "-o",
                str(yaml_file),
            ]
        )

    # ------------------------------------------------------------------ fixtures
    def test_fixture_files(self) -> None:
        """The fixtures describe the same tilt-series: one tilt angle, alignment and CTF
        per image of the stack."""
        with mrcfile.open(F.ts_03_mrcs.path, header_only=True, permissive=True) as mrc:
            self.assertEqual(int(mrc.header.nz), N_IMGS)
        angles, doses = read_tlt()
        self.assertEqual(len(angles), N_IMGS)
        self.assertEqual(np.loadtxt(F.ts_03_xf.path).shape, (N_IMGS, 6))
        for defocus in DEFOCUS_FILES:
            with self.subTest(defocus=defocus.name):
                rows = defocus.get_test_dict_list()
                self.assertEqual(len(rows), N_IMGS)
                # Their tilt angles are those of the .tlt
                np.testing.assert_allclose(
                    [row["angle_u"] for row in rows], angles, atol=0.01
                )
        self.assertGreater(stack_pixel_size(F.ts_03_mrcs.path), 0)
        self.assertGreater(stack_pixel_size(F.ts_03_tomogram.path), 0)
        # The .rawtlt has the angles of the .tlt
        np.testing.assert_allclose(np.loadtxt(F.ts_03_rawtlt.path), angles)
        # The pixel size of the parameters is the one of the stack header
        parameters = read_parameters()
        self.assertAlmostEqual(
            stack_pixel_size(F.ts_03_mrcs.path), float(parameters["--pixel-size"]), 4
        )
        for option in (
            "--voltage",
            "--spherical-aberration",
            "--amplitude-contrast",
            "--dose-rate",
            "--tilt-axis-angle",
        ):
            self.assertGreater(float(parameters[option]), 0)

    # ------------------------------------------------------------------ references
    def test_tilt_series_matches_reference(self) -> None:
        """The tilt-series, alignment and CTF are those of the reference CETS files; the
        acquisition, the values of the parameters file."""
        parameters = read_parameters()
        ts, alignment = self._ts_to_cets(
            "--xf",
            str(F.ts_03_xf.path),
            "--defocus",
            str(F.defocus_only_astigmatism.path),
            *acquisition_args(parameters),
        )
        ref_ts = load_cets_object(F.ts_03_cets_ts.path, "tilt_series", TiltSeries)
        ref_alignment = load_cets_object(
            F.ts_03_cets_alignment.path, "alignment", Alignment
        )
        ref_ctfs = load_cets_objects(F.ts_03_cets_ctf.path, "ctf", CTFMetadata)
        # The reference paths are relative to its directory
        for path in [ref_ts.path] + [ti.path for ti in ref_ts.images or []]:
            self.assertEqual(
                (F.ts_03_cets_ts.path.parent / (path or "")).resolve(),
                F.ts_03_mrcs.path,
            )
        self.assertEqual(without_paths(ts), without_paths(ref_ts))
        self.assertEqual(alignment, ref_alignment)
        self.assertEqual([ti.ctf_metadata for ti in ts.images or []], ref_ctfs)
        self.assertIs(ts.ctf_corrected, parameters["--ctf-corrected"] == "True")
        # Acquisition
        instrument = load_cets_object(self.yaml, "instrument", Instrument)
        session = load_cets_object(self.yaml, "acquisition_session", AcquisitionSession)
        self.assertEqual(ts.acquisition_session_id, session.id)
        self.assertEqual(session.instrument_id, instrument.id)
        self.assertEqual(instrument.voltage, float(parameters["--voltage"]))
        self.assertEqual(
            (
                session.spherical_aberration,
                session.amplitude_contrast,
                session.dose_rate,
                session.tilt_axis_angle,
            ),
            tuple(
                float(parameters[option])
                for option in (
                    "--spherical-aberration",
                    "--amplitude-contrast",
                    "--dose-rate",
                    "--tilt-axis-angle",
                )
            ),
        )

    def test_ctf_matches_reference(self) -> None:
        self._ctf_to_cets(F.defocus_only_astigmatism, self.yaml)
        self.assertEqual(
            load_cets_objects(self.yaml, "ctf", CTFMetadata),
            load_cets_objects(F.ts_03_cets_ctf.path, "ctf", CTFMetadata),
        )

    # ------------------------------------------------------------------ tilt-series
    def test_tilt_series_to_cets(self) -> None:
        ts, alignment = self._ts_to_cets(
            "--xf",
            str(F.ts_03_xf.path),
            "--defocus",
            str(F.defocus_astig_and_phase_shift.path),
        )
        angles, doses = read_tlt()
        pixel_size = stack_pixel_size(F.ts_03_mrcs.path)
        images = ts.images or []
        self.assertEqual(ts.id, TS_ID)
        self.assertEqual(len(images), N_IMGS)
        self.assertEqual([ti.section for ti in images], list(range(N_IMGS)))
        np.testing.assert_allclose([ti.nominal_tilt_angle for ti in images], angles)
        np.testing.assert_allclose([ti.accumulated_dose for ti in images], doses)
        # Acquisition order inferred from the dose: the 1-based rank of each image
        self.assertEqual(
            [ti.acquisition_order for ti in images],
            (np.argsort(np.argsort(doses)) + 1).tolist(),
        )
        with mrcfile.open(F.ts_03_mrcs.path, header_only=True, permissive=True) as mrc:
            shape = (int(mrc.header.nx), int(mrc.header.ny))
        for ti in images:
            self.assertEqual((ti.width, ti.height), shape)
            scale = (ti.coordinate_transformations or [])[0]
            assert isinstance(scale, Scale)
            np.testing.assert_allclose(scale.scale, [pixel_size] * 2)
        # CTF in Å
        self.assertEqual(
            ctf_values([ti.ctf_metadata for ti in images if ti.ctf_metadata]),
            expected_ctfs(F.defocus_astig_and_phase_shift),
        )
        # Alignment: the .xf rotation, then its shifts in Å
        xf = np.loadtxt(F.ts_03_xf.path)
        projections = alignment.projection_alignments or []
        self.assertEqual(alignment.tilt_series_id, TS_ID)
        self.assertEqual(len(projections), N_IMGS)
        for projection, row in zip(projections, xf):
            affine, translation = projection.sequence or []
            assert isinstance(affine, Affine) and isinstance(translation, Translation)
            np.testing.assert_allclose(
                np.array(affine.affine)[:2, :2], row[:4].reshape(2, 2), atol=ATOL
            )
            np.testing.assert_allclose(
                translation.translation, row[4:] * pixel_size, atol=ATOL
            )
            self.assertIsNone(projection.refined_tilt_angle)

    def test_tilt_series_without_alignment(self) -> None:
        """Without --xf, the alignment is the identity."""
        ts, alignment = self._ts_to_cets()
        for projection in alignment.projection_alignments or []:
            affine, translation = projection.sequence or []
            assert isinstance(affine, Affine) and isinstance(translation, Translation)
            np.testing.assert_allclose(affine.affine, np.eye(3))
            np.testing.assert_allclose(translation.translation, [0.0, 0.0])
        self.assertTrue(all(ti.ctf_metadata is None for ti in ts.images or []))
        # No acquisition values: no instrument nor acquisition session
        self.assertFalse(has_cets_key(self.yaml, "instrument"))
        self.assertFalse(has_cets_key(self.yaml, "acquisition_session"))
        self.assertIsNone(ts.acquisition_session_id)

    def test_tilt_series_refined_angles_and_pixel_size(self) -> None:
        """--refined-tilt-angles fills the alignment; --pixel-size overrides the header,
        which also scales the shifts of the alignment."""
        pixel_size = stack_pixel_size(F.ts_03_mrcs.path) / 2
        ts, alignment = self._ts_to_cets(
            "--refined-tilt-angles",
            str(F.ts_03_tlt.path),
            "--xf",
            str(F.ts_03_xf.path),
            "--pixel-size",
            str(pixel_size),
        )
        angles, _ = read_tlt()
        projections = alignment.projection_alignments or []
        np.testing.assert_allclose([p.refined_tilt_angle for p in projections], angles)
        scale = ((ts.images or [])[0].coordinate_transformations or [])[0]
        assert isinstance(scale, Scale)
        np.testing.assert_allclose(scale.scale, [pixel_size] * 2)
        translation = (projections[0].sequence or [])[1]
        assert isinstance(translation, Translation)
        np.testing.assert_allclose(
            translation.translation,
            np.loadtxt(F.ts_03_xf.path)[0, 4:] * pixel_size,
            atol=ATOL,
        )

    def test_tilt_series_rawtlt(self) -> None:
        """Tilt angles without dose: the dose and acquisition order are unknown, and
        from-cets writes back the same .rawtlt."""
        rawtlt_args = self._ts_args()
        rawtlt_args[rawtlt_args.index("--tilt-angles") + 1] = str(F.ts_03_rawtlt.path)
        self.assert_run(rawtlt_args)
        ts = load_cets_object(self.yaml, "tilt_series", TiltSeries)
        images = ts.images or []
        np.testing.assert_allclose(
            [ti.nominal_tilt_angle for ti in images], np.loadtxt(F.ts_03_rawtlt.path)
        )
        self.assertTrue(all(ti.accumulated_dose is None for ti in images))
        self.assertTrue(all(ti.acquisition_order is None for ti in images))
        rawtlt = self.out / "03.rawtlt"
        self.assert_run(
            [
                "tilt-series",
                "from-cets",
                "-i",
                str(self.yaml),
                "--tlt",
                str(self.out / "03.tlt"),
                "--rawtlt",
                str(rawtlt),
                "--no-dose",
            ]
        )
        np.testing.assert_allclose(
            np.loadtxt(rawtlt), np.loadtxt(F.ts_03_rawtlt.path), atol=1e-3
        )

    def test_tilt_series_round_trip(self) -> None:
        self._ts_to_cets("--xf", str(F.ts_03_xf.path))
        tlt, rawtlt, xf = (
            self.out / "TS_03.tlt",
            self.out / "TS_03.rawtlt",
            self.out / "TS_03.xf",
        )
        self.assert_run(
            [
                "tilt-series",
                "from-cets",
                "-i",
                str(self.yaml),
                "--tlt",
                str(tlt),
                "--rawtlt",
                str(rawtlt),
                "--xf",
                str(xf),
            ]
        )
        angles, doses = read_tlt()
        for written in (tlt, rawtlt):  # No refined angles: both are the nominal ones
            with self.subTest(file=written.name):
                values = np.loadtxt(written, ndmin=2)
                np.testing.assert_allclose(values[:, 0], angles, atol=1e-3)
                np.testing.assert_allclose(values[:, 1], doses, atol=1e-3)
        # The alignment is written back unchanged
        self.assertEqual(xf.read_text(), F.ts_03_xf.path.read_text())
        # --no-dose: only the angles
        self.assert_run(
            [
                "tilt-series",
                "from-cets",
                "-i",
                str(self.yaml),
                "--tlt",
                str(tlt),
                "--no-dose",
            ]
        )
        self.assertEqual(np.loadtxt(tlt, ndmin=2).shape, (N_IMGS, 1))

    def test_tilt_series_errors(self) -> None:
        # An alignment of other number of tilt-images
        short_xf = self.dir / "short.xf"
        short_xf.write_text("".join(F.ts_03_xf.path.read_text().splitlines(True)[:-1]))
        err = self.assert_run(self._ts_args("--xf", str(short_xf)), code=1)
        self.assertIn(str(N_IMGS), err)
        # Missing tilt angles
        args = self._ts_args()
        args[args.index("--tilt-angles") + 1] = str(self.dir / "missing.tlt")
        self.assert_run(args, code=1)
        self.assertFalse(self.yaml.exists())
        # --xf of a yaml without alignment (the output of ctf to-cets)
        self._ctf_to_cets(F.defocus_only_astigmatism, self.yaml)
        self.assert_run(
            [
                "tilt-series",
                "from-cets",
                "-i",
                str(self.yaml),
                "--tlt",
                str(self.out / "TS_03.tlt"),
                "--xf",
                str(self.out / "TS_03.xf"),
            ],
            code=1,
        )

    # ------------------------------------------------------------------ ctf
    def test_ctf_to_cets_every_format(self) -> None:
        """Every ctfplotter format, in CETS units; the cut-on frequency is not kept."""
        for defocus in DEFOCUS_FILES:
            with self.subTest(defocus=defocus.name):
                yaml_file = self.dir / f"{defocus.name}.yaml"
                self._ctf_to_cets(defocus, yaml_file)
                ctfs = load_cets_objects(yaml_file, "ctf", CTFMetadata)
                self.assertEqual(ctf_values(ctfs), expected_ctfs(defocus))

    def test_ctf_round_trip_every_format(self) -> None:
        """Writing the CTF back gives a .defocus file with the same CTF."""
        for defocus in DEFOCUS_FILES:
            with self.subTest(defocus=defocus.name):
                yaml_file = self.dir / f"{defocus.name}.yaml"
                written = self.out / f"{defocus.name}.defocus"
                self._ctf_to_cets(defocus, yaml_file)
                self.assert_run(
                    [
                        "ctf",
                        "from-cets",
                        "-i",
                        str(yaml_file),
                        "--tilt-angles",
                        str(F.ts_03_tlt.path),
                        "-o",
                        str(written),
                    ]
                )
                again = self.dir / f"{defocus.name}_again.yaml"
                self.assert_run(
                    [
                        "ctf",
                        "to-cets",
                        "--stack",
                        str(F.ts_03_mrcs.path),
                        "--defocus",
                        str(written),
                        "-o",
                        str(again),
                    ]
                )
                self.assertEqual(
                    ctf_values(load_cets_objects(again, "ctf", CTFMetadata)),
                    expected_ctfs(defocus),
                )
                # The tilt angles of the rows are those of the .tlt
                rows = np.loadtxt(written, skiprows=1, ndmin=2)
                np.testing.assert_allclose(rows[:, 2], read_tlt()[0], atol=0.01)

    def test_ctf_from_tilt_series(self) -> None:
        """The CTF of a tilt-series yaml, with its nominal tilt angles."""
        self._ts_to_cets("--defocus", str(F.defocus_only_astigmatism.path))
        written = self.out / "TS_03.defocus"
        self.assert_run(["ctf", "from-cets", "-i", str(self.yaml), "-o", str(written)])
        rows = np.loadtxt(written, skiprows=1, ndmin=2)
        expected = expected_ctfs(F.defocus_only_astigmatism)
        np.testing.assert_allclose(
            rows[:, 4], [c["defocus_u"] / NM_TO_A for c in expected], atol=0.05
        )
        np.testing.assert_allclose(rows[:, 2], read_tlt()[0], atol=0.01)

    def test_ctf_errors(self) -> None:
        # The output of ctf to-cets has no tilt angles: --tilt-angles is required
        self._ctf_to_cets(F.defocus_only_astigmatism, self.yaml)
        written = self.out / "TS_03.defocus"
        err = self.assert_run(
            ["ctf", "from-cets", "-i", str(self.yaml), "-o", str(written)], code=1
        )
        self.assertIn("--tilt-angles", err)
        # A tilt-series without CTF
        self._ts_to_cets()
        self.assert_run(
            ["ctf", "from-cets", "-i", str(self.yaml), "-o", str(written)], code=1
        )
        self.assertFalse(written.exists())

    # ------------------------------------------------------------------ tomogram
    def _tomo_to_cets(self, *extra: str) -> Tomogram:
        self.assert_run(
            [
                "tomogram",
                "to-cets",
                "--tomogram",
                str(F.ts_03_tomogram.path),
                "-o",
                str(self.yaml),
                *extra,
            ]
        )
        return load_cets_object(self.yaml, "tomogram", Tomogram)

    def test_tomogram_to_cets(self) -> None:
        tomo = self._tomo_to_cets()
        with mrcfile.open(
            F.ts_03_tomogram.path, header_only=True, permissive=True
        ) as mrc:
            header = mrc.header
            shape = (int(header.nx), int(header.ny), int(header.nz))
        self.assertEqual(tomo.id, TS_ID)
        self.assertEqual((tomo.width, tomo.height, tomo.depth), shape)
        self.assertEqual(Path(tomo.path or ""), F.ts_03_tomogram.path)
        self.assertIs(tomo.ctf_corrected, False)
        self.assertIsNone(tomo.even_path)
        scale = (tomo.coordinate_transformations or [])[0]
        assert isinstance(scale, Scale)
        np.testing.assert_allclose(
            scale.scale, [stack_pixel_size(F.ts_03_tomogram.path)] * 3
        )

    def test_tomogram_round_trip(self) -> None:
        """from-cets links the tomogram (the same file), or copies it with --copy."""
        self._tomo_to_cets()
        linked = self.out / "linked"
        self.assert_run(
            [
                "tomogram",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(linked),
            ]
        )
        self.assertTrue((linked / "TS_03.mrc").is_symlink())
        self.assertEqual((linked / "TS_03.mrc").resolve(), F.ts_03_tomogram.path)
        copied = self.out / "copied"
        self.assert_run(
            [
                "tomogram",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(copied),
                "--copy",
            ]
        )
        self.assertFalse((copied / "TS_03.mrc").is_symlink())
        self.assertEqual(
            (copied / "TS_03.mrc").read_bytes(), F.ts_03_tomogram.path.read_bytes()
        )

    def test_tomogram_pixel_size_and_halves(self) -> None:
        """A voxel size other than the one of the header: the files are copied with it in
        their header and all their data; the even/odd halves are written too."""
        voxel_size = 5.0
        tomo_path = str(F.ts_03_tomogram.path)
        tomo = self._tomo_to_cets(
            "--pixel-size",
            str(voxel_size),
            "--even",
            tomo_path,
            "--odd",
            tomo_path,
            "--ctf-corrected",
        )
        self.assertEqual((tomo.even_path, tomo.odd_path), (tomo_path, tomo_path))
        self.assertIs(tomo.ctf_corrected, True)
        self.assert_run(
            [
                "tomogram",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out),
            ]
        )
        with mrcfile.mmap(F.ts_03_tomogram.path, permissive=True) as original:
            for name in ("TS_03.mrc", "TS_03_even.mrc", "TS_03_odd.mrc"):
                with self.subTest(file=name):
                    written = self.out / name
                    self.assertFalse(written.is_symlink())
                    self.assertAlmostEqual(stack_pixel_size(written), voxel_size, 4)
                    with mrcfile.mmap(written, permissive=True) as mrc:
                        np.testing.assert_array_equal(mrc.data, original.data)

    def test_tomogram_errors(self) -> None:
        self._tomo_to_cets()
        args = [
            "tomogram",
            "from-cets",
            "-i",
            str(self.yaml),
            "--output-dir",
            str(self.out),
        ]
        self.assert_run(args)
        # The output exists and is another file
        (self.out / "TS_03.mrc").unlink()
        (self.out / "TS_03.mrc").write_bytes(b"")
        self.assert_run(args, code=1)
        # Missing tomogram
        self.assert_run(
            [
                "tomogram",
                "to-cets",
                "--tomogram",
                str(self.dir / "missing.mrc"),
                "-o",
                str(self.yaml),
            ],
            code=1,
        )

    # ------------------------------------------------------------------ command line
    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(TEST_DATA_ROOT)
        self._ts_to_cets(
            "--xf",
            str(F.ts_03_xf.path),
            "--defocus",
            str(F.defocus_astig_phase_shift_and_cutoff_freq.path),
        )
        self._tomo_to_cets("--pixel-size", "5")
        self.assert_run(
            [
                "tomogram",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.out),
            ]
        )
        self.assertEqual(tree_digest(TEST_DATA_ROOT), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["tilt-series"],
            ["subtomograms", "to-cets"],
            # --tilt-angles is required
            ["tilt-series", "to-cets", "--stack", str(F.ts_03_mrcs.path)],
            self._ts_args("--voltage", "high"),  # a number
            ["ctf", "to-cets", "--stack", str(F.ts_03_mrcs.path)],  # --defocus
            ["tomogram", "from-cets", "-i", str(self.yaml)],  # --output-dir
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (
            ["--help"],
            ["tilt-series", "to-cets", "--help"],
            ["ctf", "from-cets", "--help"],
            ["tomogram", "from-cets", "--help"],
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-imod", out.getvalue())

# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np

from imod.converters.tomogram import ImodTomogram

VOXEL_SIZE = 5.4
# Tomogram dimensions in x, y, z (the mrc data is stored as z, y, x)
TOMO_DIMS = (12, 10, 8)


def write_mrc(mrc_file: Path, voxel_size: float = VOXEL_SIZE) -> Path:
    x, y, z = TOMO_DIMS
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(np.zeros((z, y, x), dtype=np.float32))
        mrc.voxel_size = voxel_size
    return mrc_file


class CetsToImodTomogramTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_imod_tomogram"))
        self.out_dir = self.test_dir / "out"
        self.tomo_file = write_mrc(self.test_dir / "TS_03.mrc")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_linked_with_its_id(self) -> None:
        tomo = ImodTomogram(self.tomo_file).imod_to_cets()
        # Without halves, their paths are left empty
        self.assertIsNone(tomo.even_path)
        self.assertIsNone(tomo.odd_path)
        out = ImodTomogram.cets_to_imod(tomo, self.out_dir)
        self.assertEqual(out.name, "TS_03.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), self.tomo_file.resolve())
        # No halves: only the tomogram is written
        self.assertEqual([f.name for f in self.out_dir.iterdir()], ["TS_03.mrc"])
        # The exported tomogram is read back as the same CETS tomogram
        back = ImodTomogram(out).imod_to_cets()
        self.assertEqual(back, tomo)

    def test_even_odd(self) -> None:
        even = write_mrc(self.test_dir / "half1.mrc")
        odd = write_mrc(self.test_dir / "half2.mrc")
        tomo = ImodTomogram(self.tomo_file).imod_to_cets(
            even_file_name=even, odd_file_name=odd
        )
        ImodTomogram.cets_to_imod(tomo, self.out_dir)
        self.assertEqual((self.out_dir / "TS_03_even.mrc").resolve(), even.resolve())
        self.assertEqual((self.out_dir / "TS_03_odd.mrc").resolve(), odd.resolve())

    def test_from_yaml(self) -> None:
        yaml_file = self.test_dir / "tomo.yaml"
        ImodTomogram(self.tomo_file).imod_to_cets(out_yaml_file=yaml_file)
        out = ImodTomogram.cets_to_imod(yaml_file, self.out_dir, copy=True)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), VOXEL_SIZE, places=5)

    def test_copied_if_the_voxel_size_differs(self) -> None:
        """The header of the copy gets the voxel size of the CETS tomogram"""
        tomo = ImodTomogram(self.tomo_file).imod_to_cets()
        tomo.coordinate_transformations[0].scale = [10.8] * 3
        out = ImodTomogram.cets_to_imod(tomo, self.out_dir)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.8, places=5)
        with mrcfile.open(self.tomo_file, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), VOXEL_SIZE, places=5)

    def test_header_without_voxel_size(self) -> None:
        """Without voxel size in the header it must be given: it is not assumed."""
        tomo_file = write_mrc(self.test_dir / "no_apix.mrc", voxel_size=0.0)
        with self.assertRaisesRegex(ValueError, "pixel size"):
            ImodTomogram(tomo_file).imod_to_cets()
        tomo = ImodTomogram(tomo_file).imod_to_cets(pixel_size=VOXEL_SIZE)
        # The header does not have the voxel size of the CETS tomogram: copied and fixed
        out = ImodTomogram.cets_to_imod(tomo, self.out_dir)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), VOXEL_SIZE, places=5)

    def test_placeholder_voxel_size_is_not_written(self) -> None:
        """Yaml files written before the voxel size was required have 1.0 for headers
        without it: that placeholder is not written in the header."""
        tomo_file = write_mrc(self.test_dir / "no_apix.mrc", voxel_size=0.0)
        tomo = ImodTomogram(tomo_file).imod_to_cets(pixel_size=1.0)
        out = ImodTomogram.cets_to_imod(tomo, self.out_dir)
        self.assertTrue(out.is_symlink())

    def test_existing_file_is_not_overwritten(self) -> None:
        tomo = ImodTomogram(self.tomo_file).imod_to_cets()
        ImodTomogram.cets_to_imod(tomo, self.out_dir)
        ImodTomogram.cets_to_imod(tomo, self.out_dir)  # the same file again is harmless
        other_dir = self.test_dir / "other"
        other_dir.mkdir()
        other = ImodTomogram(write_mrc(other_dir / "TS_03.mrc")).imod_to_cets()
        with self.assertRaises(FileExistsError):
            ImodTomogram.cets_to_imod(other, self.out_dir)

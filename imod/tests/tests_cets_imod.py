# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

from cets_data_model.models.models import CTFMetadata
from imod.converters.ctf import ImodCtfSeries
from imod.tests import (
    ImodTestDataFiles,
    DEFOCUS_U,
    DEFOCUS_V,
    DEFOCUS_ANGLE,
    PHASE_SHIFT,
)

CETS_IMOD_CTF = "cets_imod_ctf"


class CetsImodBaseTest(TestCase):
    # Files
    ts_fn = ImodTestDataFiles.ts_03_mrcs.path
    tomo_fn = ImodTestDataFiles.ts_03_tomogram.path
    tlt_fn = ImodTestDataFiles.ts_03_tlt.path
    xf_fn = ImodTestDataFiles.ts_03_xf.path

    @classmethod
    def setUpClass(cls):
        """
        Setup test data and output directories.
        """
        cls.test_dir = Path(tempfile.mkdtemp(prefix=CETS_IMOD_CTF))
        # Change to test directory
        cls._orig_dir = os.getcwd()
        os.chdir(cls.test_dir)


class CetsImodDefocusReaderTest(CetsImodBaseTest):
    yaml_file_ctf = Path()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.yaml_file_ctf = cls.test_dir / "TS_03_cets_ctf.yaml"

    def _check_data(
        self,
        defocus_testdata: ImodTestDataFiles,
        cets_ctf_md_list: List[CTFMetadata],
        yaml_file_ctf: Path,
    ) -> None:
        test_md_dict_list = defocus_testdata.get_test_dict_list()
        self.assertEqual(len(cets_ctf_md_list), defocus_testdata.n_imgs)
        # In plain defocus and only phase shift estimation, the defocus is duplicated in both
        # defocus u and v
        defocus_v_field = (
            DEFOCUS_U if defocus_testdata.imod_flag in [0, 4] else DEFOCUS_V
        )
        for i, ctf_md in enumerate(cets_ctf_md_list):
            self.assertTrue(type(ctf_md) is CTFMetadata)
            ctf_test_md = test_md_dict_list[i]
            # Test defocus *= 10 -> from nm to angstroms
            self.assertEqual(ctf_md.defocus_u, ctf_test_md.get(DEFOCUS_U, 0.0) * 10)
            self.assertEqual(
                ctf_md.defocus_v, ctf_test_md.get(defocus_v_field, 0.0) * 10
            )
            self.assertEqual(ctf_md.defocus_angle, ctf_test_md.get(DEFOCUS_ANGLE, 0.0))
            self.assertEqual(ctf_md.phase_shift, ctf_test_md.get(PHASE_SHIFT, 0.0))
            self.assertEqual(ctf_md.defocus_handedness, -1)

        # Check if the yaml file was generated
        self.assertTrue(yaml_file_ctf.is_file())

    def _run_test_imod_to_cets(self, defocus_testdata: ImodTestDataFiles):
        print(f"\n ===> Running IMOD to CETS CTF - {defocus_testdata.description}")
        # CTF metadata
        ics = ImodCtfSeries(ts_file_name=self.ts_fn, defocus_file=defocus_testdata.path)
        cets_ctf_md_list = ics.imod_to_cets(out_yaml_file=self.yaml_file_ctf)
        # Check the metadata generated
        self._check_data(defocus_testdata, cets_ctf_md_list, self.yaml_file_ctf)

    def test_ctf_imod_to_cets_01(self):
        defocus_testdata = ImodTestDataFiles.defocus_plain_estimation
        self._run_test_imod_to_cets(defocus_testdata)

    def test_ctf_imod_to_cets_02(self):
        defocus_testdata = ImodTestDataFiles.defocus_only_astigmatism
        self._run_test_imod_to_cets(defocus_testdata)

    def test_ctf_imod_to_cets_03(self):
        defocus_testdata = ImodTestDataFiles.defocus_only_phase_shift
        self._run_test_imod_to_cets(defocus_testdata)

    def test_ctf_imod_to_cets_04(self):
        defocus_testdata = ImodTestDataFiles.defocus_astig_and_phase_shift
        self._run_test_imod_to_cets(defocus_testdata)

    def test_ctf_imod_to_cets_05(self):
        defocus_testdata = ImodTestDataFiles.defocus_astig_phase_shift_and_cutoff_freq
        self._run_test_imod_to_cets(defocus_testdata)


class CetsImodTsReaderTest(CetsImodBaseTest):
    yaml_file_ts = Path()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.yaml_file_ts = cls.test_dir / "TS_03_cets_ts.yaml"

    def test_ts_imod_to_cets(self):
        # print("\n ===> Running IMOD to CETS tilt-series")
        # # Generate the CTF metadata
        # ics = ImodCtfSeries(
        #     ts_file_name=self.ts_fn,
        #     defocus_file=ImodTestDataFiles.defocus_only_astigmatism.path,
        # )
        # cets_ctf_md_list = ics.imod_to_cets()
        # # TS Metadata
        # its = ImodTiltSeries(
        #     ts_file_name=self.ts_fn,
        #     tilt_angles=self.tlt_fn,
        #     ctf_md_list=cets_ctf_md_list,
        # )
        # cets_ts_md = its.imod_to_cets(
        #     xf_file=self.xf_fn, out_yaml_file=self.yaml_file_ts
        # )
        # Check the metadata generated
        # TODO finish this once the data model is decided
        raise Exception("finish this once the data model is decided")

# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path

CETS_IMOD_CTF = "cets_imod_ctf"


class CetsImodDefocusReaderTest(unittest.TestCase):
    def setUpClass(self):
        """
        Setup test data and output directories.
        """
        self.test_dir = Path(tempfile.mkdtemp(prefix=CETS_IMOD_CTF))
        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    # def test_imod_to_cets_01(self):
    #     ts_file_name = "test.mrc"
    #     pass

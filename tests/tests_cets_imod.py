# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path
from tests import CETS_IMOD_CTF


class CetsImodDefocusReaderTest(unittest.TestCase):
    def setUp(self):
        """
        Setup test data and output directories.
        """
        self.test_dir = Path(tempfile.mkdtemp(prefix=CETS_IMOD_CTF))
        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def _load_test_data(self, defocus_file: Path):
        pass

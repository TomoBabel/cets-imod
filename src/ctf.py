from pathlib import Path
from typing import List


class ImodCtfSeries:
    def __init__(self, ts_file_name: Path, defocus_file: Path) -> None:
        self.ts_file_name = ts_file_name
        self.defocus_file = defocus_file

    def imod_to_cets(self):
        pass

    def cets_to_imod(self):
        pass

    def _parse_defocus_file(self):
        pass

    def _get_defocus_file_flag(self) -> int:
        """This method returns the flag that indicate the
        information contained in an IMOD defocus file. The flag
        value "is the sum of:
              1 if the file has astigmatism values
              2 if the astigmatism axis angle is in radians, not degrees
              4 if the file has phase shifts
              8 if the phase shifts are in radians, not degrees
             16 if tilt angles need to be inverted to match what the
                 program expects (what Ctfplotter would produce)
                 with the -invert option
             32 if the file has cut-on frequencies attenuating the phase
                 at low frequencies"

         (from https://bio3d.colorado.edu/imod/doc/man/ctfphaseflip.html)."""

        with open(self.defocus_file) as f:
            lines = f.readlines()
        # File contains only defocus information (no astigmatism, no phase shift,
        # no cut-on frequency)
        if len(lines) == 1:
            return 0
        elif len(lines[1].split()) == 5:
            return 0
        # File contains more information apart
        return int(lines[0].split()[0])

    def _load_ctf_file(self, flag: int):
        """This method takes an IMOD-based file path containing the
        information associated to a CTF estimation and produces a set
        of dictionaries containing the information of each parameter
        for each tilt-image belonging to the tilt-series."""

        # Read info as table
        ctf_info_imod_table = self._defocus_file_to_table()

        if flag == 0:
            # Plain estimation
            return self.refactorCTFDefocusEstimationInfo(ctf_info_imod_table)

        elif flag == 1:
            # Astigmatism estimation
            return self._refactor_ctf_flag_1(ctf_info_imod_table)

        elif flag == 4:
            # Phase-shift estimation
            return self.refactorCTFDefocusPhaseShiftEstimationInfo(ctf_info_imod_table)

        elif flag == 5:
            # Astigmatism and phase shift estimation
            return self.refactorCTFDefocusAstigmatismPhaseShiftEstimationInfo(
                ctf_info_imod_table
            )

        elif flag == 37:
            # Astigmatism, phase shift and cut-on frequency estimation
            return self.refactorCTFDefocusAstigmatismPhaseShiftCutOnFreqEstimationInfo(
                ctf_info_imod_table
            )

        else:
            raise ValueError(
                "Defocus file flag do not supported. Only supported formats corresponding to flags 0, "
                "1, 4, 5, and 37."
            )

    def _defocus_file_to_table(self) -> List[List[float]]:
        """This method takes an IMOD-based ctf estimation file
        and returns a table containing the CTF estimation
        information of each tilt-image (per line) belonging to the tilt-series."""
        defocusTable = []

        with open(self.defocus_file) as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            vector = [float(i) for i in line.split()]

            if index == 0 and len(lines) == 1:
                # CTF estimation is plain (no astigmatism, no phase shift, no cut-on frequency) and is the first line.
                # Remove last element from the first line (it contains the mode of the estimation run). This case
                # considers that the estimation file only has one line.
                vector.pop()
                defocusTable.append(vector)

            elif index == 0 and len(lines[1].split()) == 5:
                # CTF estimation is plain (no astigmatism, no phase shift, no cut-on frequency) and is the first line.
                # Remove last element from the first line (it contains the mode of the estimation run).
                vector.pop()
                defocusTable.append(vector)

            elif index == 0 and len(lines[1].split()) != 5:
                # CTF estimation is not plain and is the first line.
                # Do not add this line to the table. Only contains flag and format info.
                pass

            else:
                # Any posterior line that is not the first one is added to the table .
                defocusTable.append(vector)

        return defocusTable

    def _refactor_ctf_flag_0(self, ctf_info_imod_table):
        """This method takes a table containing the information of
        an IMOD-based CTF estimation containing only defocus
        information (5 columns) and produces a dictionary for clearer
        and easier management. Flag 0 (Plain estimation)."""

        if len(ctf_info_imod_table[0]) != 5:
            raise Exception(
                "Misleading file format, CTF estimation with no astigmatism should be 5 columns long"
            )

        defocusUDict = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus = float(element[4]) * 10

            for index in range(start, end + 1):
                self._append_to_dict(defocusUDict, index, defocus)

        return defocusUDict

    def _refactor_ctf_flag_1(self, ctf_info_imod_table):
        """This method takes a table containing the information of an
        IMOD-based CTF estimation containing defocus and
        astigmatism information (7 columns) and produces a set
        of dictionaries for clearer and easier management. Flag 1
        (Astigmatism estimation)."""

        if len(ctf_info_imod_table[0]) != 7:
            raise Exception(
                "Misleading file format, CTF estimation "
                "with astigmatism should be 7 columns long"
            )
        defocusUDict = {}
        defocusVDict = {}
        defocusAngleDict = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocusU = float(element[4]) * 10  # From nm to angstroms
            defocusV = float(element[5]) * 10  # From nm to angstroms
            defocusAngle = float(element[6])

            # Segregate information from range
            for index in range(start, end + 1):
                self._append_to_dict(defocusUDict, index, defocusU)
                self._append_to_dict(defocusVDict, index, defocusV)
                self._append_to_dict(defocusAngleDict, index, defocusAngle)

        return defocusUDict, defocusVDict, defocusAngleDict

    @staticmethod
    def _append_to_dict(dictionary: dict, index: int, value: float) -> None:
        # Python Dictionary setdefault() returns the value of a key (if the key is in dictionary).
        # Else, it inserts a key with the default value to the dictionary.
        dictionary.setdefault(index, []).append(value)

    def refactorCTFDefocusEstimationInfo(self, ctf_info_imod_table):
        pass

    def refactorCTFDefocusPhaseShiftEstimationInfo(self, ctf_info_imod_table):
        pass

    def refactorCTFDefocusAstigmatismPhaseShiftEstimationInfo(
        self, ctf_info_imod_table
    ):
        pass

    def refactorCTFDefocusAstigmatismPhaseShiftCutOnFreqEstimationInfo(
        self, ctf_info_imod_table
    ):
        pass

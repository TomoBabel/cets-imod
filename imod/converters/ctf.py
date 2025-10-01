import math
import traceback
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from cets_data_model.models.models import CTFMetadata
from imod.contants import MRC_MRCS_EXT
from imod.utils.utils import (
    get_ts_no_imgs,
    standarize_defocus,
    load_md_list_yaml,
    validate_file,
    validate_new_file,
)


class ImodCtfSeries:
    def __init__(self, ts_file_name: Path, defocus_file: Path) -> None:
        self.ts_file_name = validate_file(ts_file_name, "ts_file_name", MRC_MRCS_EXT)
        self.defocus_file = validate_file(defocus_file, "defocus_file", ".defocus")

    def imod_to_cets(
        self, out_yaml_file: str | Path | None = None
    ) -> List[CTFMetadata]:
        """Reads a .defocus file and generates a list of CETS CTFMetada objects,
        one per tilt-image.

        :param out_yaml_file: name of the yaml file in which the CTF
        metadata will be written.
        :type out_yaml_file: pathlib.Path or str, optional
        """
        return self._parse_defocus_file(out_yaml_file=out_yaml_file)

    @staticmethod
    def cets_to_imod(
        ctf_yaml_file: Path | str,
        tilt_angle_list: List[float],
        defocus_file: Path,
    ) -> None:
        """Reads a .yaml file containing the serialized CETS CTFMetadata objects of
        all the tilt-images that compose a tilt-series and generates the corresponding
        IMOD defocus file."""
        # TODO: check the phase shift
        md_list = load_md_list_yaml(ctf_yaml_file, CTFMetadata)
        n_data_sec = len(md_list)
        n_angles = len(tilt_angle_list)
        if n_data_sec != n_angles:
            raise Exception(
                f"The number of data sections [{n_data_sec}] must be the same "
                f"as the number of angles [{n_angles}] provided."
            )
        with open(defocus_file, "w") as f:
            lines = ["1\t0\t0.0\t0.0\t0.0\t3\n"]
            for i, ctf_md_dict in enumerate(md_list):
                ind = i + 1  # Indices in IMOD start in 1
                tilt_angle = tilt_angle_list[i]
                newLine = "%i\t%i\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f\n" % (
                    ind,
                    ind,
                    tilt_angle,
                    tilt_angle,
                    ctf_md_dict["defocus_u"] / 10,  # Defocus value in Imod is in nm
                    ctf_md_dict["defocus_v"] / 10,
                    ctf_md_dict["defocus_angle"],
                )

                lines.append(newLine)
            f.writelines(lines)
        print(f"defocus file successfully writen! -> {defocus_file}")

    def _parse_defocus_file(
        self, out_yaml_file: str | Path | None = None
    ) -> List[CTFMetadata]:
        """Parse tilt-series ctf estimation file."""
        defocusFileFlag = self._get_defocus_file_flag()
        (defocus_u_dict, defocus_v_dict, defocus_angle_dict, phase_shift_dict) = (
            dict(),
            dict(),
            dict(),
            dict(),
        )

        if defocusFileFlag == 0:
            # Plain estimation
            defocus_u_dict = self._load_ctf_file(defocusFileFlag)

        elif defocusFileFlag == 1:
            # Astigmatism estimation
            defocus_u_dict, defocus_v_dict, defocus_angle_dict = self._load_ctf_file(
                defocusFileFlag
            )

        elif defocusFileFlag == 4:
            # Phase-shift information
            defocus_u_dict, phase_shift_dict = self._load_ctf_file(defocusFileFlag)

        elif defocusFileFlag == 5:
            # Astigmatism and phase shift estimation
            defocus_u_dict, defocus_v_dict, defocus_angle_dict, phase_shift_dict = (
                self._load_ctf_file(defocusFileFlag)
            )

        elif defocusFileFlag == 37:
            # Astigmatism, phase shift and cut-on frequency estimation "
            defocus_u_dict, defocus_v_dict, defocus_angle_dict, phase_shift_dict, _ = (
                self._load_ctf_file(defocusFileFlag)
            )

        else:
            raise ValueError(
                f"Defocus file flag {defocusFileFlag} is not supported. Only supported formats "
                "correspond to flags 0, 1, 4, 5, and 37."
            )

        n_imgs = get_ts_no_imgs(self.ts_file_name)
        ctf_md_list = []
        for i in range(1, n_imgs + 1):
            defocus_u_list = defocus_u_dict.get(i, [])
            defocus_v_list = defocus_v_dict.get(i, [])
            phase_shift_list = phase_shift_dict.get(i, [])
            defocus_angle_list = defocus_angle_dict.get(i, [])
            len_defocus_angle_list = len(defocus_angle_list)
            len_phase_shift_list = len(phase_shift_list)

            # DEFOCUS INFORMATION --------------------------------------------------------------------------------------
            if defocus_u_list and defocus_v_list:
                # Check that the three list are equally long
                len_defocus_u_list = len(defocus_u_list)
                if (
                    len_defocus_u_list + len(defocus_v_list) + len_defocus_angle_list
                ) % 3 != 0:
                    raise Exception(
                        "defocus_u_list, defocus_v_list and defocus_angle_list lengths must be equal."
                    )

                # DefocusU, DefocusV and DefocusAngle are set equal to the middle estimation of the list
                middlePoint = math.trunc(len_defocus_u_list / 2)

                # If the size of the defocus list is even, mean the 2 centre values
                if len_defocus_u_list % 2 == 0:
                    defocus_u = (
                        defocus_u_list[middlePoint] + defocus_u_list[middlePoint - 1]
                    ) / 2
                    defocus_v = (
                        defocus_v_list[middlePoint] + defocus_v_list[middlePoint - 1]
                    ) / 2
                    defocus_angle = (
                        defocus_angle_list[middlePoint]
                        + defocus_angle_list[middlePoint - 1]
                    ) / 2
                else:
                    defocus_u = defocus_u_list[middlePoint]
                    defocus_v = defocus_v_list[middlePoint]
                    defocus_angle = defocus_angle_list[middlePoint]

            else:
                defocus_list = defocus_u_list if defocus_u_list else defocus_v_list
                len_defocus_list = len(defocus_list)
                defocus_angle = 0
                # DefocusU and DefocusV are set at the same value, equal to the middle
                # estimation of the list
                middle_point = math.trunc(len_defocus_list / 2)
                # If the size of the defocus list is even, mean the 2 centre values
                if len_defocus_list % 2 == 0:
                    defocus_u = (
                        defocus_list[middle_point] + defocus_list[middle_point - 1]
                    ) / 2
                else:
                    # The size of defocus estimation is odd, get the centre value
                    defocus_u = defocus_list[middle_point]
                defocus_v = defocus_u

            # PHASE SHIFT INFORMATION ----------------------------------------------------------------------------------
            # Check that all the lists are equally long
            phase_shift = 0
            if phase_shift_list:
                if (
                    len(defocus_u_list) + len_phase_shift_list + len_defocus_angle_list
                ) % 3 != 0:
                    raise Exception(
                        f"phase_shift_list length [{len_phase_shift_list}] must be equal to "
                        f"defocus_u_list [{len(defocus_u_list)}], "
                        f"defocus_v_list [{len(defocus_v_list)}] and "
                        f"defocus_angle_list [{len_defocus_angle_list}] lengths."
                    )

                # PhaseShift is set equal to the middle estimation of the list
                middlePoint = math.trunc(len(phase_shift_list) / 2)

                # If the size of the phase shift list is even, mean the 2 centre values
                if len(phase_shift_list) % 2 == 0:
                    phase_shift = (
                        phase_shift_list[middlePoint]
                        + phase_shift_list[middlePoint - 1]
                    ) / 2
                else:
                    # If the size of phase shift list estimation is odd, get the centre value
                    phase_shift = phase_shift_list[middlePoint]

            defocus_u, defocus_v, defocus_angle = standarize_defocus(
                defocus_u, defocus_v, defocus_angle
            )
            ctf_md_list.append(
                CTFMetadata(
                    defocus_u=defocus_u,
                    defocus_v=defocus_v,
                    defocus_angle=defocus_angle,
                    phase_shift=phase_shift,
                )
            )
        # Write the output yaml file if requested
        self._write_ctf_yaml(ctf_md_list, out_yaml_file)
        return ctf_md_list

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
            return self._refactor_ctf_flag_0(ctf_info_imod_table)

        elif flag == 1:
            # Astigmatism estimation
            return self._refactor_ctf_flag_1(ctf_info_imod_table)

        elif flag == 4:
            # Phase-shift estimation
            return self._refactor_ctf_flag_4(ctf_info_imod_table)

        elif flag == 5:
            # Astigmatism and phase shift estimation
            return self._refactor_ctf_flag_5(ctf_info_imod_table)

        elif flag == 37:
            # Astigmatism, phase shift and cut-on frequency estimation
            return self._refactor_ctf_flag_37(ctf_info_imod_table)

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

    @staticmethod
    def _append_to_dict(dictionary: dict, index: int, value: float) -> None:
        # Python Dictionary setdefault() returns the value of a key (if the key is in dictionary).
        # Else, it inserts a key with the default value to the dictionary.
        dictionary.setdefault(index, []).append(value)

    def _refactor_ctf_flag_0(
        self, ctf_info_imod_table: List[List[float]]
    ) -> dict[int, list[float]]:
        """This method takes a table containing the information of
        an IMOD-based CTF estimation containing only defocus
        information (5 columns) and produces a dictionary for clearer
        and easier management. Flag 0 (Plain estimation)."""

        if len(ctf_info_imod_table[0]) != 5:
            raise Exception(
                "Misleading file format, CTF estimation with no astigmatism should be 5 columns long"
            )

        defocus_u_dict: Dict[int, List[float]] = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus = float(element[4]) * 10

            for index in range(start, end + 1):
                self._append_to_dict(defocus_u_dict, index, defocus)

        return defocus_u_dict

    def _refactor_ctf_flag_1(
        self, ctf_info_imod_table: List[List[float]]
    ) -> Tuple[dict[int, list[float]], dict[int, list[float]], dict[int, list[float]]]:
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
        defocus_u_dict: Dict[int, List[float]] = {}
        defocus_v_dict: Dict[int, List[float]] = {}
        defocus_angle_dict: Dict[int, List[float]] = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus_u = float(element[4]) * 10  # From nm to angstroms
            defocus_v = float(element[5]) * 10  # From nm to angstroms
            defocus_angle = float(element[6])

            # Segregate information from range
            for index in range(start, end + 1):
                self._append_to_dict(defocus_u_dict, index, defocus_u)
                self._append_to_dict(defocus_v_dict, index, defocus_v)
                self._append_to_dict(defocus_angle_dict, index, defocus_angle)

        return defocus_u_dict, defocus_v_dict, defocus_angle_dict

    def _refactor_ctf_flag_4(
        self, ctf_info_imod_table: List[List[float]]
    ) -> Tuple[dict[int, list[float]], dict[int, list[float]]]:
        """This method takes a table containing the information of
        an IMOD-based CTF estimation containing defocus, and phase
        shift information (6 columns) and produces a new set of
        dictionaries for clearer and easier management. Flag 4
        (Phase-shift estimation)."""

        if len(ctf_info_imod_table[0]) != 6:
            raise Exception(
                "Misleading file format, CTF estimation with defocus "
                "and phase shift should be 6 columns long"
            )

        defocus_u_dict: Dict[int, List[float]] = {}
        phase_shift_dict: Dict[int, List[float]] = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus_u = float(element[4]) * 10
            phase_shift = float(element[5])

            for index in range(start, end + 1):
                self._append_to_dict(defocus_u_dict, index, defocus_u)
                self._append_to_dict(phase_shift_dict, index, phase_shift)

        return defocus_u_dict, phase_shift_dict

    def _refactor_ctf_flag_5(
        self, ctf_info_imod_table: List[List[float]]
    ) -> Tuple[
        dict[int, list[float]],
        dict[int, list[float]],
        dict[int, list[float]],
        dict[int, list[float]],
    ]:
        """This method takes a table containing the information of
        an IMOD-based CTF estimation containing defocus, astigmatism
        and phase shift information (8 columns) and produces a new
        set of dictionaries for clearer and easier management.
        Flag 5 (Astigmatism and phase shift estimation)."""

        if len(ctf_info_imod_table[0]) != 8:
            raise Exception(
                "Misleading file format, CTF estimation with astigmatism and phase "
                "shift should be 8 columns long"
            )

        defocus_u_dict: Dict[int, List[float]] = {}
        defocus_v_dict: Dict[int, List[float]] = {}
        defocus_angle_dict: Dict[int, List[float]] = {}
        phase_shift_dict: Dict[int, List[float]] = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus_u = float(element[4]) * 10
            defocus_v = float(element[5]) * 10
            angle = float(element[6])
            phase_shift = float(element[7])

            for index in range(start, end + 1):
                self._append_to_dict(defocus_u_dict, index, defocus_u)
                self._append_to_dict(defocus_v_dict, index, defocus_v)
                self._append_to_dict(defocus_angle_dict, index, angle)
                self._append_to_dict(phase_shift_dict, index, phase_shift)

        return defocus_u_dict, defocus_v_dict, defocus_angle_dict, phase_shift_dict

    def _refactor_ctf_flag_37(
        self, ctf_info_imod_table: List[List[float]]
    ) -> Tuple[
        dict[int, list[float]],
        dict[int, list[float]],
        dict[int, list[float]],
        dict[int, list[float]],
    ]:
        """This method takes a table containing the information of an
        IMOD-based CTF estimation containing defocus, astigmatism, phase
        shift information and cut-on frequency (8 columns) and produces a
        new set of dictionaries for clearer and easier management.
        Flag 37 (Astigmatism, phase shift and cut-on frequency estimation)."""

        if len(ctf_info_imod_table[0]) != 9:
            raise Exception(
                "Misleading file format, CTF estimation with astigmatism, "
                "phase shift and cut-on frequency should be 9 columns long"
            )

        defocus_u_dict: Dict[int, List[float]] = {}
        defocus_v_dict: Dict[int, List[float]] = {}
        defocus_angle_dict: Dict[int, List[float]] = {}
        phase_shift_dict: Dict[int, List[float]] = {}

        for element in ctf_info_imod_table:
            start, end = int(element[0]), int(element[1])
            defocus_u = float(element[4]) * 10
            defocus_v = float(element[5]) * 10
            angle = float(element[6])
            phase_shift = float(element[7])

            for index in range(start, end + 1):
                self._append_to_dict(defocus_u_dict, index, defocus_u)
                self._append_to_dict(defocus_v_dict, index, defocus_v)
                self._append_to_dict(defocus_angle_dict, index, angle)
                self._append_to_dict(phase_shift_dict, index, phase_shift)

        return (
            defocus_u_dict,
            defocus_v_dict,
            defocus_angle_dict,
            phase_shift_dict,
        )

    @staticmethod
    def _write_ctf_yaml(
        cets_ctf_md_list: List[CTFMetadata], yaml_file: Path | str | None
    ) -> None:
        if yaml_file is None:
            print("write_yaml -> yaml_file is None. Skipping...")
            return
        try:
            yaml_file = validate_new_file(yaml_file)
            with open(yaml_file, "a") as file:
                for metadata in cets_ctf_md_list:
                    metadata_dict = metadata.model_dump()
                    yaml.dump(metadata_dict, file, sort_keys=False, explicit_start=True)
            print(f"yaml file successfully written! -> {yaml_file}")
        except Exception as e:
            print(
                f"Unable to write the output yaml file {yaml_file} with "
                f"the exception -> {e}"
            )
            print(traceback.format_exc())


# # READER EXAMPLE
# import yaml
# from os.path import exists
# from os import remove
#
# f_path = Path(
#     "/home/jjimenez/scipion3/data/tests/relion40_sta_tutorial_data/tomograms/TS_03"
# )
# ts_file = f_path.joinpath("03.mrc")
# f_path_defocus = Path(
#     "/home/jjimenez/scipion3/data/tests/relion40_sta_tutorial_data/testImodCtf"
# )
# defocus_f = f_path_defocus.joinpath("TS_03.defocus")
#
# ics = ImodCtfSeries(ts_file_name=Path(ts_file), defocus_file=Path(defocus_f))
# mdList = ics.imod_to_cets()
#
# output_path = Path("/home/jjimenez/CZII/cets_scratch_dir")
# yaml_file = output_path.joinpath("TS_03_cets_ctf.yaml")
# if exists(yaml_file):
#     remove(yaml_file)
#
# with open(yaml_file, "a") as file:
#     for metadata in mdList:
#         metadata_dict = metadata.model_dump()
#         yaml_output = yaml.dump(
#             metadata_dict, file, sort_keys=False, explicit_start=True
#         )
#
# for metadata in mdList:
#     metadata_dict = metadata.model_dump()
#     yaml_output = yaml.dump(metadata_dict, sort_keys=False, explicit_start=True)
#     print(yaml_output)
#
# # WRITER EXAMPLE
# tlt_file = f_path.joinpath("03.rawtlt")
# with open(tlt_file, "r") as f:
#     tilt_angle_list = [float(line.strip()) for line in f if line.strip()]
#
# output_defocus_file = output_path.joinpath("TS_03.defocus")
# if exists(output_defocus_file):
#     remove(output_defocus_file)
# ics.cets_to_imod(yaml_file, tilt_angle_list, output_defocus_file)
#
# print(f"Defocus file written written from CETS {output_defocus_file}")

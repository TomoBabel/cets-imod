import math
import traceback
from pathlib import Path
from typing import List, Dict
import yaml
from cets_data_model.models.models import CTFMetadata
from imod.contants import (
    MRC_MRCS_EXT,
    DEFOCUS_FLAG_ASTIGMATISM,
    DEFOCUS_FLAG_ANGLE_IN_RADIANS,
    DEFOCUS_FLAG_PHASE_SHIFT,
    DEFOCUS_FLAG_PHASE_IN_RADIANS,
    DEFOCUS_FLAG_CUT_ON,
    DEFOCUS_FILE_VERSION_3,
)
from imod.utils.utils import (
    get_ts_no_imgs,
    standarize_defocus,
    load_md_list_yaml,
    validate_file,
    validate_new_file,
)

# Names of the columns of the IMOD defocus files
START_VIEW = "start_view"
END_VIEW = "end_view"
START_ANGLE = "start_angle"
END_ANGLE = "end_angle"
DEFOCUS_U = "defocus_u"
DEFOCUS_V = "defocus_v"
DEFOCUS_ANGLE = "defocus_angle"
PHASE_SHIFT = "phase_shift"
CUT_ON_FREQUENCY = "cut_on_frequency"


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
        IMOD defocus file (version 3, with astigmatism and, if any image has a non-zero
        phase shift, phase shifts in degrees)."""
        md_list = load_md_list_yaml(ctf_yaml_file, CTFMetadata)
        n_data_sec = len(md_list)
        n_angles = len(tilt_angle_list)
        if n_data_sec != n_angles:
            raise Exception(
                f"The number of data sections [{n_data_sec}] must be the same "
                f"as the number of angles [{n_angles}] provided."
            )
        has_phase_shift = any(md.get(PHASE_SHIFT) for md in md_list)
        flag = DEFOCUS_FLAG_ASTIGMATISM
        if has_phase_shift:
            flag += DEFOCUS_FLAG_PHASE_SHIFT
        with open(defocus_file, "w") as f:
            lines = [f"{flag}\t0\t0.0\t0.0\t0.0\t{DEFOCUS_FILE_VERSION_3}\n"]
            for i, ctf_md_dict in enumerate(md_list):
                ind = i + 1  # Indices in IMOD start in 1
                tilt_angle = tilt_angle_list[i]
                newLine = "%i\t%i\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f" % (
                    ind,
                    ind,
                    tilt_angle,
                    tilt_angle,
                    ctf_md_dict[DEFOCUS_U] / 10,  # Defocus value in Imod is in nm
                    ctf_md_dict[DEFOCUS_V] / 10,
                    ctf_md_dict[DEFOCUS_ANGLE],
                )
                if has_phase_shift:
                    newLine += "\t%.2f" % (ctf_md_dict.get(PHASE_SHIFT) or 0.0)
                lines.append(newLine + "\n")
            f.writelines(lines)
        print(f"defocus file successfully writen! -> {defocus_file}")

    def _parse_defocus_file(
        self, out_yaml_file: str | Path | None = None
    ) -> List[CTFMetadata]:
        """Parse tilt-series ctf estimation file.

        Every combination of the flags of the IMOD defocus file format is supported (see
        ``_get_defocus_file_flag``):

        * Defoci are converted from nm to Å (underfocus positive in both).
        * The astigmatism angle and the phase shift are converted to degrees when the flags
          say they are in radians.
        * The tilt-angle inversion flag does not change the values read, as they are
          assigned to the views by their number, not by their angle.
        * The phase shift is 0 when the file has none.
        * The cut-on frequencies are not kept, as CTFMetadata has no field for them.

        When a view is covered by several lines (ranges of views fitted together), the
        middle estimation is taken (the mean of the two middle ones if there is an even
        number of them).
        """
        flag = self._get_defocus_file_flag()
        values_per_view = self._load_ctf_file(flag)

        n_imgs = get_ts_no_imgs(self.ts_file_name)
        ctf_md_list = []
        for view in range(1, n_imgs + 1):  # Views in IMOD are numbered from 1
            view_values = values_per_view.get(view, {})
            defocus_u = self._middle_value(view_values, DEFOCUS_U, view)
            if DEFOCUS_V in view_values:
                defocus_v = self._middle_value(view_values, DEFOCUS_V, view)
                defocus_angle = self._middle_value(view_values, DEFOCUS_ANGLE, view)
            else:
                # No astigmatism: both defoci are the same
                defocus_v = defocus_u
                defocus_angle = 0.0
            phase_shift = (
                self._middle_value(view_values, PHASE_SHIFT, view)
                if PHASE_SHIFT in view_values
                else 0.0
            )
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

         (from https://bio3d.colorado.edu/imod/doc/man/ctfphaseflip.html).

         Only version 3 files, whose first line is the header "flags 0 0. 0. 0 3", have
         flags. Versions 1 and 2 only have defocus values, which corresponds to flag 0."""
        with open(self.defocus_file) as f:
            first_line = f.readline().split()
        if self._is_version_3_header(first_line):
            return int(float(first_line[0]))
        return 0

    @staticmethod
    def _is_version_3_header(fields: List[str]) -> bool:
        """The header of version 3 files is "flags 0 0. 0. 0 3". Data lines cannot be
        confused with it, as their second value (ending view) is numbered from 1."""
        try:
            return (
                len(fields) == 6
                and float(fields[1]) == 0
                and float(fields[5]) == DEFOCUS_FILE_VERSION_3
            )
        except ValueError:
            return False

    @staticmethod
    def _get_defocus_file_columns(flag: int) -> List[str]:
        """Names of the columns of the data lines of a defocus file with the given flag:
        starting and ending view, starting and ending tilt angle and defocus, followed by the
        optional columns in the order the format defines them."""
        columns = [START_VIEW, END_VIEW, START_ANGLE, END_ANGLE, DEFOCUS_U]
        if flag & DEFOCUS_FLAG_ASTIGMATISM:
            columns += [DEFOCUS_V, DEFOCUS_ANGLE]
        if flag & DEFOCUS_FLAG_PHASE_SHIFT:
            columns.append(PHASE_SHIFT)
        if flag & DEFOCUS_FLAG_CUT_ON:
            columns.append(CUT_ON_FREQUENCY)
        return columns

    def _load_ctf_file(self, flag: int) -> Dict[int, Dict[str, List[float]]]:
        """Reads the data lines of the defocus file and returns, for each view (numbered
        from 1), the list of estimations of each value (defoci in Å, angles in degrees). A
        view has several estimations when it is covered by several lines."""
        columns = self._get_defocus_file_columns(flag)
        values_per_view: Dict[int, Dict[str, List[float]]] = {}
        for row in self._defocus_file_to_table(flag):
            if len(row) != len(columns):
                raise ValueError(
                    f"The data lines of {self.defocus_file} (flag {flag}) should have "
                    f"{len(columns)} columns ({', '.join(columns)}), but one has "
                    f"{len(row)}: {row}"
                )
            values = dict(zip(columns, row))
            values[DEFOCUS_U] *= 10  # From nm to angstroms
            if DEFOCUS_V in values:
                values[DEFOCUS_V] *= 10  # From nm to angstroms
            if flag & DEFOCUS_FLAG_ANGLE_IN_RADIANS and DEFOCUS_ANGLE in values:
                values[DEFOCUS_ANGLE] = math.degrees(values[DEFOCUS_ANGLE])
            if flag & DEFOCUS_FLAG_PHASE_IN_RADIANS and PHASE_SHIFT in values:
                values[PHASE_SHIFT] = math.degrees(values[PHASE_SHIFT])
            for view in range(int(values[START_VIEW]), int(values[END_VIEW]) + 1):
                view_values = values_per_view.setdefault(view, {})
                for field in (DEFOCUS_U, DEFOCUS_V, DEFOCUS_ANGLE, PHASE_SHIFT):
                    if field in values:
                        view_values.setdefault(field, []).append(values[field])
        return values_per_view

    def _defocus_file_to_table(self, flag: int) -> List[List[float]]:
        """This method takes an IMOD-based ctf estimation file and returns a table
        containing the values of each data line. The header line of version 3 files is
        skipped, and the version number that the first line of version 1 and 2 files may
        have at its end is removed."""
        with open(self.defocus_file) as f:
            lines = [line.split() for line in f if line.strip()]
        if lines and self._is_version_3_header(lines[0]):
            lines = lines[1:]
        elif lines and len(lines[0]) > len(self._get_defocus_file_columns(flag)):
            lines[0] = lines[0][:-1]  # Version number
        return [[float(value) for value in line] for line in lines]

    def _middle_value(
        self, view_values: Dict[str, List[float]], field: str, view: int
    ) -> float:
        """Middle estimation of a value of a view, or the mean of the two middle ones if
        there is an even number of them."""
        values = view_values.get(field, [])
        if not values:
            raise ValueError(
                f"The view {view} has no {field} in the defocus file {self.defocus_file}"
            )
        middle = len(values) // 2
        if len(values) % 2 == 0:
            return (values[middle - 1] + values[middle]) / 2
        return values[middle]

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

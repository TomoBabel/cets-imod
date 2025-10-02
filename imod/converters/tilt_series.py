import traceback
from pathlib import Path
from typing import Optional, List
import numpy as np
import yaml

from cets_data_model.models.models import (
    Affine,
    CTFMetadata,
    TiltSeries,
    TiltImage,
    CoordinateTransformation,
)
from cets_data_model.utils.image_utils import get_mrc_info
from imod.contants import MRC_MRCS_EXT
from imod.utils.utils import (
    validate_file,
    validate_tilt_angle_list,
    parse_tlt_file,
    parse_xf_file,
    validate_ctf_md_list,
    validate_even_odd_files,
    write_tlt,
    write_xf,
    validate_new_file,
    load_md_list_yaml,
)


class ImodTiltSeries:
    def __init__(
        self,
        ts_file_name: str | Path,
        tilt_angles: str | Path | List[float],
        ctf_md_list: Optional[List[CTFMetadata]] = None,
    ) -> None:
        self.ts_file_name = validate_file(ts_file_name, "ts_file_name", MRC_MRCS_EXT)
        if type(tilt_angles) is List[float]:
            tilt_angles = validate_tilt_angle_list(self.ts_file_name, tilt_angles)
            tlt_file, dose_list, acq_orders = None, None, None
        else:
            tlt_file = validate_file(
                str(tilt_angles), "tilt_angles", [".tlt", ".rawtlt"]
            )
            tilt_angles, dose_list, acq_orders = parse_tlt_file(tlt_file)
        n_imgs = len(tilt_angles)
        self.ctf_md_list = validate_ctf_md_list(ctf_md_list, n_imgs)
        self.tlt_file = tlt_file
        self.tilt_angles = tilt_angles
        self.dose_list = dose_list
        self.acq_orders = acq_orders
        self.n_imgs = n_imgs

    def imod_to_cets(
        self,
        xf_file: str | Path | None = None,
        even_file_name: str | Path | None = None,
        odd_file_name: str | Path | None = None,
        ctf_corrected: bool = False,
        out_yaml_file: str | Path | None = None,
    ) -> TiltSeries:
        """Converts an IMOD tilt-series into CETS metadata.

        :param xf_file: xf alignment file. If not provided, the Identity matrix
        will be used as alignment data.
        :type xf_file: pathlib.Path or str, optional

        :param even_file_name: path of the even tomogram,
        :type even_file_name: pathlib.Path or str, optional

        :param odd_file_name: path of the even tomogram,
        :type odd_file_name: pathlib.Path or str, optional

        :param ctf_corrected: xFlag to indicate if the tomogram was reconstructed
        from a tilt-series with the ctf corrected.
        :type ctf_corrected: bool, optional

        :param out_yaml_file: name of the yaml file in which the tilt-series
        metadata will be written.
        :type out_yaml_file: pathlib.Path or str, optional
        """
        # Validate even/odd
        even_file_name, odd_file_name = validate_even_odd_files(
            even_file_name, odd_file_name
        )
        # Read image info
        img_info = get_mrc_info(self.ts_file_name)
        width = img_info.size_x
        height = img_info.size_y
        pix_size = img_info.apix_x
        # Parse xf file
        xf_file = validate_file(xf_file, "xf_file", ".xf")
        in_transform_matrix = parse_xf_file(xf_file)

        ts_filename = str(self.ts_file_name)
        ts_id = self.ts_file_name.stem
        pixel_size = img_info.apix_x
        ti_list = []
        for index in range(self.n_imgs):
            output_transform_matrix = in_transform_matrix[:, :, index]
            ti = TiltImage(
                path=ts_filename,
                section=index,
                nominal_tilt_angle=self.tilt_angles[index],
                accumulated_dose=self.dose_list[index] if self.dose_list else None,
                ctf_metadata=self.ctf_md_list[index] if self.ctf_md_list else None,
                width=width,
                height=height,
                coordinate_systems=None,  # TODO: fill this
                coordinate_transformations=self._genTransform(
                    output_transform_matrix, pix_size
                ),
                ts_id=ts_id,
                acq_order=self.acq_orders[index] if self.acq_orders else None,
                pixel_size=pixel_size,
            )
            ti_list.append(ti)
        ts = TiltSeries(
            path=ts_filename,
            ts_id=ts_id,
            pixel_size=pixel_size,
            ctf_corrected=ctf_corrected,
            even_path=str(even_file_name),
            odd_path=str(odd_file_name),
            images=ti_list,
        )
        # Write the output yaml file if requested
        self._write_ts_yaml(ts, out_yaml_file)
        return ts

    @staticmethod
    def cets_to_imod(
        cets_ts: TiltSeries | Path | str,
        tlt_file: str | Path,
        add_dose_to_tlt: bool = True,
        xf_file: str | Path | None = None,
    ):
        """Converts CETS Tilt-series metadata into IMOD files.

        :param cets_ts: CETS tilt-series metadata or a yaml file written from it.
        :type cets_ts: TiltSeries or pathlib.Path or str

        :param tlt_file: output tlt file to be generated.
        :type tlt_file: pathlib.Path or str

        :param add_dose_to_tlt: used to indicate if the generated tlt file should also
        contain a second column with the dose.
        :type add_dose_to_tlt: bool, optional, Defaults to True

        :param xf_file: output xf file to be generated.
        :type: pathlib.Path or str, optional, Defaults to None
        """
        if type(cets_ts) is not TiltSeries:
            cets_ts = load_md_list_yaml(cets_ts, TiltSeries)
        # Write the tlt file
        write_tlt(cets_ts, tlt_file, add_dose_to_tlt=add_dose_to_tlt)
        if xf_file is not None:
            # Write the xf file
            write_xf(cets_ts, xf_file)

    def _genTransform(
        self, xf_matrix: np.ndarray, pix_size: float
    ) -> list[CoordinateTransformation]:
        return [
            Affine(
                affine=self._get_affine_values(xf_matrix, pix_size),
                name="IMOD roto-translation from a .xf file. Shifts in angstroms.",
                input="Aligned and projected movie frames (unaligned tilt-image)",
                output="Aligned projection (tilt-image)",
            )
        ]

    @staticmethod
    def _get_affine_values(xf_matrix: np.ndarray, pix_size: float) -> List[List[float]]:
        """Gets the rotation angle in degrees, and the shifts in X and Y directions,
        in angstroms."""
        xf_matrix = xf_matrix.tolist()
        row1 = xf_matrix[0]
        row2 = xf_matrix[1]
        row1[-1] *= pix_size  # shift_x: convert to angstroms
        row2[-1] *= pix_size  # shift_y: convert to angstroms
        return [row1, row2, xf_matrix[2]]

    @staticmethod
    def _write_ts_yaml(cets_ts_md: TiltSeries, yaml_file: Path | str | None) -> None:
        if yaml_file is None:
            print("write_yaml -> yaml_file is None. Skipping...")
            return
        try:
            yaml_file = validate_new_file(yaml_file)
            metadata_dict = cets_ts_md.model_dump(mode="json")
            with open(yaml_file, "a") as f:
                yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
            print(f"yaml file successfully written! -> {yaml_file}")
        except Exception as e:
            print(
                f"Unable to write the output yaml file {yaml_file} with "
                f"the exception -> {e}"
            )
            print(traceback.format_exc())

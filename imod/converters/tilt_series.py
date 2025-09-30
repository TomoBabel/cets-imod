from pathlib import Path
from typing import Optional, List
import numpy as np
from cets_data_model.models.ctf_model import CTFMetadata
from cets_data_model.models.models import Affine
from cets_data_model.models.tilt_images_model import TiltImage, TiltSeries
from cets_data_model.utils.image_utils import get_mrc_info
from imod.contants import MRC_MRCS_EXT
from imod.utils.utils import (
    validate_file,
    validate_tilt_angle_list,
    parse_tlt_file,
    parse_xf_file,
    validate_ctf_md_list,
    validate_even_odd_files,
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
        xf_file: str | Path = "",
        binning: int = 1,
        even_file_name: str | Path = "",
        odd_file_name: str | Path = "",
        ctf_corrected: bool = False,
    ) -> TiltSeries:
        """Converts an IMOD tilt-series into CETS metadata.

        :param xf_file: xf alignment file. If not provided, the Identity matrix
        will be used as alignment data.
        :type xf_file: pathlib.Path, optional

        :param binning: binning factor desired to be applied to scale the transformation
        shifts.
        :type binning: int, optional

        :param even_file_name: path of the even tomogram,
        :type even_file_name: pathlib.Path, optional

        :param odd_file_name: path of the even tomogram,
        :type odd_file_name: pathlib.Path, optional

        :param ctf_corrected: xFlag to indicate if the tomogram was reconstructed
        from a tilt-series with the ctf corrected.
        :type ctf_corrected: bool, optional
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
        ti_list = []
        for index in range(self.n_imgs):
            output_transform_matrix = in_transform_matrix[:, :, index]
            if binning > 1:  # Re-scale the shifts if necessary
                output_transform_matrix[0][2] *= binning
                output_transform_matrix[1][2] *= binning
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
                ts_id=self.ts_file_name.stem,
                acq_order=self.acq_orders[index] if self.acq_orders else None,
                pixel_size=img_info.apix_x,
                ctf_corrected=ctf_corrected,
                even_path=str(even_file_name),
                odd_path=str(odd_file_name),
            )
            ti_list.append(ti)
        return TiltSeries(
            images=ti_list,
            path=ts_filename,
        )

    def cets_to_imod(
        self,
        tlt_file: str | Path,
        add_dose_to_tlt: Optional[bool] = True,
        xf_file: Optional[str | Path] = None,
    ):
        """Converts CETS Tilt-series metadata into IMOD files.

        :param tlt_file: output tlt file to be generated.
        :type tlt_file: pathlib.Path

        :param add_dose_to_tlt: used to indicate if the generated tlt file should also
        contain a second column with the dose.
        :type add_dose_to_tlt: bool, optional, Defaults to True

        :param xf_file: output xf file to be generated.
        :type: pathlib.Path, optional
        """
        pass

    def _genTransform(self, xf_matrix: np.ndarray, pix_size: float) -> list[Affine]:
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
        sx = xf_matrix[0, 2] * pix_size
        sy = xf_matrix[1, 2] * pix_size
        row1 = xf_matrix[0]
        row1[-1] = sx
        row2 = xf_matrix[1]
        row2[-1] = sy
        return [row1, row2, xf_matrix[2]]

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np


def load_mrc_file(mrc_file: Path) -> np.ndarray:
    """Loads an MRC file"""
    with mrcfile.mmap(mrc_file, mode="r", permissive=True) as mrc:
        data = mrc.data
    return data


def get_ts_no_imgs(ts_file_name: Path):
    """Loads a tilt-series mrc file and returns the number of images"""
    data = load_mrc_file(ts_file_name)
    dims = data.shape
    return 1 if len(dims) < 2 else min(dims)


def standarize_defocus(
    defocus_u: float, defocus_v: float, defocus_angle: float
) -> Tuple[float, float, float]:
    """Modify defocusU, defocusV and defocusAngle to conform
    the EMX standard: defocusU > defocusV, 0 <= defocusAngle < 180
    and the defocusAnges is between x-axis and defocusU. Also
    determine the defocusRatio(defocusU/defocusV).
    For more details see:
    http://i2pc.cnb.csic.es/emx/LoadDictionaryFormat.htm?type=Convention#ctf
    """
    if defocus_v > defocus_u:
        out_defocus_u = defocus_v  # exchange defocusU by defocusV
        out_defocus_v = defocus_u
        defocus_angle += 90.0
    else:
        out_defocus_u = defocus_u
        out_defocus_v = defocus_v
    if defocus_angle >= 180.0:
        defocus_angle -= 180.0
    elif defocus_angle < 0.0:
        defocus_angle += 180.0

    return out_defocus_u, out_defocus_v, defocus_angle

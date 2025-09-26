import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, get_args, get_origin, Union, get_type_hints, List, Dict, Type
import mrcfile
import numpy as np


def validate_file(filename: Path | str, field_name: str) -> Path:
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{field_name} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{field_name} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {field_name}: {p}")

    return p


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


# TODO: if yaml is used, generalize it to any Pydantic ConfiguredBaseModel and move to the main repo
def _resolve_type(tp):
    """Helper to resolve type from Optional/Union as get_args(Optional[float]) returns (float, NoneType)
    as it is a shorthand for Union[float, None]"""
    if get_origin(tp) is Union:
        args = get_args(tp)
        non_none_args = [arg for arg in args if arg is not type(None)]
        return non_none_args[0] if non_none_args else type(None)
    return tp


def _get_resolved_types(model_cls: type[BaseModel]) -> dict[str, type]:
    """Generates a dictionary from a Pydantic model where the keys are the field names
    and the values are their corresponding data types."""
    type_hints = get_type_hints(model_cls)
    return {name: _resolve_type(tp) for name, tp in type_hints.items()}


def _cast_value(value: str, target_type: type):
    """Casting helper function."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return None


def load_md_list_yaml(yaml_file: Path, model_cls: Type[BaseModel]) -> List[Dict]:
    """Loads a .yaml file containing a list of"""
    with open(yaml_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    field_names = list(model_cls.model_fields.keys())
    resolved_types_dict = _get_resolved_types(model_cls)
    metadata_list = []

    token = ":"
    section_len = len(field_names)
    position = 0
    while position < len(lines):
        if token in lines[position]:
            block = lines[position : position + section_len]
            entry = {}
            for line in block:
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                target_type = resolved_types_dict.get(key, str)
                entry[key] = _cast_value(value, target_type)
            metadata_list.append(entry)
            position += section_len
        else:
            position += 1
    return metadata_list

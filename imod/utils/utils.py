import os
import traceback
from pathlib import Path

from pydantic import BaseModel
from typing import Tuple, get_args, get_origin, Union, get_type_hints, List, Dict, Type
import mrcfile
import numpy as np

from cets_data_model.models.models import CTFMetadata, TiltSeries
from imod.contants import MRC_MRCS_EXT


def validate_file(
    filename: Path | str | None, field_name: str, expected_ext: str | List[str]
) -> Path:
    if filename is None:
        raise ValueError(f"File read from field {field_name} cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{field_name} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{field_name} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {field_name}: {p}")

    ext = p.suffix
    expected_ext_list: List[str] = (
        [expected_ext] if isinstance(expected_ext, str) else expected_ext
    )
    if ext not in expected_ext_list:
        raise ValueError(
            f"Invalid file extension '{ext}'. Expected one of: {expected_ext_list}"
        )
    return p


def validate_even_odd_files(
    even_file_name: Path | str | None, odd_file_name: Path | str | None
) -> Tuple[Path | str | None, Path | str | None]:
    even_provided = even_file_name is not None
    odd_provided = odd_file_name is not None
    if even_provided ^ odd_provided:  # Xor
        raise ValueError(
            "Both files 'even_file_name' and 'odd_file_name' should be provided, or none of them."
        )
    if even_provided:
        even_file_name = validate_file(even_file_name, "even_file_name", MRC_MRCS_EXT)
    if odd_provided:
        odd_file_name = validate_file(odd_file_name, "odd_file_name", MRC_MRCS_EXT)
    return even_file_name, odd_file_name


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


def validate_tilt_angle_list(
    ts_filename: Path, tilt_angle_list: List[float]
) -> List[float]:
    n_imgs = get_ts_no_imgs(ts_filename)
    n_angles = len(tilt_angle_list)
    if n_imgs != n_angles:
        raise ValueError(f"Expected {n_imgs} tilt angles, but got {n_angles}.")
    return tilt_angle_list


def validate_ctf_md_list(
    ctf_md_list: List[CTFMetadata] | None, expected_n_elements: int
) -> List[CTFMetadata] | None:
    if ctf_md_list is None:
        return ctf_md_list
    if not all(type(elem) is CTFMetadata for elem in ctf_md_list):
        raise TypeError(
            "All the elements in the ctf metadata provided must be of type CTFMetadata"
        )
    n_ctf_md = len(ctf_md_list)
    if n_ctf_md != expected_n_elements:
        raise ValueError(
            f"Expected {expected_n_elements} CTFMetadata elements, but got {n_ctf_md}."
        )
    return ctf_md_list


def parse_tlt_file(tlt_file_name) -> Tuple[List[float], List[float], List[int]]:
    """Parse the IMOD tlt file, that can contain 1 column (tilt-angles), 2
    (tilt-angles and accumulated dose) or 3 (tilt-angles, accumulated dose and
    acquisition order)."""
    angles, doses, orders = [], [], []
    with open(tlt_file_name) as f:
        for line in f:
            strippedLine = line.strip()
            if not strippedLine:
                print(f"Empty line found in {tlt_file_name}. Ignoring it.")
                continue
            columns = strippedLine.split()
            if columns:
                angles.append(float(columns[0]))
                # If there is a second column, we take it as dose
                if len(columns) > 1:
                    doses.append(float(columns[1]))
                # If there is a third column, we take it as tilt order
                if len(columns) > 2:
                    orders.append(int(columns[2]))

    print(f"Angles found: {angles}")

    if doses:
        print(f"Doses found: {doses}")

    if orders:
        print(f"Acquisition order found: {orders}")
    elif doses:
        # Calculate tilt order based on the dose
        orders = get_acq_order_from_doses(doses)
        print(f"Tilt orders inferred from dose list: {orders}")

    return angles, doses, orders


def parse_xf_file(xf_file: Path) -> np.ndarray:
    """This method takes an IMOD-based transformation matrix file (.xf) path and
    returns a 3D matrix containing the transformation matrices for
    each tilt-image belonging to the tilt-series."""

    matrix = np.loadtxt(xf_file, dtype=float, comments="#")
    n_lines = matrix.shape[0]
    transform_matrix = np.empty([3, 3, n_lines])

    for row in range(n_lines):
        transform_matrix[0, 0, row] = matrix[row][0]
        transform_matrix[1, 0, row] = matrix[row][2]
        transform_matrix[0, 1, row] = matrix[row][1]
        transform_matrix[1, 1, row] = matrix[row][3]
        transform_matrix[0, 2, row] = matrix[row][4]
        transform_matrix[1, 2, row] = matrix[row][5]
        transform_matrix[2, 0, row] = 0.0
        transform_matrix[2, 1, row] = 0.0
        transform_matrix[2, 2, row] = 1.0

    return transform_matrix


def get_acq_order_from_doses(dose_list: List[float]) -> List[int]:
    """Generate the acquisition order list of a tilt-series based on the provided
    dose_list, which may be unsorted.
    """
    dose_list_sorted = np.argsort(dose_list)
    acq_order_list = [0] * len(dose_list)
    for i in range(len(dose_list)):
        finalIndex = dose_list_sorted[i]
        acq_order_list[finalIndex] = i + 1

    return acq_order_list


def write_tlt(
    cets_ts_md: TiltSeries, tlt_file: Path | str | None, add_dose_to_tlt: bool = False
) -> None:
    if tlt_file is None:
        print("write_tlt -> tlt_file is None. Skipping...")
        return
    try:
        tlt_file = validate_new_file(tlt_file)
        tilt_angles: list[float]
        dose_list: list[float] = []
        # Read the required data
        tilt_angles = [ti.nominal_tilt_angle for ti in cets_ts_md.images]
        if add_dose_to_tlt:
            dose_list = [ti.accumulated_dose for ti in cets_ts_md.images]
            # Check the dose_list as it can be None if not provided by the cets_ts_md
            if all(dose in [None, "None"] for dose in dose_list):
                print(
                    f"Dose was requested to be added to the tlt file "
                    f"\n{tlt_file}, "
                    f"but it will not be added because it is None in all "
                    f"the tilt images contained in the current tilt-series "
                    f"\n{cets_ts_md.path}"
                )
                dose_list = []

        # Write the file
        with open(tlt_file, "w") as f:
            if dose_list:
                f.writelines(
                    f"{angle:0.3f} {dose:0.4f}\n"
                    for angle, dose in zip(tilt_angles, dose_list)
                )
            else:
                f.writelines(f"{angle:0.3f}\n" for angle in tilt_angles)
        print(f"tlt file successfully written! -> {tlt_file}")
    except Exception as e:
        print(
            f"Unable to write the output tlt file {tlt_file} with the exception -> {e}"
        )
        print(traceback.format_exc())


def write_xf(cets_ts_md: TiltSeries, xf_file: Path | str | None) -> None:
    if xf_file is None:
        print("write_xf -> xf_file is None. Skipping...")
        return
    try:
        xf_file = validate_new_file(xf_file)
        # Read the required data
        pixel_size = cets_ts_md.images[0].pixel_size
        transform_list = []
        for ti in cets_ts_md.images:
            transform = ti.coordinate_transformations[0].affine
            matrix_elements = np.array(transform).flatten()
            # The shifts are stored in angstroms in CETS, but in pixels in IMOD
            sx = matrix_elements[2] / pixel_size
            sy = matrix_elements[5] / pixel_size
            transform_list.append(
                [
                    f"{matrix_elements[0]:.7f}",
                    f"{matrix_elements[1]:.7f}",
                    f"{matrix_elements[3]:.7f}",
                    f"{matrix_elements[4]:.7f}",
                    f"{float(f'{sx:.3g}'):>6}",
                    f"{float(f'{sy:.3g}'):>6}",
                ]
            )
        # write the xf_file
        with open(xf_file, "w") as f:
            for row in transform_list:
                f.write("\t".join(str(item) for item in row) + "\n")
        print(f"xf file successfully written! -> {xf_file}")
    except Exception as e:
        print(f"Unable to write the output xf file {xf_file} with the exception -> {e}")
        print(traceback.format_exc())


def validate_new_file(in_file: Path | str) -> Path:
    in_file = Path(in_file).expanduser()
    if in_file.exists():
        in_file.unlink()  # Remove the file
    return in_file


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


# YAML STUFF ############################################################
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


def load_md_list_yaml(yaml_file: Path | str, model_cls: Type[BaseModel]) -> List[Dict]:
    """Loads a .yaml file containing a list of"""
    yaml_file = validate_file(yaml_file, "yaml_file", ".yaml")
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


# def load_md_list_yaml(yaml_file: Path | str, model_cls: Type[BaseModel]) -> List[Dict]:
#     """
#     Loads a .yaml file which may contain a single document or MULTIPLE documents
#     (separated by '---'), and validates the 'images' list within each against
#     the Pydantic model.
#     """
#     yaml_file = validate_file(yaml_file, "yaml_file", ".yaml")
#
#     all_metadata_list = []
#
#     with open(yaml_file, "r") as f:
#         # Usa yaml.load_all() para iterar sobre todos los documentos en el archivo
#         yaml_documents = yaml.load_all(f, Loader=yaml.FullLoader)
#
#         for data in yaml_documents:
#             # Manejar el caso de un documento vacío (que load_all podría devolver como None)
#             if data is None:
#                 continue
#
#             # Asumimos que cada documento tiene la estructura esperada: un dict con clave 'images'
#             if not isinstance(data, dict) or 'images' not in data or not isinstance(data['images'], list):
#                 print(f"Warning: Skipping a document with unexpected structure: {data}")
#                 continue
#
#             # Iterar sobre la lista de entradas de 'images' de este documento
#             for entry_data in data['images']:
#                 try:
#                     # Validar y estructurar con Pydantic
#                     validated_model = model_cls.model_validate(entry_data)
#
#                     # Añadir a la lista total
#                     all_metadata_list.append(validated_model.model_dump())
#
#                 except ValidationError as e:
#                     print(f"Validation failed for an entry in the YAML file:\n{entry_data}\nError: {e}")
#                     # Manejo de error: podrías continuar o lanzar la excepción
#
#     return all_metadata_list

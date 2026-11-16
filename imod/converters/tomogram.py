import traceback
from pathlib import Path

import mrcfile
import yaml

from cets_data_model.models.models import Tomogram
from cets_data_model.utils.image_utils import get_mrc_info
from imod.utils.utils import (
    export_mrc,
    get_image_pixel_size,
    validate_even_odd_files,
    validate_new_file,
    validate_file,
    gen_coordinate_systems,
    gen_array_to_physical,
)


class ImodTomogram:
    def __init__(self, tomo_file: Path | str) -> None:
        self.file_name = validate_file(tomo_file, "tomo_file", ".mrc")

    def imod_to_cets(
        self,
        even_file_name: Path | str | None = None,
        odd_file_name: Path | str | None = None,
        ctf_corrected: bool = False,
        out_yaml_file: str | Path | None = None,
    ) -> Tomogram:
        """Converts an IMOD tomogran into CETS metadata.

        :param even_file_name: path of the even tomogram,
        :type even_file_name: pathlib.Path, optional

        :param odd_file_name: path of the even tomogram,
        :type odd_file_name: pathlib.Path, optional

        :param ctf_corrected: xFlag to indicate if the tomogram was reconstructed
        from a tilt-series with the ctf corrected.
        :type ctf_corrected: bool, optional

        :param out_yaml_file: name of the yaml file in which the tomogram
        metadata will be written.
        :type out_yaml_file: pathlib.Path or str, optional
        """
        # Validate even/odd
        even_file_name, odd_file_name = validate_even_odd_files(
            even_file_name, odd_file_name
        )
        # Read image info
        tomo_filename = str(self.file_name)
        image_info_obj = get_mrc_info(tomo_filename)
        tomo_id = self.file_name.stem
        # Every image (volume) gets an array (voxel, unitless) and a physical (Å) coordinate
        # system plus exactly one canonical array_to_physical scale (the voxel size), per spec.
        voxel_size = image_info_obj.apix_x if image_info_obj.apix_x else 1.0
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        array_to_physical = gen_array_to_physical(
            voxel_size, array_cs.name, physical_cs.name, ndim=3
        )
        tomo = Tomogram(
            id=tomo_id,
            path=tomo_filename,
            tilt_series_id=tomo_id,
            width=image_info_obj.size_x,
            height=image_info_obj.size_y,
            depth=image_info_obj.size_z,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[array_to_physical],
            ctf_corrected=ctf_corrected,
            even_path=str(even_file_name) if even_file_name else None,
            odd_path=str(odd_file_name) if odd_file_name else None,
        )
        # Write the output yaml file if requested
        self._write_ts_yaml(tomo, out_yaml_file)
        return tomo

    @staticmethod
    def cets_to_imod(
        cets_tomo: Tomogram | Path | str,
        output_directory: Path | str,
        copy: bool = False,
    ) -> Path:
        """Makes a CETS tomogram available as IMOD files: ``<id>.mrc`` and, if the
        tomogram has them, its even/odd halves as ``<id>_even.mrc`` and ``<id>_odd.mrc``.

        An IMOD tomogram is an mrc file, the one the CETS tomogram points to, so nothing
        is converted: each file is linked, or copied if requested or if the voxel size of
        its header is not the one of the CETS tomogram (the header of the copy is then
        updated). The rest of the CETS metadata (ctf_corrected, tilt_series_id) has no place
        in the file.

        :param cets_tomo: CETS tomogram metadata or a yaml file written from it.
        :type cets_tomo: Tomogram or pathlib.Path or str

        :param output_directory: directory in which the files will be written.
        :type output_directory: pathlib.Path or str

        :param copy: copy the files instead of linking them.
        :type copy: bool, optional, Defaults to False

        :returns: the path of the tomogram.
        """
        if not isinstance(cets_tomo, Tomogram):
            with open(cets_tomo) as f:
                cets_tomo = Tomogram(**next(yaml.safe_load_all(f)))
        if not cets_tomo.path:
            raise ValueError(f"The tomogram [{cets_tomo.id}] has no file.")
        out_dir = Path(output_directory)
        voxel_size = get_image_pixel_size(cets_tomo)
        with mrcfile.open(cets_tomo.path, header_only=True, permissive=True) as mrc:
            header_voxel_size = float(mrc.voxel_size.x)
        if not header_voxel_size and voxel_size == 1.0:
            # imod_to_cets sets the voxel size to 1.0 when the header does not have it:
            # it is a placeholder, not a value to be written in the header.
            voxel_size = None
        tomo_file = export_mrc(
            cets_tomo.path, out_dir / f"{cets_tomo.id}.mrc", voxel_size, copy
        )
        for half_path, suffix in (
            (cets_tomo.even_path, "even"),
            (cets_tomo.odd_path, "odd"),
        ):
            # Yaml files written before the halves were left empty hold the string "None"
            if half_path and half_path != "None":
                export_mrc(
                    half_path,
                    out_dir / f"{cets_tomo.id}_{suffix}.mrc",
                    voxel_size,
                    copy,
                )
        print(f"tomogram successfully written! -> {tomo_file}")
        return tomo_file

    @staticmethod
    def _write_ts_yaml(cets_ts_md: Tomogram, yaml_file: Path | str | None) -> None:
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


# # READER EXAMPLE
# import yaml
#
# f_path = Path(
#     "/home/jjimenez/ExtraDataSets"
# )
# tomo_file = f_path.joinpath("TS_03.mrc")
# it = ImodTomogram(tomo_file_name=tomo_file)
# tomo_metadata = it.imod_to_cets()
#
# output_file = "/home/jjimenez/CZII/cets_scratch_dir/TS_03_cets_tomogram.yaml"
# metadata_dict = tomo_metadata.model_dump()
# with open(output_file, "w") as file:
#     yaml.dump(metadata_dict, file, sort_keys=False, explicit_start=True)
#
# print(yaml.dump(metadata_dict, sort_keys=False, explicit_start=True))

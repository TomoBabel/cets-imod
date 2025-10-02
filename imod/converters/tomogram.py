import traceback
from pathlib import Path

import yaml

from cets_data_model.models.models import Tomogram
from cets_data_model.utils.image_utils import get_mrc_info
from imod.utils.utils import validate_even_odd_files, validate_new_file, validate_file


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
        tomo = Tomogram(
            path=tomo_filename,
            tomo_id=self.file_name.stem,
            width=image_info_obj.size_x,
            height=image_info_obj.size_y,
            depth=image_info_obj.size_z,
            voxel_size=image_info_obj.apix_x,
            ctf_corrected=ctf_corrected,
            even_path=str(even_file_name),
            odd_path=str(odd_file_name),
        )
        # Write the output yaml file if requested
        self._write_ts_yaml(tomo, out_yaml_file)
        return tomo

    # def cets_to_imod(self):
    #     pass

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

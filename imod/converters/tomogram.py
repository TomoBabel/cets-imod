from pathlib import Path

from cets_data_model.models.models import Tomogram
from cets_data_model.utils.image_utils import get_mrc_info
from imod.utils.utils import validate_even_odd_files


class ImodTomogram:
    def __init__(self, tomo_file_name: Path | str) -> None:
        self.file_name = tomo_file_name

    def imod_to_cets(
        self,
        even_file_name: Path | str | None = None,
        odd_file_name: Path | str | None = None,
        ctf_corrected: bool = False,
    ) -> Tomogram:
        """Converts an IMOD tomogran into CETS metadata.

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
        tomo_filename = str(self.file_name)
        image_info_obj = get_mrc_info(tomo_filename)
        return Tomogram(
            path=tomo_filename,
            width=image_info_obj.size_x,
            height=image_info_obj.size_y,
            depth=image_info_obj.size_z,
            voxel_size=image_info_obj.apix_x,
            ctf_corrected=ctf_corrected,
            even_path=str(even_file_name),
            odd_path=str(odd_file_name),
        )

    # def cets_to_imod(self):
    #     pass


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

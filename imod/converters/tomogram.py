from pathlib import Path
from typing import Optional
from datamodels.models.tomogram_model import Tomogram
from datamodels.utils.em_file_readers import read_mrc


class ImodTomogram:
    def __init__(self, tomo_file_name: Path) -> None:
        self.file_name = tomo_file_name

    def imod_to_cets(
        self,
        even_file_name: Optional[Path] = None,
        odd_file_name: Optional[Path] = None,
        ctf_corrected: Optional[bool] = False,
    ) -> Tomogram:
        """Converts an IMOD tomogran into CETS metadata.

        :param even_file_name: path of the even tomogram,
        :type even_file_name: pathlib.Path, optional

        :param odd_file_name: path of the even tomogram,
        :type even_file_name: pathlib.Path, optional

        :param ctf_corrected: xFlag to indicate if the tomogram was reconstructed
        from a tilt-series with the ctf corrected.
        :type ctf_corrected: bool, optional
        """
        nx, ny, nz, apix = read_mrc(self.file_name)
        return Tomogram(
            path=str(self.file_name),
            width=nx,
            height=ny,
            depth=nz,
            voxel_size=apix,
            ctf_corrected=ctf_corrected,
            even_path=even_file_name,
            odd_path=odd_file_name,
        )

    def cets_to_imod(self):
        pass


# import yaml
#
# f_path = Path('/home/jjimenez/ScipionUserData/projects/TestImodTomoReconstruction/Runs/000441_ProtImodTomoReconstruction/extra/TS_03/')
# tomo_file = f_path.joinpath('TS_03.mrc')
# it = ImodTomogram(tomo_file_name=tomo_file)
# tomo_metadata = it.imod_to_cets()
#
# metadata_dict = tomo_metadata.model_dump()
# yaml_output = yaml.dump(metadata_dict, sort_keys=False)
# print(yaml_output)

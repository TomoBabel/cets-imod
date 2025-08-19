from pathlib import Path
from typing import Optional


class ImodTiltSeries:

    def __init__(self, file_name: Path) -> None:
        self.file_name = file_name

    def imod_to_cets(self,
                     tlt_file: Path,
                     xf_file: Optional[Path] = None):
        """Converts an IMOD tilt-series into CETS metadata.

        :param tlt_file: tlt file containing the angles in a column. It may also
        contain the dose in a second column.
        :type tlt_file: pathlib.Path

        :param xf_file: xf alignment file. If not provided, the Identity matrix
        will be used as alignment data.
        :type xf_file: pathlib.Path, optional
        """
        pass

    def cets_to_imod(self,
                     tlt_file: Path,
                     add_dose_to_tlt: Optional[bool] = True,
                     xf_file: Optional[Path] = None):
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

from enum import Enum
from pathlib import Path

CETS_IMOD = "cets_imod"
CETS_IMOD_CTF = f"{CETS_IMOD}_ctf"
_TEST_DATA_DIR = Path(__file__).parent / "test_data" / "imod_defocus_files"


class ImodCtfTestData(str, Enum):  # Value is usable as str
    file_plain_estimation = "TS_03_plain_estimation.defocus"
    file_only_astigmatism = "TS_03_only_astigmatism.defocus"
    file_only_phase_shift = "TS_03_only_phase_shift.defocus"
    file_astig_and_phase_shift = "TS_03_astigmatism_and_phase_shift.defocus"
    file_astig_phase_shift_and_cutoff_freq = (
        "TS_03_astigmatism_phase_shift_and_cutoff_freq.defocus"
    )

    @property
    def path(self) -> Path:
        # Usage:
        # ImodCtfTestData.file_only_phase_shift.path
        p = _TEST_DATA_DIR / self
        if not p.is_file():
            available = ", ".join(sorted(f.name for f in _TEST_DATA_DIR.glob("*")))
            raise FileNotFoundError(
                f"Test file [{p}] does not exist. Available: {available}"
            )
        return p

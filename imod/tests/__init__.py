from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
from typing import Final, List

# Test root
TEST_DATA_ROOT: Final[Path] = Path(
    os.getenv("TEST_DATA_ROOT", Path(__file__).parent / "test_data")
)
# Subfolders under TEST_DATA_ROOT
CTF_DATA_DIR: Final[Path] = Path("imod_defocus_files")
TS_DATA_DIR: Final[Path] = Path("tilt_series")
ALI_DATA_DIR: Final[Path] = Path("alignment_files")
TOMOGRAMS_DIR: Final[Path] = Path("tomograms")

# Dataset-wide metadata
N_IMGS_TS03: Final[int] = 40
DEFOCUS_U: Final[str] = "defocus_u"
DEFOCUS_COMMON_FIELDS: Final[tuple[str, ...]] = (
    "index_u",
    "index_v",
    "angle_u",
    "angle_v",
    DEFOCUS_U,
)
DEFOCUS_V: Final[str] = "defocus_v"
DEFOCUS_ANGLE: Final[str] = "defocus_angle"
PHASE_SHIFT: Final[str] = "phase_shift"
CUTOFF_FREQ: Final[str] = "cutoff_freq"


# Immutable metadata per fixture
@dataclass(frozen=True)
class Fixture:
    relpath: Path
    imod_flag: int = -1
    column_names: tuple[str, ...] = ()
    description: str = ""
    # Shared dataset attributes:
    n_imgs: int = N_IMGS_TS03


class ImodTestDataFiles(Enum):
    defocus_plain_estimation = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_plain_estimation.defocus",
        imod_flag=0,
        column_names=DEFOCUS_COMMON_FIELDS,
        description="CTF - plain estimation",
    )
    defocus_only_astigmatism = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_only_astigmatism.defocus",
        imod_flag=1,
        column_names=DEFOCUS_COMMON_FIELDS + (DEFOCUS_V, DEFOCUS_ANGLE),
        description="CTF - only astigmatism",
    )
    defocus_only_phase_shift = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_only_phase_shift.defocus",
        imod_flag=4,
        column_names=DEFOCUS_COMMON_FIELDS + (PHASE_SHIFT,),
        description="CTF - only phase shift",
    )
    defocus_astig_and_phase_shift = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_astigmatism_and_phase_shift.defocus",
        imod_flag=5,
        column_names=DEFOCUS_COMMON_FIELDS + (DEFOCUS_V, DEFOCUS_ANGLE, PHASE_SHIFT),
        description="CTF - astigmatism and phase shift",
    )
    defocus_astig_phase_shift_and_cutoff_freq = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_astigmatism_phase_shift_and_cutoff_freq.defocus",
        imod_flag=37,
        column_names=DEFOCUS_COMMON_FIELDS
        + (DEFOCUS_V, DEFOCUS_ANGLE, PHASE_SHIFT, CUTOFF_FREQ),
        description="CTF - astigmatism, phase shift and cutoff frequency",
    )
    ts_03_mrcs = Fixture(relpath=TS_DATA_DIR / "TS_03.mrcs")
    ts_03_tlt = Fixture(relpath=TS_DATA_DIR / "TS_03.tlt")
    ts_03_xf = Fixture(relpath=ALI_DATA_DIR / "TS_03.xf")
    ts_03_tomogram = Fixture(relpath=TOMOGRAMS_DIR / "TS_03.mrc")

    @property
    def relpath(self) -> Path:
        """Relative path (from TEST_DATA_ROOT)."""
        return self.value.relpath

    @property
    def path(self) -> Path:
        """Absolute, validated path on disk."""
        p = (TEST_DATA_ROOT / self.value.relpath).resolve()
        if not p.is_file():
            # Show siblings in the expected folder (concise and actionable)
            folder = p.parent
            siblings = (
                "\n  - "
                + "\n  - ".join(sorted(f.name for f in folder.glob("*") if f.is_file()))
                if folder.exists()
                else " (folder missing)"
            )
            raise FileNotFoundError(
                f"Test file not found: {p}\nSiblings in {folder}:{siblings}"
            )
        return p

    @property
    def imod_flag(self) -> int:
        return self.value.imod_flag

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.value.column_names

    @property
    def n_imgs(self) -> int:
        return self.value.n_imgs

    @property
    def description(self) -> str:
        return self.value.description

    # def open(self, mode: str = "rb"):
    #     return self.path.open(mode)

    def get_test_dict_list(self) -> List[dict]:
        with open(self.path, "r") as f:
            lines = f.readlines()

        # Convert lines to list of lists of values
        data = []
        for line in lines:
            parts = line.strip().split()
            row = []
            for part in parts:
                value = float(part)
                if value.is_integer():
                    value = int(value)
                row.append(value)
            data.append(row)

        # Adjust based on imod_flag
        keys = self.column_names
        if self.imod_flag == 0:
            if len(data) > 0 and len(data[0]) > len(keys):
                data[0] = data[0][: len(keys)]
        elif self.imod_flag > 0:
            data = data[1:]

        # Build the list of dictionaries
        dict_list = []
        for row in data:
            row_dict = dict(zip(keys, row))
            dict_list.append(row_dict)

        return dict_list


# # Access file + metadata
# F = ImodCtfTestDataFiles
# print("Defocus path:", F.only_phase_shift.path)
# print("Columns:", F.only_phase_shift.column_names)
# print("IMOD flag:", F.only_phase_shift.imod_flag)
#
# # Read first 2 lines of the defocus file
# with F.only_phase_shift.open("rt") as f:
#     for _ in range(2):
#         print(">", f.readline().rstrip())
#
# # Tilt-series (shared)
# print("Tilt-series:", F.only_phase_shift.ts_path)
# print("Num images:", F.only_phase_shift.n_imgs)

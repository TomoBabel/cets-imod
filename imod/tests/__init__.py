from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
from typing import Final

# Test root
TEST_DATA_ROOT: Final[Path] = Path(
    os.getenv("TEST_DATA_ROOT", Path(__file__).parent / "test_data")
)
# Subfolders under TEST_DATA_ROOT
CTF_DATA_DIR: Final[Path] = Path("imod_defocus_files")
TS_DATA_DIR: Final[Path] = Path("tilt_series")
ALI_DATA_DIR: Final[Path] = Path("alignment_files")

# Dataset-wide metadata
N_IMGS_TS03: Final[int] = 40
DEFOCUS_COMMON_FIELDS: Final[tuple[str, ...]] = (
    "index_u",
    "index_v",
    "angle_u",
    "angle_v",
    "defocus_u",
)
DEFOCUS_V: Final[str] = "defocus_v"
DEFOCUS_ANGLE: Final[str] = "defocus_angle"
PHASE_SHIFT: Final[str] = "phase_shift"
CUTOFF_FREQ: Final[str] = "cutoff_freq"


# Immutable metadata per fixture
@dataclass(frozen=True)
class Fixture:
    relpath: Path
    imod_flag: int | None = None
    column_names: tuple[str, ...] = ()
    # Shared dataset attributes:
    n_imgs: int = N_IMGS_TS03


class ImodTestDataFiles(Enum):
    defocus_plain_estimation = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_plain_estimation.defocus",
        imod_flag=0,
        column_names=DEFOCUS_COMMON_FIELDS,
    )
    defocus_only_astigmatism = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_only_astigmatism.defocus",
        imod_flag=1,
        column_names=DEFOCUS_COMMON_FIELDS + (DEFOCUS_V, DEFOCUS_ANGLE),
    )
    defocus_only_phase_shift = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_only_phase_shift.defocus",
        imod_flag=4,
        column_names=DEFOCUS_COMMON_FIELDS + (PHASE_SHIFT,),
    )
    defocus_astig_and_phase_shift = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_astigmatism_and_phase_shift.defocus",
        imod_flag=5,
        column_names=DEFOCUS_COMMON_FIELDS + (DEFOCUS_V, DEFOCUS_ANGLE, PHASE_SHIFT),
    )
    defocus_astig_phase_shift_and_cutoff_freq = Fixture(
        relpath=CTF_DATA_DIR / "TS_03_astigmatism_phase_shift_and_cutoff_freq.defocus",
        imod_flag=37,
        column_names=DEFOCUS_COMMON_FIELDS
        + (DEFOCUS_V, DEFOCUS_ANGLE, PHASE_SHIFT, CUTOFF_FREQ),
    )
    ts_03_mrcs = Fixture(relpath=TS_DATA_DIR / "TS_03.mrcs")
    ts_03_tlt = Fixture(relpath=TS_DATA_DIR / "TS_03.tlt")
    ts_03_xf = Fixture(relpath=ALI_DATA_DIR / "TS_03.xf")

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
    def imod_flag(self) -> int | None:
        return self.value.imod_flag

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.value.column_names

    @property
    def n_imgs(self) -> int:
        return self.value.n_imgs

    def open(self, mode: str = "rb"):
        return self.path.open(mode)


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

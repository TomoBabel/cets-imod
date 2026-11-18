"""Command line of the IMOD <-> CETS converters (``cets-imod``).

Examples::

    cets-imod tilt-series to-cets --stack TS_03.mrc --tilt-angles TS_03.rawtlt \\
        -o TS_03_cets.yaml
    cets-imod tilt-series to-cets --stack TS_03.mrc --tilt-angles TS_03.rawtlt \\
        --refined-tilt-angles TS_03.tlt --xf TS_03.xf --defocus TS_03.defocus \\
        --voltage 300 --tilt-axis-angle 85.3 -o TS_03_cets.yaml
    cets-imod tilt-series from-cets -i TS_03_cets.yaml --tlt out/TS_03.tlt --xf out/TS_03.xf
    cets-imod ctf to-cets --stack TS_03.mrc --defocus TS_03.defocus -o TS_03_ctf.yaml
    cets-imod ctf from-cets -i TS_03_cets.yaml -o out/TS_03.defocus
    cets-imod tomogram to-cets --tomogram TS_03_rec.mrc -o TS_03_tomo.yaml
    cets-imod tomogram from-cets -i TS_03_tomo.yaml --output-dir out

``to-cets`` writes a CETS yaml file with one entry per kind of object (``tilt_series``,
``alignment``, ``instrument``, ``acquisition_session``, ``ctf``, ``tomogram``), which is what
``from-cets`` reads. ``instrument`` and ``acquisition_session`` are only written when the
acquisition values they hold are given (IMOD files do not record them).
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import (
    Alignment,
    CTFMetadata,
    TiltSeries,
    Tomogram,
)
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    check_outputs,
    prepare_outputs,
    run_cli,
)
from cets_data_model.utils.yaml_io import (
    has_cets_key,
    load_cets_object,
    load_cets_objects,
    write_cets_yaml,
)
from imod.contants import (
    ACQUISITION_SESSION_KEY,
    ALIGNMENT_KEY,
    CTF_KEY,
    INSTRUMENT_KEY,
    TILT_SERIES_KEY,
    TOMOGRAM_KEY,
)
from imod.converters.ctf import ImodCtfSeries
from imod.converters.tilt_series import ImodTiltSeries
from imod.converters.tomogram import ImodTomogram
from imod.utils.utils import parse_tlt_file


# ------------------------------------------------------------------------- tilt-series
def tilt_series_to_cets(args: argparse.Namespace) -> None:
    ctf_md_list = (
        ImodCtfSeries(args.stack, args.defocus).imod_to_cets() if args.defocus else None
    )
    converter = ImodTiltSeries(
        args.stack,
        args.tilt_angles,
        ctf_md_list=ctf_md_list,
        voltage=args.voltage,
        spherical_aberration=args.spherical_aberration,
        amplitude_contrast=args.amplitude_contrast,
        dose_rate=args.dose_rate,
        tilt_axis_angle=args.tilt_axis_angle,
        refined_tilt_angles=args.refined_tilt_angles,
    )
    ts, alignment, instrument, session = converter.imod_to_cets(
        xf_file=args.xf,
        even_stack_file_name=args.even,
        odd_stack_file_name=args.odd,
        ctf_corrected=args.ctf_corrected,
        pixel_size=args.pixel_size,
    )
    out = write_cets_yaml(
        args.output,
        {
            TILT_SERIES_KEY: ts,
            ALIGNMENT_KEY: alignment,
            INSTRUMENT_KEY: instrument,
            ACQUISITION_SESSION_KEY: session,
        },
    )
    print(f"CETS tilt-series written -> {out}")


def tilt_series_from_cets(args: argparse.Namespace) -> None:
    ts = load_cets_object(args.input, TILT_SERIES_KEY, TiltSeries)
    alignment = (
        load_cets_object(args.input, ALIGNMENT_KEY, Alignment)
        if has_cets_key(args.input, ALIGNMENT_KEY)
        else None
    )
    if args.xf is not None and alignment is None:
        raise ValueError(f"{args.input} has no alignment to write the xf file.")
    outputs = prepare_outputs(args.tlt, args.xf, args.rawtlt)
    ImodTiltSeries.cets_to_imod(
        ts,
        args.tlt,
        alignment=alignment,
        add_dose_to_tlt=not args.no_dose,
        xf_file=args.xf,
        rawtlt_file=args.rawtlt,
    )
    check_outputs(outputs)


def _add_tilt_series(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types, "tilt-series", "Tilt-series: stack, tilt angles and alignment."
    )

    to_cets = directions.add_parser(
        TO_CETS, help="IMOD tilt-series -> CETS yaml (tilt-series, alignment, ...)."
    )
    to_cets.add_argument(
        "--stack", required=True, type=Path, help="Stack (.mrc/.mrcs)."
    )
    to_cets.add_argument(
        "--tilt-angles",
        required=True,
        type=Path,
        help="Nominal tilt angles: .rawtlt (or a .tlt holding them), optionally with the "
        "dose and acquisition order columns.",
    )
    to_cets.add_argument(
        "--refined-tilt-angles",
        type=Path,
        help="Tilt angles refined by the alignment (.tlt written by tiltalign).",
    )
    to_cets.add_argument(
        "--xf", type=Path, help="Alignment (.xf). Without it, the identity is used."
    )
    to_cets.add_argument(
        "--defocus", type=Path, help="CTF of the tilt-images (.defocus)."
    )
    to_cets.add_argument("--even", type=Path, help="Even stack.")
    to_cets.add_argument("--odd", type=Path, help="Odd stack.")
    to_cets.add_argument(
        "--ctf-corrected",
        action="store_true",
        help="The stack is corrected for the CTF.",
    )
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Pixel size of the stack (Å). Default: the one of its header.",
    )
    # IMOD files do not record the acquisition: these values are optional, and only the
    # objects they describe are written (instrument: --voltage; acquisition session: any
    # of them).
    acquisition = to_cets.add_argument_group(
        "acquisition (optional)",
        "IMOD files do not record them. Without --voltage the yaml has no instrument; "
        "without any of them, no acquisition session either.",
    )
    acquisition.add_argument("--voltage", type=float, help="Voltage (kV).")
    acquisition.add_argument(
        "--spherical-aberration", type=float, help="Spherical aberration (mm)."
    )
    acquisition.add_argument(
        "--amplitude-contrast", type=float, help="Amplitude contrast (fraction)."
    )
    acquisition.add_argument("--dose-rate", type=float, help="Dose rate (e-/Å²/s).")
    acquisition.add_argument(
        "--tilt-axis-angle",
        type=float,
        help="Tilt axis angle (degrees, from the detector x-axis).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=tilt_series_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS,
        help="CETS yaml -> IMOD tilt angles (.tlt/.rawtlt) and alignment (.xf).",
    )
    add_input_yaml(
        from_cets, "CETS yaml with the tilt-series (and its alignment, for --xf)."
    )
    from_cets.add_argument(
        "--tlt",
        required=True,
        type=Path,
        help="Tilt angles to write: the refined ones if the alignment has them, else "
        "the nominal ones.",
    )
    from_cets.add_argument("--rawtlt", type=Path, help="Nominal tilt angles to write.")
    from_cets.add_argument("--xf", type=Path, help="Alignment to write (.xf).")
    from_cets.add_argument(
        "--no-dose",
        action="store_true",
        help="Do not write the dose and acquisition order columns in the tilt angles.",
    )
    from_cets.set_defaults(func=tilt_series_from_cets)


# ------------------------------------------------------------------------- ctf
def ctf_to_cets(args: argparse.Namespace) -> None:
    ctf_list = ImodCtfSeries(args.stack, args.defocus).imod_to_cets()
    out = write_cets_yaml(args.output, {CTF_KEY: ctf_list})
    print(f"CETS CTF written -> {out}")


def ctf_from_cets(args: argparse.Namespace) -> None:
    ts: Optional[TiltSeries] = (
        load_cets_object(args.input, TILT_SERIES_KEY, TiltSeries)
        if has_cets_key(args.input, TILT_SERIES_KEY)
        else None
    )
    if ts is not None:
        images = ts.images or []
        if any(ti.ctf_metadata is None for ti in images):
            raise ValueError(f"Some tilt-images in {args.input} have no CTF.")
        ctf_list: List[CTFMetadata] = [
            ti.ctf_metadata for ti in images if ti.ctf_metadata is not None
        ]
    else:
        ctf_list = load_cets_objects(args.input, CTF_KEY, CTFMetadata)

    if args.tilt_angles is not None:
        tilt_angles, _, _ = parse_tlt_file(args.tilt_angles)
    elif ts is not None:
        nominal = [ti.nominal_tilt_angle for ti in ts.images or []]
        if any(angle is None for angle in nominal):
            raise ValueError(
                f"Some tilt-images in {args.input} have no nominal tilt angle: "
                f"provide --tilt-angles."
            )
        tilt_angles = [float(angle) for angle in nominal if angle is not None]
    else:
        raise ValueError(
            f"{args.input} has no tilt-series to take the tilt angles from: provide "
            f"--tilt-angles."
        )
    outputs = prepare_outputs(args.output)
    ImodCtfSeries.cets_to_imod(ctf_list, tilt_angles, args.output)
    check_outputs(outputs)


def _add_ctf(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(data_types, "ctf", "CTF of the tilt-images (.defocus).")

    to_cets = directions.add_parser(TO_CETS, help="IMOD .defocus -> CETS yaml.")
    to_cets.add_argument(
        "--stack", required=True, type=Path, help="Stack the CTF was estimated on."
    )
    to_cets.add_argument("--defocus", required=True, type=Path, help=".defocus file.")
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=ctf_to_cets)

    from_cets = directions.add_parser(FROM_CETS, help="CETS yaml -> IMOD .defocus.")
    add_input_yaml(
        from_cets,
        "CETS yaml with the CTF: a tilt-series with the CTF of its tilt-images, or the "
        "output of 'ctf to-cets'.",
    )
    from_cets.add_argument(
        "--tilt-angles",
        type=Path,
        help="Tilt angles (.tlt/.rawtlt). Default: the nominal tilt angles of the "
        "tilt-series in the input.",
    )
    from_cets.add_argument(
        "-o", "--output", required=True, type=Path, help=".defocus file to write."
    )
    from_cets.set_defaults(func=ctf_from_cets)


# ------------------------------------------------------------------------- tomogram
def tomogram_to_cets(args: argparse.Namespace) -> None:
    tomo = ImodTomogram(args.tomogram).imod_to_cets(
        even_file_name=args.even,
        odd_file_name=args.odd,
        ctf_corrected=args.ctf_corrected,
        pixel_size=args.pixel_size,
    )
    out = write_cets_yaml(args.output, {TOMOGRAM_KEY: tomo})
    print(f"CETS tomogram written -> {out}")


def tomogram_from_cets(args: argparse.Namespace) -> None:
    tomo = load_cets_object(args.input, TOMOGRAM_KEY, Tomogram)
    ImodTomogram.cets_to_imod(tomo, args.output_dir, copy=args.copy)


def _add_tomogram(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(data_types, "tomogram", "Tomogram (.mrc).")

    to_cets = directions.add_parser(TO_CETS, help="IMOD tomogram -> CETS yaml.")
    to_cets.add_argument(
        "--tomogram", required=True, type=Path, help="Tomogram (.mrc)."
    )
    to_cets.add_argument("--even", type=Path, help="Even tomogram.")
    to_cets.add_argument("--odd", type=Path, help="Odd tomogram.")
    to_cets.add_argument(
        "--ctf-corrected",
        action="store_true",
        help="Reconstructed from a CTF-corrected tilt-series.",
    )
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Voxel size of the tomogram (Å). Default: the one of its header.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=tomogram_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> IMOD tomogram files (<id>.mrc, even/odd)."
    )
    add_input_yaml(from_cets, "CETS yaml with the tomogram.")
    from_cets.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the tomogram files in.",
    )
    from_cets.add_argument(
        "--copy",
        action="store_true",
        help="Copy the files instead of linking them.",
    )
    from_cets.set_defaults(func=tomogram_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-imod",
        description="Converts IMOD files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{tilt-series,ctf,tomogram}", required=True
    )
    _add_tilt_series(data_types)
    _add_ctf(data_types)
    _add_tomogram(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())

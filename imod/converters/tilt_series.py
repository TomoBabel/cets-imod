import traceback
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import yaml
from cets_data_model.models.models import (
    Affine,
    Alignment,
    CTFMetadata,
    ProjectionAlignment,
    TiltSeries,
    TiltImage,
    CoordinateTransformation,
    CoordinateSystem,
    Axis,
    # SpaceAxis,
    # AxisUnit,
    AxisType,
    Translation,
    Vector3D,
    Matrix3x3,
    Instrument,
    AcquisitionSession,
)
from cets_data_model.utils.image_utils import get_mrc_info
from imod.contants import MRC_MRCS_EXT
from imod.utils.utils import (
    validate_file,
    validate_tilt_angle_list,
    parse_tlt_file,
    parse_xf_file,
    validate_ctf_md_list,
    validate_even_odd_files,
    write_tlt,
    write_xf,
    validate_new_file,
    load_md_list_yaml,
)


class ImodTiltSeries:
    def __init__(
        self,
        ts_file_name: str | Path,
        tilt_angles: str | Path | List[float],
        ctf_md_list: Optional[List[CTFMetadata]] = None,
        voltage: Optional[float] = None,
        spherical_aberration: Optional[float] = None,
        amplitude_contrast: Optional[float] = None,
        dose_rate: Optional[float] = None,
    ) -> None:
        """:param voltage: acceleration voltage in kV.
        :param spherical_aberration: spherical aberration (Cs) in mm.
        :param amplitude_contrast: amplitude contrast fraction (dimensionless).
        :param dose_rate: dose rate during acquisition in e-/A^2/s.

        The four acquisition constants above are session/microscope values that are
        not present in the IMOD tilt-series files (.tlt/.xf) parsed here, so they are
        accepted as optional inputs. ``imod_to_cets`` emits them as an ``Instrument``
        (voltage, spherical aberration) and an ``AcquisitionSession`` (amplitude contrast,
        dose rate) rather than storing them on the tilt-images.
        """
        self.ts_file_name = validate_file(ts_file_name, "ts_file_name", MRC_MRCS_EXT)
        if type(tilt_angles) is List[float]:
            tilt_angles = validate_tilt_angle_list(self.ts_file_name, tilt_angles)
            tlt_file, dose_list, acq_orders = None, None, None
        else:
            tlt_file = validate_file(
                str(tilt_angles), "tilt_angles", [".tlt", ".rawtlt"]
            )
            tilt_angles, dose_list, acq_orders = parse_tlt_file(tlt_file)
        n_imgs = len(tilt_angles)
        self.ctf_md_list = validate_ctf_md_list(ctf_md_list, n_imgs)
        self.tlt_file = tlt_file
        self.tilt_angles = tilt_angles
        self.dose_list = dose_list
        self.acq_orders = acq_orders
        self.n_imgs = n_imgs
        # Acquisition constants (constant across the tilt-series). Not available from
        # the .tlt/.xf files, so supplied by the caller; default to None otherwise.
        self.voltage = voltage
        self.spherical_aberration = spherical_aberration
        self.amplitude_contrast = amplitude_contrast
        self.dose_rate = dose_rate

    def imod_to_cets(
        self,
        xf_file: str | Path | None = None,
        even_stack_file_name: str | Path | None = None,
        odd_stack_file_name: str | Path | None = None,
        ctf_corrected: bool = False,
        out_yaml_file: str | Path | None = None,
    ) -> Tuple[TiltSeries, Alignment, Instrument, AcquisitionSession]:
        """Converts an IMOD tilt-series into CETS metadata.

        In the current data model the per-projection alignment is NOT stored inside each
        tilt-image's ``coordinate_transformations``. Instead it is represented with the
        dedicated ``ProjectionAlignment`` structure (one per tilt-image, holding the
        ``[Translation, Affine]`` pair in its ``sequence``), and all of them are aggregated
        into a single ``Alignment`` for the tilt-series. ``Alignment`` objects live under
        ``Region.alignments``; since this converter emits a bare ``TiltSeries`` (not a
        ``Region``), the ``Alignment`` is returned alongside it for higher-level assembly.
        The i-th ``ProjectionAlignment`` corresponds to the i-th ``TiltSeries.images`` entry
        (positional binding, matching the schema's ordered-list convention).

        Microscope/session acquisition metadata is no longer stored on the tilt-images: the
        constructor-supplied constants are emitted as an ``Instrument`` (voltage, spherical
        aberration) and an ``AcquisitionSession`` (amplitude contrast, dose rate; linked to the
        instrument via ``instrument_id``). The tilt-series references the session via
        ``acquisition_session_id``. Both are returned as the 3rd and 4th elements for
        higher-level assembly onto ``Dataset.instruments`` / ``Dataset.acquisition_sessions``.

        :param xf_file: xf alignment file. If not provided, the Identity matrix
        will be used as alignment data.
        :type xf_file: pathlib.Path or str, optional

        :param even_stack_file_name: path of the even tomogram,
        :type even_stack_file_name: pathlib.Path or str, optional

        :param odd_stack_file_name: path of the even tomogram,
        :type odd_stack_file_name: pathlib.Path or str, optional

        :param ctf_corrected: xFlag to indicate if the tomogram was reconstructed
        from a tilt-series with the ctf corrected.
        :type ctf_corrected: bool, optional

        :param out_yaml_file: name of the yaml file in which the tilt-series
        metadata will be written.
        :type out_yaml_file: pathlib.Path or str, optional
        """
        # Validate even/odd
        even_stack_file_name, odd_stack_file_name = validate_even_odd_files(
            even_stack_file_name, odd_stack_file_name
        )
        # Read image info
        img_info = get_mrc_info(self.ts_file_name)
        width = img_info.size_x
        height = img_info.size_y
        # pix_size = img_info.apix_x
        # Parse xf file
        xf_file = validate_file(xf_file, "xf_file", ".xf")
        in_rotation_matrix_pile, in_translation_vector_pile = parse_xf_file(xf_file)

        ts_filename = str(self.ts_file_name)
        ts_id = self.ts_file_name.stem
        pixel_size = img_info.apix_x
        # Dataset-level acquisition metadata (constant across the tilt-series): the
        # microscope hardware (Instrument) and the session parameters (AcquisitionSession),
        # linked to the instrument via instrument_id. IDs are derived from the tilt-series id
        # to stay unique when several tilt-series are assembled into a single Dataset.
        instrument = Instrument(
            id=f"{ts_id}_instrument",
            voltage=self.voltage,
            spherical_aberration=self.spherical_aberration,
        )
        acquisition_session = AcquisitionSession(
            id=f"{ts_id}_session",
            instrument_id=instrument.id,
            amplitude_contrast=self.amplitude_contrast,
            dose_rate=self.dose_rate,
        )
        axis_z = Axis(name="Z", axis_unit="angstrom", axis_type=AxisType.space)
        coordinate_systems = CoordinateSystem(name="IMOD", axes=[axis_z])
        ti_list = []
        projection_alignments = []
        for index in range(self.n_imgs):
            output_translation_transform = in_translation_vector_pile[:, index]
            output_rotation_matrix = in_rotation_matrix_pile[:, :, index]
            # Unique tilt-image id within the tilt-series (derived from the ts id + section).
            tilt_image_id = f"{ts_id}_{index}"
            ti = TiltImage(
                id=tilt_image_id,
                movie_stack_id=ts_id,  # TODO: define this
                path=ts_filename,
                section=index,
                nominal_tilt_angle=self.tilt_angles[index],
                accumulated_dose=self.dose_list[index] if self.dose_list else None,
                ctf_metadata=self.ctf_md_list[index] if self.ctf_md_list else None,
                # Acquisition constants no longer live on the tilt-image; they are emitted
                # as the Instrument / AcquisitionSession built above.
                width=width,
                height=height,
                coordinate_systems=[coordinate_systems],
                # Alignment is no longer stored here; it now lives in the ProjectionAlignment
                # aggregated in the Alignment returned with this tilt-series.
            )
            ti_list.append(ti)
            # One ProjectionAlignment per projection, linked to its tilt-image by tilt_image_id
            # (index-aligned with ti_list / images).
            projection_alignments.append(
                self._gen_projection_alignment(
                    output_translation_transform,
                    output_rotation_matrix,
                    pixel_size,
                    projection_alignment_id=f"{ts_id}_align_{index}",
                    tilt_image_id=tilt_image_id,
                )
            )
        ts = TiltSeries(
            id=ts_id,  # TODO: define this
            movie_stack_series_id=ts_id,  # TODO: define this
            path=ts_filename,
            even_path=even_stack_file_name,
            odd_path=odd_stack_file_name,
            ctf_corrected=ctf_corrected,
            images=ti_list,
            # Link the tilt-series to its acquisition session.
            acquisition_session_id=acquisition_session.id,
        )
        # The tilt-series alignment (one ProjectionAlignment per tilt-image). It is meant to be
        # placed under Region.alignments together with this tilt-series, and links the whole set
        # back to the tilt-series via tilt_series_id.
        alignment = Alignment(
            tilt_series_id=ts_id, projection_alignments=projection_alignments
        )
        # Write the output yaml files if requested (tilt-series + alignment + instrument + session)
        self._write_ts_yaml(ts, out_yaml_file)
        if out_yaml_file is not None:
            self._write_ts_yaml(alignment, self._alignment_yaml_path(out_yaml_file))
            self._write_ts_yaml(instrument, self._instrument_yaml_path(out_yaml_file))
            self._write_ts_yaml(
                acquisition_session, self._session_yaml_path(out_yaml_file)
            )
        return ts, alignment, instrument, acquisition_session

    @staticmethod
    def cets_to_imod(
        cets_ts: TiltSeries | Path | str,
        tlt_file: str | Path,
        alignment: Optional[Alignment] = None,
        add_dose_to_tlt: bool = True,
        xf_file: str | Path | None = None,
    ):
        """Converts CETS Tilt-series metadata into IMOD files.

        :param cets_ts: CETS tilt-series metadata or a yaml file written from it.
        :type cets_ts: TiltSeries or pathlib.Path or str

        :param tlt_file: output tlt file to be generated.
        :type tlt_file: pathlib.Path or str

        :param alignment: CETS Alignment holding one ProjectionAlignment per tilt-image
        (index-aligned with ``cets_ts.images``). Required to write the xf file, since the
        alignment is no longer stored inside the tilt-image ``coordinate_transformations``.
        :type alignment: Alignment, optional, Defaults to None

        :param add_dose_to_tlt: used to indicate if the generated tlt file should also
        contain a second column with the dose.
        :type add_dose_to_tlt: bool, optional, Defaults to True

        :param xf_file: output xf file to be generated.
        :type: pathlib.Path or str, optional, Defaults to None
        """
        if type(cets_ts) is not TiltSeries:
            cets_ts = load_md_list_yaml(cets_ts, TiltSeries)
        # Write the tlt file
        write_tlt(cets_ts, tlt_file, add_dose_to_tlt=add_dose_to_tlt)
        if xf_file is not None:
            if alignment is None:
                print(
                    "cets_to_imod -> an xf_file was requested but no alignment was "
                    "provided. Skipping the xf file."
                )
            else:
                # Write the xf file from the ProjectionAlignment structure
                write_xf(alignment, xf_file)

    def _gen_projection_alignment(
        self,
        translation_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        pix_size: float = 1.0,
        projection_alignment_id: str = "",
        tilt_image_id: str | None = None,
    ) -> ProjectionAlignment:
        """Builds the per-projection alignment as a ProjectionAlignment whose ``sequence``
        holds the translation and the affine rotation (order preserved from the previous
        coordinate_transformations layout: translation first, affine second).

        :param projection_alignment_id: unique id for this ProjectionAlignment.
        :param tilt_image_id: id of the TiltImage this alignment applies to.
        """
        return ProjectionAlignment(
            id=projection_alignment_id,
            tilt_image_id=tilt_image_id,
            sequence=[
                self._gen_translation_transform(translation_matrix, pix_size),
                self._gen_affine_transform(rotation_matrix),
            ],
            name="IMOD projection alignment from a .xf file.",
            input="Tilt-image",
            output="Aligned tilt-image",
        )

    @staticmethod
    def _alignment_yaml_path(ts_yaml_file: str | Path) -> Path:
        """Derives the sibling yaml path for the alignment from the tilt-series yaml path."""
        p = Path(ts_yaml_file)
        return p.with_name(f"{p.stem}_alignment{p.suffix}")

    @staticmethod
    def _instrument_yaml_path(ts_yaml_file: str | Path) -> Path:
        """Derives the sibling yaml path for the instrument from the tilt-series yaml path."""
        p = Path(ts_yaml_file)
        return p.with_name(f"{p.stem}_instrument{p.suffix}")

    @staticmethod
    def _session_yaml_path(ts_yaml_file: str | Path) -> Path:
        """Derives the sibling yaml path for the acquisition session from the tilt-series
        yaml path."""
        p = Path(ts_yaml_file)
        return p.with_name(f"{p.stem}_acquisition_session{p.suffix}")

    def _gen_affine_transform(
        self,
        rotation_matrix: np.ndarray,
    ) -> CoordinateTransformation:
        return Affine(
            affine=self._get_affine_values(rotation_matrix),
            name="IMOD rotation from a .xf file.",
            input="Tilt-image",
            output="Aligned tilt-image (rotation-corrected)",
        )

    def _gen_translation_transform(
        self, translation_matrix: np.ndarray, pix_size: float = 1.0
    ) -> Translation:
        return Translation(
            translation=self._get_translation_values(translation_matrix, pix_size),
            name="IMOD translation from a .xf file. Shifts in angstroms.",
            input="Tilt-image",
            output="Aligned tilt-image (translation-corrected)",
        )

    @staticmethod
    def _get_affine_values(xf_matrix: np.ndarray) -> Matrix3x3:
        """Gets the rotation angle in degrees."""
        xf_matrix = xf_matrix.tolist()
        row1: Vector3D = xf_matrix[0]
        row2: Vector3D = xf_matrix[1]
        row3: Vector3D = [0, 0, 1]
        affine_matrix: Matrix3x3 = [row1, row2, row3]
        return affine_matrix

    @staticmethod
    def _get_translation_values(
        translation_matrix: np.ndarray, pix_size: float = 1.0
    ) -> Vector3D:
        """Gets the shifts in X and Y directions, in angstroms."""
        translation_matrix *= pix_size  # convert to angstroms
        translation_vector: Vector3D = translation_matrix.tolist()
        return translation_vector

    @staticmethod
    def _write_ts_yaml(
        cets_ts_md: TiltSeries | Alignment | Instrument | AcquisitionSession,
        yaml_file: Path | str | None,
    ) -> None:
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

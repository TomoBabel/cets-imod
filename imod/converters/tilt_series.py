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
    Translation,
    Vector3D,
    Matrix3x3,
    Instrument,
    AcquisitionSession,
)
from cets_data_model.utils.image_utils import get_mrc_info
from cets_data_model.utils.yaml_io import load_cets_object
from imod.contants import ALIGNMENT_KEY, MRC_MRCS_EXT, TILT_SERIES_KEY
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
    gen_coordinate_systems,
    gen_array_to_physical,
    get_image_pixel_size,
    resolve_pixel_size,
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
        tilt_axis_angle: Optional[float] = None,
        refined_tilt_angles: str | Path | List[float] | None = None,
    ) -> None:
        """:param tilt_angles: nominal tilt angles (the ones reported by the microscope),
        as a list or as an IMOD tilt angles file (.rawtlt, or a .tlt holding them).
        :param refined_tilt_angles: tilt angles refined by the alignment (the .tlt written by
        tiltalign), as a list or as a .tlt file. They are stored in the
        ``ProjectionAlignment``s, as they are a result of the alignment.
        :param voltage: acceleration voltage in kV.
        :param spherical_aberration: spherical aberration (Cs) in mm.
        :param amplitude_contrast: amplitude contrast fraction (dimensionless).
        :param dose_rate: dose rate during acquisition in e-/A^2/s.
        :param tilt_axis_angle: angle of the tilt axis relative to the detector x-axis,
        in degrees (IMOD's RotationAngle / the TiltAxisAngle of the mdoc).

        The acquisition constants above are session/microscope values that are not
        present in the IMOD tilt-series files (.tlt/.xf) parsed here, so they are
        accepted as optional inputs. ``imod_to_cets`` emits them as an ``Instrument``
        (voltage) and an ``AcquisitionSession`` (spherical aberration, amplitude contrast,
        dose rate, tilt axis angle) rather than storing them on the tilt-images.
        """
        self.ts_file_name = validate_file(ts_file_name, "ts_file_name", MRC_MRCS_EXT)
        if isinstance(tilt_angles, (list, tuple)):
            tilt_angles = validate_tilt_angle_list(
                self.ts_file_name, [float(angle) for angle in tilt_angles]
            )
            tlt_file, dose_list, acq_orders = None, None, None
        else:
            tlt_file = validate_file(
                str(tilt_angles), "tilt_angles", [".tlt", ".rawtlt"]
            )
            tilt_angles, dose_list, acq_orders = parse_tlt_file(tlt_file)
        n_imgs = len(tilt_angles)
        self.refined_tilt_angles = self._load_refined_tilt_angles(
            refined_tilt_angles, n_imgs
        )
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
        self.tilt_axis_angle = tilt_axis_angle

    @staticmethod
    def _load_refined_tilt_angles(
        refined_tilt_angles: str | Path | List[float] | None, n_imgs: int
    ) -> Optional[List[float]]:
        """Refined tilt angles as a list (one per tilt-image), or None if not given."""
        if refined_tilt_angles is None:
            return None
        if isinstance(refined_tilt_angles, (list, tuple)):
            angles = [float(angle) for angle in refined_tilt_angles]
        else:
            tlt_file = validate_file(
                str(refined_tilt_angles), "refined_tilt_angles", ".tlt"
            )
            angles, _, _ = parse_tlt_file(tlt_file)
        if len(angles) != n_imgs:
            raise ValueError(
                f"Expected {n_imgs} refined tilt angles, but got {len(angles)}."
            )
        return angles

    def imod_to_cets(
        self,
        xf_file: str | Path | None = None,
        even_stack_file_name: str | Path | None = None,
        odd_stack_file_name: str | Path | None = None,
        ctf_corrected: bool = False,
        out_yaml_file: str | Path | None = None,
        pixel_size: float | None = None,
    ) -> Tuple[
        TiltSeries, Alignment, Optional[Instrument], Optional[AcquisitionSession]
    ]:
        """Converts an IMOD tilt-series into CETS metadata.

        In the current data model the per-projection alignment is NOT stored inside each
        tilt-image's ``coordinate_transformations``. Instead it is represented with the
        dedicated ``ProjectionAlignment`` structure (one per tilt-image, holding the
        ``[Affine, Translation]`` pair in its ``sequence``), and all of them are aggregated
        into a single ``Alignment`` for the tilt-series. ``Alignment`` objects live under
        ``Region.alignments``; since this converter emits a bare ``TiltSeries`` (not a
        ``Region``), the ``Alignment`` is returned alongside it for higher-level assembly.
        The i-th ``ProjectionAlignment`` corresponds to the i-th ``TiltSeries.images`` entry
        (positional binding, matching the schema's ordered-list convention).

        Microscope/session acquisition metadata is no longer stored on the tilt-images: the
        constructor-supplied constants are emitted as an ``Instrument`` (voltage) and an
        ``AcquisitionSession`` (spherical aberration, amplitude contrast, dose rate; linked to
        the instrument via ``instrument_id``). The tilt-series references the session via
        ``acquisition_session_id``. Both are returned as the 3rd and 4th elements for
        higher-level assembly onto ``Dataset.instruments`` / ``Dataset.acquisition_sessions``.
        IMOD files do not record any of these values, so they are only emitted when given:
        the ``Instrument`` is None if the voltage is not given, and the
        ``AcquisitionSession`` is None (and the tilt-series has no
        ``acquisition_session_id``) if neither a session value nor the voltage is given.
        Objects with every field empty would claim an acquisition nothing is known about.
        The tilt axis angle is not in the .tlt/.xf files either, so it is the one given to
        the constructor; ``cets_to_imod`` does not write it, as no IMOD file written there
        has a place for it.

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

        :param pixel_size: pixel size of the stack (Å). Defaults to the one of its header;
        if the header does not have it either, a ValueError is raised.
        :type pixel_size: float, optional

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
        # Parse xf file (identity alignment if not provided: no rotation, no shift)
        if xf_file is None:
            in_rotation_matrix_pile = np.zeros([2, 3, self.n_imgs])
            in_rotation_matrix_pile[0, 0, :] = 1.0
            in_rotation_matrix_pile[1, 1, :] = 1.0
            in_translation_vector_pile = np.zeros([3, self.n_imgs])
        else:
            xf_file = validate_file(xf_file, "xf_file", ".xf")
            in_rotation_matrix_pile, in_translation_vector_pile = parse_xf_file(xf_file)
            n_xf = in_rotation_matrix_pile.shape[2]
            if n_xf != self.n_imgs:
                raise ValueError(
                    f"The xf file {xf_file} has {n_xf} transformations, but there are "
                    f"{self.n_imgs} tilt angles."
                )

        ts_filename = str(self.ts_file_name)
        ts_id = self.ts_file_name.stem
        # Merged (tilt-series) pixel size in Å; backs the array_to_physical scale and the
        # conversion of alignment shifts to Å.
        pixel_size = resolve_pixel_size(
            pixel_size, img_info.apix_x, str(self.ts_file_name)
        )
        # Dataset-level acquisition metadata (constant across the tilt-series): the
        # microscope hardware (Instrument) and the session parameters (AcquisitionSession),
        # linked to the instrument via instrument_id. IDs are derived from the tilt-series id
        # to stay unique when several tilt-series are assembled into a single Dataset.
        # Only emitted when something is known about them (see the docstring). The tilt-series
        # reaches the instrument through its session, so a session is also emitted when only
        # the voltage is given.
        instrument = (
            Instrument(id=f"{ts_id}_instrument", voltage=self.voltage)
            if self.voltage is not None
            else None
        )
        session_values = (
            self.amplitude_contrast,
            self.dose_rate,
            self.spherical_aberration,
            self.tilt_axis_angle,
        )
        acquisition_session = (
            AcquisitionSession(
                id=f"{ts_id}_session",
                instrument_id=instrument.id if instrument else None,
                amplitude_contrast=self.amplitude_contrast,
                dose_rate=self.dose_rate,
                spherical_aberration=self.spherical_aberration,
                tilt_axis_angle=self.tilt_axis_angle,
            )
            if instrument is not None
            or any(value is not None for value in session_values)
            else None
        )
        ti_list = []
        projection_alignments = []
        for index in range(self.n_imgs):
            output_translation_transform = in_translation_vector_pile[:, index]
            output_rotation_matrix = in_rotation_matrix_pile[:, :, index]
            # Unique tilt-image id within the tilt-series (derived from the ts id + section).
            tilt_image_id = f"{ts_id}_{index}"
            # Every image gets an array (pixel, unitless) and a physical (Å) coordinate system
            # plus exactly one canonical array_to_physical scale transformation (spec).
            image_cs_name = f"tilt_image_{index:03d}"
            array_cs, physical_cs = gen_coordinate_systems(image_cs_name, ndim=2)
            array_to_physical = gen_array_to_physical(
                pixel_size, array_cs.name, physical_cs.name, ndim=2
            )
            ti = TiltImage(
                id=tilt_image_id,
                movie_stack_id=ts_id,  # TODO: define this
                path=ts_filename,
                section=index,
                nominal_tilt_angle=self.tilt_angles[index],
                accumulated_dose=self.dose_list[index] if self.dose_list else None,
                # Temporal collection order (from the .tlt 3rd column, or inferred from
                # the dose list by parse_tlt_file). Stored explicitly rather than left to
                # be re-derived from the dose downstream.
                acquisition_order=self.acq_orders[index] if self.acq_orders else None,
                ctf_metadata=self.ctf_md_list[index] if self.ctf_md_list else None,
                # Acquisition constants no longer live on the tilt-image; they are emitted
                # as the Instrument / AcquisitionSession built above.
                width=width,
                height=height,
                coordinate_systems=[array_cs, physical_cs],
                coordinate_transformations=[array_to_physical],
            )
            ti_list.append(ti)
            # One ProjectionAlignment per projection: maps this projection's physical frame to
            # the tilt-series' shared aligned physical frame (index-aligned with ti_list).
            projection_alignments.append(
                self._gen_projection_alignment(
                    output_translation_transform,
                    output_rotation_matrix,
                    pixel_size,
                    projection_alignment_id=f"{ts_id}_align_{index}",
                    tilt_image_id=tilt_image_id,
                    input_cs=physical_cs.name,
                    output_cs=f"tilt_series_{ts_id}_physical",
                    section=index,
                    refined_tilt_angle=(
                        self.refined_tilt_angles[index]
                        if self.refined_tilt_angles
                        else None
                    ),
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
            # Link the tilt-series to its acquisition session, if any.
            acquisition_session_id=(
                acquisition_session.id if acquisition_session else None
            ),
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
            if instrument is not None:
                self._write_ts_yaml(
                    instrument, self._instrument_yaml_path(out_yaml_file)
                )
            if acquisition_session is not None:
                self._write_ts_yaml(
                    acquisition_session, self._session_yaml_path(out_yaml_file)
                )
        return ts, alignment, instrument, acquisition_session

    @staticmethod
    def cets_to_imod(
        cets_ts: TiltSeries | Path | str,
        tlt_file: str | Path,
        alignment: Alignment | Path | str | None = None,
        add_dose_to_tlt: bool = True,
        xf_file: str | Path | None = None,
        rawtlt_file: str | Path | None = None,
    ):
        """Converts CETS Tilt-series metadata into IMOD files.

        :param cets_ts: CETS tilt-series metadata or a yaml file containing it (written by
        imod_to_cets or by the command line).
        :type cets_ts: TiltSeries or pathlib.Path or str

        :param tlt_file: output tlt file to be generated, with the tilt angles IMOD uses to
        reconstruct: the refined ones of the alignment if it has them for every tilt-image
        (as the .tlt written by tiltalign), or else the nominal ones.
        :type tlt_file: pathlib.Path or str

        :param alignment: CETS Alignment holding one ProjectionAlignment per tilt-image
        (index-aligned with ``cets_ts.images``). Required to write the xf file, since the
        alignment is no longer stored inside the tilt-image ``coordinate_transformations``.
        It can also be a yaml file containing it.
        :type alignment: Alignment or pathlib.Path or str, optional, Defaults to None

        :param add_dose_to_tlt: used to indicate if the generated tlt file should also
        contain a second column with the dose.
        :type add_dose_to_tlt: bool, optional, Defaults to True

        :param xf_file: output xf file to be generated.
        :type: pathlib.Path or str, optional, Defaults to None

        :param rawtlt_file: output file with the nominal tilt angles (IMOD's .rawtlt).
        :type: pathlib.Path or str, optional, Defaults to None
        """
        if not isinstance(cets_ts, TiltSeries):
            cets_ts = load_cets_object(cets_ts, TILT_SERIES_KEY, TiltSeries)
        if alignment is not None and not isinstance(alignment, Alignment):
            alignment = load_cets_object(alignment, ALIGNMENT_KEY, Alignment)
        # Write the tlt file: refined angles if the alignment has them, else nominal ones
        refined_angles = ImodTiltSeries._get_refined_tilt_angles(cets_ts, alignment)
        write_tlt(
            cets_ts,
            tlt_file,
            add_dose_to_tlt=add_dose_to_tlt,
            tilt_angles=refined_angles,
        )
        if rawtlt_file is not None:
            write_tlt(cets_ts, rawtlt_file, add_dose_to_tlt=add_dose_to_tlt)
        if xf_file is not None:
            if alignment is None:
                print(
                    "cets_to_imod -> an xf_file was requested but no alignment was "
                    "provided. Skipping the xf file."
                )
            else:
                # Per-projection pixel sizes (Å) let write_xf convert the CETS shifts
                # (stored in Å) back to pixels, as native IMOD .xf files use.
                images = getattr(cets_ts, "images", None) or []
                pixel_sizes = [get_image_pixel_size(img) for img in images]
                write_xf(alignment, xf_file, pixel_sizes=pixel_sizes)

    @staticmethod
    def _get_refined_tilt_angles(
        cets_ts: TiltSeries, alignment: Optional[Alignment]
    ) -> Optional[List[float]]:
        """Refined tilt angle of each tilt-image (in the order of ``cets_ts.images``), taken
        from the ProjectionAlignment that references it (tilt_image_id) or, if they do not
        reference the tilt-images, from the one in the same position. None if there is no
        alignment or it does not have a refined tilt angle for every tilt-image."""
        if alignment is None or not alignment.projection_alignments:
            return None
        images = cets_ts.images or []
        projections = alignment.projection_alignments
        by_image = {p.tilt_image_id: p for p in projections if p.tilt_image_id}
        if by_image and all(ti.id in by_image for ti in images):
            matched = [by_image[ti.id] for ti in images]
        elif len(projections) == len(images):
            matched = list(projections)
        else:
            return None
        angles = [p.refined_tilt_angle for p in matched]
        if any(angle is None for angle in angles):
            return None
        return [float(angle) for angle in angles if angle is not None]

    def _gen_projection_alignment(
        self,
        translation_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        pix_size: float = 1.0,
        projection_alignment_id: str = "",
        tilt_image_id: str | None = None,
        input_cs: str | None = None,
        output_cs: str | None = None,
        section: int = 0,
        refined_tilt_angle: float | None = None,
    ) -> ProjectionAlignment:
        """Builds the per-projection alignment as a ProjectionAlignment whose ``sequence``
        is ``[Affine, Translation]`` (rotation applied first, then shift, shifts in Å),
        mapping the tilt-image physical frame (``input_cs``) to the tilt-series' shared
        aligned physical frame (``output_cs``).

        :param projection_alignment_id: unique id for this ProjectionAlignment.
        :param tilt_image_id: id of the TiltImage this alignment applies to.
        :param input_cs: name of the tilt-image physical coordinate system.
        :param output_cs: name of the shared tilt-series aligned physical coordinate system.
        :param section: 0-based projection index, used for the alignment name.
        :param refined_tilt_angle: tilt angle refined by the alignment, in degrees.
        """
        return ProjectionAlignment(
            id=projection_alignment_id,
            tilt_image_id=tilt_image_id,
            refined_tilt_angle=refined_tilt_angle,
            sequence=[
                self._gen_affine_transform(rotation_matrix),
                self._gen_translation_transform(translation_matrix, pix_size),
            ],
            name=f"alignment_tilt_{section:03d}",
            input=input_cs,
            output=output_cs,
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
    ) -> Affine:
        """Per-projection rotation as a homogeneous 3x3 affine (dimensionless)."""
        return Affine(affine=self._get_affine_values(rotation_matrix))

    def _gen_translation_transform(
        self, translation_matrix: np.ndarray, pix_size: float = 1.0
    ) -> Translation:
        """Per-projection shift, converted from pixels to Å (physical frame)."""
        return Translation(
            translation=self._get_translation_values(translation_matrix, pix_size)
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
    ) -> List[float]:
        """Gets the shifts in X and Y directions, in angstroms: a 2D vector, as the tilt-image
        frame is 2D (the same as cets-scipion)."""
        return [float(shift) * pix_size for shift in translation_matrix[:2]]

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

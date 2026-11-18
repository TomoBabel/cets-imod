MRC_MRCS_EXT = [".mrc", ".mrcs"]

# Flags of the IMOD defocus files (version 3), whose value is the sum of these bits
# (https://bio3d.colorado.edu/imod/doc/man/ctfphaseflip.html#Defocus%20File%20Format)
DEFOCUS_FLAG_ASTIGMATISM = 1  # columns of defocus V and astigmatism angle
DEFOCUS_FLAG_ANGLE_IN_RADIANS = 2  # astigmatism angle in radians, not degrees
DEFOCUS_FLAG_PHASE_SHIFT = 4  # column of phase shift
DEFOCUS_FLAG_PHASE_IN_RADIANS = 8  # phase shift in radians, not degrees
DEFOCUS_FLAG_INVERTED_ANGLES = 16  # tilt angles inverted (ctfplotter -invert)
DEFOCUS_FLAG_CUT_ON = 32  # column of cut-on frequency
# Version written at the end of the header line of version 3 defocus files
DEFOCUS_FILE_VERSION_3 = 3

# Keys of the objects in the CETS yaml files (see cets_data_model.utils.yaml_io)
TILT_SERIES_KEY = "tilt_series"
ALIGNMENT_KEY = "alignment"
INSTRUMENT_KEY = "instrument"
ACQUISITION_SESSION_KEY = "acquisition_session"
CTF_KEY = "ctf"
TOMOGRAM_KEY = "tomogram"

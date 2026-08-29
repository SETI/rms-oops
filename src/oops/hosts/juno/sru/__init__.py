##########################################################################################
# oops/hosts/juno/sru/__init__.py
##########################################################################################

import numpy as np
import julian
import pdsparser
import astropy.io.fits as pyfits
import oops

from oops.hosts.juno import Juno

from filecache import FCPath

##########################################################################################
# Standard class methods
##########################################################################################

def from_file(filespec, return_all_planets=False, method='strict', **parameters):
    """A general, static method to return a Snapshot object based on a given
    Juno SRU EDR image file.

    Inputs:
        filespec            The full path to a Juno SRU FITS image file or its
                            detached PDS label.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.

        method              Label reading method to be passed to Pds3Label.
    """
    SRU.initialize()    # Define everything the first time through; use
                        # defaults unless initialize() is called explicitly.

    filespec = FCPath(filespec)

    # Load the PDS label; given a data file path, Pds3Label reads the
    # detached .LBL/.lbl label alongside it
    label = pdsparser.Pds3Label(filespec, method=method).as_dict()

    # Get metadata
    meta = _Metadata(label)

    # Locate the data file; when given the label, take the file name from the
    # label's ^IMAGE pointer
    if filespec.suffix.upper() == '.LBL':
        pointer = label['^IMAGE']
        name = pointer[0] if isinstance(pointer, (tuple, list)) else pointer
        datspec = filespec.parent / name
    else:
        datspec = filespec

    # Load time-dependent kernels
    Juno.load_cks(meta.tstart, meta.tstop)
    Juno.load_spks(meta.tstart, meta.tstop)

    # Load the data array
    data = _load_data(datspec, meta)

    # Define the inertially fixed camera frame for this observation
    frame = SRU.create_frame(meta.unit, meta.tstart)

    # Construct the Snapshot
    obs = oops.obs.Snapshot(('v','u'),
                            meta.tstart, meta.exposure, SRU.fov(),
                            'JUNO', frame,
                            instrument = 'SRU' + str(meta.unit),
                            target = meta.target,
                            tdi_on = meta.tdi_on,
                            data = data)

#    obs.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(obs.time, 'sru', return_all_planets))
    obs.insert_subfield('filespec', filespec)
    obs.insert_subfield('basename', filespec.name)
    obs.insert_subfield('dict', label)

    return obs

def _load_data(datspec, meta):
    """Load the image array from the FITS file.

    Parameters:
        datspec (str or FCPath): Full path to the FITS data file.
        meta (object): Image Metadata object.

    Returns:
        A Numpy array containing the data in axis order (line, sample), where lines
            and samples correspond to the CCD rows and columns defined in the SIS. Dummy
            pixels (rows 510-511, columns 0-1) and any pixels not downlinked contain zero.
    """
    local_path = datspec.retrieve()
    with pyfits.open(local_path) as hdulist:
        data = hdulist[0].data

    if data.shape != (meta.nlines, meta.nsamples):
        raise ValueError('SRU data shape %s does not match label (%d,%d)'
                         % (data.shape, meta.nlines, meta.nsamples))

    return data


#*******************************************************************************
class _Metadata(object):

    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Parameters:
            label (dict): The label dictionary.

        Attributes:
            nlines          Number of lines (CCD rows). nsamples        Number of samples
            per line (CCD columns). exposure        Exposure duration in seconds. tstart
            Image start time in seconds TDB. tstop           Image stop time in seconds
            TDB. unit            SRU unit number, 1 or 2. tdi_on          True if
            time-delay integration was used to compensate for the spacecraft spin. target
            Target name.
        """

        # Image dimensions
        self.nlines = label['IMAGE']['LINES']
        self.nsamples = label['IMAGE']['LINE_SAMPLES']

        # Timing
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))

        # Exposure time
        try:
            self.exposure = label['EXPOSURE_DURATION']
        except KeyError:
            self.exposure = self.tstop - self.tstart

        # SRU unit number, from e.g. PRODUCT_ID = "SRU_1_2024100T045333_60_V01"
        self.unit = int(label['PRODUCT_ID'].split('_')[1])

        # Time-delay integration
        self.tdi_on = label.get('JNO:TDI_ON', 'UNK') == 'YES'

        # Target
        self.target = label['TARGET_NAME']

        return


#*******************************************************************************
class SRU(object):
    """An instance-free class to hold SRU instrument parameters.

    The Juno Stellar Reference Unit (SRU) is a star tracker operated as a broadband
    visible (450-1100 nm) science imager. Values here are from the SRU EDR/CRT SIS,
    JUNO_SRU_EDR_CRT_SIS_V01_2.
    """

    SAMPLES = 512               # CCD columns; columns 0-1 are dummy pixels
    LINES = 512                 # CCD rows; rows 510-511 are dummy pixels
    UV_LOS = (255.5, 255.5)     # boresight pixel
    FL_PIXELS = 1760.21137      # focal length in pixel units (~29.924 mm)

    # Radial distortion correction f(R) = a0 + a1*R + a2*R**2 + a3*R**4,
    # where R is the tangent of the undistorted radial angle and f(R) scales
    # the pinhole-projected tangents into distortion-corrected ones.
    DISTORTION = (0.999432579, -0.0295412410, 0.2733020107, -1.9368112951)

    _fov = None
    spice_frames = {}
    initialized = False

    @staticmethod
    def initialize(asof=None, **kwargs):
        """Initialize key information about the SRU instrument.

        Must be called first. After the first call, later calls to this function are
        ignored.

        Parameters:
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore. kwargs:     Arguments for juno.initialize() and
                Body.define_solar_system()
        """

        # Quick exit after first call
        if SRU.initialized:
            return

        # initialize Juno
        Juno.initialize(asof=asof, **kwargs)
        Juno.load_instruments(asof=asof)

        SRU.initialized = True

    @staticmethod
    def fov():
        """The SRU field of view, common to both units.

        The SIS distortion model scales the pinhole tangents (from pixel
        offsets relative to the boresight) by f(R); in BarrelFOV terms the
        radial distance polynomial is f(R)*R, so the R**4 term of f becomes
        the fifth-order coefficient.
        """
        if SRU._fov is None:
            scale = 1./SRU.FL_PIXELS
            (a0, a1, a2, a3) = SRU.DISTORTION
            SRU._fov = oops.fov.BarrelFOV((scale, scale),
                                          (SRU.SAMPLES, SRU.LINES),
                                          coefft_xy_from_uv=(a0, a1, a2, 0., a3),
                                          uv_los=SRU.UV_LOS)
        return SRU._fov

    @staticmethod
    def create_frame(unit, time):
        """Create the camera frame for an SRU observation.

        The frame is inertially fixed at the SRU orientation for the given time. With the
        spacecraft spinning at ~2 rpm, the SRU uses time-delay integration (TDI) to shift
        the accumulating image charge in step with the scene, so the recorded scene is
        frozen at the orientation the camera had when the exposure began, rather than
        rotating with the spacecraft during the exposure. (Without TDI, the scene at the
        start of exposure smears along the CCD columns.)

        The SPICE frames JUNO_SRU1/JUNO_SRU2 have the boresight along +X, with the CCD x
        axis (the along-row direction, in which samples are counted) along +Y and the CCD
        y axis (the along-column direction, in which lines are counted) along +Z; the SIS
        maps a distortion-corrected position (tx,ty) to the unit vector (1,-tx,-ty). The
        fixed rotation applied here re-labels those axes to the OOPS camera convention:
        boresight along +Z, x along increasing sample, y along increasing line.

        Parameters:
            unit: SRU unit number, 1 or 2.
            time (Scalar): Time at which to define the inertially fixed frame, in seconds
                TDB; normally the image start time.

        Returns:
            An unregistered, per-observation Frame object.
        """
        spice_frame = 'JUNO_SRU' + str(unit)
        if unit not in SRU.spice_frames:
            SRU.spice_frames[unit] = oops.frame.SpiceFrame(spice_frame)

        # rotation to reorganize axis vectors
        rot = oops.Matrix3([[ 0,-1, 0],
                            [ 0, 0,-1],
                            [ 1, 0, 0]])

        # Define fixed frame relative to J2000 from the SRU orientation at
        # the given time
        xform = SRU.spice_frames[unit].transform_at_time(time)
        return oops.frame.Cmatrix(rot * xform.matrix)

    @staticmethod
    def reset():
        """Reset the internal SRU parameters.

        Can be useful for debugging.
        """
        SRU._fov = None
        SRU.spice_frames = {}
        SRU.initialized = False

        Juno.reset()

##########################################################################################

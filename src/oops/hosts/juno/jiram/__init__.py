##########################################################################################
# oops/inst/juno/jiram/__init__.py
##########################################################################################

import julian
import pdsparser
import oops

from oops.hosts.juno import Juno

from filecache import FCPath

__all__ = ['from_file', 'JIRAM']

##########################################################################################
# Standard class methods
##########################################################################################
def from_file(filespec, return_all_planets=False, method='strict', **parameters):
    """A Snapshot object based on a given JIRAM image or spectrum file.

    Parameters:
        return_all_planets (bool, optional): Include kernels for all planets not just
            Jupiter or Saturn.
        method (str, optional): Label reading method to be passed to Pds3Label.
    """
    JIRAM.initialize()    # Define everything the first time through; use
                          # defaults unless initialize() is called explicitly.

    filespec = FCPath(filespec)

    # Load the PDS label
    label = pdsparser.Pds3Label(filespec, method=method).as_dict()

    # Get common metadata
    meta = _Metadata(label)

    # Load time-dependent kernels
    Juno.load_cks(meta.tstart, meta.tstart + 3600.)
    Juno.load_spks(meta.tstart, meta.tstart + 3600.)

    # Determine which observation type and load data
    ext = filespec.suffix

    # Image
    if ext.upper() == '.IMG':
        from . import img
        return img.from_file(filespec, label,
                             return_all_planets=return_all_planets, **parameters)

    # Spectrum
    if ext.upper() == '.DAT':
        from . import spe
        return spe.from_file(filespec, label,
                             return_all_planets=return_all_planets, **parameters)

    return None


#*******************************************************************************
class _Metadata(object):

    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Parameters:
            label (dict): The label dictionary.

        Attributes:
            tstart (float): Image start time in seconds TDB.
            tstop (float): Image stop time in seconds TDB.
        """

        # Default timing for unprocessed frame
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))

        return


#*******************************************************************************
class JIRAM(object):
    """A instance-free class to hold JIRAM instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    def initialize(asof=None, **kwargs):
        """Initialize key information about the JIRAM instrument.

        Key information about the WAC and NAC is filled in.

        Must be called first. After the first call, later calls to this function are
        ignored.

        Parameters:
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            **kwargs: Arguments for `juno.initialize()` and
                :meth:`~oops.Body.define_solar_system`.
        """

        # Quick exit after first call
        if JIRAM.initialized:
            return

        # initialize Juno
        Juno.initialize(asof=asof, **kwargs)
        Juno.load_instruments(asof=asof)

        JIRAM.initialized = True

    @staticmethod
    def create_frame(time, name):
        """Create a frame for a JIRAM component.

        Parameters:
            time (Scalar): Time at which to define the inertialy fixed mirror-corrected
                frame.
            name (str): Name of the component.
        """
        spice_frame = 'JUNO_JIRAM_' + name

        # rotation to reorganize axes vectors
        rot = oops.Matrix3([[ 0,-1, 0],
                            [-1, 0, 0],
                            [ 0, 0, 1]])

        # Define fixed frame relative to J2000 from JIRAM orientation at
        # given time
        jiram_raw = oops.frame.SpiceFrame(spice_frame,
                                          frame_id=spice_frame+'_RAW')
        xform = jiram_raw.transform_at_time(time)

        jiram_raw_j2000 = oops.frame.Cmatrix(xform.matrix,
                                             frame_id=spice_frame+'_RAW_J2000')
        jiram_frame = oops.frame.Cmatrix(rot,
                                         jiram_raw_j2000,
                                         frame_id=spice_frame)

    @staticmethod
    def reset():
        """Reset the internal JIRAM parameters.

        Can be useful for debugging.
        """
        JIRAM.instrument_kernel = None
        JIRAM.fovs = {}
        JIRAM.initialized = False

        Juno.reset()

##########################################################################################

##########################################################################################
# oops/inst/juno/jiram/spe.py
##########################################################################################

import numpy as np
import julian
import cspyce
import oops

from oops.hosts.juno.jiram import JIRAM

from filecache import FCPath

##########################################################################################
# Standard class methods
##########################################################################################

#### TODO: verify label input needed or whether it is covered by filespec, as
####       other host modules.
def from_file(filespec, label, fast_distortion=True,
                               return_all_planets=False, **parameters):
    """A Snapshot object based on a given JIRAM image or spectrum file.

    Parameters:
        filespec (str, Path, or FCPath): The full path to a Juno JIRAM spectral file or
            its PDS label.
        fast_distortion (bool or None, optional): True to use a pre-inverted polynomial;
            False to use a dynamically solved polynomial; None to use a FlatFOV.
        return_all_planets (bool, optional): Include kernels for all planets not just
            Jupiter or Saturn.
    """

    filespec = FCPath(filespec)

    # Get metadata
    meta = _Metadata(label)

    # Define everything the first time through
    SPE.initialize(meta.tstart)

    # Load the data array as separate framelets, with associated labels
    data = _load_data(filespec, label, meta)

    # Construct Snapshots for slit in each band
    slits = []
    for i in range(meta.nsamples):
        item = oops.obs.Snapshot(('v','u'),
                                 meta.tstart, meta.exposure, meta.fov,
                                 'JUNO', 'JUNO_JIRAM_S',
                                 data=np.reshape(data[:,i],(1,meta.nlines)) )

#        item.insert_subfield('spice_kernels',
#                   Juno.used_kernels(item.time, 'jiram', return_all_planets))
        item.insert_subfield('filespec', filespec)
        item.insert_subfield('basename', filespec.name)
        slits.append(item)

#    return slits

    # Construct Slit1D for all bands
    obs = oops.obs.Slit1D(('u','b'),
                          meta.tstart, meta.exposure, meta.fov,
                          'JUNO', 'JUNO_JIRAM_S', data=data )

#    obs.insert_subfield('spice_kernels',
#               Juno.used_kernels(item.time, 'jiram', return_all_planets))
    obs.insert_subfield('filespec', filespec)
    obs.insert_subfield('basename', filespec.name)

    return (obs, slits)

def _load_data(filespec, label, meta):
    """Load the data array from the file and splits into individual framelets.

    Parameters:
        filespec (str or FCPath): Full path to the data file.
        label (str): Label for composite image.
        meta (object): Image Metadata object.

    Returns:
        numpy.ndarray: The individual spectra in wavelength order, with axes
        (spectrum #, sample).
    """

    # Read data
    local_path = filespec.retrieve()
    data = np.fromfile(local_path, dtype='<f4').reshape(meta.nlines,meta.nsamples)

    return data


#*******************************************************************************
class _Metadata(object):

    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Parameters:
            label (dict): The label dictionary.

        Attributes:
            nlines (int): Number of rows in the spectral table.
            nsamples (int): Number of columns in the spectral table.
            exposure (float): Exposure duration in seconds.
            tstart (float): Observation start time in seconds TDB.
            tstop (float): Observation stop time in seconds TDB.
            target (str): Target name.
            fov (FOV): The field of view of one slit.
        """

        # dimensions
        self.nlines = label['FILE']['TABLE']['ROWS']
        self.nsamples = label['FILE']['TABLE']['COLUMNS']

        # Exposure time
        self.exposure = label['EXPOSURE_DURATION']

        # Default timing
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))

        # target
        self.target = label['TARGET_NAME']

        # Kernel FOV params
        cross_angle = cspyce.gdpool('INS-61420_FOV_CROSS_ANGLE', 0)[0]
        fo = cspyce.gdpool('INS-61420_FOCAL_LENGTH', 0)[0]
        px = cspyce.gdpool('INS-61420_PIXEL_SIZE', 0)[0]
        cxy = cspyce.gdpool('INS-61420_CCD_CENTER', 0)
        scale = px/1000/fo

        # FOVs
        self.fov = oops.fov.FlatFOV(scale, (self.nlines, 1), uv_los=cxy)

        return


#*******************************************************************************
class SPE(object):
    """A instance-free class to hold SPE instrument parameters."""

    initialized = False

    @staticmethod
    def initialize(time, asof=None, **kwargs):
        """        Initialize key information about the SPE instrument.

        Must be called first. After the first call, later calls to this function are
        ignored.

        Parameters:
            time (Scalar): Time at which to define the inertialy fixed mirror- corrected
                frame.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore. kwargs:     Arguments for juno.initialize() and
                Body.define_solar_system()
        """

        # Quick exit after first call
        if SPE.initialized:
            return

        # initialize JIRAM
        JIRAM.initialize(asof=asof, **kwargs)

        # Construct the SpiceFrame
        JIRAM.create_frame(time, 'S')

        SPE.initialized = True

    @staticmethod
    def reset():
        """Reset the internal SPE parameters.

        Can be useful for debugging.
        """
        SPE.initialized = False

        JIRAM.reset()

##########################################################################################

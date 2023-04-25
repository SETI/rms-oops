################################################################################
# hosts/galileo/ssi/__init__.py
################################################################################
import numpy as np
import julian
import cspyce
import vicar
import pdstable
import pdsparser
import oops

from hosts         import pds3
from hosts.galileo import Galileo

################################################################################
# Standard class methods
################################################################################
def from_file(filespec,
              return_all_planets=False, full_fov=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    Galileo SSI image file.  By default, only the valid image region is
    returned.

    Inputs:
        full_fov:           If True, the full image is returned with a mask
                            describing the regions with no data.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """

    SSI.initialize()    # Define everything the first time through; use defaults
                        # unless initialize() is called explicitly.

    # Load the PDS label
    label = pds3.get_label(filespec)

    # Load the data array
    vic = vicar.VicarImage.from_file(filespec, extraneous='warn')
    vicar_dict = vic.as_dict()

    # Get image metadata
    meta = Metadata(label)

    # Load time-dependent kernels
    Galileo.load_cks(meta.tstart, meta.tstart + meta.exposure)
    Galileo.load_spks(meta.tstart, meta.tstart + meta.exposure)

    # Define the field of view
    FOV = meta.fov(full_fov=full_fov)

    # Trim the image
    data = meta.trim(vic.data_2d, full_fov=full_fov)

    # Create a Snapshot
    result = oops.obs.Snapshot(('v','u'), meta.tstart, meta.exposure,
                               FOV,
                               path = 'GLL',
                               frame = 'GLL_SCAN_PLATFORM',
                               dict = vicar_dict,       # Add the VICAR dict
                               data = data,             # Add the data array
                               instrument = 'SSI',
                               filter = meta.filter,
                               filespec = filespec,
                               basename = os.path.basename(filespec))

    result.insert_subfield('spice_kernels',
                           Galileo.used_kernels(result.time, 'ssi',
                                                return_all_planets))

    return result

#===============================================================================
def initialize(planets=None, asof=None,
               mst_pck=True, irregulars=True):
    """Initialize key information about the SSI instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

    Input:
        planets     A list of planets to pass to define_solar_system. None or
                    0 means all.
        asof        Only use SPICE kernels that existed before this date; None
                    to ignore.
        mst_pck     True to include MST PCKs, which update the rotation models
                    for some of the small moons.
        irregulars  True to include the irregular satellites;
                    False otherwise.
    """
    SSI.initialize(planets=planets, asof=asof,
                   mst_pck=mst_pck, irregulars=irregulars)


#===============================================================================
class Metadata(object):

    #===========================================================================
    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Input:
            label           The label dictionary.
        """

        # Image dimensions
        self.nlines = label['IMAGE']['LINES']
        self.nsamples = label['IMAGE']['LINE_SAMPLES']

        # Exposure time
        exposure_ms = label['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        # Filters
        self.filter = label['FILTER_NAME']

        #TODO: determine whether IMAGE_TIME is the start time or the mid time..
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['IMAGE_TIME']))
        self.tstop = self.tstart + self.exposure

        # Target
        self.target = label['TARGET_NAME']

        # Telemetry mode
        self.mode = label['TELEMETRY_FORMAT_ID']

        # Window
        if 'CUT_OUT_WINDOW' in label:
            self.window = label['CUT_OUT_WINDOW']
        else:
            self.window = None

    #===========================================================================
    def trim(self, data, full_fov=False):
        """Trim image to label window

        Input:
            full_fov        If True, the image is not trimmed.

        Output:
            Data array trimmed to the data window.
        """

        if full_fov:
            return None

        window = self.window

        if window is None:
            return data

        return data[window[0]:window[2], window[1]:window[3]]

    #===========================================================================
    def fov(self, full_fov=False):
        """Construct the field of view based on the metadata.

        Input:
            full_fov        If False, the FOV is cropped to the dimensions
                            given by the cutout window.

        Attributes:
            FOV object.
        """

        # FOV Kernel pool variables
        cf_var = 'INS-77036_DISTORTION_COEFF'
        fo_var = 'INS-77036_FOCAL_LENGTH'
        px_var = 'INS-77036_PIXEL_SIZE'
        cxy_var = 'INS-77036_FOV_CENTER'

        cf = cspyce.gdpool(cf_var, 0)[0]
        fo = cspyce.gdpool(fo_var, 0)[0]
        px = cspyce.gdpool(px_var, 0)[0]
        cxy = cspyce.gdpool(cxy_var, 0)

        # Construct FOV
        scale = px/fo
        distortion_coeff = [1,0,cf]

        # Direct summation modes
        if self.mode=='HIS' or self.mode=='AI8':
            scale = scale*2
            cxy = cxy/2

        # Construct full FOV
        fov_full = oops.fov.BarrelFOV(scale,
                                      (self.nsamples, self.nlines),
                                      coefft_uv_from_xy=distortion_coeff,
                                      uv_los=(cxy[0], cxy[1]))

        # Apply cutout window if full fov not requested
        if not full_fov and self.window is not None:
            window = np.array(self.window)
            fov = oops.fov.SliceFOV(fov_full,
                                    window[[0,1]],
                                    window[2:] - window[0:2])
        else:
            fov = fov_full

        return fov



#===============================================================================
class SSI(object):
    """An instance-free class to hold Galileo SSI instrument parameters."""

    instrument_kernel = None
    fov = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(planets=None, asof=None,
                   mst_pck=True, irregulars=True):
        """Initialize key information about the SSI instrument.

        Fills in key information about the camera.  Must be called first.
        After the first call, later calls to this function are ignored.

        Input:
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """

        # Quick exit after first call
        if SSI.initialized:
            return

        # Initialize Galileo
        Galileo.initialize(planets=planets, asof=asof,
                           mst_pck=mst_pck, irregulars=irregulars)
        Galileo.load_instruments(asof=asof)

        # Construct the SpiceFrame
        _ = oops.frame.SpiceFrame("GLL_SCAN_PLATFORM")

        SSI.initialized = True
        return

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal Galileo SSI parameters.

        Can be useful for debugging.
        """

        SSI.instrument_kernel = None
        SSI.fov = {}
        SSI.initialized = False

        Galileo.reset()

################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path
import oops.backplane.gold_master as gm

from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY

#===============================================================================
class Test_Galileo_SSI_GoldMaster(unittest.TestCase):

    #===========================================================================
    def setUp(self):
        from hosts.galileo.ssi import standard_obs

    #===========================================================================
    def test_C0349632100R(self):
        gm.execute_as_unittest(self, 'C0349632100R')

    #===========================================================================
    def test_C0368369200R(self):
        gm.execute_as_unittest(self, 'C0368369200R')

    #===========================================================================
    def test_C0061455700R(self):
        gm.execute_as_unittest(self, 'C0061455700R')

############################################

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

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
        filespec            The full path to a Galileo SSI file or its PDS label.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.

        full_fov:           If True, the full image is returned with a mask
                            describing the regions with no data.
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
def from_index(filespec, supplemental_filespec, full_fov=False, **parameters):
    """A static method to return a list of Snapshot objects.

    One object for each row in an SSI index file. The filespec refers to the
    label of the index file.
    """
    SSI.initialize()    # Define everything the first time through

    # Read the index file
    COLUMNS = []        # Return all columns
    TIMES = ['START_TIME']
    table = pdstable.PdsTable(filespec, columns=COLUMNS, times=TIMES)
    row_dicts = table.dicts_by_row()

    # Read the supplemental index file
    table = pdstable.PdsTable(supplemental_filespec)
    supplemental_row_dicts = table.dicts_by_row()
    for row_dict, supplemental_row_dict in zip(row_dicts, supplemental_row_dicts):
        row_dict.update(supplemental_row_dict)


####TODO: CUT_OUT_WINDOW should be represented in the index label as a single
####      column with 4 values as:
####
####        OBJECT                        = COLUMN
####          NAME                        = CUT_OUT_WINDOW
####          ITEMS                       = 4
####          DATA_TYPE                   = ASCII_INTEGER
####          START_BYTE                  = 52
####          BYTES                       = 3
####          FORMAT                      = I3
####          DESCRIPTION                 = "xxxxxxxx."
####        END_OBJECT                    = COLUMN
####
####       However, it appears that the code to handle this in psdtable.py
####       starting at line 257 does not work.  Therefore, at present,
####       CUT_OUT_WINDOW is represented as four separate columns.
####       (actally should return integer values)
    for row_dict in row_dicts:
        row_dict['CUT_OUT_WINDOW'] = [ row_dict['CUT_OUT_WINDOW_0'],
                                       row_dict['CUT_OUT_WINDOW_1'],
                                       row_dict['CUT_OUT_WINDOW_2'],
                                       row_dict['CUT_OUT_WINDOW_3'] ]



    # Create a list of Snapshot objects
    snapshots = []
    for row_dict in row_dicts:


        print(row_dict['FILE_SPECIFICATION_NAME'])
        if row_dict['FILE_SPECIFICATION_NAME'] == 'G2/IO/C0359986604R.LBL':
            continue
        ###--> hits limit on # loaded kernels

        # Get image metadata
        meta = Metadata(row_dict)

        # Load time-dependent kernels
        Galileo.load_cks(meta.tstart, meta.tstart + meta.exposure)
        Galileo.load_spks(meta.tstart, meta.tstart + meta.exposure)

        # Define the field of view
        FOV = meta.fov(full_fov=full_fov)

        # Create a Snapshot
        item = oops.obs.Snapshot(('v','u'), meta.tstart, meta.exposure,
                                 FOV,
                                 path = 'GLL',
                                 frame = 'GLL_SCAN_PLATFORM',
                                 dict = row_dict,         # Add the index dict
                                 instrument = 'SSI',
                                 filter = meta.filter,
                                 filespec = filespec,
                                 basename = os.path.basename(filespec))

        item.spice_kernels = Galileo.used_kernels(item.time, 'ssi')

        item.filespec = os.path.join(row_dict['VOLUME_ID'],
                                     row_dict['FILE_SPECIFICATION_NAME'])
#############        item.basename = row_dict['FILE_NAME']

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[ 0]['IMAGE_TIME']
    tdb1 = row_dicts[-1]['IMAGE_TIME']

    Galileo.load_cks( tdb0, tdb1)
    Galileo.load_spks(tdb0, tdb1)

    return snapshots

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
    def __init__(self, meta_dict):
        """Use the label or index dict to assemble the image metadata."""

        # Image dimensions
        self.nlines = 800
        self.nsamples = 800

        # Exposure time
        exposure_ms = meta_dict['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        # Filters
        self.filter = meta_dict['FILTER_NAME']

        #TODO: determine whether IMAGE_TIME is the start time or the mid time..
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(meta_dict['IMAGE_TIME']))
        self.tstop = self.tstart + self.exposure

        # Target
        self.target = meta_dict['TARGET_NAME']

        # Telemetry mode
        self.mode = meta_dict['TELEMETRY_FORMAT_ID']

        # Window
        if 'CUT_OUT_WINDOW' in meta_dict:
            self.window = np.array(meta_dict['CUT_OUT_WINDOW'])
            self.window_origin = self.window[0:2]-1
            self.window_shape = self.window[2:]
            self.window_uv_origin = np.flip(self.window_origin)
            self.window_uv_shape = np.flip(self.window_shape)
        else:
            self.window = None

    #===========================================================================
    def trim(self, data, full_fov=False):
        """Trim image to label window.

        Input:
            data            Numpy array containing the image data.

            full_fov        If True, the image is not trimmed.

        Output:
            Data array trimmed to the data window.
        """
        if full_fov:
            return data

        if self.window is None:
            return data

        origin = self.window_origin
        shape = self.window_shape

        return data[origin[0]:origin[0]+shape[0],
                    origin[1]:origin[1]+shape[1]]

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
        distortion_coeff = [1, 0, cf]

        # Direct summation modes
        if self.mode in ('HIS', 'AI8'):
            scale = scale*2
            cxy = cxy/2

        # Construct full FOV
        fov_full = oops.fov.BarrelFOV(scale,
                                      (self.nsamples, self.nlines),
                                      coefft_uv_from_xy=distortion_coeff,
                                      uv_los=(cxy[0], cxy[1]))

        # Apply cutout window if full fov not requested
        if not full_fov and self.window is not None:
            uv_origin = self.window_uv_origin
            uv_shape = self.window_uv_shape
            fov = oops.fov.SliceFOV(fov_full, uv_origin, uv_shape)
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

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

#===============================================================================
class Test_AAA_Galileo_SSI_index_file(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        dir = '/home/spitale/SETI/RMS/metadata/GO_0xxx/GO_0017'

        obs = from_index(os.path.join(dir, 'GO_0017_index.lbl'),
                         os.path.join(dir, 'GO_0017_supplemental_index.lbl'))


#        dir = os.path.join(TESTDATA_PARENT_DIRECTORY, 'galileo/GO_0017')
#
#        obs = from_index(os.path.join(dir, 'GO_0017_index.lbl'),
#                         os.path.join(dir, 'GO_0017_supplemental_index.lbl'))


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

    #===========================================================================
    def test_C0374685140R(self):
        gm.execute_as_unittest(self, 'C0374685140R')

############################################

if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

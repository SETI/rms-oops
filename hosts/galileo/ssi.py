################################################################################
# hosts/galileo/iss.py
################################################################################
import numpy as np
import julian
import cspyce
import vicar
import pdstable
import pdsparser
import oops

from hosts.galileo import Galileo

################################################################################
# Standard class methods
################################################################################
def from_file(filespec, fast_distortion=True,
              return_all_planets=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    Galileo SSI image file.

    Inputs:
        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """
    SSI.initialize()    # Define everything the first time through; use defaults
                        # unless initialize() is called explicitly.

    # Load the PDS label
    lbl_filespec = filespec.replace('.img', '.LBL')
    recs = pdsparser.PdsLabel.load_file(lbl_filespec)
    label = pdsparser.PdsLabel.from_string(recs).as_dict()

    # Load the dta array
    vic = vicar.VicarImage.from_file(filespec, extraneous='warn')
    vicar_dict = vic.as_dict()

    # Get image metadata
    meta = Metadata(label)

    # Load time-dependent kernels
    Galileo.load_cks(meta.tstart, meta.tstart + meta.exposure)
    Galileo.load_spks(meta.tstart, meta.tstart + meta.exposure)

    # Define the field of view
    FOV = meta.fov(label)

    # Define the mask
    mask = meta.mask(label)

    # Create a Snapshot
    result = oops.obs.Snapshot(('v','u'), meta.tstart, meta.exposure,
                               FOV,
                               path = 'GLL',
                               frame = 'GLL_SCAN_PLATFORM',
                               dict = vicar_dict,       # Add the VICAR dict
                               data = vic.data_2d,      # Add the data array
                               instrument = 'SSI',
                               filter = meta.filter,
                               mask = mask,
                               filespec = filespec,
                               basename = os.path.basename(filespec))

    result.insert_subfield('spice_kernels',
                           Galileo.used_kernels(result.time, 'iss',
                                                return_all_planets))

    return result

#===============================================================================
def from_index(filespec, **parameters):
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

    # Create a list of Snapshot objects
    snapshots = []
    for row_dict in row_dicts:

        tstart = julian.tdb_from_tai(row_dict['START_TIME'])
        texp = max(1.e-3, row_dict['EXPOSURE_DURATION']) / 1000.
        mode = row_dict['INSTRUMENT_MODE_ID']

        name = row_dict['INSTRUMENT_NAME']

        item = oops.obs.Snapshot(('v','u'), tstart, texp,
                                 SSI.fov[mode, False],
                                 'GLL', 'GLL_SSI',
                                 dict = row_dict,       # Add index dictionary
                                 index_dict = row_dict, # Old name
                                 instrument = 'SSI',
                                 sampling = mode)

        item.spice_kernels = Galileo.used_kernels(item.time, 'iss')

        item.filespec = os.path.join(row_dict['VOLUME_ID'],
                                     row_dict['FILE_SPECIFICATION_NAME'])
        item.basename = row_dict['FILE_NAME']

        snapshots.append(item)

    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[ 0]['START_TIME']
    tdb1 = row_dicts[-1]['START_TIME']

    Galileo.load_cks( tdb0, tdb1)
    Galileo.load_spks(tdb0, tdb1)

    return snapshots

#===============================================================================
def initialize(ck='reconstructed', planets=None, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    """Initialize key information about the SSI instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

    Input:
        ck,spk      'predicted', 'reconstructed', or 'none', depending on which
                    kernels are to be used. Defaults are 'reconstructed'. Use
                    'none' if the kernels are to be managed manually.
        planets     A list of planets to pass to define_solar_system. None or
                    0 means all.
        asof        Only use SPICE kernels that existed before this date; None
                    to ignore.
        gapfill     True to include gapfill CKs. False otherwise.
        mst_pck     True to include MST PCKs, which update the rotation models
                    for some of the small moons.
        irregulars  True to include the irregular satellites;
                    False otherwise.
    """
    SSI.initialize(ck=ck, planets=planets, asof=asof,
                   spk=spk, gapfill=gapfill,
                   mst_pck=mst_pck, irregulars=irregulars)


#===============================================================================
class Metadata(object):

    #===========================================================================
    def __init__(self, label):
        """Use the label to assemble the image metadata.

        Input:
            label           The label dictionary.

        Attributes:
            nlines          A Numpy array containing the data in axis order
                            (line, sample).
            nsamples        The time sampling array in (line, sample) axis
                            order, or None if no time backplane is found in
                            the file.
            nframelets

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

    #===========================================================================
    def mask(self, label):
        """Create a Galileo SSI mask.

        Can be useful for debugging.
        """
        window = label.get('CUT_OUT_WINDOW')

        if window is None:
            return None

        grid = np.mgrid[0:self.nlines, 0:self.nsamples] + 1
        mask = np.where((grid[0] >= window[0]) & (grid[0] <= window[2]) &
                        (grid[1] >= window[1]) & (grid[1] <= window[3]), False, True)
        return mask

    #===========================================================================
    def fov(self, label):
        """Use the label to assemble the image metadata.

        Input:
            label           The label dictionary.

        Attributes:
            nlines          A Numpy array containing the data in axis order
                            (line, sample).
            nsamples        The time sampling array in (line, sample) axis
                            order, or None if no time backplane is found in
                            the file.
            nframelets

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
        mode = label['TELEMETRY_FORMAT_ID']
        if mode=='HIS' or mode=='AI8':
            scale = scale*2
            cxy = cxy/2

        self.fov = oops.fov.RadialFOV(scale,
                                      (self.nsamples, self.nlines),
                                      coefft_uv_from_xy=distortion_coeff,
                                      uv_los=(cxy[0], cxy[1]))

        return self.fov



#===============================================================================
class SSI(object):
    """An instance-free class to hold Galileo SSI instrument parameters."""

    instrument_kernel = None
    fov = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """Initialize key information about the SSI instrument.

        Fills in key information about the camera.  Must be called first.
        After the first call, later calls to this function are ignored.

        Input:
            ck,spk      'predicted', 'reconstructed', or 'none', depending on
                        which kernels are to be used. Defaults are
                        'reconstructed'. Use 'none' if the kernels are to be
                        managed manually.
            planets     A list of planets to pass to define_solar_system. None
                        or 0 means all.
            asof        Only use SPICE kernels that existed before this date;
                        None to ignore.
            gapfill     True to include gapfill CKs. False otherwise.
            mst_pck     True to include MST PCKs, which update the rotation
                        models for some of the small moons.
            irregulars  True to include the irregular satellites;
                        False otherwise.
        """

        # Quick exit after first call
        if SSI.initialized:
            return

        # Initialize Galileo
        Galileo.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                           gapfill=gapfill,
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

from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
from oops.backplane.exercise_backplanes import exercise_backplanes
from oops.backplane.unittester_support  import Backplane_Settings


#*******************************************************************************
class Test_Galileo_SSI(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        snapshots = from_index(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                            'galileo/SSI/index.lbl'))
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                          'galileo/SSI/W1575634136_1.IMG'))
        snapshot3940 = snapshots[3940]  #should be same as snapshot

        self.assertTrue(abs(snapshot.time[0] - snapshot3940.time[0]) < 1.e-3)
        self.assertTrue(abs(snapshot.time[1] - snapshot3940.time[1]) < 1.e-3)


#*******************************************************************************
class Test_Galileo_SSI_Backplane_Exercises(unittest.TestCase):

    #===========================================================================
    def runTest(self):

        if Backplane_Settings.NO_EXERCISES:
            self.skipTest('')

        root = os.path.join(TESTDATA_PARENT_DIRECTORY, 'galileo/SSI')
        file = os.path.join(root, 'N1460072401_1.IMG')
        obs = from_file(file)
        exercise_backplanes(obs, use_inventory=True, inventory_border=4,
                                 planet_key='SATURN')


############################################
from oops.backplane.unittester_support import backplane_unittester_args

if __name__ == '__main__':
    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################

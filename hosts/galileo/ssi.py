################################################################################
# hosts/galileo/iss.py
################################################################################
import numpy as np
import julian
import vicar
import pdstable
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

    # Load the VICAR file
    vic = vicar.VicarImage.from_file(filespec)
    vicar_dict = vic.as_dict()

    # Get key information from the header
    tstart = julian.tdb_from_tai(julian.tai_from_iso(vic['START_TIME']))
    texp = max(1.e-3, vicar_dict['EXPOSURE_DURATION']) / 1000.
    mode = vicar_dict['INSTRUMENT_MODE_ID']

    name = vicar_dict['INSTRUMENT_NAME']

    filter1, filter2 = vicar_dict['FILTER_NAME']

    gain_mode = None

    if vicar_dict['GAIN_MODE_ID'][:3] == '215':
        gain_mode = 0
    elif vicar_dict['GAIN_MODE_ID'][:2] == '95':
        gain_mode = 1
    elif vicar_dict['GAIN_MODE_ID'][:2] == '29':
        gain_mode = 2
    elif vicar_dict['GAIN_MODE_ID'][:2] == '12':
        gain_mode = 3

    # Make sure the SPICE kernels are loaded
    Galileo.load_cks( tstart, tstart + texp)
    Galileo.load_spks(tstart, tstart + texp)

    # Create a Snapshot
    result = oops.obs.Snapshot(('v','u'), tstart, texp,
                               SSI.fov[mode, fast_distortion],
                               path = 'GALILEO',
                               frame = 'GALILEO_SSI',
                               dict = vicar_dict,       # Add the VICAR dict
                               data = vic.data_2d,      # Add the data array
                               instrument = 'SSI',
                               sampling = mode,
                               filter1 = filter1,
                               filter2 = filter2,
                               gain_mode = gain_mode)

    result.insert_subfield('spice_kernels',
                           Galileo.used_kernels(result.time, 'iss',
                                                return_all_planets))
    result.insert_subfield('filespec', filespec)
    result.insert_subfield('basename', os.path.basename(filespec))

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
                                 'GALILEO', 'GALILEO_SSI',
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
class SSI(object):
    """An instance-free class to hold Galileo SSI instrument parameters."""

    instrument_kernel = None
    fov = {}
    initialized = False

    # Create a master version of the distortion model from
    #   Owen Jr., W.M., 2003. Galileo SSI Geometric Calibration of April 2003.
    #   JPL IOM 312.E-2003.
    # These polynomials convert from X,Y (radians) to U,V (pixels).

    F = 2002.703    # mm
    E2 = 8.28e-6    # / mm2
    E5 = 5.45e-6    # / mm
    E6 = -19.67e-6  # / mm
    KX = 83.33333   # samples/mm
    KY = 83.3428    # lines/mm

    COEFF = np.zeros((4,4,2))
    COEFF[1,0,0] = KX    * F
    COEFF[3,0,0] = KX*E2 * F**3
    COEFF[1,2,0] = KX*E2 * F**3
    COEFF[1,1,0] = KX*E5 * F**2
    COEFF[2,0,0] = KX*E6 * F**2

    COEFF[0,1,1] = KY    * F
    COEFF[2,1,1] = KY*E2 * F**3
    COEFF[0,3,1] = KY*E2 * F**3
    COEFF[0,2,1] = KY*E5 * F**2
    COEFF[1,1,1] = KY*E6 * F**2

    DISTORTION_COEFF_XY_TO_UV = COEFF

    # Create a master version of the inverse distortion model.
    # These coefficients were computed by numerically solving the above
    # polynomials.
    # These polynomials convert from U,V (pixels) to X,Y (radians).
    # Maximum errors from applying the original distortion model and
    # then inverting:
    #   X DIFF MIN MAX -7.01489382641e-10 1.15568299657e-10
    #   Y DIFF MIN MAX -7.7440587623e-10 8.81658628916e-10
    #   U DIFF MIN MAX -0.00138077186011 1.94695478513e-05
    #   V DIFF MIN MAX -0.000474712833352 0.000563339044731

    INV_COEFF = np.zeros((4,4,2))
    INV_COEFF[:,:,0] = [[ -1.14799845e-10,  7.80494024e-14,  1.73312704e-15, -5.95242349e-19],
                        [  5.99190197e-06, -3.91823615e-13, -7.12054264e-15,  0.00000000e+00],
                        [  1.42455664e-12, -2.15242793e-18,  0.00000000e+00,  0.00000000e+00],
                        [ -7.11199158e-15,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    INV_COEFF[:,:,1] = [[ -4.86573092e-12,  5.99122009e-06, -3.91358768e-13, -7.12774967e-15],
                        [  1.03832114e-13,  1.41728538e-12, -1.55778300e-18,  0.00000000e+00],
                        [  2.03725690e-16, -7.11687723e-15,  0.00000000e+00,  0.00000000e+00],
                        [  1.67086926e-21,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

    DISTORTION_COEFF_UV_TO_XY = INV_COEFF

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

        Galileo.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                           gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Galileo.load_instruments(asof=asof)

        # Load the instrument kernel
        SSI.instrument_kernel = Galileo.spice_instrument_kernel('SSI')[0]

        # Construct a Polynomial FOV
        info = SSI.instrument_kernel['INS']['GALILEO_SSI']

        # Full field of view
        lines = info['PIXEL_LINES']
        samples = info['PIXEL_SAMPLES']

        xfov = info['FOV_REF_ANGLE']
        yfov = info['FOV_CROSS_ANGLE']
        assert info['FOV_ANGLE_UNITS'] == 'DEGREES'

        uscale = np.arctan(np.tan(xfov * oops.RPD) / (samples/2.))
        vscale = np.arctan(np.tan(yfov * oops.RPD) / (lines/2.))

        # Display directions: [u,v] = [right,down]
        full_fov = oops.fov.PolynomialFOV((samples,lines),
                                       coefft_uv_from_xy=
                               SSI.DISTORTION_COEFF_XY_TO_UV,
                                       coefft_xy_from_uv=None)
        full_fov_fast = oops.fov.PolynomialFOV((samples,lines),
                                       coefft_uv_from_xy=
                               SSI.DISTORTION_COEFF_XY_TO_UV,
                                       coefft_xy_from_uv=
                               SSI.DISTORTION_COEFF_UV_TO_XY)
        full_fov_none = oops.fov.FlatFOV((uscale,vscale), (samples,lines))

        # Load the dictionary, include the subsampling modes
        SSI.fov['FULL', False] = full_fov
        SSI.fov['SUM2', False] = oops.fov.SubsampledFOV(full_fov, 2)
        SSI.fov['SUM4', False] = oops.fov.SubsampledFOV(full_fov, 4)
        SSI.fov['FULL', True] = full_fov_fast
        SSI.fov['SUM2', True] = oops.fov.SubsampledFOV(full_fov_fast, 2)
        SSI.fov['SUM4', True] = oops.fov.SubsampledFOV(full_fov_fast, 4)
        SSI.fov['FULL', None] = full_fov_none
        SSI.fov['SUM2', None] = oops.fov.SubsampledFOV(full_fov_none, 2)
        SSI.fov['SUM4', None] = oops.fov.SubsampledFOV(full_fov_none, 4)

        SSI.fov['FULL'] = full_fov_none
        SSI.fov['SUM2'] = oops.fov.SubsampledFOV(full_fov_none, 2)
        SSI.fov['SUM4'] = oops.fov.SubsampledFOV(full_fov_none, 4)

        # Construct a SpiceFrame
        # Deal with the fact that the instrument's internal
        # coordinate  system is rotated 180 degrees
        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        flipped = oops.frame.SpiceFrame('GALILEO_SSI',
                                        frame_id='GALILEO_SSI_FLIPPED')
        frame = oops.frame.Cmatrix(rot180, flipped,
                                       frame_id='GALILEO_SSI')

        SSI.initialized = True

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

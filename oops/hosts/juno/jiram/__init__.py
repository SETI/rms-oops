################################################################################
# oops/inst/juno/jiram/__init__.py
################################################################################

import numpy as np
import julian
import cspyce
from polymath import *
import os.path
import pdsparser
import oops

from oops.hosts.juno import Juno
################################################################################
# Standard class methods
################################################################################
def from_file(filespec, return_all_planets=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    JIRAM image or spectrum file.

    Inputs:
        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """
    JIRAM.initialize()    # Define everything the first time through; use
                          # defaults unless initialize() is called explicitly.

    # Load the PDS label
    base = os.path.splitext(filespec)[0]
    lbl_filespec = base + '.LBL'
    recs = pdsparser.PdsLabel.load_file(lbl_filespec)
    label = pdsparser.PdsLabel.from_string(recs).as_dict()

    # Get common metadata
    meta = Metadata(label)

    # Load time-dependent kernels
    Juno.load_cks(meta.tstart, meta.tstart + 3600.)
    Juno.load_spks(meta.tstart, meta.tstart + 3600.)

    # Determine which observation type and load data
    ext = os.path.splitext(filespec)[1]

    # Image
    if ext.upper() == '.IMG':
        from . import img
        return img.from_file(filespec, label,
                             return_all_planets=False, **parameters)

    # Spectrum
    if ext.upper() == '.DAT':
        from . import spe
        return spe.from_file(filespec, label,
                             return_all_planets=False, **parameters)

    return None


#*******************************************************************************
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

    #===========================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """Initialize key information about the JIRAM instrument.

        Must be called first. After the first call, later calls to this function
        are ignored.

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
        if JIRAM.initialized: return

        # initialize Juno
        Juno.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                        gapfill=gapfill,
                        mst_pck=mst_pck, irregulars=irregulars)
        Juno.load_instruments(asof=asof)

        JIRAM.initialized = True

    #===========================================================================
    @staticmethod
    def create_frame(time, name):
        """Create a frame for a JIRAM component.

        Input:
            time  time at which to define the inertialy fixed mirror-corrected
                  frame.

            name  name of the component.
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

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal JIRAM parameters.

        Can be useful for debugging.
        """
        JIRAM.instrument_kernel = None
        JIRAM.fovs = {}
        JIRAM.initialized = False

        Juno.reset()

################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path
import oops.backplane.gold_master as gm
import oops.hosts.juno.jiram as jiram

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


#===============================================================================
class Test_Juno_JIRAM_GoldMaster(unittest.TestCase):

    #===========================================================================
    def setUp(self):
        gm.define_standard_obs('JIR_IMG_RDR_2013282T133843_V03',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2000/DATA/'
                                       'JIR_IMG_RDR_2013282T133843_V03.IMG'),
                index   = 1,
                module  = 'hosts.juno.jiram',
                planet  = '',
                moon    = 'MOON',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_IMG_RDR_2017244T104633_V01',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2008/DATA/'
                                       'JIR_IMG_RDR_2017244T104633_V01.IMG'),
                index   = 1,
                module  = 'hosts.juno.jiram',
                planet  = '',
                moon    = 'EUROPA',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_IMG_RDR_2018197T055537_V01',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2014/DATA/'
                                       'JIR_IMG_RDR_2018197T055537_V01.IMG'),
                index   = 0,
                module  = 'hosts.juno.jiram',
                planet  = 'JUPITER',
                moon    = '',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})

        gm.define_standard_obs('JIR_SPE_RDR_2013282T133845_V03',
                obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                       'juno/jiram/JNOJIR_2000/DATA/'
                                       'JIR_SPE_RDR_2013282T133845_V03.DAT'),
                index   = 0,
                module  = 'hosts.juno.jiram',
                planet  = '',
                moon    = 'MOON',
                ring    = '',
                kwargs  = {'inventory':False, 'border':4})


    #===========================================================================
    def test_1(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2013282T133843_V03')

    #===========================================================================
    def test_2(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2017244T104633_V01')

    #===========================================================================
    def test_3(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_IMG_RDR_2018197T055537_V01')

    #===========================================================================
    def test_4(self):
        gm.execute_standard_unittest(unittest.TestCase, 'JIR_SPE_RDR_2013282T133845_V03')


##############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################

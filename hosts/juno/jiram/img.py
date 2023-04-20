################################################################################
# oops/inst/juno/jiram/img.py
################################################################################

import numpy as np
import julian
import cspyce
from polymath import *
import os.path
import oops

from hosts.juno.jiram import JIRAM

################################################################################
# Standard class methods
################################################################################

#===============================================================================
def from_file(filespec, label, fast_distortion=True,
                               return_all_planets=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    JIRAM image or spectrum file.

    Inputs:
        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """

    # Get metadata
    meta = Metadata(label)

    # Define everything the first time through
    IMG.initialize(meta.tstart)

    # Load the data array as separate framelets, with associated labels
    (framelets, flabels) = _load_data(filespec, label, meta)

    # Construct a Snapshot for each framelet
    obs = []
    for i in range(meta.nframelets):
        fmeta = Metadata(flabels[i])

        item = oops.obs.Snapshot(('v','u'),
                                 fmeta.tstart, fmeta.exposure, fmeta.fov,
                                 'JUNO', 'JUNO_JIRAM_I_' + fmeta.filter_frame,
                                 instrument = 'JIRAM_I',
                                 filter = fmeta.filter,
                                 data = framelets[:,:,i])

#        item.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(item.time, 'jiram', return_all_planets))
        item.insert_subfield('filespec', filespec)
        item.insert_subfield('basename', os.path.basename(filespec))
        obs.append(item)

    return obs

#===============================================================================
def _load_data(filespec, label, meta):
    """Load the data array from the file and splits into individual framelets.

    Input:
        filespec        Full path to the data file.
        label           Label for composite image.
        meta            Image Metadata object.

    Return:             (framelets, framelet_labels)
        framelets       A Numpy array containing the individual frames in
                        axis order (line, sample, framelet #).
        framelet_labels List of labels for each framelet.
    """

    # Read data
    # seems like this should be handled in a readpds-style function somewhere
    data = np.fromfile(filespec, dtype='<f4').reshape(meta.nlines,meta.nsamples)

    # Split into framelets:
    #   - Add framelet number and filter index to label
    #   - Change image dimensions in label
    filters = meta.filter
    nf = len(filters)
    framelets = np.empty([meta.frlines,meta.nsamples,meta.nframelets])
    framelet_labels = []

    for i in range(meta.nframelets):
        framelets[:,:,i] = data[meta.frlines*i:meta.frlines*(i+1),:]

        framelet_label = label.copy()
        frn = i//nf
        ifl = i%nf

        framelet_label['FRAMELET'] = {}
        framelet_label['FRAMELET']['FRAME_NUMBER'] = frn
        framelet_label['FRAMELET']['FRAMELET_NUMBER'] = i
        framelet_label['FRAMELET']['FRAMELET_FILTER'] = filters[ifl]

        label['LINES'] = meta.frlines
        label['LINE_SAMPLES'] = meta.nsamples

        framelet_labels.append(framelet_label)

    return (framelets, framelet_labels)


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

        # image dimensions
        self.nlines = label['FILE']['IMAGE']['LINES']
        self.nsamples = label['FILE']['IMAGE']['LINE_SAMPLES']
        self.frlines = 128
        self.nframelets = int(self.nlines/self.frlines)
        self.size = [self.nsamples, self.frlines]

        # Exposure time
        self.exposure = 0
        try:
            self.exposure = label['EXPOSURE_DURATION']
        except:
##            print('No exposure information')
            self.exposure = 1.      # TODO: This should go away after the labels are
                                    # redelivered

        # Filters
        self.filter = []

        Lparm = label['L_BAND_PARAMETERS']['LINE_FIRST_PIXEL']
        if type(Lparm) is int: self.filter.append('L_BAND')

        Mparm = label['M_BAND_PARAMETERS']['LINE_FIRST_PIXEL']
        if type(Mparm) is int: self.filter.append('M_BAND')

        # Default timing for unprocessed frame
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))
        if self.exposure == 0: self.exposure = self.tstop - self.tstart

        # target
        self.target = label['TARGET_NAME']

        # Framelet-specific parameters, if applicable
        if 'FRAMELET' in label.keys():
            frn = label['FRAMELET']['FRAME_NUMBER']

            # Filter
            self.filter = label['FRAMELET']['FRAMELET_FILTER']

            # Filter-specific instrument id
            if self.filter == 'L_BAND':
                self.instc = -61411
                self.filter_frame = 'LBAND'
            if self.filter == 'M_BAND':
                self.instc = -61412
                self.filter_frame = 'MBAND'
            sinstc = str(self.instc)

            # FOV
            prefix = 'INS' + sinstc
            cross_angle = cspyce.gdpool(prefix + '_FOV_CROSS_ANGLE', 0)[0]
            fo = cspyce.gdpool(prefix + '_FOCAL_LENGTH', 0)[0]
            px = cspyce.gdpool(prefix + '_PIXEL_SIZE', 0)[0]
            cxy = cspyce.gdpool(prefix + '_CCD_CENTER', 0)
            scale = px/1000/fo

            self.fov = oops.fov.FlatFOV(scale,
                                        (self.nsamples, self.frlines), cxy)

        return


#*******************************************************************************
class IMG(object):
    """An instance-free class to hold IMG instrument parameters."""

    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(time, ck='reconstructed', planets=None, asof=None,
                         spk='reconstructed', gapfill=True,
                         mst_pck=True, irregulars=True):
        """Initialize key information about the IMG instrument.

        Must be called first. After the first call, later calls to this function
        are ignored.

        Input:
            time        time at which to define the inertialy fixed mirror-
                        corrected frame.
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

        # Quick exit after first call
        if IMG.initialized: return

        # initialize JIRAM
        JIRAM.initialize(ck=ck, planets=planets, asof=asof,
                         spk=spk, gapfill=gapfill,
                         mst_pck=mst_pck, irregulars=irregulars)

        # Construct the SpiceFrames
        JIRAM.create_frame(time, 'I_MBAND')
        JIRAM.create_frame(time, 'I_LBAND')

        IMG.initialized = True

    #===========================================================================
    @staticmethod
    def reset():
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Reset the internal IMG parameters.

        Can be useful for debugging.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        IMG.initialized = False

        JIRAM.reset()

################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path
import oops.backplane.gold_master as gm
import hosts.juno.jiram as jiram

from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY


#===============================================================================
class Test_Juno_JIRAM_IMG(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        pass


##############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

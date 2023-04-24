################################################################################
# oops/inst/juno/junocam.py
################################################################################

import re
import numpy as np
import julian
import cspyce
from polymath import *
### Avoid from ... import *; just import what you need.
import os.path
import pdsparser
import oops

from hosts.juno import Juno
### BTW, we have standards for how to order imports. Often not important, but
### this is a long enough list.

################################################################################
# Standard class methods
################################################################################

#===============================================================================
### Avoid two horizontal separators in a row. One should suffice.
def from_file(filespec, fast_distortion=True,
              return_all_planets=False, snap=False, **parameters):
    """A general, static method to return a Pushframe object based on a given
    JUNOCAM image file.

    Inputs:
        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.

        snap                True to model the image as a Snapshot rather than as
                            a TimedImage.
    """
    JUNOCAM.initialize()    # Define everything the first time through; use
                            # defaults unless initialize() is called explicitly.
### I think we recommend a blank line after a multi-line docstring.

    # Load the PDS label
    lbl_filespec = filespec.replace('.img', '.LBL')
### This failed for me because the files come off the PDS archive volumes in
### upper case, so they end in '.IMG', not 'img'. You need to find a way to make
### this work regardless of the case of either file extension.
    recs = pdsparser.PdsLabel.load_file(lbl_filespec)
    label = pdsparser.PdsLabel.from_string(recs).as_dict()

    # Get composite image metadata
    meta = Metadata(label)
### A class named "Metadata" makes me nervous--it's too generic. If this class is
### unique to this file, and not intended for use elsewhere, then it should have
### an underscore in front: "_Metadata".

    # Load the data array as separate framelets, with associated labels
    (framelets, flabels) = _load_data(filespec, label, meta)

    # Load time-dependent kernels
    Juno.load_cks(meta.tstart0, meta.tstart0 + 3600.)
    Juno.load_spks(meta.tstart0, meta.tstart0 + 3600.)
### Just curious--is there a reason to define the required time frame as one hour?
### You do have the stop time, after all.

    # Construct a Pushframe for each framelet
    obs = []
    for i in range(meta.nframelets):
        fmeta = Metadata(flabels[i])

        if snap:
            item = oops.obs.Snapshot(('v','u'),
                                     fmeta.tstart, fmeta.tdi_texp,
                                     fov = fmeta.fov,
                                     path = 'JUNO',
                                     frame = 'JUNO_JUNOCAM',
                                     instrument = 'JUNOCAM',
                                     filter = fmeta.filter,
                                     data = framelets[:,:,i])
### See discussion below in _load_data.
### BTW, having the framelet index i last is not a good idea, because it means
### that framelets[:,:,i] is a discontiguous array. Contiguous arrays are often
### much more efficient in NumPy, for a variety of reasons.
### None of this will matter if you just reshape the array, in which case the
### index will be [i,:,:], or just [i].

        if not snap:
### Should be "else"

            tdi_lines = fmeta.fov.uv_shape.vals[0]
            cadence = oops.cadence.TDICadence(tdi_lines, fmeta.tstart,
                                              fmeta.tdi_texp, fmeta.tdi_stages)
            fov = oops.fov.TDIFOV(fmeta.fov, tstop=fmeta.tstop,
                                  tdi_texp=fmeta.tdi_texp, tdi_axis='-u')
            item = oops.obs.TimedImage(('v','ut'),
                                       cadence = cadence,
                                       fov = fov,
                                       path = 'JUNO',
                                       frame = 'JUNO_JUNOCAM',
                                       instrument = 'JUNOCAM',
                                       filter = fmeta.filter,
                                       data = framelets[:,:,i])
### The fov value of fov.meta was wrong here. Each framelet needs its own
### TDIFOV, not the generic fov, and each will have its own unique tstop value:
###     fov = TDIFOV(fmeta.fov, tstop, tdi_texp=fmeta.tdi_texp, tdi_axis='-u')
### where tstop is the end time of that individual framelet. I suspect it is
### close enough  that the backplanes look reasonable, but it should be fixed
### and checked.

#        item.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(item.time, 'junocam', return_all_planets))
        item.insert_subfield('filespec', filespec)
        item.insert_subfield('basename', os.path.basename(filespec))
        obs.append(item)

    return obs

#===============================================================================
def initialize(ck='reconstructed', planets=None, offset_wac=True, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    """Initialize key information about the JUNOCAM instrument.

    Must be called first. After the first call, later calls to this function
    are ignored.

    Input:
        ck,spk      'predicted', 'reconstructed', or 'none', depending on which
                    kernels are to be used. Defaults are 'reconstructed'. Use
                    'none' if the kernels are to be managed manually.
        planets     A list of planets to pass to define_solar_system. None or
                    0 means all.
        offset_wac  True to offset the WAC frame relative to the NAC frame as
                    determined by star positions.
        asof        Only use SPICE kernels that existed before this date; None
                    to ignore.
        gapfill     True to include gapfill CKs. False otherwise.
        mst_pck     True to include MST PCKs, which update the rotation models
                    for some of the small moons.
        irregulars  True to include the irregular satellites;
                    False otherwise.
    """

    JUNOCAM.initialize(ck=ck, planets=planets, offset_wac=offset_wac, asof=asof,
                   spk=spk, gapfill=gapfill,
                   mst_pck=mst_pck, irregulars=irregulars)

### mst_pck has no relevance here as an option; it's specific to Saturn.
### unless JunoCam ever targeted a Jupiter irregular, you can also omit that option.
### Really, I think you only need to ck and spk options, and only so we can be
### ready for some future date when/if this info is regenerated.

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
    bits = label['IMAGE']['SAMPLE_BITS']
    dtype = '>u' +str(int(bits/8))
### Add space before "str"; change "int(bits/8)" to "bits//8" (floor division is handy!)
    data = np.fromfile(filespec, dtype=dtype).reshape(meta.nlines,meta.nsamples)
### Add a space after the comma

### BTW, unless the things around operators are extremely short (like "bits//8"),
### you should surround all operators with spaces. Even in this case, I would
### probably write "bits // 8".

### Your re-shaping of the framelets below is unnecessary! This is much better:
###     framelets = data.reshape((meta.nframelets, 128, meta.nsamples))
### or even...
###     framelets = data.reshape((-1, 128, meta.nsamples))
### because a "-1" axis in reshape() just figures out whatever it needs to be.
### BTW, reshape does not touch the data array; it just changes the way it is indexed,
### aka the "strides", which is a very very fast operation. This is one of the great
### features of NumPy and we use it a lot.

    # Split into framelets:
    #   - Add framelet number and filter index to label
    #   - Change image dimensions in label
    nf = len(meta.filter)
    framelets = np.empty([meta.frlines,meta.nsamples,meta.nframelets])
### You should almost always use spaces after commas. Really the only common
### exceptions are tuples containing short integers (e.g., "shape = (128,128)"
### and indexing with very short variable names (e.g., "x = array[i,j,3]"

### As noted above, you don't need to create a new array at all. But if you did,
### please preserve the dtype of input arrays. There's no reason to convert them--
### that's up to the user.

    framelet_labels = []

    for i in range(meta.nframelets):
        framelets[:,:,i] = data[meta.frlines*i:meta.frlines*(i+1),:]
### See above. Note that framelets is now indexed with the frame number first.

        framelet_label = label.copy()
        frn = i//nf
        ifl = i%nf
### Add spaces around operators

        framelet_label['FRAMELET'] = {}
        framelet_label['FRAMELET']['FRAME_NUMBER'] = frn
        framelet_label['FRAMELET']['FRAMELET_NUMBER'] = i

        filters = label['FILTER_NAME']
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
        self.nlines = label['IMAGE']['LINES']
        self.nsamples = label['IMAGE']['LINE_SAMPLES']
        self.frlines = 128
        self.nframelets = int(self.nlines/self.frlines)
### Change to "self.nlines // self.frlines". There's no need for "int".

        # Exposure time
        exposure_ms = label['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        # Filters
        self.filter = label['FILTER_NAME']

        # Default timing for unprocessed frame
        self.tinter = label['INTERFRAME_DELAY']
        self.tinter0 = self.tinter

        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstart0 = self.tstart
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))

        self.tdi_stages = label['JNO:TDI_STAGES_COUNT']
        self.tdi_texp = self.exposure/self.tdi_stages
        if self.exposure < self.tdi_texp: self.tdi_texp = self.exposure
### use two lines

        # target
        self.target = label['TARGET_NAME']

        # Framelet-specific parameters, if applicable
        if 'FRAMELET' in label.keys():
            frn = label['FRAMELET']['FRAME_NUMBER']

            # Filter
            self.filter = label['FRAMELET']['FRAMELET_FILTER']

            # Filter-specific instrument id
            if self.filter == 'RED': self.instc = -61503
            if self.filter == 'GREEN': self.instc = -61502
            if self.filter == 'BLUE': self.instc = -61501
            if self.filter == 'METHANE': self.instc = -61504
            sinstc = str(self.instc)
### Convert this to a dictionary lookup.

            # Timing
            prefix = 'INS' + sinstc
            delta_var = prefix + '_INTERFRAME_DELTA'
            bias_var = prefix + '_START_TIME_BIAS'

            self.delta = cspyce.gdpool(delta_var, 0)[0]
            self.bias = cspyce.gdpool(bias_var, 0)[0]

            self.tinter = self.tinter0 + self.delta
            self.tstart = self.tstart0 + self.bias + frn*self.tinter

            self.tstop = self.tstart + self.exposure

            # FOV
            k1_var = 'INS' + sinstc + '_DISTORTION_K1'
            k2_var = 'INS' + sinstc + '_DISTORTION_K2'
            cx_var = 'INS' + sinstc + '_DISTORTION_X'
            cy_var = 'INS' + sinstc + '_DISTORTION_Y'
            fo_var = 'INS' + sinstc + '_FOCAL_LENGTH'
            px_var = 'INS' + sinstc + '_PIXEL_SIZE'

            k1 = cspyce.gdpool(k1_var, 0)[0]
            k2 = cspyce.gdpool(k2_var, 0)[0]
            cx = cspyce.gdpool(cx_var, 0)[0]
            cy = cspyce.gdpool(cy_var, 0)[0]
            fo = cspyce.gdpool(fo_var, 0)[0]
            px = cspyce.gdpool(px_var, 0)[0]

            # Check RATIONALE_DESC in label for modification to methane
            # DISTORTION_Y
            if self.filter== 'METHANE': cy = self.update_cy(label, cy)
### line break after colon

            # Construct FOV
            scale = px/fo
            distortion_coeff = [1,0,k1,0,k2]

            self.fov = oops.fov.BarrelFOV(scale,
                                          (self.nsamples, self.frlines),
                                          coefft_uv_from_xy=distortion_coeff,
                                          uv_los=(cx, cy))

        return

    #===========================================================================
    def update_cy(self, label, cy):
        """Look at label RATIONALE_DESC for a correction to DISTORTION_Y for
        some methane images.

        Input:
            label           The label dictionary.
            cy              Uncorrected cy value.

        Output:
            cy              Corrected cy value.

        """
        desc = label['RATIONALE_DESC']
        desc = re.sub('\s+',' ', desc)                     # compress whitespace
### There's a standard way to do this without re:
###     desc = ' '.join(desc.split())
        kv = desc.partition('INS-61504_DISTORTION_Y = ')   # parse keyword
        return float(kv[2].split()[0])                     # parse/convert value

### OK, if all you need to do is get the value following the equal sign, a single
### regular expression will handle the whole task:
###     match = re.match(r'.*\n.*INS-61504_DISTORTION_Y = ([\d\.]+)', RATIONALE_DESC)
###     return float(match.group(1))
###
### I would probably define this outside the function:
###     RATIONALE_RE = re.compile(r'.*\n.*INS-61504_DISTORTION_Y = ([\d\.]+)')
### then...
###     match = _Metadata.RATIONALE_RE.match(RATIONALE_DESC)
###     return float(match.group(1))

#*******************************************************************************
class JUNOCAM(object):
    """A instance-free class to hold JUNOCAM instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    #===========================================================================
    @staticmethod
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        """
        Initialize key information about the JUNOCAM instrument; fill in key
        information about the WAC and NAC.

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
### See my comments above about the extraneous input parameters

        # Quick exit after first call
        if JUNOCAM.initialized: return

        # Initialize Juno
        Juno.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                        gapfill=gapfill,
                        mst_pck=mst_pck, irregulars=irregulars)
        Juno.load_instruments(asof=asof)

        # Construct the SpiceFrame
        ignore = oops.frame.SpiceFrame("JUNO_JUNOCAM")

        JUNOCAM.initialized = True

    #===========================================================================
    @staticmethod
    def reset():
        """Reset the internal JUNOCAM parameters.

        Can be useful for debugging.
        """

        JUNOCAM.instrument_kernel = None
        JUNOCAM.fovs = {}
        JUNOCAM.initialized = False

        Juno.reset()

################################################################################
# UNIT TESTS
################################################################################
import os
import unittest
import oops.backplane.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

#===============================================================================
class Test_Juno_Junocam_GoldMaster(unittest.TestCase):

    #===========================================================================
    def setUp(self):
        from hosts.juno.junocam import standard_obs

    #===========================================================================
    def test_JNCR_2016347_03C00192_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2016347_03C00192_V01')

    #===========================================================================
    def test_JNCR_2020366_31C00065_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2020366_31C00065_V01')

    #===========================================================================
    def test_JNCR_2019096_19M00012_V02(self):
        gm.execute_as_unittest(self, 'JNCR_2019096_19M00012_V02')

    #===========================================================================
    def test_JNCR_2019149_20G00008_V01(self):
        gm.execute_as_unittest(self, 'JNCR_2019149_20G00008_V01')



##############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

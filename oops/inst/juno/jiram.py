################################################################################
# oops/inst/juno/jiram.py
################################################################################

from IPython import embed   ## TODO: remove

import numpy as np
import julian
import pdstable
import cspyce
import oops
from polymath import *
import os.path
import pdsparser

from oops.inst.juno.juno_ import Juno

################################################################################
# Standard class methods
################################################################################

#===============================================================================
# from_file
#===============================================================================
def from_file(filespec, fast_distortion=True,
              return_all_planets=False, **parameters):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A general, static method to return a Snapshot object based on a given
    JIRAM image file.

    Inputs:
        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    JIRAM.initialize()    # Define everything the first time through; use 
                            # defaults unless initialize() is called explicitly.

    #-----------------------
    # Load the PDS label 
    #-----------------------
    lbl_filespec = filespec.replace(".img", ".LBL")
    recs = pdsparser.PdsLabel.load_file(lbl_filespec)
    label = pdsparser.PdsLabel.from_string(recs).as_dict()

    #---------------------------------
    # Get composite image metadata 
    #---------------------------------
    meta = Metadata(label)

    #------------------------------------------------------------------
    # Load the data array as separate framelets, with associated labels
    #------------------------------------------------------------------
    (framelets, flabels) = _load_data(filespec, label, meta)

    #--------------------------------
    # Load time-dependent kernels 
    #--------------------------------
    Juno.load_cks(meta.tstart, meta.tstart + 3600.)
    Juno.load_spks(meta.tstart, meta.tstart + 3600.)

    #-----------------------------------------
    # Construct a Snapshot for each framelet
    #-----------------------------------------

    obs = []
    for i in range(meta.nframelets):
        fmeta = Metadata(flabels[i])

        item = (oops.obs.Snapshot(("v","u"), 
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#                                 {'tstart':fmeta.tstart, 'texp':fmeta.exposure}, fmeta.fov,
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
                                 fmeta.tstart, fmeta.exposure, fmeta.fov,
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

                                 "JUNO", "JUNO_JIRAM_I_" + fmeta.filter_frame, 
                                 instrument = "JIRAM_I",
                                 filter = fmeta.filter, 
                                 data = framelets[:,:,i]))


#        item.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(item.time, 'jiram', return_all_planets))


        item.insert_subfield('filespec', filespec)
        item.insert_subfield('basename', os.path.basename(filespec))
        obs.append(item)

    return obs

#===============================================================================



#===============================================================================
# initialize
#===============================================================================
def initialize(ck='reconstructed', planets=None, offset_wac=True, asof=None,
               spk='reconstructed', gapfill=True,
               mst_pck=True, irregulars=True):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Initialize key information about the JIRAM instrument.

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
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    JIRAM.initialize(ck=ck, planets=planets, offset_wac=offset_wac, asof=asof,
                   spk=spk, gapfill=gapfill,
                   mst_pck=mst_pck, irregulars=irregulars)

#===============================================================================



#===============================================================================
# _load_data
#===============================================================================
def _load_data(filespec, label, meta):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Loads the data array from the file and splits into individual framelets. 

    Input:
        filespec        Full path to the data file.
        label           Label for composite image.
        meta            Image Metadata object.

    Return:             (framelets, framelet_labels)
        framelets       A Numpy array containing the individual frames in 
                        axis order (line, sample, framelet #).
        framelet_labels List of labels for each framelet.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #----------------
    # Read data 
    #----------------
    # seems like this should be handled in a readpds-style function somewhere
    data = np.fromfile(filespec, dtype='<f4').reshape(meta.nlines,meta.nsamples)

    #--------------------------------------------------------
    # Split into framelets:
    #   - Add framelet number and filter index to label
    #   - Change image dimensions in label
    #--------------------------------------------------------
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
#===============================================================================



#*******************************************************************************
# Metadata 
#*******************************************************************************
class Metadata(object):

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, label):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Uses the label to assemble the image metadata.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------
        # image dimensions
        #---------------------
        self.nlines = label['FILE']['IMAGE']['LINES']
        self.nsamples = label['FILE']['IMAGE']['LINE_SAMPLES']
        self.frlines = 128
        self.nframelets = int(self.nlines/self.frlines)
        self.size = [self.nsamples, self.frlines]

        #-----------------
        # Exposure time
        #-----------------
        self.exposure = label['EXPOSURE_DURATION']

        #-------------
        # Filters
        #-------------
        self.filter = ['L_BAND', 'M_BAND']

        #--------------------------------------------
        # Default timing for unprocessed frame
        #--------------------------------------------
        self.tstart = julian.tdb_from_tai(
                        julian.tai_from_iso(label['START_TIME']))
        self.tstop = julian.tdb_from_tai(
                       julian.tai_from_iso(label['STOP_TIME']))
      
        #-------------
        # target
        #-------------
        self.target = label['TARGET_NAME']

        #----------------------------------------------
        # Framelet-specific parameters, if applicable
        #----------------------------------------------
        if 'FRAMELET' in label.keys():
            frn = label['FRAMELET']['FRAME_NUMBER']
            
            #- - - - - - - - - - - - - - - - - - - - 
            # Filter
            #- - - - - - - - - - - - - - - - - - - - 
            self.filter = label['FRAMELET']['FRAMELET_FILTER']

            #- - - - - - - - - - - - - - - - - - - - 
            # Filter-specific instrument id
            #- - - - - - - - - - - - - - - - - - - - 
            if self.filter == 'L_BAND': 
                self.instc = -61411
                self.filter_frame = 'LBAND'
            if self.filter == 'M_BAND': 
                self.instc = -61412
                self.filter_frame = 'MBAND'
            sinstc = str(self.instc)

            #- - - - - - - - - - - - - - - - - - - - - 
            # Timing
            #- - - - - - - - - - - - - - - - - - - - - 
            prefix = 'INS' + sinstc

            #- - - - - - - 
            # FOV
            #- - - - - - - 
            cross_angle = cspyce.gdpool(prefix + '_FOV_CROSS_ANGLE', 0)[0]
            fo = cspyce.gdpool(prefix + '_FOCAL_LENGTH', 0)[0]
            px = cspyce.gdpool(prefix + '_PIXEL_SIZE', 0)[0]
            cxy = cspyce.gdpool(prefix + '_CCD_CENTER', 0)
            scale = px/1000/fo

            self.fov = oops.fov.FlatFOV(scale, 
	                                (self.nsamples, self.frlines), cxy)

        return
    #===========================================================================

#*******************************************************************************





#*******************************************************************************
# JIRAM 
#*******************************************************************************
class JIRAM(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A instance-free class to hold JIRAM instrument parameters.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    #===========================================================================
    # initialize
    #===========================================================================
    def initialize(ck='reconstructed', planets=None, asof=None,
                   spk='reconstructed', gapfill=True,
                   mst_pck=True, irregulars=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Initialize key information about the JIRAM instrument; fill in key
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Quick exit after first call
        if JIRAM.initialized: return

        Juno.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                           gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Juno.load_instruments(asof=asof)

        #-----------------------------------
        # Construct the SpiceFrames
        #-----------------------------------
        rot90 = oops.Matrix3([[0,-1,0],[-1,0,0],[0,0,1]])
        mband_rot = oops.frame.SpiceFrame("JUNO_JIRAM_I_MBAND", 
	                                            id="JUNO_JIRAM_I_MBAND_ROT")
        mband = oops.frame.Cmatrix(rot90, mband_rot, id="JUNO_JIRAM_I_MBAND")

        lband_rot = oops.frame.SpiceFrame("JUNO_JIRAM_I_LBAND", 
	                                            id="JUNO_JIRAM_I_LBAND_ROT")
        lband = oops.frame.Cmatrix(rot90, lband_rot, id="JUNO_JIRAM_I_LBAND")


        JIRAM.initialized = True
    #===========================================================================



    #===========================================================================
    # reset
    #===========================================================================
    @staticmethod
    def reset():
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Resets the internal JIRAM parameters. Can be useful for
        debugging.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        JIRAM.instrument_kernel = None
        JIRAM.fovs = {}
        JIRAM.initialized = False

        Juno.reset()
    #============================================================================

#*****************************************************************************




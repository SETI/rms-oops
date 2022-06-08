################################################################################
# oops/inst/juno/junocam.py
################################################################################

from IPython import embed   ## TODO: remove

import numpy as np
import julian
from PIL import Image
import pdstable
import json
import cspyce
import oops
from polymath import *
import os.path
import pdsparser

from oops.inst.juno.juno_ import Juno
from oops import TWOPI, HALFPI, PI

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
    A general, static method to return a Pushframe object based on a given
    JUNOCAM image file.

    Inputs:
        fast_distortion     True to use a pre-inverted polynomial;
                            False to use a dynamically solved polynomial;
                            None to use a FlatFOV.

        return_all_planets  Include kernels for all planets not just
                            Jupiter or Saturn.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    JUNOCAM.initialize()    # Define everything the first time through; use 
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
    Juno.load_cks(meta.tstart0, meta.tstart0 + 3600.)
    Juno.load_spks(meta.tstart0, meta.tstart0 + 3600.)

    #-----------------------------------------
    # Construct a Pushframe for each framelet
    #-----------------------------------------

    snap = False


    obs = []
    for i in range(meta.nframelets):
        fmeta = Metadata(flabels[i])

        if snap:
            item = (oops.obs.Snapshot(("v","u"), 
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#                                 {'tstart':fmeta.tstart, 'texp':fmeta.tdi_texp}, fmeta.fov,
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#                                 fmeta.tstart, fmeta.exposure, fmeta.fov,
                                 fmeta.tstart, fmeta.tdi_texp, fmeta.fov,
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

                                 "JUNO", "JUNO_JUNOCAM", 
                                 instrument = "JUNOCAM",
                                 filter = fmeta.filter, 
                                 data = framelets[:,:,i]))




        if not snap:
            item = (oops.obs.Pushframe(("vt","u"), 
                                {'nexp':fmeta.tdi_stages, 
                                 'tstart':fmeta.tstart, 
                                 'texp':fmeta.tdi_texp}, 
                                 fmeta.fov,
                                 "JUNO", "JUNO_JUNOCAM", 
                                 instrument = "JUNOCAM",
                                 filter = fmeta.filter, 
                                 data = framelets[:,:,i]))

#            item = (oops.obs.Pushbroom(("vt","u"), Pair.ONES, 
#                                 cadence, fmeta.fov,
#                                 "JUNO", "JUNO_JUNOCAM", 
#                                 instrument = "JUNOCAM",
#                                 filter = fmeta.filter, 
#                                 data = framelets[:,:,i]))




#        item.insert_subfield('spice_kernels', \
#                   Juno.used_kernels(item.time, 'junocam', return_all_planets))


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
    Initialize key information about the JUNOCAM instrument.

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
    JUNOCAM.initialize(ck=ck, planets=planets, offset_wac=offset_wac, asof=asof,
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
    data = np.fromfile(filespec, \
                      dtype='>u2').reshape(meta.nlines,meta.nsamples)

    #--------------------------------------------------------
    # Split into framelets:
    #   - Add framelet number and filter index to label
    #   - Change image dimensions in label
    #--------------------------------------------------------
    nf = len(meta.filter)
    framelets = np.empty([meta.frlines,meta.nsamples,meta.nframelets])
    framelet_labels = []
    
    for i in range(meta.nframelets):
        framelets[:,:,i] = data[meta.frlines*i:meta.frlines*(i+1),:]

        framelet_label = label.copy()
        frn = i//nf
        ifl = i%nf
        framelet_label['FRAME_NUMBER'] = frn
        framelet_label['FRAMELET_NUMBER'] = i
        framelet_label['FRAMELET_FILTER_INDEX'] = ifl

        filters = label['FILTER_NAME']
        framelet_label['FRAMELET_FILTER'] = filters[ifl]

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
        self.nlines = label['IMAGE']['LINES']
        self.nsamples = label['IMAGE']['LINE_SAMPLES']
        self.frlines = 128
        self.nframelets = int(self.nlines/self.frlines)

        #-----------------
        # Exposure time
        #-----------------
        exposure_ms = label['EXPOSURE_DURATION']
        self.exposure = exposure_ms/1000.

        #-------------
        # Filters
        #-------------
        self.filter = label['FILTER_NAME']
 
        #--------------------------------------------
        # Default timing for unprocessed frame
        #--------------------------------------------
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
      
        #-------------
        # target
        #-------------
        self.target = label['TARGET_NAME']

        #----------------------------------------------
        # Framelet-specific parameters, if applicable
        #----------------------------------------------
        if 'FRAME_NUMBER' in label.keys():
            frn = label['FRAME_NUMBER']
            ifl = label['FRAMELET_FILTER_INDEX']
            
            #- - - - - - - - - - - - - - - - - - - - 
            # Filter
            #- - - - - - - - - - - - - - - - - - - - 
            self.filter = (label['FILTER_NAME'])[ifl]

            #- - - - - - - - - - - - - - - - - - - - 
            # Filter-specific instrument id
            #- - - - - - - - - - - - - - - - - - - - 
            if self.filter == 'RED': self.instc = -61503
            if self.filter == 'GREEN': self.instc = -61502
            if self.filter == 'BLUE': self.instc = -61501
            if self.filter == 'METHANE': self.instc = -61504
            sinstc = str(self.instc)

            #- - - - - - - - - - - - - - - - - - - - - 
            # Timing
            #- - - - - - - - - - - - - - - - - - - - - 
            prefix = 'INS' + sinstc
            delta_var = prefix + '_INTERFRAME_DELTA'
            bias_var = prefix + '_START_TIME_BIAS'
            
            self.delta = cspyce.gdpool(delta_var)
            self.bias = cspyce.gdpool(bias_var)

            self.tinter = self.tinter0 + self.delta
            self.tstart = self.tstart0 + self.bias + frn*self.tinter

            self.tstop = self.tstart + self.exposure
            
            #- - - - - - - 
            # FOV
            #- - - - - - - 
            embed()
            k1_var = 'INS' + sinstc + '_DISTORTION_K1'
            k2_var = 'INS' + sinstc + '_DISTORTION_K2'
            cx_var = 'INS' + sinstc + '_DISTORTION_X'
            cy_var = 'INS' + sinstc + '_DISTORTION_Y'
            fo_var = 'INS' + sinstc + '_FOCAL_LENGTH'
            px_var = 'INS' + sinstc + '_PIXEL_SIZE'
            k1 = cspyce.gdpool(k1_var)[0]
            k2 = cspyce.gdpool(k2_var)[0]
            cx = cspyce.gdpool(cx_var)[0]
            cy = cspyce.gdpool(cy_var)[0]
            fo = cspyce.gdpool(fo_var)[0]
            px = cspyce.gdpool(px_var)[0]

            scale = px/fo
            distortion_coeff = [1,0,k1,0,k2]

            self.fov = oops.fov.RadialFOV(scale, 
                                            (self.nsamples, self.frlines),
                                            coefft_uv_from_xy=distortion_coeff, 
                                            uv_los=(cx, cy))

        return
    #===========================================================================

#*******************************************************************************





#*******************************************************************************
# JUNOCAM 
#*******************************************************************************
class JUNOCAM(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A instance-free class to hold JUNOCAM instrument parameters.
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Quick exit after first call
        if JUNOCAM.initialized: return

        Juno.initialize(ck=ck, planets=planets, asof=asof, spk=spk,
                           gapfill=gapfill,
                           mst_pck=mst_pck, irregulars=irregulars)
        Juno.load_instruments(asof=asof)



        #-----------------------------------
        # Construct the SpiceFrame
        #-----------------------------------
        ignore = oops.frame.SpiceFrame("JUNO_JUNOCAM")

        JUNOCAM.initialized = True
    #===========================================================================



    #===========================================================================
    # reset
    #===========================================================================
    @staticmethod
    def reset():
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Resets the internal JUNOCAM parameters. Can be useful for
        debugging.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        JUNOCAM.instrument_kernel = None
        JUNOCAM.fovs = {}
        JUNOCAM.initialized = False

        Juno.reset()
    #============================================================================

#*****************************************************************************




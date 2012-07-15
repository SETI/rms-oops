################################################################################
# oops/inst/cassini/vims.py
################################################################################

import numpy as np
import julian
import vicar
import pdstable
import cspice
import oops

from oops.inst.cassini.cassini_ import Cassini
#from oops.inst.cassini.vims import VIMS

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
        Cassini VIMS image file."""
    
    VIMS.initialize()    # Define everything the first time through
    
    # Load the VICAR file
    vic = vicar.VicarImage.from_file(filespec)
    
    # Get key information from the header
    time = vic["START_TIME"]
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)
    
    time = vic["STOP_TIME"]
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)
    
    mode = vic["INSTRUMENT_MODE_ID"]
    
    name = vic["INSTRUMENT_NAME"]
    # for the moment we are testing V
    camera = "V"
    
    # Make sure the SPICE kernels are loaded
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)
    
    # Create a Snapshot
    result = oops.obs.Snapshot((tdb0, tdb1),            # time
                               VIMS.fovs[camera,mode],   # fov
                               "CASSINI",               # path_id
                               "CASSINI_VIMS_" + camera) # frame_id
    
    # Insert the Vicar object as asubfield in case more info is needed.
    # This object behaves like a dictionary for most practical purposes.
    result.insert_subfield("vicar_dict", vic)
    # make consistent with from_index()
    #result.insert_subfield("index_dict", vic)
    result.insert_subfield("data", vic.get_2d_array())
    
    return result

################################################################################

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an VIMS index file. The filespec refers to the label of the index file.
        """
    
    VIMS.initialize()    # Define everything the first time through
    #test
    #VIMS.initialize()
    
    # Read the index file
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"])
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:
        
        tdb0 = julian.tdb_from_tai(dict["START_TIME"])
        tdb1 = julian.tdb_from_tai(dict["STOP_TIME"])
        
        mode = dict["INSTRUMENT_MODE_ID"]
        
        name = dict["INSTRUMENT_NAME"]
        # for the moment we are testing V
        camera = "V"
        
        item = oops.obs.Snapshot((tdb0, tdb1),              # time
                                 VIMS.fovs[camera,mode],     # fov
                                 "CASSINI",                 # path_id
                                 "CASSINI_VIMS_" + camera)   # frame_id
        
        # Tack on the dictionary in case more info is needed
        item.insert_subfield("index_dict", dict)
        
        snapshots.append(item)
    
    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]
    
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)
    
    return snapshots

def obs_from_index(filespec, index, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an VIMS index file. The filespec refers to the label of the index file.
        """
    
    VIMS.initialize()    # Define everything the first time through
    
    # Read the index file
    table = pdstable.PdsTableRow(filespec, index, ["START_TIME", "STOP_TIME"])
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    snapshots = []
    dict = row_dicts[0]
    
    tdb0 = julian.tdb_from_tai(dict["START_TIME"])
    tdb1 = julian.tdb_from_tai(dict["STOP_TIME"])
    
    mode = dict["INSTRUMENT_MODE_ID"]
    
    name = dict["INSTRUMENT_NAME"]
    # for the moment we are testing V
    camera = "V"
    
    item = oops.obs.Snapshot((tdb0, tdb1),              # time
                             VIMS.fovs[camera,mode],     # fov
                             "CASSINI",                 # path_id
                             "CASSINI_VIMS_" + camera)   # frame_id
    
    # Tack on the dictionary in case more info is needed
    item.insert_subfield("index_dict", dict)
    
    # Make sure all the SPICE kernels are loaded
    tdb0 = dict["START_TIME"]
    tdb1 = dict["STOP_TIME"]
    
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)
    
    return item

################################################################################

class VIMS(object):
    """A instance-free class to hold Cassini VIMS instrument parameters."""
    
    instrument_kernel = None
    fovs = {}
    initialized = False
    
    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called first.
            """
        
        # Quick exit after first call
        if VIMS.initialized: return
        
        Cassini.initialize()
        Cassini.load_instruments()
        
        # Load the instrument kernel
        VIMS.instrument_kernel = Cassini.spice_instrument_kernel("VIMS")[0]
        
        # Construct a flat FOV for each camera
        for detector in ["IR", "V"]:
            info = VIMS.instrument_kernel["INS"]["CASSINI_VIMS_" + detector]
            
            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]
            
            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"
            
            uscale = np.arctan(np.tan(xfov * np.pi/180.) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * np.pi/180.) / (lines/2.))
            
            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Flat((uscale,vscale), (samples,lines))
            
            # Load the dictionary, include the subsampling modes
            VIMS.fovs[detector, "FULL"] = full_fov
            VIMS.fovs[detector, "SUM2"] = oops.fov.Subsampled(full_fov, 2)
            VIMS.fovs[detector, "SUM4"] = oops.fov.Subsampled(full_fov, 4)
        
        # Construct a SpiceFrame for each camera
        # Deal with the fact that the instrument's internal coordinate system is
        # rotated 180 degrees
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_IRC",
                                       id="CASSINI_VIMS_IR_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_VIMS_V",
                                       id="CASSINI_VIMS_V_FLIPPED")
        
        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_IR",
                                    "CASSINI_VIMS_IR_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_VIMS_V",
                                    "CASSINI_VIMS_V_FLIPPED")
        
        VIMS.initialized = True
    
    @staticmethod
    def reset():
        """Resets the internal Cassini VIMS parameters. Can be useful for
            debugging."""
        
        VIMS.instrument_kernel = None
        VIMS.fovs = {}
        VIMS.initialized = False
        
        Cassini.reset()

################################################################################
# Initialize at load time
################################################################################

VIMS.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cassini_VIMS(unittest.TestCase):
    
    def runTest(self):
        
        # Create the pushbroom objects
        #pbs = from_file("test_data/cassini/VIMS/V1546355125_1.QUB")
        
        ob = from_file("test_data/cassini/VIMS/V1546355804_1.QUB")
        print "observation time:"
        print ob[0].time
        resolution = 8.0
        meshgrid = Meshgrid.for_fov(ob.fov, undersample=resolution, swap=True)
        bp = oops.Backplane(ob, meshgrid)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

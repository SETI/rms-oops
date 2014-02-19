################################################################################
# oops/inst/nh/lorri.py
#
# 2/9/14 Created (RSF)
################################################################################

import numpy as np
import julian
import pyfits
import pdstable
import oops
import oops.registry as registry
import solar

from oops.inst.nh.nh_ import NewHorizons


################################################################################
# Standard class methods
################################################################################

def from_file(filespec, use_fits_geom=False, **parameters):
    """A general, static method to return a Snapshot object based on a given
    NewHorizons LORRI image file.
    
    If parameters["data"] is False, no data or associated arrays are loaded.
    If parameters["calibration"] is False, no calibration objects are created.
    If parameters["headers"] is False, no header dictionary is returned.

    If parameters["astrometry"] is True, this is equivalent to data=False,
    calibration=False, headers=False.

    If parameters["solar_range"] is specified, it overrides the distance from the
    Sun to the target body (in AU) for calibration purposes.
    """

    LORRI.initialize()    # Define everything the first time through

    # Load the FITS file
    nh_file = pyfits.open(filespec)

    # Get key information from the header
    texp = nh_file[0].header["EXPTIME"]
    tdb_midtime = nh_file[0].header["SPCSCET"]
    tstart = tdb_midtime-texp/2
    binning_mode = nh_file[0].header["SFORMAT"]
    fov = LORRI.fovs[binning_mode]
    target_name = nh_file[0].header["SPCCBTNM"]
    target_body = None
    if registry.body_exists(target_name):
        target_body = registry.body_lookup(target_name)
    
    # Make sure the SPICE kernels are loaded
    NewHorizons.load_cks( tstart, tstart + texp)
    NewHorizons.load_spks(tstart, tstart + texp)

    path_id = "NEW HORIZONS"
    frame_id = "NH_LORRI_"+binning_mode
    
    if use_fits_geom: # Don't use the SPICE information to figure out where we're looking
        # First construct a path from the Sun to NH
        vecx = -nh_file[0].header["SPCSSCX"]
        vecy = -nh_file[0].header["SPCSSCY"]
        vecz = -nh_file[0].header["SPCSSCZ"]
        path_id = "NH_LORRI_PATH_"+str(tstart) # The path_id has to be unique to this observation
        sc_path = oops.path.FixedPath(oops.Vector3([vecx, vecy, vecz]), "SUN", "J2000", path_id)

        # Next create a frame based on the boresight
        ra = nh_file[0].header["SPCBRRA"]
        dec = nh_file[0].header["SPCBRDEC"]
        north_clk = nh_file[0].header["SPCEMEN"]
        scet = nh_file[0].header["SPCSCET"]
        frame_id = "NH_LORRI_FRAME_"+str(tstart) # The frame_id has to be unique to this observation
        oops.frame.Cmatrix.from_ra_dec(ra, dec, north_clk, frame_id+"_FLIPPED", "J2000")

        # We have to flip the X/Y axes because the LORRI standard has X vertical and Y horizontal
        flipxy = oops.Matrix3([[0,1,0],
                               [1,0,0],
                               [0,0,1]])
        oops.frame.Cmatrix(flipxy, frame_id, frame_id+"_FLIPPED")
    
    # Create a Snapshot
    snapshot = oops.obs.Snapshot(("v","u"), tstart, texp, fov,
                                 path_id, frame_id,
                                 target = target_name,
                                 instrument = "LORRI")

    # Interpret loader options
    if parameters.has_key("astrometry") and parameters["astrometry"]:
        include_data = False
        include_calibration = False
        include_headers = False

    else:
        include_data = (not parameters.has_key("data") or
                        parameters["data"])
        include_calibration = (not parameters.has_key("calibration") or
                        parameters["calibration"])
        include_headers = (not parameters.has_key("headers") or
                        parameters["headers"])

    if include_data:
        data = nh_file[0].data
        error = nh_file[1].data
        quality = nh_file[2].data

        snapshot.insert_subfield("data", data)
        snapshot.insert_subfield("error", error)
        snapshot.insert_subfield("quality", quality)

    if include_calibration:
        spectral_name = target_name
        if parameters.has_key("calib_body"):
            spectral_name = parameters["calib_body"]
    
        # Look up the solar range...
        try:
            solar_range = parameters["solar_range"]
        except KeyError:
            solar_range = None
    
        # If necessary, get the solar range from the target name
        if solar_range is None and target_body is not None:
            target_sun_path = oops.registry.connect_paths(target_body.path_id,
                                                          "SUN")
            # Paths of the relevant bodies need to be defined in advance!
    
            sun_event = target_sun_path.event_at_time(tdb_midtime)
            solar_range = sun_event.pos.norm().vals / solar.AU
    
        if solar_range is None:
            raise IOError("Calibration can't figure out range from Sun to target body "+target_name+" in file "+filespec)

        extended_calib = {}
        point_calib = {}

        for spectral_name in ["SOLAR", "PLUTO", "PHOLUS", "CHARON", "JUPITER"]:        
            spectral_radiance = nh_file[0].header["R"+spectral_name] # Extended source
            spectral_irradiance = nh_file[0].header["P"+spectral_name] # Point source
    
            F_solar = 176 # pivot 6076.2 A at 1 AU
            extended_factor = 1/texp/spectral_radiance * np.pi * solar_range**2 / F_solar # Conversion to I/F
            point_factor = 1/texp/spectral_irradiance
        
            extended_calib[spectral_name] = oops.calib.ExtendedSource("I/F", extended_factor)
            point_calib[spectral_name] = oops.calib.PointSource("I/F", point_factor, fov)
        
        snapshot.insert_subfield("point_calib", point_calib)
        snapshot.insert_subfield("extended_calib", extended_calib)

    if include_headers:
        headers = []
        for objects in nh_file:
            headers.append(objects.header)

        snapshot.insert_subfield("headers", headers)
                               
    return snapshot

################################################################################

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
    in an LORRI index file. The filespec refers to the label of the index file.
    """

    assert False # Not implemented

#    LORRI.initialize()    # Define everything the first time through
#
#    # Read the index file
#    COLUMNS = []        # Return all columns
#    TIMES = ["START_TIME"]
#    table = pdstable.PdsTable(filespec, columns=COLUMNS, times=TIMES)
#    row_dicts = table.dicts_by_row()
#
#    # Create a list of Snapshot objects
#    snapshots = []
#    for dict in row_dicts:
#
#        tstart = julian.tdb_from_tai(dict["START_TIME"])
#        texp = max(1.e-3, dict["EXPOSURE_DURATION"]) / 1000.
#        mode = dict["INSTRUMENT_MODE_ID"]
#
#        name = dict["INSTRUMENT_NAME"]
#        if "WIDE" in name:
#            camera = "WAC"
#        else:
#            camera = "NAC"
#
#        item = oops.obs.Snapshot(("v","u"), tstart, texp, LORRI.fovs[camera,mode],
#                                 "NEWHORIZONS", "NEWHORIZONS_LORRI_" + camera,
#                                 dict = dict,       # Add index dictionary
#                                 index_dict = dict, # Old name
#                                 instrument = "LORRI",
#                                 detector = camera,
#                                 sampling = mode)
#
#        snapshots.append(item)
#
#    # Make sure all the SPICE kernels are loaded
#    tdb0 = row_dicts[ 0]["START_TIME"]
#    tdb1 = row_dicts[-1]["START_TIME"]
#
#    NewHorizons.load_cks( tdb0, tdb1)
#    NewHorizons.load_spks(tdb0, tdb1)
#
#    return snapshots

################################################################################

class LORRI(object):
    """A instance-free class to hold NewHorizons LORRI instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    @staticmethod
    def initialize():
        """Fills in key information about LORRI. Must be called first.
        """

        # Quick exit after first call
        if LORRI.initialized: return

        NewHorizons.initialize()
        NewHorizons.load_instruments()

        # Load the instrument kernel
        LORRI.instrument_kernel = NewHorizons.spice_instrument_kernel("LORRI")[0]

        # Construct a flat FOV for each camera
        for binning_mode, binning_id in [("1X1", -98301), ("4X4", -98302)]:
            info = LORRI.instrument_kernel["INS"][binning_id]

            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]

            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            assert info["FOV_ANGLE_UNITS"] == "DEGREES"
            
            uscale = np.arctan(np.tan(xfov * np.pi/180.) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * np.pi/180.) / (lines/2.))
            
            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Flat((-uscale,-vscale), (samples,lines))

            # Load the dictionary
            LORRI.fovs[binning_mode] = full_fov

        # Construct a SpiceFrame for each camera
        ignore = oops.frame.SpiceFrame("NH_LORRI_1x1", id='NH_LORRI_1X1_FLIPPED')
        ignore = oops.frame.SpiceFrame("NH_LORRI_4x4", id='NH_LORRI_4X4_FLIPPED')

        # For some reason the SPICE IK gives the boresight along -Z instead of +Z,
        # so we have to flip all axes.
        flipxyz = oops.Matrix3([[-1, 0, 0],
                                [ 0,-1, 0],
                                [ 0, 0,-1]])
        oops.frame.Cmatrix(flipxyz, "NH_LORRI_1X1",
                           "NH_LORRI_1X1_FLIPPED")
        oops.frame.Cmatrix(flipxyz, "NH_LORRI_4X4",
                           "NH_LORRI_4X4_FLIPPED")

        LORRI.initialized = True

    @staticmethod
    def reset():
        """Resets the internal NewHorizons LORRI parameters. Can be useful for
        debugging."""

        LORRI.instrument_kernel = None
        LORRI.fovs = {}
        LORRI.initialized = False

        NewHorizons.reset()

################################################################################
# Initialize at load time
################################################################################

LORRI.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

class Test_NewHorizons_LORRI(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
        import cspice
        
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             astrometry=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             data=False, calibration=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))
    
        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             data=True, calibration=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             headers=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"))
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        self.assertTrue(snapshot.data.shape == (1024,1024))
        self.assertTrue(snapshot.quality.shape == (1024,1024))
        self.assertTrue(snapshot.error.shape == (1024,1024))
        
        self.assertAlmostEqual(snapshot.time[1]-snapshot.time[0], snapshot.texp)
        self.assertAlmostEqual(snapshot.time[0]+snapshot.texp/2,
                               cspice.utc2et(snapshot.headers[0]["SPCUTCID"]),
                               places=3)
        self.assertEqual(snapshot.target, "EUROPA")

        snapshot_fits = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY, "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                                  use_fits_geom=True)
        
        self.assertEqual(snapshot.time, snapshot_fits.time)
        self.assertEqual(snapshot.texp, snapshot_fits.texp)

        meshgrid = oops.Meshgrid.for_fov(snapshot.fov, (0,0), limit=(1,1))
        bp = oops.Backplane(snapshot, meshgrid=meshgrid)
        bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
        ra =          bp.right_ascension().vals.astype('float')
        ra_fits =     bp_fits.right_ascension().vals.astype('float')
        dec =         bp.declination().vals.astype('float')
        dec_fits =    bp_fits.declination().vals.astype('float')
        europa =      bp.where_intercepted("europa").vals.astype('float')
        europa_fits = bp_fits.where_intercepted("europa").vals.astype('float')

        self.assertAlmostEqual(ra, ra_fits, places=3)
        self.assertAlmostEqual(dec, dec_fits, places=3)
        self.assertEqual(europa, 0.0)
        self.assertEqual(europa_fits, 0.0)

        meshgrid = oops.Meshgrid.for_fov(snapshot.fov, (642,451), limit=(643,452))
        bp = oops.Backplane(snapshot, meshgrid=meshgrid)
        bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
        long =        bp.longitude("europa").vals.astype('float')
        long_fits =   bp_fits.longitude("europa").vals.astype('float')
        lat =         bp.latitude("europa").vals.astype('float')
        lat_fits =    bp_fits.latitude("europa").vals.astype('float')
        europa =      bp.where_intercepted("europa").vals.astype('float')
        europa_fits = bp_fits.where_intercepted("europa").vals.astype('float')
        
        self.assertAlmostEqual(long, long_fits, places=1)
        self.assertAlmostEqual(lat, lat_fits, places=1)
        self.assertEqual(europa, 1.0)
        self.assertEqual(europa_fits, 1.0)

        europa_iof = snapshot.extended_calib["CHARON"].value_from_dn(snapshot.data[451,642])
        self.assertGreater(europa_iof, 0.4)
        self.assertLess(europa_iof, 0.6)
        
############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

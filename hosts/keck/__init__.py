################################################################################
# hosts/keck/__init__.py: Class Keck
#
# This is an initial implementation of a Keck II FITS reader.  It does not
# support distortion models or instruments other than NIRC2.
################################################################################

import numpy as np
import os
import re
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
import glob
import unittest

import julian
import hosts.solar as solar
import tabulation as tab
import oops

########################################
# Global Variables
########################################

# A handy constant
RADIANS_PER_ARCSEC = oops.RPD / 3600.

# This should be a reasonably complete procedure for mapping the first three
# letters of the P.I.'s target name to the SPICE name of the target body.
KECK_TARGET_DICT = {"MAR": "MARS",
                    "JUP": "JUPITER",
                    "SAT": "SATURN",
                    "URA": "URANUS",
                    "NEP": "NEPTUNE",
                    "PLU": "PLUTO",
                    "IO" : "IO",
                    "EUR": "EUROPA",
                    "GAN": "GANYMEDE",
                    "CAL": "CALLISTO",
                    "ENC": "ENCELADUS",
                    "TIT": "TITAN",
                    "PHO": "PHOEBE"}

# Define some important paths and frames
oops.Body.define_solar_system("1990-01-01", "2020-01-01")

################################################################################
# Standard instrument methods
################################################################################

def from_file(filespec, **parameters):
    """A general, static method to return an Observation object based on a given
    data file generated by the Keck Telescope."""

    keck_file = pyfits.open(filespec)
    return Keck.from_opened_fitsfile(keck_file, **parameters)

################################################################################
# Class Keck
################################################################################

class Keck(object):
    """This class defines functions and properties unique to the Keck
    Telescope.

    Objects of this class are empty; they only exist to support inheritance.
    """

    def filespec(self, keck_file):
        """The full directory path and name of the file."""

        # Found by poking around inside a pyfits object
        return keck_file._HDUList__file._File__file.name

    #===========================================================================
    def telescope_name(self, keck_file):
        """The name of the telescope from which the observation was obtained."""

        return keck_file[0].header["TELESCOP"]

    #===========================================================================
    def instrument_name(self, keck_file):
        """The name of the Keck instrument associated with the file."""

        return keck_file[0].header["CURRINST"]

    #===========================================================================
    def detector_name(self, keck_file, **parameters):
        """The name of the detector on the Keck instrument that was used to
        obtain this file.
        """

        return keck_file[0].header["CAMNAME"]

    #===========================================================================
    def data_array(self, keck_file, **parameters):
        """Array containing the data."""

        return keck_file[0].data

    #===========================================================================
    # This works for Snapshot observations. Others must override.
    def time_limits(self, hst_file, **parameters):
        """A tuple containing the overall start and end times of the
        observation.
        """

        date_obs = hst_file[0].header["DATE-OBS"]
        time_obs_start = hst_file[0].header["EXPSTART"]
        time_obs_end = hst_file[0].header["EXPSTOP"]

        tdb0 = julian.tdb_from_tai(julian.tai_from_iso(date_obs + "T" +
                                                       time_obs_start))
        tdb1 = julian.tdb_from_tai(julian.tai_from_iso(date_obs + "T" +
                                                       time_obs_end))

        return (tdb0, tdb1)

    #===========================================================================
    def target_body(self, keck_file):
        """The body object defining the image target.

        It is based on educated guesses from the target name used by the PI.
        """

        global KECK_TARGET_DICT

        targname = keck_file[0].header["OBJECT"].upper()

        if len(targname) >= 2:      # Needed to deal with 2-letter "IO"
            key2 = targname[0:2]
            key3 = key2
        if len(targname) >= 3:
            key3 = targname[0:3]

        # Raises a KeyError on failure
        try:
            body_name = KECK_TARGET_DICT[key3]
        except KeyError:
            body_name = KECK_TARGET_DICT[key2]

        return oops.Body.lookup(body_name)

    #===========================================================================
    def construct_snapshot(self, keck_file, **parameters):
        """A Snapshot object for the data found in the specified image."""

        fov = self.define_fov(keck_file, **parameters)

        # Create a list of FITS header objects for a subfield
        headers = []
        for objects in keck_file:
            headers.append(objects.header)

        times = self.time_limits(keck_file, **parameters)

        uv_center = fov.uv_from_xy((0,0))
        xy_center = fov.xy_from_uv(uv_center, derivs=True)

        v_wrt_y_deg = np.arctan(xy_center.d_duv.vals[0,1] /
                                xy_center.d_duv.vals[1,1]) * oops.DPR

        orient = 0. # XXX
        clock = 0.
#        clock = orient - v_wrt_y_deg

        frame_id = keck_file[0].header["FILENAME"]

        ra = keck_file[0].header["RA"]
        dec = keck_file[0].header["DEC"]
        raoff = keck_file[0].header["RAOFF"]
        decoff = keck_file[0].header["DECOFF"]
        ra -= raoff
        dec -= decoff

        cmatrix = oops.frame.Cmatrix.from_ra_dec(ra, dec, clock,
                                                 frame_id + "_CMATRIX")

        times = self.time_limits(keck_file)
        frame = oops.frame.TrackerFrame(cmatrix,
                                        self.target_body(keck_file).path_id,
                                        "EARTH", times[0], frame_id)

        return oops.obs.Snapshot(
                        axes = ("v","u"),
                        tstart = times[0],
                        texp = times[1] - times[0],
                        fov = fov,
                        frame = frame,
                        path_id = "EARTH",
                        frame_id = frame_id,
                        data = self.data_array(keck_file, **parameters),
                        target = self.target_body(keck_file),
                        telescope = self.telescope_name(keck_file),
                        instrument = self.instrument_name(keck_file),
                        detector = self.detector_name(keck_file),
                        filter = self.filter_name(keck_file),
                        headers = headers)

    #===========================================================================
    @staticmethod
    def from_opened_fitsfile(keck_file, **parameters):
        """A general, static method to return an Observation object based on an
        Keck data file generated by the Keck Telescope.
        """

        # Make an instance of the Keck class
        this = Keck()

        # Confirm that the telescope is Keck
        if this.telescope_name(keck_file) != "Keck II":
            raise IOError("not a Keck II file: " + this.filespec(keck_file))

        # Figure out the instrument
        instrument = this.instrument_name(keck_file)

        if instrument == "NIRC2":
            from .nirc2 import NIRC2
            obs = NIRC2.from_opened_fitsfile(keck_file, **parameters)

        else:
            raise IOError("unsupported instrument in Keck file " +
                          this.filespec(keck_file) + ": " + instrument)

        return obs

################################################################################
# UNIT TESTS
################################################################################

class Test_Keck(unittest.TestCase):

    def runTest(self):

        import cspyce
        from .keck import Keck

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

# oops/instrument/hst/__init__.py

import numpy as np
import unittest

import os
import re
import pyfits
import cspice

import oops
import oops.instrument.hst.acs.hrc
import oops.instrument.hst.acs.wfc
import oops.instrument.hst.wfc3.uvis

################################################################################
# Global Variables
################################################################################

# A handy constant
RADIANS_PER_ARCSEC = np.pi / 180. / 3600.

# This table enables a small correction to pixel locations within the distortion
# model when dealing with subarrays. The difference they make is probably
# infinitesimal.

OVERSCAN_CENTER_OFFSET = {
    ("HST","ACS",   "HRC" ): oops.Pair((19,10)), # Consistent with the IHB
    ("HST","ACS",   "SBC" ): 0.,                 # No SBC overscan
    ("HST","ACS",   "WFC" ): oops.Pair((24,10)), # Uncertain, but probably close
    ("HST","NICMOS","NIC1"): 0.,                 # No NICMOS overscan (?)
    ("HST","NICMOS","NIC2"): 0.,
    ("HST","NICMOS","NIC3"): 0.,
    ("HST","WFC3",  "IR"  ): 0.,                 # No IR overscan (?)
    ("HST","WFC3",  "UVIS"): oops.Pair((13, 0)), # ?, consistent with C512C
    ("HST","WFPC2", ""    ): 0.                  # No info in IHB, probably OK
}

# For each instrument and detector, this dictionary returns the list of fields
# in an IDC record that define the key for that record

IDC_FILE_KEY_LISTS = {
    ("HST","ACS",   "HRC" ): ["FILTER1","FILTER2"],
    ("HST","ACS",   "SBC" ): ["FILTER1","FILTER2"],
    ("HST","ACS",   "WFC" ): ["DETCHIP","FILTER1","FILTER2"],
    ("HST","NICMOS","NIC1"): ["FILTER"],
    ("HST","NICMOS","NIC2"): ["FILTER"],
    ("HST","NICMOS","NIC3"): ["FILTER"],
    ("HST","WFC3",  "IR"  ): ["FILTER"],
    ("HST","WFC3",  "UVIS"): ["DETCHIP","FILTER"],
    ("HST","WFPC2", ""    ): ["FILTER1","FILTER2","DETCHIP"]
}

# For each instrument and detector, this dictionary returns the list of
# keyword and header number where the information can be found in the header of
# a FITS data file.

DATA_FILE_KEY_LISTS = {
    ("HST","ACS",   "HRC" ): [("FILTER1",0),("FILTER2",0)],
    ("HST","ACS",   "SBC" ): [("FILTER1",0),("FILTER2",0)],
    ("HST","ACS",   "WFC" ): [("CCDCHIP",0),("FILTER1",0),("FILTER2",0)],
    ("HST","NICMOS","NIC1"): [("FILTER", 0)],
    ("HST","NICMOS","NIC2"): [("FILTER", 0)],
    ("HST","NICMOS","NIC3"): [("FILTER", 0)],
    ("HST","WFC3",  "IR"  ): [("FILTER", 0)],
    ("HST","WFC3",  "UVIS"): [("CCDCHIP",1),("FILTER",0)],
    ("HST","WFPC2", ""    ): [("FILTNAM1",0),("FILTNAM2",0)]
}

# After a call to set_idc_path(), these global variables will be defined:

HST_IDC_PATH = None
    # The directory prefix pointing to the location where all HST IDC files
    # reside.

IDC_FILE_NAME_DICT = None
    # A dictionary that associates each instrument and detector with the name of
    # a particular IDC file.

IDC_DICTS_LOADED = None
    # A dictionary that returns True if the IDC file for a particular instrument
    # and detector has been loaded; False otherwise.

# This is the global dictionary containing all IDC information as function of
# instrument, detector, chip and filter(s). It is updated by load_idc_dict()
# every time a new instrument/detector combination is required.

GLOBAL_IDC_DICT = {}

################################################################################
# from_file()
################################################################################

def from_file(file_spec):
    """Given the name of a file, this function returns an associated Observation
    object describing the data found in the file."""

    (host, instrument, detector) = get_file_info(file_spec)
    if host != "HST":
        raise IOError("instrument host is not HST: " + file_spec)

    if instrument == "ACS":
        if detector == "HRC":
            return oops.instrument.hst.acs.hrc.from_file(file_spec)
        elif detector == "WFC":
            return oops.instrument.hst.acs.wfc.from_file(file_spec)
        else:
            raise RuntimeError("HST/ACS/" + detector +
                               " is not supported: " + file_spec)

    elif instrument == "WFC3":
        if detector == "UVIS":
            return oops.instrument.hst.wfc3.uvis.from_file(file_spec)
        else:
            raise RuntimeError("HST/WFC3/" + detector +
                               " is not supported: " + file_spec)

    else:
        raise RuntimeError("HST instrument " + instrument +
                           " is not supported: " + file_spec)

################################################################################
# Helper functions for most HST images
################################################################################

def get_file_info(file_spec):
    """Returns the instrument_host, instrument and detector associated with
    a data file.

    Input:
        file_spec       the full path to a data file.

    Return:             a tuple containing (instrument_host, instrument
                        detector).
    """

    # See if it is a FITS file
    try:
        hst_file = pyfits.open(file_spec)
    except IOError:
        # Replace the uninformative error message from pyfits
        raise IOError("unrecognized file format: " + file_spec)

    info = get_hst_info(hst_file)
    hst_file.close()

    return info

def get_hst_info(hst_file):
    """Returns the instrument_host, instrument and detector associated with
    a data file already opened by pyfits.open().
    """

    # Extract the instrument_host
    try:
        host = hst_file[0].header["TELESCOP"]
    except KeyError:
        fits_file.close()
        raise IOError("unidentified instrument host: " + file_spec)

    # Confirm the instrument host
    if host != "HST":
        raise IOError("instrument host is not HST: " + file_spec)

    # Get the instrument
    try:
        instrument = hst_file[0].header["INSTRUME"]
    except KeyError:
        raise IOError("unidentified instrument: " + file_spec)

    # Get the detector
    detector = ""
    if instrument == "WFPC2":
        detector == ""
    elif instrument == "NICMOS":
        detector = hst_file[0].header["APERTURE"][0:4]
    else:
        try:
            detector = hst_file[0].header["DETECTOR"]
        except KeyError:
            raise IOError("unidentified detector: " + file_spec)

    return ("HST", instrument, detector)

def from_hst_image_file(file_spec):
    """This version of from_file() should work for most HST images but has only
    been tested for WFC3 and ACS."""

    hst_file = pyfits.open(file_spec)

    times = get_times(hst_file)
    fov   = get_fov(hst_file)
    frame = get_frame(hst_file, fov.uv_los)
    data  = get_data(hst_file)
    axes  = ("v","u")

    return oops.Observation(data, axes, times, fov, "EARTH", frame.frame_id)

def set_idc_path(idc_path):
    """Defines the directory path to the IDC files. It must be called before
    any HST files are loaded. The alternative is to define the environment
    variable HST_IDC_PATH."""

    global HST_IDC_PATH
    global IDC_FILE_NAME_DICT
    global IDC_DICTS_LOADED

    # Save the argument as a global variable. Make sure it ends with slash.
    HST_IDC_PATH = idc_path

    if HST_IDC_PATH[-1] != "/":
        HST_IDC_PATH += "/"

    # We associate files with specific instruments and detector combinations
    # via a dictionary. The definition of this dictionary resides in the
    # same directory as the IDC files themselves, in a file called
    #   IDC_FILE_NAME_DICT.txt
    # Every time we update an IDC file, we will need to update this file
    # too.
    #
    # The file has this syntax:
    #   ("HST","ACS",   "HRC"): "p7d1548qj_idc.fits"
    #   ("HST","WFPC2", ""   ): "v5r1512gi_idc.fits"
    # etc., where each row defines a dictionary entry comprising a key
    # ("HST", instrument name, detector name) and the name of the associated
    # IDC file in FITS format.

    # Compile a regular expression that ensures nothing bad is contained in
    # this file.
    regex = re.compile(r' *\( *("\w*" *, *)*"\w*" *\) *: *"\w+\.fits" *$',
                       re.IGNORECASE)

    # Read the key:value pairs and make sure they are clean
    f = open(HST_IDC_PATH + "IDC_FILE_NAME_DICT.txt")
    lines = []
    for line in f:
        if regex.match(line) is False:
            raise IOError("syntax error in IDC definition: " + line)

        lines.append(line)
    f.close()

    # Define the global dictionary
    IDC_FILE_NAME_DICT = eval("{" + ", ".join(lines) + "}")

    # This dictionary tracks the IDC files loaded

    IDC_DICTS_LOADED = {}
    for key in IDC_FILE_NAME_DICT.keys():
        IDC_DICTS_LOADED[key] = False

    return

def load_idc_dict(detector_key):
    """Loads the IDC dictionary information for a specified HST detector.
    If the dictionary has already been loaded, it does nothing.

    Input:
        detector_key    a tuple ("HST", instrument, detector) as returned by
                        get_hst_file_info()
    """

    # If the dictionary was never initialized, find the HST_IDC_PATH
    # environment variable and initialize it now.
    if HST_IDC_PATH is None:
        set_idc_path(os.environ["HST_IDC_PATH"])

    # See if the info has been loaded already
    if IDC_DICTS_LOADED[detector_key]: return

    # Get the key name info for the dictionary
    key_name_list = IDC_FILE_KEY_LISTS[detector_key]

    # Open the file
    hst_file = pyfits.open(HST_IDC_PATH + IDC_FILE_NAME_DICT[detector_key])

    # Get the names of columns
    fits_obj = hst_file[1]
    ncolumns = fits_obj.header["TFIELDS"]
    names = []
    for c in range(ncolumns):
        key = "TTYPE" + str(c+1)
        names.append(fits_obj.header[key])

    # Read the rows of the IDC table...
    nrows = fits_obj.header["NAXIS2"]
    for r in range(nrows):

        row_dict = {}
        is_forward = False

        # For each column...
        for c in range(ncolumns):

            # Convert the value to a standard Python type
            value = fits_obj.data[r][c]
            dtype = str(fits_obj.data.dtype[c])

            if   "S" in dtype: value = str(value)
            elif "i" in dtype: value = int(value)
            elif "f" in dtype: value = float(value)
            else:
                raise ValueError("Unrecognized dtype: " + dtype)

            # Make sure this is a FORWARD transform
            if names[c] == "DIRECTION" and value == "FORWARD":
                is_forward = True

            # Append to the row's list and dictionary
            row_dict[names[c]] = value

        # Only save forward transforms
        if is_forward:

            # Make the dictionary key for the row
            key = []
            for name in key_name_list:
                key.append(row_dict[name])

            key = detector_key + tuple(key)

            # Add to the dictionary
            GLOBAL_IDC_DICT[key] = row_dict

    IDC_DICTS_LOADED[detector_key] = True
    return

def idc_key(hst_file, quadrant=0):
    """Returns the key into the IDC dictionary associated with this file.
    The file must already have been opened via pyfits.open(). Use
    quadrant = 0-3 for the four quadrants of WFPC2"""

    key = get_hst_info(hst_file)

    key_name_tuples = DATA_FILE_KEY_LISTS[key]
    for (name,i) in key_name_tuples:
        key += (hst_file[i].header[name],)

    if key[1] == "WFPC2": key += (quadrant,)

    return key

def get_fov(hst_file, quadrant=0):
    """Returns the FOV object associated with the file. The FOV includes the
    HST distortion model, and allows for possible subarrays and possible
    sub-sampling. 

    Input:
        hst_file    the FITS file object, as returned by pyfits.open().
        quadrant    for WFPC2, enter 0-3 to select PC1, WF2, WF3 or WF4;
                    otherwise, ignored.

    Return:         the associated FOV object.
    """

    header1 = hst_file[1].header

    # Get the IDC dictionary entry for this file
    detector_key = get_hst_info(hst_file)

    # Load the IDC information if necessary
    load_idc_dict(detector_key)

    # Get the associated dictionary entry
    idc_dict = GLOBAL_IDC_DICT[idc_key(hst_file, quadrant)]

    # If this is a drizzled file, we can treat it as a FlatFOV
    try:
        drizcorr = hst_file[0].header["DRIZCORR"]
    except KeyError:
        drizcorr = ""

    if drizcorr == "COMPLETE":

        # Define the field of view without sub-sampling
        # *** SHOULD BE CHECKED ***
        scale = idc_dict["SCALE"] * RADIANS_PER_ARCSEC
        fov = oops.FlatFOV((scale,-scale),
                      (header1["NAXIS1"], header1["NAXIS2"]))

        # Apply the sub-sampling if necessary
        binaxis1 = header1["BINAXIS1"]
        binaxis2 = header1["BINAXIS2"]
        if binaxis1 != 1 or binaxis2 != 1:
            fov = oops.SubsampledFOV(fov, (binaxis1, binaxis2))

        return fov

    # Determine the order of the transform
    if "CX11" in idc_dict: order = 1
    if "CX22" in idc_dict: order = 2
    if "CX33" in idc_dict: order = 3
    if "CX44" in idc_dict: order = 4
    if "CX55" in idc_dict: order = 5
    if "CX66" in idc_dict: order = 6

    # Create arrays of the coefficients
    cxy = np.zeros((order+1, order+1, 2))

    # The first index is the order of the term.
    # The second index is the coefficient on the sample axis.
    for   i in range(1, order+1):
      for j in range(i+1):
        try:
            # In these arrays, the indices are the powers of x (increasing
            # rightward) and y (increasing upward).
            cxy[j,i-j,0] =  idc_dict["CX" + str(i) + str(j)]
            cxy[j,i-j,1] = -idc_dict["CY" + str(i) + str(j)]
        except KeyError: pass

    # Figure out the shape and center, which might have been altered by
    # overscan pixels
    idc_shape  = oops.Pair((idc_dict["XSIZE"], idc_dict["YSIZE"]))
    true_shape = oops.Pair((header1["NAXIS1"], header1["NAXIS2"]))

    # If the image is not smaller than the full FOV, center it appropriately
    # and return the FOV. It can be larger because of overscan pixels.
    if np.all(true_shape.vals >= idc_shape.vals):
        return oops.PolynomialFOV(cxy * RADIANS_PER_ARCSEC,
                        true_shape,
                        (header1["CRPIX1"], header1["CRPIX2"]),
                        (idc_dict["SCALE"] * RADIANS_PER_ARCSEC)**2)

    # Otherwise, first define the nominal full FOV using IDC info alone
    fov = oops.PolynomialFOV(cxy * RADIANS_PER_ARCSEC,
                    (idc_dict["XSIZE"], idc_dict["YSIZE"]),
                    (idc_dict["XREF" ], idc_dict["YREF" ]),
                    (idc_dict["SCALE"] * RADIANS_PER_ARCSEC)**2)

    # Shift it to the actual subarray
    # For ACS and WFC3, apply a small correction for the overscan pixels,
    # which are included in the CENTERA1/2 parameters but not in the IDC
    # model.
    fov = oops.SubarrayFOV(fov,
                    (oops.Pair((header1["CENTERA1"],header1["CENTERA2"]))
                                - OVERSCAN_CENTER_OFFSET[detector_key]),
                    true_shape,
                    oops.Pair(((header1["CRPIX1"],header1["CRPIX2"]))))

    # Apply the sub-sampling if necessary
    binaxis1 = header1["BINAXIS1"]
    binaxis2 = header1["BINAXIS2"]
    if binaxis1 != 1 or binaxis2 != 1:
        fov = oops.SubsampledFOV(fov, (binaxis1, binaxis2))

    return fov

def get_times(hst_file):
    """Returns the times in seconds at the beginning and end of the
    exposure.

    Input:
        hst_file    the FITS file object, as returned by pyfits.open().

    Return:         a tuple containing the start and end times in seconds
                    TDB.
    """

    date_obs = hst_file[0].header["DATE-OBS"]
    time_obs = hst_file[0].header["TIME-OBS"]
    exptime  = hst_file[0].header["EXPTIME"]

    time0 = cspice.str2et(date_obs + " " + time_obs)
    time1 = time0 + exptime

    return (time0, time1)

def get_frame(hst_file, uv_los, quadrant=0):
    """Returns the CmatrixFrame that rotates from J2000 coordinates into the
    frame of the HST observation. 

    Input:
        hst_file    the FITS file object, as returned by pyfits.open().
        uv_los      the pixel coordinates in the image that correspond to the
                    optic axis of the image or subarray.
        quadrant    for WFPC2, enter 0-3 to select PC1, WF2, WF3 or WF4;
                    otherwise, ignored.

    Return:         the associated FOV object.
    """

    u_ref   = hst_file[1].header["CRPIX1"]
    v_ref   = hst_file[1].header["CRPIX2"]
    ra_ref  = hst_file[1].header["CRVAL1"]
    dec_ref = hst_file[1].header["CRVAL2"]

    dra_du  = hst_file[1].header["CD1_1"]
    dra_dv  = hst_file[1].header["CD1_2"]
    ddec_du = hst_file[1].header["CD2_1"]
    ddec_dv = hst_file[1].header["CD2_2"]

    # We need (ra,dec) at the optic axis coordinates uv_los
    # dra  = dra_du  * (u_los - u_ref) + dra_dv  * (v_los - v_ref)
    # ddec = ddec_du * (u_los - u_ref) + ddec_dv * (v_los - v_ref)

    duv = uv_los - (u_ref,v_ref)
    ra_los  = ra_ref  + duv.dot((dra_du,  dra_dv)).vals
    dec_los = dec_ref + duv.dot((ddec_du, ddec_dv)).vals

    clock = hst_file[1].header["ORIENTAT"]
    file_name = hst_file[0].header["FILENAME"]

    return oops.Cmatrix(ra_los, dec_los, clock, file_name)

def get_data(hst_file, quadrant=0):
    """Returns the data array associated with the image. 

    Input:
        hst_file    the FITS file object, as returned by pyfits.open().
        quadrant    for WFPC2, enter 0-3 to select PC1, WF2, WF3 or WF4;
                    otherwise, ignored. ***NOT YET IMPLEMENTED***

    Return:         the associated FOV object.
    """

    return hst_file[1].data

################################################################################
# UNIT TESTS
################################################################################

class Test_Instrument_HST(unittest.TestCase):

    def runTest(self):

        APR = 180./np.pi * 3600.

        prefix = "unittest_data/hst/"
        self.assertEqual(get_file_info(prefix + "ibht07svq_drz.fits"),
                         ('HST', 'WFC3', 'IR'))
        self.assertEqual(get_file_info(prefix + "ibht07svq_ima.fits"),
                         ('HST', 'WFC3', 'IR'))
        self.assertEqual(get_file_info(prefix + "ibht07svq_raw.fits"),
                         ('HST', 'WFC3', 'IR'))
        self.assertEqual(get_file_info(prefix + "ibu401nnq_flt.fits"),
                         ('HST', 'WFC3', 'UVIS'))
        self.assertEqual(get_file_info(prefix + "j9dh35h7q_raw.fits"),
                         ('HST', 'ACS', 'HRC'))
        self.assertEqual(get_file_info(prefix + "j96o01ioq_raw.fits"),
                         ('HST', 'ACS', 'WFC'))
        self.assertEqual(get_file_info(prefix + "n43h05b3q_raw.fits"),
                         ('HST', 'NICMOS', 'NIC2'))
        self.assertEqual(get_file_info(prefix + "ua1b0101m_d0f.fits"),
                         ('HST', 'WFPC2', ''))
        self.assertEqual(get_file_info(prefix + "ua1b0309m_d0m.fits"),
                         ('HST', 'WFPC2', ''))

        self.assertRaises(IOError, get_file_info, prefix + "a.b.c.d")

        # Raw ACS, full-frame with overscan pixels
        prefix = "unittest_data/hst/"
        hst_file = pyfits.open(prefix + "j9dh35h7q_raw.fits")

        # Test get_time()
        (time0, time1) = get_times(hst_file)

        self.assertTrue(time1 - time0 - hst_file[0].header["EXPTIME"] > -1.e-8)
        self.assertTrue(time1 - time0 - hst_file[0].header["EXPTIME"] <  1.e-8)

        str0 = cspice.et2utc(time0, "ISOC", 0)
        self.assertEqual(str0, hst_file[0].header["DATE-OBS"] + "T" +
                               hst_file[0].header["TIME-OBS"])

        # Test get_fov()
        key = idc_key(hst_file)
        self.assertEqual(key, ("HST", "ACS", "HRC", "F475W", "CLEAR2S"))
        fov = get_fov(hst_file)
        self.assertEqual(GLOBAL_IDC_DICT[key]["DETCHIP"], 1)

        shape = tuple(fov.uv_shape.vals)
        buffer = np.empty(shape + (2,))
        buffer[:,:,0] = np.arange(shape[0])[..., np.newaxis] + 0.5
        buffer[:,:,1] = np.arange(shape[1]) + 0.5
        pixels = oops.Pair(buffer)

        self.assertTrue(np.all(fov.uv_is_inside(pixels).vals))

        # Confirm that a PolynomialFOV is reversible

        # This is SLOW for a million pixels but it works. I have done a bit of
        # optimization and appear to have reached the point of diminishing
        # returns.

        # los = fov.los_from_uv(pixels)
        # test_pixels = fov.uv_from_los(los)

        # Faster version, 1/64 pixels
        NSTEP = 8
        pixels = oops.Pair(buffer[::NSTEP,::NSTEP])
        los = fov.los_from_uv(pixels)
        test_pixels = fov.uv_from_los(los)

        self.assertTrue(np.max(np.abs(test_pixels.vals - pixels.vals)) < 1.e-12)

        # Separations between pixels in arcsec are around 0.025
        seps = los[1:].sep(los[:-1])
        self.assertTrue(np.min(seps.vals) * APR > 0.028237 * NSTEP)
        self.assertTrue(np.max(seps.vals) * APR < 0.028648 * NSTEP)

        seps = los[:,1:].sep(los[:,:-1])
        self.assertTrue(np.min(seps.vals) * APR > 0.024547 * NSTEP)
        self.assertTrue(np.max(seps.vals) * APR < 0.025186 * NSTEP)

        # Pixel area factors are near unity
        areas = fov.area_factor(pixels)
        self.assertTrue(np.min(areas.vals) > 1.102193)
        self.assertTrue(np.max(areas.vals) < 1.149735)

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

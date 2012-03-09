################################################################################
# oops_/inst/hst/wfc3/ir.py: HST/WFC3 subclass IR
################################################################################

import pyfits
from oops.inst.hst.wfc3 import WFC3

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/WFC3/IR."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the SBN class
    this = IR()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is ACS
    if this.instrument_name(hst_file) != "WFC3":
        raise IOError("not an HST/WFC3 file: " + this.filespec(hst_file))

    # Confirm that the detector is IR
    if this.detector_name(hst_file) != "IR":
        raise IOError("not an HST/WFC3/IR file: " + this.filespec(hst_file))

    return IR.from_opened_fitsfile(hst_file, parameters)

################################################################################
# Class IR
################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ["OTA/hst_ota_???_syn.fits",
                     "WFC3/wfc3_ir_cor_???_syn.fits",
                     "WFC3/wfc3_ir_mask_???_syn.fits",
                     "WFC3/wfc3_ir_mir1_???_syn.fits",
                     "WFC3/wfc3_ir_mir2_???_syn.fits",
                     "WFC3/wfc3_ir_rcp_???_syn.fits",
                     "WFC3/wfc3_ir_primary_???_syn.fits",
                     "WFC3/wfc3_ir_secondary_???_syn.fits",
                     "WFC3/wfc3_ir_win_???_syn.fits"]

CCD_SYN_FILE_PARTS    = ["WFC3/wfc3_ir_ccd", "_???_syn.fits"]
FILTER_SYN_FILE_PARTS = ["WFC3/wfc3_ir_",    "_???_syn.fits"]

class IR(WFC3):
    """This class defines functions and properties unique to the WFC3/IR
    detector. Everything else is inherited from higher levels in the class
    hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    # The IR detector has a single filter wheel. The name is identified by
    # FITS parameter FILTER.
    def filter_name_from_file(self, hst_file):
        """Returns the name of the filter for the WFC3/IR detector.
        """

        return hst_file[0].header["FILTER"]

    # The IDC dictionaries for WFC3/IR are all keyed by (FILTER,).
    def define_fov(self, hst_file, parameters={}):
        """Returns an FOV object defining the field of view of the given image
        file.
        """

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hst_file, ("FILTER",))

        # Define the key into the dictionary
        idc_key = (hst_file[0].header["FILTER"],)

        return self.construct_fov(IDC_DICT[idc_key], hst_file)

    def select_syn_files(self, hst_file, parameters={}):
        """Returns the list of SYN files containing profiles that are to be
        multiplied together to obtain the throughput of the given instrument,
        detector and filter combination."""

        global GENERAL_SYN_FILES, CCD_SYN_FILE_PARTS, FILTER_SYN_FILE_PARTS

        # Copy all the standard file names
        syn_filenames = []
        for filename in GENERAL_SYN_FILES:
            syn_filenames.append(filename)

        # Add the filter file name
        syn_filenames.append(FILTER_SYN_FILE_PARTS[0] +
                             hst_file[0].header["FILTER"].lower() +
                             FILTER_SYN_FILE_PARTS[1])

        return syn_filenames

    @staticmethod
    def from_opened_fitsfile(hst_file, parameters={}):
        """A general class method to return an Observation object based on an
        HST data file generated by HST/WFC3/IR."""
    
        return IR().construct_snapshot(hst_file, parameters)

################################################################################
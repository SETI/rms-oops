################################################################################
# oops/hosts/hst/nicmos/nic1.py: HST/NICMOS subclass NIC1
################################################################################

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
from . import NICMOS

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, **parameters):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/NICMOS/NIC1.
    """

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the NIC1 class
    this = NIC1()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is NICMOS
    if this.instrument_name(hst_file) != "NICMOS":
        raise IOError("not an HST/NICMOS file: " + this.filespec(hst_file))

    # Confirm that the detector is NIC1
    if this.detector_name(hst_file) != "IR":
        raise IOError("not an HST/NICMOS/NIC1 file: " + this.filespec(hst_file))

    return NIC1.from_opened_fitsfile(hst_file, **parameters)

#===============================================================================
#===============================================================================
class NIC1(NICMOS):
    """This class defines functions and properties unique to the NIC1 detector.
    Everything else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    # Used by select_syn_files, defined in NICMOS.py
    DETECTOR_SYN_FILES = ["NICMOS/nic1_bend_???_syn.fits",
                          "NICMOS/nic1_cmask_???_syn.fits",
                          "NICMOS/nic1_dewar_???_syn.fits",
                          "NICMOS/nic1_image_???_syn.fits",
                          "NICMOS/nic1_para1_???_syn.fits",
                          "NICMOS/nic1_para2_???_syn.fits"]

    FILTER_SYN_FILE_PARTS = ["NICMOS/nic1_", "_???_syn.fits"]

    @staticmethod
    def from_opened_fitsfile(hst_file, **parameters):
        """A general class method to return an Observation object based on an
        HST data file generated by HST/NICMOS/NIC1.
        """

        return NIC1().construct_snapshot(hst_file, **parameters)

################################################################################

################################################################################
# oops_/inst/hst/acs/__init__.py: HST subclass ACS
################################################################################

import pyfits
import oops_.all as oops
from oops_.inst.hst import HST

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/ACS."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the ACS class
    this = ACS()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is ACS
    if this.instrument_name(hst_file) != "ACS":
        raise IOError("not an HST/ACS file: " + this.filespec(hst_file))

    return ACS.from_opened_fitsfile(hst_file, parameters)

################################################################################
# Class ACS
################################################################################

class ACS(HST):
    """This class defines functions and properties unique to the ACS instrument.
    Everything else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    # The HRC and WFC detectors have two filter wheels. Names are identified by
    # FITS parameters FILTER1 and FILTER2 in the first header. SBC overrides
    # this method because only FILTER1 is used.

    def filter_name(self, hst_file):
        """Returns the name of the filter for this particular ACS detector.
        Overlapped filters are separated by a plus sign."""

        name1 = hst_file[0].header["FILTER1"]
        name2 = hst_file[0].header["FILTER2"]

        if name1[0:5] == "CLEAR":
            if name2[0:5] == "CLEAR":
                return "CLEAR"
            else:
                return name2
        else:
            if name2[0:5] == "CLEAR":
                return name1
            else:
                return name1 + "+" + name2

    @staticmethod
    def from_opened_fitsfile(hst_file, parameters={}):
        """A general, static method to return an Observation object based on an
        HST data file generated by HST/ACS."""
    
        # Make an instance of the ACS class
        this = ACS()
    
        # Figure out the detector
        detector = this.detector_name(hst_file)
    
        if detector == "HRC":
            from oops_.inst.hst.acs.hrc import HRC
            obs = HRC.from_opened_fitsfile(hst_file, parameters)
    
        elif detector == "WFC":
            from oops_.inst.hst.acs.wfc import WFC
            obs = WFC.from_opened_fitsfile(hst_file, parameters)
    
        elif detector == "SBC":
            from oops_.inst.hst.acs.sbc import SBC
            obs = SBC.from_opened_fitsfile(hst_file, parameters)
    
        else:
            raise IOError("unsupported detector in HST/ACS file " +
                          this.filespec(hst_file) + ": " + detector)
    
        # Insert subfields common to all ACS images
        obs.insert_subfield("detector", detector)
        obs.insert_subfield("filter", ACS().filter_name(hst_file))
        obs.insert_subfield("quality", hst_file[2].data)
        obs.insert_subfield("error", hst_file[3].data)
    
        return obs

################################################################################

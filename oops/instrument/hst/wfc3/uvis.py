import oops

################################################################################
# Functions for instrument.hst.wfc3.uvis
################################################################################

def from_file(file_spec):
    """This function returns an Observation object for an HST/WFC3/UVIS data
    file."""

    return oops.instrument.hst.from_hst_image_file(file_spec)

################################################################################

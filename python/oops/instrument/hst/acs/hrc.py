import oops

################################################################################
# Functions for instrument.hst.acs.hrc
################################################################################

def from_file(file_spec):
    """This function returns an Observation object for an HST/ACS/HRC data
    file."""

    return oops.instrument.hst.from_hst_image_file(file_spec)

################################################################################

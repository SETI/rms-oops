##########################################################################################
# oops/hosts/hst/acs/hrc.py
##########################################################################################

import astropy.io.fits as pyfits

from . import ACS

from filecache import FCPath

##########################################################################################
# Standard class methods
##########################################################################################

def from_file(filespec, **parameters):
    """A general, static method to return an Observation object based on a given data file
    generated by HST/ACS/HRC.
    """

    filespec = FCPath(filespec)

    # Open the file
    local_path = filespec.retrieve()
    hdulist = pyfits.open(local_path)

    # Make an instance of the HRC class
    this = HRC()

    # Confirm that the telescope is HST
    if this.telescope_name(hdulist) != 'HST':
        raise IOError(f'not an HST file: {filespec}')

    # Confirm that the instrument is ACS
    if this.instrument_name(hdulist) != 'ACS':
        raise IOError(f'not an HST/ACS file: {filespec}')

    # Confirm that the detector is HRC
    if this.detector_name(hdulist) != 'HRC':
        raise IOError(f'not an HST/ACS/HRC file: {filespec}')

    return HRC.from_opened_fitsfile(hdulist, **parameters)

##########################################################################################
# Class HRC
##########################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ['OTA/hst_ota_???_syn.fits',
                     'ACS/acs_hrc_win_???_syn.fits',
                     'ACS/acs_hrc_m12_???_syn.fits',
                     'ACS/acs_hrc_m3_???_syn.fits',
                     'ACS/acs_hrc_ccd_mjd_???_syn.fits']

CORONOGRAPH_SYN_FILE = 'ACS/acs_hrc_coron_???_syn.fits'

FILTER_SYN_FILE = ['ACS/acs_', '_???_syn.fits']

class HRC(ACS):
    """This class defines functions and properties unique to the NIC1 detector. Everything
    else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    def define_fov(self, hdulist, **parameters):
        """An FOV object defining the field of view of the given image file."""

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hdulist, ('FILTER1', 'FILTER2'))

        # Define the key into the dictionary
        idc_key = (hdulist[0].header['FILTER1'], hdulist[0].header['FILTER2'])

        # Define the plate scale
        if 'platescale' in parameters:
            platescale = parameters['platescale']
        elif idc_key == ('CLEAR1S', 'CLEAR2S'):
            platescale = 0.9987     # determined empirically for Mab's orbit
        else:
            platescale = 1.

        # Use the default function defined at the HST level for completing the
        # definition of the FOV
        return self.construct_fov(IDC_DICT[idc_key], hdulist, platescale)

    def select_syn_files(self, hdulist, **parameters):
        """The list of SYN files containing profiles that are to be multiplied together to
        obtain the throughput of the given instrument, detector, and filter combination.
        """

        global GENERAL_SYN_FILES, CORONOGRAPH_SYN_FILE, FILTER_SYN_FILE

        # Copy all the standard file names
        syn_filenames = []
        for filename in GENERAL_SYN_FILES:
            syn_filenames.append(filename)

        # Add the filter file names
        for filter_name in (hdulist[0].header['FILTER1'], hdulist[0].header['FILTER2']):

            if filter_name[0:3] == 'POL':
                if filter_name[-2:] == 'UV':
                    filter_name = 'POL_UV'
                else:
                    filter_name = 'POL_V'

            if filter_name[0:5] != 'CLEAR':
                syn_filenames.append(FILTER_SYN_FILE[0] + filter_name.lower() +
                                     FILTER_SYN_FILE[1])

        # Add the coronograph file name if necessary
        if hdulist[0].header['APERTURE'][0:9] == 'HRC-CORON':
            syn_filenames.append(CORONOGRAPH_SYN_FILE)

        return syn_filenames

    @staticmethod
    def from_hdulist(hdulist, **parameters):
        """A general class method to return an Observation object based on an HST data
        file generated by HST/ACS/HRC.
        """

        return HRC().construct_snapshot(hdulist, **parameters)

##########################################################################################

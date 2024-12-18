##########################################################################################
# oops/hosts/hst/wfc3/uvis.py: HST/WFC3 subclass UVIS
##########################################################################################

import astropy.io.fits as pyfits
from . import WFC3

from filecache import FCPath

##########################################################################################
# Standard class method
##########################################################################################

def from_file(filespec, **parameters):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/WFC3/UVIS.
    """

    filespec = FCPath(filespec)

    # Open the file
    local_path = filespec.retrieve()
    hdulist = pyfits.open(local_path)

    # Make an instance of the UVIS class
    this = UVIS()

    # Confirm that the telescope is HST
    if this.telescope_name(hdulist) != 'HST':
        raise IOError(f'not an HST file: {filespec}')

    # Confirm that the instrument is ACS
    if this.instrument_name(hdulist) != 'WFC3':
        raise IOError(f'not an HST/WFC3 file: {filespec}')

    # Confirm that the detector is UVIS
    if this.detector_name(hdulist) != 'UVIS':
        raise IOError(f'not an HST/WFC3/UVIS file: {filespec}')

    return UVIS.from_opened_fitsfile(hdulist, **parameters)

##########################################################################################
# UVIS class
##########################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ['OTA/hst_ota_???_syn.fits',
                     'WFC3/wfc3_uvis_cor_???_syn.fits',
                     'WFC3/wfc3_uvis_iwin_???_syn.fits',
                     'WFC3/wfc3_uvis_mir1_???_syn.fits',
                     'WFC3/wfc3_uvis_mir2_???_syn.fits',
                     'WFC3/wfc3_uvis_owin_???_syn.fits',
                     'WFC3/wfc3_uvis_qyc_???_syn.fits']

CCD_SYN_FILE_PARTS    = ['WFC3/wfc3_uvis_ccd', '_???_syn.fits']
FILTER_SYN_FILE_PARTS = ['WFC3/wfc3_uvis_',    '_???_syn.fits']


class UVIS(WFC3):
    """This class defines functions and properties unique to the WFC3/UVIS
    detector. Everything else is inherited from higher levels in the class
    hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    # The IDC dictionaries for WFC3/IR are all keyed by (FILTER,).
    def define_fov(self, hdulist, **parameters):
        """An FOV object defining the full field of view (ignoring any subarray)
        for the given image file.
        """

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hdulist, ('DETCHIP', 'FILTER'))

        # Define the key into the dictionary
        idc_key = (hdulist[1].header['CCDCHIP'],hdulist[0].header['FILTER'])

        if 'platescale' in parameters:
            platescale = parameters['platescale']
        else:
            platescale = 1.

        return self.construct_fov(IDC_DICT[idc_key], hdulist, platescale=platescale)

    def select_syn_files(self, hdulist, **parameters):
        """The list of SYN files containing profiles that are to be multiplied together to
        obtain the throughput of the given instrument, detector and filter combination.
        """

        global GENERAL_SYN_FILES, CCD_SYN_FILE_PARTS, FILTER_SYN_FILE_PARTS

        # Copy all the standard file names
        syn_filenames = []
        for filename in GENERAL_SYN_FILES:
            syn_filenames.append(filename)

        # Add the filter file name
        syn_filenames.append(FILTER_SYN_FILE_PARTS[0] +
                             hdulist[0].header['FILTER'].lower() +
                             FILTER_SYN_FILE_PARTS[1])

        # Determine the layer of the FITS file to read
        try:
            layer = parameters['layer']
            assert hdulist[layer].header['EXTTYPE'] == 'SCI'
        except KeyError:
            layer = 1

        # Add the CCD file name
        syn_filenames.append(CCD_SYN_FILE_PARTS[0] +
                             str(hdulist[layer].header['CCDCHIP']) +
                             CCD_SYN_FILE_PARTS[1])

        return syn_filenames

    def dn_per_sec_factor(self, hdulist):
        """The factor that converts a pixel value to DN per second.

        Input:
            hdulist    the object returned by pyfits.open()

        Return          the factor to multiply a pixel value by to get DN/sec
        """

        return 1. / hdulist[0].header['EXPTIME']

    @staticmethod
    def from_hdulist(hdulist, **parameters):
        """A general, static method to return an Observation object based on an
        HST data file generated by HST/WFC3/UVIS.
        """

        return UVIS().construct_snapshot(hdulist, **parameters)

##########################################################################################

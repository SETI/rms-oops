##########################################################################################
# oops/host/_fits_support.py
##########################################################################################
"""Shared FITS tools."""

from oops._exceptions import OopsIndexError, OopsKeyError
from ._errors import HostFormatError

__all__ = ['get_fits_image_hdu']


def _hdu_is_image(hdu):
    """True if the given HDU describes a FITS IMAGE."""
    if 'XTENSION' in hdu.header:
        return hdu.header['XTENSION'] == 'IMAGE'
    if hdu.header['NAXIS'] < 2:
        return False
    if 'TFIELDS' in hdu.header:
        return False
    return True


def get_fits_image_hdu(hdulist, index=None):
    """The selected HDU from a FITS HDUList.

    Parameters:
        hdulist (HDUList): HDUList of the FITS file.
        index (int | str, optional): The numeric index or EXTNAME of the HDU to select.
            The default is to return the first IMAGE HDU.

    Returns:
        PrimaryHDU | ImageHDU: The selected image HDU.

    Raises:
        HostFormatError: If `hdulist` does not contain an image array.
        OopsIndexError: If the integer value of `index` is out of range.
        OopsKeyError: If the string value of `index` is not the name of an IMAGE object.
    """

    if index is None:
        image_hdus = [hdu for hdu in hdulist if _hdu_is_image(hdu)]
        if not image_hdus:
            raise HostFormatError('FITS file does not contain an image: '
                                  f'{hdulist.filename()}')
        return image_hdus[0]

    try:
        hdu = hdulist[index]
    except IndexError as err:
        raise OopsIndexError(*err.args) from err
    except KeyError as err:
        raise OopsKeyError(*err.args) from err

    if not _hdu_is_image(hdu):
        raise HostFormatError(f'HDU {index!r} does not contain an image: '
                              f'{hdulist.filename()}')

    return hdu

##########################################################################################

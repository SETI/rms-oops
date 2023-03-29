################################################################################
# hosts/solar/stis.py: HST/STIS model for the solar flux density at 1 AU.
#
# From Bohlin, Dickinson, & Calzetti 2001, Astron. J.
################################################################################

import os

import astropy.io.fits as pyfits
import hosts.solar as solar
import tabulation as tab

# Read the file
filepath = os.path.join(os.path.split(solar.__file__)[0],
                        'stis-sun_reference_stis_002.fits')
hdulist = pyfits.open(filepath)
try:
    table = hdulist[1].data
    wavelength = table['WAVELENGTH']    # Angstroms
    flux = table['FLUX']                # erg/s/cm^2/A
finally:
    hdulist.close()

FLUX_DENSITY = tab.Tabulation(wavelength, flux)
UNITS = 'erg/s/cm^2/A'
XUNITS = 'A'

################################################################################

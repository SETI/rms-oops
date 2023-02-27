################################################################################
# hosts/solar/kurucz.py
#
# From http://kurucz.harvard.edu/stars/sun/, file fsunallp.2000resam125.txt.
################################################################################

import numpy as np
import os

import hosts.solar as solar
import tabulation as tab

# From http://kurucz.harvard.edu/stars/sun/00asun.readme
#
#                        Irradiance
#
#      The fluxes I compute from my model atmosphres are flux
# moments at the surface of the star in ergs/cm**2/s/ster/nm.
# The solar irradiance is the flux at 1 AU from the center of
# the sun.
#
#      To convert from flux moment to irradiance multiply by
# 4*pi*(Rsun/AU)**2 = 2.720E-4 .
#
#      To convert from erg/cm**2/s/nm to W/m**2/micron
# multiply by 1.E-7*1.E4*1.E3 = 1.

RSUN = 695700.
FACTOR = 4 * np.pi * (RSUN/solar.AU)**2

if 'FLUX_DENSITY' not in globals(): # pragma: no cover

    # Read the file
    filepath = os.path.join(os.path.split(solar.__file__)[0],
                            'kurucz-fsunallp.2000resam125.txt')
    array = np.fromfile(filepath, sep=' ')
    array = array.reshape(-1,3)

    # column 1 is wavelength in nm
    # column 2 "flux moment"; see notes above for conversion

    wavelength = array[:,0]
    flux = array[:,1] * FACTOR

    FLUX_DENSITY = tab.Tabulation(wavelength, flux)
    UNITS = 'W/m^2/um'
    XUNITS = 'nm'

################################################################################

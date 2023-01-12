################################################################################
# hosts/solar/__init__.py: Models for the solar flux density at 1 AU.
#
# Models currently supported are:
#   Colina      0.14   to 2.5 micron
#   Kurucz      0.15   to 300 micron
#   Rieke       0.2    to 30  micron
#   STIS        0.1195 to 2.7 micron
#   STIS_Rieke  0.1195 to 30  micron
################################################################################

__all__ = []    # don't import any solar models by default, only as requested.

import functools
import importlib
import numpy as np
import tabulation as tab

# Class constants to be used externally as needed
AU = 149597870.7    # km
C = 299792.458      # km/sec

# Converts from W/m^2 to erg/s/cm^2
TO_CGS = 1.e7 / 1.e4

# Converts from flux per micron to flux per Angstrom
TO_PER_ANGSTROM = 1.e-4

# Converts from flux per micron to flux per nanometer
TO_PER_NM = 1.e-3

# First UNIT_DICT item is conversion factor from W/m^2/um or from W/m^2/Hz.
# Second item is True if the units are per wavelength, False if per frequency.
UNIT_DICT = {
    'W/m^2/um'     : (1.   , True),     # default units
    'W/m^2/nm'     : (1.e-3, True),
    'W/m^2/A'      : (1.e-4, True),
    'erg/s/cm^2/um': (1.e+3, True),
    'erg/s/cm^2/nm': (1.   , True),
    'erg/s/cm^2/A' : (1.e-1, True),

    'W/m^2/Hz'     : (1.   , False),
    'erg/s/cm^2/Hz': (1.e+3, False),
    'Jy'           : (1.e26, False),
    'uJy'          : (1.e32, False),
}

# First XUNIT_DICT item is conversion factor from um or from Hz.
# Second item is True if the units are wavelength, False if frequency.
XUNIT_DICT = {
    'um': (1.  , True),
    'nm': (1.e3, True),
    'A' : (1.e4, True),
    'Hz': (1.  , False),
}

C_IN_UM_HZ = C * 1.e9

#===============================================================================
@functools.lru_cache(maxsize=4)
def flux_density(model='STIS_Rieke', units='W/m^2/um', xunits='um',
                 sun_range=1., solar_f=False):
    """A Tabulation of solar flux density at 1 AU in the specified units.

    Note that the tabulation is always returned in units of microns.

    Input:
        model           name of the model, default "STIS_Rieke".
        units           units to provide, default "W/m^2/um". Options are:
                        "W/m^2/um", "W/m^2/nm", "W/m^2/A", "erg/s/cm^2/um",
                        "erg/s/cm^2/nm", "erg/s/cm^2/A", "W/m^2/Hz",
                        "erg/s/cm^2/Hz", "Jy", "uJy".
        xunits          units for the x-axis, default "um" (for microns).
                        Options are: "um", "nm", "A", or "Hz".
        sun_range       optional distance from Sun to target in AU.
        solar_f         True to divide by pi, providing solar F instead of
                        solar flux density.

    Return:             a Tabulation of the model solar flux density in the
                        specified units.
    """

    # The first act of importing a model causes its value of FLUX_DENSITY to
    # become defined.
    try:
        module = importlib.import_module('hosts.solar.' + model.lower())
    except ImportError:
        raise ValueError('undefined solar model: ' + model)

    # Check units
    if units not in UNIT_DICT:
        raise ValueError('invalid units: ' + units)

    if xunits not in XUNIT_DICT:
        raise ValueError('invalid units: ' + xunits)

    # Get the tabulation
    tabulation = module.FLUX_DENSITY

    # If we have the desired units, return
    if units == module.UNITS and xunits == module.XUNITS:
        return tabulation * ((1./np.pi if solar_f else 1.) / sun_range**2)

    # Gather unit info
    (scale, per_wavelength) = UNIT_DICT[units]
    (xscale, x_is_wavelength) = XUNIT_DICT[xunits]

    (model_scale, model_per_wavelength) = UNIT_DICT[module.UNITS]
    (model_xscale, model_x_is_wavelength) = XUNIT_DICT[module.XUNITS]

    # Create the new x-values
    if x_is_wavelength == model_x_is_wavelength:
        new_x = (xscale / model_xscale) * tabulation.x
    else:
        new_x = (xscale * model_xscale * C_IN_UM_HZ) / tabulation.x

    # Create the new y-values
    factor = scale/model_scale * (1./np.pi if solar_f else 1.) / sun_range**2

    if per_wavelength == model_per_wavelength:
        new_y = factor * tabulation.y

    else:
        # w = wavelength in microns
        # f = frequency in Hz
        #
        # We must satisfy:
        #   flux_w dw = flux_f df
        # so
        #   flux_w = flux_f |df/dw|
        # or
        #   flux_f = flux_w |dw/df|
        #
        # We have
        #   f = C/w
        # so
        #   |df/dw| = C/w^2 = f^2/C
        # or
        #   |dw/df| = C/f^2 = w^2/C

        if per_wavelength:  # we need df/dw
            if model_x_is_wavelength:
                new_y = ((factor * C_IN_UM_HZ * model_xscale**2) *
                         tabulation.y / tabulation.x**2)
            else:
                new_y = ((factor / C_IN_UM_HZ / model_xscale**2) *
                         tabulation.y * tabulation.x**2)

        else:               # we need dw/df
            if model_x_is_wavelength:
                new_y = ((factor / C_IN_UM_HZ / model_xscale**2) *
                         tabulation.y * tabulation.x**2)
            else:
                new_y = ((factor * C_IN_UM_HZ * model_xscale**2) *
                         tabulation.y / tabulation.x**2)

    return tab.Tabulation(new_x, new_y)

#===============================================================================
def bandpass_flux_density(bandpass, model='STIS_Rieke', units='W/m^2/um',
                                    xunits='um', sun_range=1., solar_f=False):
    """The solar flux density averaged over a filter bandpass.

    Input:
        bandpass        the Tabulation of the filter bandpass, with wavelength
                        in microns. Alternatively, a tuple of two arrays
                        (wavelength, flux), each of the same size.
        model           name of the model, default "STIS_Rieke". Alternatively,
                        a Tabulation of the solar flux density, in which case it
                        must be in the desired units already, and must be
                        tabulated in the same units as the bandpass.
        units           units to provide, default "W/m^2/um". Options are:
                        "W/m^2/um", "W/m^2/nm", "W/m^2/A", "erg/s/cm^2/um",
                        "erg/s/cm^2/nm", "erg/s/cm^2/A", "W/m^2/Hz",
                        "erg/s/cm^2/Hz", "Jy", "uJy". Ignored if the solar model
                        is a Tabulation.
        xunits          units for the x-axis of the bandpass, default "um" (for
                        microns). Options are: "um", "nm", "A", or "Hz".
        sun_range       optional distance from Sun to target in AU.
        solar_f         True to divide by pi, providing solar F instead of
                        solar flux density.

    Return:             the mean solar flux density or solar F within the filter
                        bandpass.
    """

    if not isinstance(bandpass, tab.Tabulation):
        bandpass = tab.Tabulation(*bandpass)

    if isinstance(model, tab.Tabulation):
        flux = model * (1./np.pi if solar_f else 1.) / sun_range**2
    else:
        flux = flux_density(model, units=units, xunits=xunits,
                                   sun_range=sun_range, solar_f=solar_f)

    # Multiply together the bandpass and the solar spectrum Tabulations
    product = bandpass * flux

    # Resample the bandpass at the same wavelengths for a more reliable
    # normalization
    bandpass = bandpass.resample(product.x)

    # Return the ratio of integrals
    return product.integral() / bandpass.integral()

#===============================================================================
def mean_flux_density(center, width, model='STIS_Rieke', units='W/m^2/um',
                                     xunits='um', sun_range=1., solar_f=False):
    """The solar flux density averaged over the bandpass of a "boxcar" filter,
    given its center and full width.

    Input:
        center          the center of the bandpass.
        width           the full width of the bandpass.
        model           name of the model, default "STIS_Rieke". Alternatively,
                        a Tabulation of the solar flux density, in which case it
                        must be in the desired units already, and must be
                        tabulated in the same units as the bandpass.
        units           units to provide, default "W/m^2/um". Options are:
                        "W/m^2/um", "W/m^2/nm", "W/m^2/A", "erg/s/cm^2/um",
                        "erg/s/cm^2/nm", "erg/s/cm^2/A", "W/m^2/Hz",
                        "erg/s/cm^2/Hz", "Jy", "microJy", "uJy". Ignored if the
                        solar model is a Tabulation.
        xunits          units for bandpass center and width. Options are: "um"
                        (for microns), "nm", "A", or "Hz". Default is "um".
        sun_range       optional distance from Sun to target in AU.
        solar_f         True to divide by pi, providing solar F instead of
                        solar flux density.

    Return:             the mean solar flux density or solar F within the filter
                        bandpass.
    """

    # Create a boxcar filter Tabulation
    bandpass = tab.Tabulation((center - width/2., center + width/2.), (1.,1.))

    # Return the mean over the filter
    return bandpass_flux_density(bandpass, model=model, units=units,
                                           xunits=xunits, sun_range=sun_range,
                                           solar_f=solar_f)

#===============================================================================
def bandpass_f(bandpass, model='STIS_Rieke', units='W/m^2/um', xunits='um',
                         sun_range=1.):
    """Solar F averaged over a filter bandpass.

    F is defined such that pi*F is the solar flux density.

    Input:
        bandpass        the Tabulation of the filter bandpass, with wavelength
                        in microns. Alternatively, a tuple of two arrays
                        (wavelength, flux), each of the same size.
        model           name of the model, default "STIS_Rieke". Alternatively,
                        a Tabulation of the solar flux density, in which case it
                        must be in the desired units already, and must be
                        tabulated in the same units as the bandpass.
        units           units to provide, default "W/m^2/um". Options are:
                        "W/m^2/um", "W/m^2/nm", "W/m^2/A", "erg/s/cm^2/um",
                        "erg/s/cm^2/nm", "erg/s/cm^2/A", "W/m^2/Hz",
                        "erg/s/cm^2/Hz", "Jy", "uJy". Ignored if the solar model
                        is a Tabulation.
        xunits          units for the x-axis of the bandpass, default "um" (for
                        microns). Options are: "um", "nm", "A", or "Hz".
        sun_range       optional distance from Sun to target in AU.

    Return:             the mean solar F within the filter bandpass.
    """

    return bandpass_flux_density(bandpass, model='STIS_Rieke',
                                           units='W/m^2/um', xunits='um',
                                           sun_range=1., solar_f=True)

#===============================================================================
def mean_f(center, width, model='STIS_Rieke', units='W/m^2/um', xunits='um',
                          sun_range=1.):
    """The solar F averaged over the bandpass of a "boxcar" filter, given its
    center and full width.

    Input:
        center          the center of the bandpass.
        width           the full width of the bandpass.
        model           name of the model, default "STIS_Rieke". Alternatively,
                        a Tabulation of the solar flux density, in which case it
                        must be in the desired units already, and must be
                        tabulated in the same units as the bandpass.
        units           units to provide, default "W/m^2/um". Options are:
                        "W/m^2/um", "W/m^2/nm", "W/m^2/A", "erg/s/cm^2/um",
                        "erg/s/cm^2/nm", "erg/s/cm^2/A", "W/m^2/Hz",
                        "erg/s/cm^2/Hz", "Jy", "uJy". Ignored if the solar model
                        is a Tabulation.
        xunits          units for bandpass center and width. Options are: "um"
                        (for microns), "nm", "A", or "Hz". Default is "um".
        sun_range       optional distance from Sun to target in AU.

    Return:             the mean solar F within the specified bandpass.
    """

    return mean_flux_density(center, width, model='STIS_Rieke',
                                            units='W/m^2/um', xunits='um',
                                            sun_range=1., solar_f=True)

################################################################################

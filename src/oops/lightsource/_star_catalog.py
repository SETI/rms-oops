##########################################################################################
# oops/lightsource/_star_catalog.py
##########################################################################################
"""Support for stars as LightSource objects."""

from collections import namedtuple

import numpy as np

from oops._exceptions import OopsKeyError, OopsValueError
from oops.body import Body
from oops.constants import RPD, RPS, SPD
from oops.lightsource import LightSource
from oops.lightsource.starsource import StarSource

__all__ = ['star_lookup', 'add_star']

# One catalog entry: the star's primary name, its common name or None, its J2000 right
# ascension and declination in degrees, its type information or None, its magnitude or
# None, the band of that magnitude ("V", "Ks" or "UV1565") or None, its parallax in
# radians, and its proper motion as d(ra,dec)/dt in radians per second (the rate of right
# ascension not scaled by cos(dec)) or None.
_StarEntry = namedtuple('_StarEntry', ['name', 'common_name', 'ra', 'dec', 'type_info',
                                       'magnitude', 'mag_type', 'parallax',
                                       'proper_motion'])

_STAR_LOOKUP = {}           # _StarEntry keyed by every upper-case name of the star

_RADIANS_PER_MAS = RPS / 1000.
_SECONDS_PER_YEAR = 365.25 * SPD        # the Julian year of catalog proper motions

_GREEK_LETTER_ABBREVS = {
    'Alpha'  : ('alf', 'alp'),
    'Beta'   : ('bet',),
    'Gamma'  : ('gam',),
    'Delta'  : ('del',),
    'Epsilon': ('eps',),
    'Zeta'   : ('zet',),
    'Eta'    : ('eta',),
    'Theta'  : ('tet', 'the'),
    'Iota'   : ('iot',),
    'Kappa'  : ('kap',),
    'Lambda' : ('lam',),
    'Mu'     : ('mu.', 'mu'),
    'Nu'     : ('nu.', 'nu'),
    'Xi'     : ('ksi', 'xi'),
    'Omicron': ('omi',),
    'Pi'     : ('pi.', 'pi'),
    'Rho'    : ('rho',),
    'Sigma'  : ('sig',),
    'Tau'    : ('tau',),
    'Upsilon': ('ups',),
    'Phi'    : ('phi',),
    'Chi'    : ('khi', 'chi'),
    'Psi'    : ('psi',),
    'Omega'  : ('ome',),
}


def _initialize_star_lookup():
    """Populate the _STAR_LOOKUP dictionary."""

    from ._BRIGHTEST_STARS    import _BRIGHTEST_STARS
    from ._BRIGHTEST_IR_STARS import _BRIGHTEST_IR_STARS
    from ._BRIGHTEST_UV_STARS import _BRIGHTEST_UV_STARS
    from ._MESSIER_OBJECTS    import _MESSIER_OBJECTS
    from ._CONSTELLATIONS     import _CONSTELLATION_ABBREVS

    # IR and UV stars first so a matching star's V magnitude takes precedence
    stars = (_BRIGHTEST_IR_STARS + _BRIGHTEST_UV_STARS + _BRIGHTEST_STARS
             + _MESSIER_OBJECTS)
    alt_keys = {}
    for (full_name, common_name, ra, dec, type_info, mag, mag_type, pm_ra, pm_dec,
         parallax_mas) in stars:
        keys = []
        if common_name:
            keys.append(common_name)

        # Separate first word from remainder
        first, _, remainder = full_name.partition(' ')

        # For Messier names, "Messier 31" -> "M31"
        if first == 'Messier':
            short_names = ['M' + remainder]

        # For stars, generate short names with and without "." and "0"
        else:
            letter_plus_digit = False
            if first.isdigit():                         # a number alone is unchanged
                letter = ''
                nums = [first]
            elif first[-1].isdigit():                   # separate a trailing digit
                nums = ['0' + first[-1], first[-1]]     # "1" might be "01"
                letter = first[:-1]
                letter_plus_digit = True
            else:
                letter = first
                nums = ['']
            letters = _GREEK_LETTER_ABBREVS.get(letter, [letter])
            constellation = _CONSTELLATION_ABBREVS.get(remainder, remainder)

            short_names = []
            for letter in letters:
                for num in nums:
                    short_name = letter + num + '_' + constellation
                    short_names.append(short_name)
                if letter_plus_digit:                   # include desig without digit
                    alt_key = (letter + '_' + constellation).upper()
                    if alt_key in alt_keys:             # ...but only if unambiguous
                        alt_keys[alt_key] = ''          # sentinel value
                    else:
                        alt_keys[alt_key] = short_name.upper()

        keys += short_names

        # The catalog's RA rate is scaled by cos(dec), as is conventional; StarSource's
        # is not
        scale = _RADIANS_PER_MAS / _SECONDS_PER_YEAR
        proper_motion = (pm_ra / np.cos(dec * RPD) * scale, pm_dec * scale)
        parallax = max(parallax_mas, 0.) * _RADIANS_PER_MAS

        result = _StarEntry(short_names[0], common_name, ra, dec, type_info, mag,
                            mag_type, parallax, proper_motion)
        for key in keys:
            _STAR_LOOKUP[key.upper().replace(' ', '_')] = result

        for alt_key, key in alt_keys.items():
            if key:  # if not sentinel value
                _STAR_LOOKUP[alt_key] = _STAR_LOOKUP[key]


def star_lookup(key):
    """The light source associated with the given star identifier.

    Parameters:
        key (str): The star's abbreviated name (e.g., "alf CMa"), or its common name
            (e.g., "Sirius"). Case is ignored; spaces are replaced with underscores.

    Returns:
        LightSource: The LightSource already registered under `key`, if any; otherwise, a
        new :class:`~oops.lightsource.StarSource` at the star's J2000 position, with the
        catalog's magnitude, magnitude type, type, parallax and proper motion, registered
        under `key` in upper case and with spaces replaced with underscores. A star whose
        catalog parallax is not positive is given zero parallax.

    Raises:
        OopsKeyError: If `key` does not identify one of the stars in the table.
        OopsValueError: If `key` is already the name of a Body that is not a LightSource.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    key = key.upper().replace(' ', '_')
    if key not in _STAR_LOOKUP:
        # Try missing underscore, e.g., "ALPVIR" -> "ALP_VIR"
        if (4 <= len(key) <= 6) and '_' not in key:
            alt_key = key[:-3] + '_' + key[-3:]
            if alt_key not in _STAR_LOOKUP:
                raise OopsKeyError(f'unknown star "{key}"')
            key = alt_key

    entry = _STAR_LOOKUP[key]

    if Body.exists(key):
        lightsource = Body.lookup(key)
        if not isinstance(lightsource, LightSource):
            raise OopsValueError(f'Star name is also a Body name: {key}')
        return lightsource

    return StarSource(key, entry.ra, entry.dec, magnitude=entry.magnitude,
                      mag_type=entry.mag_type, stellar_type=entry.type_info,
                      parallax=entry.parallax, proper_motion=entry.proper_motion)


def add_star(names, ra, dec, mag=None, type_info=None, *, mag_type=None, parallax=0.,
             proper_motion=None):
    """Add one star to the catalog.

    Each name is registered in upper case with spaces replaced by underscores. A name
    already in the catalog at the same coordinates is left unchanged, keeping its
    existing names, magnitude, magnitude type, type information, parallax, and proper
    motion; any new names are added. A host can therefore add its stars every time it is
    initialized.

    Parameters:
        names (str | list[str]): One or more names for the star.
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        mag (float, optional): Star magnitude.
        type_info (str, optional): Star type information.
        mag_type (str, optional): The band of `mag`, such as "V" for a visual magnitude.
        parallax (float, optional): The annual parallax of the star in radians; default
            zero.
        proper_motion (PairLike, optional): The time derivative of the star's (right
            ascension, declination) in radians per second, as
            :class:`~oops.lightsource.StarSource` takes it; default zero. The rate of
            right ascension is not scaled by the cosine of the declination.

    Raises:
        OopsValueError: If any of `names` is already in the catalog at a different right
            ascension or declination. No names are added in that case.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    if isinstance(names, str):
        names = [names]

    keys = [name.upper().replace(' ', '_') for name in names]

    common_name = names[1] if len(names) > 1 else None
    result = _StarEntry(names[0], common_name, ra, dec, type_info, mag, mag_type,
                        parallax, proper_motion)

    for key in keys:
        entry = _STAR_LOOKUP.get(key)
        if entry and (entry.ra, entry.dec) != (ra, dec):
            raise OopsValueError(f'Star {key} already exists with different coordinates')

    for key in keys:
        _STAR_LOOKUP.setdefault(key, result)

##########################################################################################

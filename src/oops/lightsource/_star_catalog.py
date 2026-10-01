##########################################################################################
# oops/lightsource/_star_catalog.py
##########################################################################################
"""Support for stars as LightSource objects."""

from oops._exceptions import OopsKeyError, OopsValueError
from oops.body import Body
from oops.lightsource import LightSource

__all__ = ['star_lookup', 'add_star']

_STAR_LOOKUP = {}

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
    for (full_name, common_name, ra, dec, type_info, vmag) in stars:
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
            if first.isdigit():                         # a number alone is unchanged
                letter = ''
                nums = [first]
            elif first[-1].isdigit():                   # separate a trailing digit
                nums = ['0' + first[-1], first[-1]]     # "1" might be "01"
                letter = first[:-1]
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

        keys += short_names

        result = (short_names[0], common_name, ra, dec, type_info, vmag)
        for key in keys:
            _STAR_LOOKUP[key.upper().replace(' ', '_')] = result


def star_lookup(key):
    """The LightSource object associated with the given star identifier.

    Parameters:
        key (str): The star's abbreviated name (e.g., "alf CMa"), or its common name
            (e.g., "Sirius"). Case is ignored; spaces are replaced with underscores.

    Returns:
        LightSource: The LightSource already registered under `key`, if any; otherwise, a
        new :class:`~oops.lightsource.LightSource` at the star's J2000 position,
        registered under `key` in upper case and with spaces replaced with underscores.

    Raises:
        OopsKeyError: If `key` does not identify one of the stars in the table.
        OopsValueError: If `key` is already the name of a Body that is not a LightSource.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    key = key.upper().replace(' ', '_')
    if key not in _STAR_LOOKUP:
        # Try missing underscore, e.g., "ALPVIR" -> "ALP_VIR"
        if (5 <= len(key) <= 7) and '_' not in key:
            alt_key = key[:-3] + '_' + key[-3]
            if alt_key not in _STAR_LOOKUP:
                raise OopsKeyError(f'unknown star "{key}"')
            key = alt_key

    (_, _, ra, dec, _, _) = _STAR_LOOKUP[key]

    if Body.exists(key):
        lightsource = Body.lookup(key)
        if not isinstance(lightsource, LightSource):
            raise OopsValueError(f'Star name is also a Body name: {key}')
        return lightsource

    return LightSource(key, (ra, dec))


def add_star(names, ra, dec, mag=None, type_info=None):
    """Add one star to the catalog.

    Each name is registered in upper case with spaces replaced by underscores. A name
    already in the catalog at the same coordinates is left unchanged, keeping its
    existing names, magnitude, and type information; any new names are added. A host can
    therefore add its stars every time it is initialized.

    Parameters:
        names (str | list[str]): One or more names for the star.
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        mag (float, optional): Star magnitude.
        type_info (str, optional): Star type information.

    Raises:
        OopsValueError: If any of `names` is already in the catalog at a different right
            ascension or declination. No names are added in that case.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    if isinstance(names, str):
        names = [names]

    keys = [name.upper().replace(' ', '_') for name in names]

    names = list(names)
    while len(names) < 2:
        names.append(None)
    result = tuple(names[:2] + [ra, dec, type_info, mag])

    for key in keys:
        if key in _STAR_LOOKUP and _STAR_LOOKUP[key][2:4] != result[2:4]:
            raise OopsValueError(f'Star {key} already exists with different coordinates')

    for key in keys:
        _STAR_LOOKUP.setdefault(key, result)

##########################################################################################

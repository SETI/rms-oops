##########################################################################################
# oops/lightsource/_star_catalog.py
##########################################################################################
"""Support for stars as LightSource objects."""

from collections import namedtuple

from oops._exceptions import OopsKeyError, OopsValueError
from oops.body import Body
from oops.lightsource import LightSource
from oops.lightsource.starsource import StarSource

__all__ = ['star_lookup', 'add_star']

# One catalog entry: the star's primary name, its common name or None, its J2000 right
# ascension and declination in degrees, its type information or None, its magnitude or
# None, the band of that magnitude ("V", "Ks" or "UV1565") or None, its proper motion in
# mas/yr as d(ra)/dt * cos(dec) and d(dec)/dt, as catalogs give them, and its parallax in
# mas. The fields follow the column order of the catalog tables. A parallax that is not
# positive is kept as published.
_StarTuple = namedtuple('_StarTuple', ['name', 'more', 'ra', 'dec', 'type_info',
                                       'magnitude', 'mag_type', 'ra_prop', 'dec_prop',
                                       'parallax'])

_STAR_LOOKUP = {}           # _StarTuple keyed by LightSource.as_key of every star name
_FOLDED_KEYS = {}           # the _STAR_LOOKUP keys sharing each upper-case key

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
    from ._EXTRA_STARS        import _EXTRA_STARS
    from ._MESSIER_OBJECTS    import _MESSIER_OBJECTS
    from ._CONSTELLATIONS     import _CONSTELLATION_ABBREVS

    # IR and UV stars first so a matching star's V magnitude takes precedence
    stars = (_BRIGHTEST_IR_STARS + _BRIGHTEST_UV_STARS + _BRIGHTEST_STARS
             + _MESSIER_OBJECTS + _EXTRA_STARS)
    alt_names = {}
    for star in stars:
        star = _StarTuple(*star)

        # Separate first word from remainder
        first, _, remainder = star.name.partition(' ')

        # For Messier names, "Messier 31" -> "M31"
        if first == 'Messier':
            names = ['M' + remainder]

        # For stars, generate short names with and without "." and "0"
        else:
            letter_plus_digit = False
            if first.isdigit():                         # a number alone is unchanged
                letter = ''
                nums = [first]
            elif first[0] == 'V' and first[1:].isdigit():   # "V744" is one designation
                letter = first
                nums = ['']
            elif first[-1].isdigit():                   # separate a trailing digit
                nums = ['0' + first[-1], first[-1]]     # "1" might be "01"
                letter = first[:-1]
                letter_plus_digit = True
            else:
                letter = first
                nums = ['']
            letters = _GREEK_LETTER_ABBREVS.get(letter, [letter])
            constellation = _CONSTELLATION_ABBREVS.get(remainder, remainder)

            names = []
            for letter in letters:
                for num in nums:
                    name = letter + num + ' ' + constellation
                    names.append(name)
                if letter_plus_digit:                   # include desig without digit
                    alt_name = letter + '_' + constellation
                    if alt_name in alt_names:           # ...but only if unambiguous
                        alt_names[alt_name] = ''        # sentinel value
                    else:
                        alt_names[alt_name] = names[0]

        names.append(star.name)
        if star.more:
            if isinstance(star.more, str):
                names.append(star.more)
            else:
                names += star.more

        result = _StarTuple(names[0], names[1:], *star[2:])
        for name in names:
            _set_star(LightSource.as_key(name), result)

        for alt_name, name in alt_names.items():
            if name:  # if not sentinel value
                result = _STAR_LOOKUP[LightSource.as_key(name)]
                _set_star(LightSource.as_key(alt_name), result)


def _set_star(key, star, *, replace=True):
    """Enter one key of a star into _STAR_LOOKUP and _FOLDED_KEYS.

    Parameters:
        key (str): The key, as given by LightSource.as_key.
        star (_StarTuple): The star.
        replace (bool, optional): True to replace a star already under this key; False
            to keep it.
    """

    if replace or key not in _STAR_LOOKUP:
        _STAR_LOOKUP[key] = star
    _FOLDED_KEYS.setdefault(key.upper(), set()).add(key)


def _find_star(key, name):
    """The catalog entry for a key, matched case-insensitively unless that is ambiguous.

    Parameters:
        key (str): The key, as given by LightSource.as_key.
        name (str): The name as given, for the error message.

    Returns:
        _StarTuple | None: The entry whose key is `key`; otherwise, the one entry whose key
        differs from it only in case; otherwise, None.

    Raises:
        OopsKeyError: If `key` is not in the catalog and the keys that differ from it only
            in case belong to more than one star.
    """

    if key in _STAR_LOOKUP:
        return _STAR_LOOKUP[key]

    # A case-insensitive match is accepted only if every matching key names one star, so
    # that "t Cep" finds "T Cep" while "B Cen" and "b Cen" stay distinct
    keys = _FOLDED_KEYS.get(key.upper(), set())
    stars = {_STAR_LOOKUP[k].name: _STAR_LOOKUP[k] for k in keys}
    if len(stars) > 1:
        raise OopsKeyError(f'ambiguous star "{name}": could be '
                           + ', '.join(sorted(stars)))
    return next(iter(stars.values()), None)


def star_lookup(name):
    """The light source associated with the given star identifier.

    Parameters:
        name (str): The star's abbreviated name (e.g., "alf CMa"), or its common name
            (e.g., "Sirius"). Case matters only where two stars have names that differ
            only in case, such as "B Cen" and "b Cen"; otherwise "t cep" finds "T Cep".

    Returns:
        LightSource: The LightSource already registered under the star's primary catalog
        name, if any; otherwise, a new :class:`~oops.lightsource.StarSource` at the
        star's J2000 position, with the catalog's magnitude, magnitude type, type,
        parallax and proper motion. It is named and registered under the primary name
        (e.g., "alf CMa", whose key is "ALF_CMA"), and the star's other catalog names
        become its alternative names. Every name of the star therefore returns the same
        object. A star whose catalog parallax is not positive is given zero parallax.

    Raises:
        OopsKeyError: If `name` does not identify one of the stars in the table, or if it
            matches no name exactly and matches the names of more than one star when
            case is ignored.
        OopsValueError: If the star's primary name is already the name of a Body that is
            not a LightSource.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    star = _find_star(LightSource.as_key(name), name)
    if star is None:
        # Try missing underscore, e.g., "ALFVIR" -> "ALF_VIR", "bCen" -> "b_CEN"
        if (4 <= len(name) <= 7) and ' ' not in name and '_' not in name:
            star = _find_star(LightSource.as_key(name[:-3] + ' ' + name[-3:]), name)
        if star is None:
            raise OopsKeyError(f'unknown star "{name}"')

    # The source is registered under its primary name, whichever name it was looked up by
    primary_key = LightSource.as_key(star.name)
    if primary_key in Body.BODY_REGISTRY:
        lightsource = Body.BODY_REGISTRY[primary_key]
        if not isinstance(lightsource, LightSource):
            raise OopsValueError(f'Star name is also a Body name: {primary_key}')
        return lightsource

    source = StarSource(star.name, star.ra, star.dec, magnitude=star.magnitude,
                        mag_type=star.mag_type, stellar_type=star.type_info,
                        parallax=max(star.parallax, 0.),
                        proper_motion=(star.ra_prop, star.dec_prop))
    source.add_name(star.more)
    return source


def add_star(names, ra, dec, mag=None, type_info=None, *, mag_type=None, parallax=0.,
             proper_motion=(0., 0.)):
    """Add one star to the catalog.

    Each name is registered under its key, as given by
    :meth:`~oops.lightsource.LightSource.as_key`. A name already in the catalog at the
    same coordinates is left unchanged, keeping its existing names, magnitude, magnitude
    type, type information, parallax, and proper motion; any new names are added. A host
    can therefore add its stars every time it is initialized.

    Parameters:
        names (str | list[str]): One or more names for the star.
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        mag (float, optional): Star magnitude.
        type_info (str, optional): Star type information.
        mag_type (str, optional): The band of `mag`, such as "V" for a visual magnitude.
        parallax (float, optional): The annual parallax of the star in milliarcseconds;
            default zero. A value that is not positive is treated as zero.
        proper_motion (tuple[float, float], optional): The proper motion of the star in
            milliarcseconds per year, as (d(ra)/dt * cos(dec), d(dec)/dt), the form in
            which catalogs give it; default zero.

    Raises:
        OopsValueError: If any of `names` is already in the catalog at a different right
            ascension or declination. No names are added in that case.
    """

    if not _STAR_LOOKUP:
        _initialize_star_lookup()

    if isinstance(names, str):
        names = [names]

    keys = [LightSource.as_key(name) for name in names]
    for i, key in enumerate(keys):
        star = _STAR_LOOKUP.get(key)
        if star and (star.ra, star.dec) != (ra, dec):
            raise OopsValueError(f'Star {names[i]} already exists with different '
                                 'coordinates')

    result = _StarTuple(names[0], names[1:], ra, dec, type_info, mag, mag_type,
                        proper_motion[0], proper_motion[1], parallax)
    for key in keys:
        _set_star(key, result, replace=False)

##########################################################################################

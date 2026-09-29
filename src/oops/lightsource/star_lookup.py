##########################################################################################
# oops/lightsource/star_lookup.py
##########################################################################################
"""Support for stars as LightSource objects."""

from oops.body import Body
from oops.lightsource import LightSource

__all__ = ['star_lookup']

_STAR_LOOKUP = {}

_GREEK_LETTER_ABBREVS = {
    'Alpha'  : 'alf',
    'Beta'   : 'bet',
    'Gamma'  : 'gam',
    'Delta'  : 'del',
    'Epsilon': 'eps',
    'Zeta'   : 'zet',
    'Eta'    : 'eta',
    'Theta'  : 'tet',
    'Iota'   : 'iot',
    'Kappa'  : 'kap',
    'Lambda' : 'lam',
    'Mu'     : 'mu.',
    'Nu'     : 'nu.',
    'Xi'     : 'ksi',
    'Omicron': 'omi',
    'Pi'     : 'pi.',
    'Rho'    : 'rho',
    'Sigma'  : 'sig',
    'Tau'    : 'tau',
    'Upsilon': 'ups',
    'Phi'    : 'phi',
    'Chi'    : 'khi',
    'Psi'    : 'psi',
    'Omega'  : 'ome',
}


def _initialize_star_lookup():
    """Populate the _STAR_LOOKUP dictionary."""

    from ._BRIGHTEST_STARS import _BRIGHTEST_STARS
    from ._CONSTELLATIONS import _CONSTELLATION_ABBREVS

    for (full_name, common_name, ra, dec, spectype, vmag) in _BRIGHTEST_STARS:
        keys = [full_name]
        if common_name:
            keys.append(common_name)
        letter, _, constellation = full_name.partition(' ')
        if letter[-1].isdigit():
            digit = '0' + letter[-1]
            letter = letter[:-1]
        else:
            digit = ''
        const = _CONSTELLATION_ABBREVS.get(constellation, constellation)
        short_name = _GREEK_LETTER_ABBREVS.get(letter, letter) + digit + ' ' + const
        keys.append(short_name)

        result = (short_name, common_name, ra, dec, spectype, vmag)
        for key in keys:
            _STAR_LOOKUP[key.upper()] = result


def star_lookup(key):
    """The LightSource object associated with the given star identifier."""
    if not _STAR_LOOKUP:
        _initialize_star_lookup()
    key_upper = key.upper()
    if key_upper not in _STAR_LOOKUP:
        raise KeyError(f'unknown star "{key}"')
    (_, _, ra, dec, _, _) = _STAR_LOOKUP[key_upper]
    if Body.exists(key):
        return Body.lookup(key)
    return LightSource(key, (ra, dec))

##########################################################################################

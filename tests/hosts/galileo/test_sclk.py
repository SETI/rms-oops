##########################################################################################
# tests/hosts/galileo/test_sclk.py
##########################################################################################
"""Conversion of Galileo spacecraft clock counts to seconds TDB.

The conversion must work without the mission initializer, loading only the leap-second
and spacecraft clock kernels, so that an index generator can call it cheaply.
"""

import json
import subprocess
import sys

import cspyce
import julian
import pytest

from oops.hosts.galileo import Galileo

# GO_0002/RAW_CAL/C0003061200R.LBL, a checkout frame nine days after launch
LABEL_COUNT = '00030612.00'
SPICE_COUNT = '30612:0:0:0'
EXPECTED_ISO_PREFIX = '1989-10-27T21:11:'

# One mod-91 tick of the Galileo clock, in seconds
MOD91_TICK = 60.667 / 91

# Run in a fresh interpreter: convert a count twice and report what SPICE has loaded
FRESH_PROCESS_SCRIPT = f"""
import json, cspyce
from oops.hosts.galileo import Galileo
tdb = Galileo.tdb_from_sclk({LABEL_COUNT!r})
assert Galileo.tdb_from_sclk({LABEL_COUNT!r}) == tdb
loaded = [cspyce.kdata(i, 'ALL')[0] for i in range(cspyce.ktotal('ALL'))]
print(json.dumps({{'tdb': tdb, 'loaded': loaded, 'initialized': Galileo.initialized}}))
"""


@pytest.fixture(scope='module', autouse=True)
def _sclk_kernels() -> None:
    """Furnish the clock kernels; skip if the SPICE resources are unavailable."""

    try:
        Galileo.load_sclk()
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.skip('Galileo SPICE kernels unavailable: ' + str(e))


def test_label_form_matches_direct_conversion() -> None:
    """The label form of the count gives exactly the SPICE conversion of that count."""

    tdb = Galileo.tdb_from_sclk(LABEL_COUNT)

    assert tdb == cspyce.scs2e(Galileo.SPACECRAFT_ID, SPICE_COUNT)
    iso = julian.iso_from_tai(julian.tai_from_tdb(tdb), digits=0)
    assert iso.startswith(EXPECTED_ISO_PREFIX)


@pytest.mark.parametrize('count', ['00030612:00:0:0', '30612.0', '30612',
                                   ' 00030612.00 ', '30612/0/0/0'],
                         ids=['spice-form', 'short-label', 'rim-only',
                              'padded', 'other-delimiter'])
def test_equivalent_forms_agree(count: str) -> None:
    """Delimiters, leading zeros, surrounding blanks and omitted fields do not matter."""

    assert Galileo.tdb_from_sclk(count) == Galileo.tdb_from_sclk(LABEL_COUNT)


def test_trailing_fields_advance_the_time() -> None:
    """A nonzero mod-91 field moves the time forward by one tick."""

    later = Galileo.tdb_from_sclk('30612:1') - Galileo.tdb_from_sclk(LABEL_COUNT)

    assert later == pytest.approx(MOD91_TICK, abs=0.01)


@pytest.mark.parametrize('count', ['', 'abc.00', '1:2:3:4:5', '30612.x1'],
                         ids=['empty', 'letters', 'five-fields', 'bad-field'])
def test_invalid_count_raises(count: str) -> None:
    """A malformed count raises ValueError naming the offending string."""

    with pytest.raises(ValueError, match='invalid Galileo spacecraft clock count') as e:
        Galileo.tdb_from_sclk(count)

    assert repr(count) in str(e.value)


def test_conversion_needs_no_initializer() -> None:
    """In a fresh process, converting a count loads two text kernels and nothing else."""

    result = subprocess.run([sys.executable, '-c', FRESH_PROCESS_SCRIPT],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    report = json.loads(result.stdout)
    assert report['tdb'] == cspyce.scs2e(Galileo.SPACECRAFT_ID, SPICE_COUNT)
    assert not report['initialized']

    basenames = sorted(path.rsplit('/', 1)[-1] for path in report['loaded'])
    assert basenames == ['mk00062a.tsc', 'naif0012.tls']

##########################################################################################

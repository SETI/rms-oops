##########################################################################################
# tests/test_exceptions.py: The OopsSpiceError exception
##########################################################################################

import cspyce
import pytest

from oops import OopsSpiceError

# The error that cspyce raises from ckgp when no C-kernel record covers the time, as it
# reads for Cassini kernels at a time outside the loaded CK
CKGP_MESSAGE = ('SPICE(CKINSUFFDATA) -- sigerr -- insufficient C kernel data to evaluate '
                'instrument/spacecraft -82000 (CASSINI_SC_COORD) at spacecraft clock '
                'time 242374092064.07803 with tolerance 0.0')

# 2010-01-01T00:00:00.000 UTC in seconds TDB
TDB = 315576066.18392
ISO = '2010-01-01T00:00:00.000'

DESCRIPTION = 'no Cassini ISS NAC pointing'
FILEPATH = '/data/COISS_2001/N1454725799_1.IMG'


def test_the_constructor_sets_the_attributes() -> None:
    """The keywords become attributes, and default to None."""

    error = OopsSpiceError('no pointing', spice_code='SPICE(CKINSUFFDATA)',
                           filepath=FILEPATH)
    bare = OopsSpiceError('no pointing')

    assert str(error) == 'no pointing'
    assert error.spice_code == 'SPICE(CKINSUFFDATA)'
    assert error.filepath == FILEPATH
    assert bare.spice_code is None
    assert bare.filepath is None


@pytest.mark.parametrize(('time', 'filepath', 'expected'), [
    (TDB, FILEPATH, f'{DESCRIPTION} at {ISO} for {FILEPATH}: SPICE(CKINSUFFDATA)'),
    (None, None, f'{DESCRIPTION}: SPICE(CKINSUFFDATA)'),
], ids=['all_parts', 'no_time_or_file'])
def test_the_message_form(time: float | None, filepath: str | None,
                          expected: str) -> None:
    """The message is the description, then the time, the file and the SPICE code."""

    error = OopsSpiceError.from_spice_error(OSError(CKGP_MESSAGE),
                                            description=DESCRIPTION, time=time,
                                            filepath=filepath)

    assert str(error) == expected
    assert error.spice_code == 'SPICE(CKINSUFFDATA)'


def test_the_code_is_taken_from_a_raised_cspyce_error() -> None:
    """A KeyError that cspyce raises, whose text is quoted, still yields its code."""

    with pytest.raises(KeyError, match='UNKNOWNFRAME') as info:
        cspyce.pxform('J2000', 'NOT_A_FRAME_OF_ANY_KERNEL', 0.)

    error = OopsSpiceError.from_spice_error(info.value, description=DESCRIPTION)

    assert error.spice_code == 'SPICE(UNKNOWNFRAME)'


def test_text_without_a_code_gives_none() -> None:
    """Without a SPICE short error in the text, spice_code is None and the message has no
    code."""

    error = OopsSpiceError.from_spice_error(RuntimeError('kernel pool is empty'),
                                            description=DESCRIPTION)

    assert error.spice_code is None
    assert str(error) == DESCRIPTION

##########################################################################################

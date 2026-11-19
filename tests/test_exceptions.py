##########################################################################################
# tests/test_exceptions.py: The OopsSpiceError exception
##########################################################################################

import pathlib
import pickle

import cspyce
import pytest

import oops
from oops      import OopsException, OopsSpiceError
from oops.host import HostError, HostFormatError

# The error that cspyce raises from ckgp when no C-kernel record covers the time, as it
# reads for Cassini kernels at a time outside the loaded CK
CKGP_MESSAGE = ('SPICE(CKINSUFFDATA) -- sigerr -- insufficient C kernel data to evaluate '
                'instrument/spacecraft -82000 (CASSINI_SC_COORD) at spacecraft clock '
                'time 242374092064.07803 with tolerance 0.0')

# The start of the error that cspyce raises from pxform and sxform, and so from an oops
# SpiceFrame, when no frame connection can be made at the time
PXFORM_MESSAGE = ('SPICE(NOFRAMECONNECT) -- sxform -- At epoch 3.1557606618392E+08 TDB '
                  '(2010 JAN 01 00:01:06.183 TDB), there is insufficient information '
                  'available to transform from reference frame 1 (J2000) to reference '
                  'frame -82360 (CASSINI_ISS_NAC).')

# The epoch of PXFORM_MESSAGE in seconds TDB, and the same moment as an ISO UTC time
TDB = 315576066.18392
ISO = '2010-01-01T00:00:00.000'

DESCRIPTION = 'no Cassini ISS NAC pointing'
FILEPATH = '/data/COISS_2001/N1454725799_1.IMG'


def test_the_constructor_sets_the_message() -> None:
    """The first argument is the message."""

    error = OopsSpiceError('no pointing', spice_code='SPICE(CKINSUFFDATA)',
                           filepath=FILEPATH)

    assert str(error) == 'no pointing'


def test_the_constructor_sets_the_attributes() -> None:
    """The spice_code and filepath keywords become attributes."""

    path = pathlib.Path(FILEPATH)
    error = OopsSpiceError('no pointing', spice_code='SPICE(CKINSUFFDATA)',
                           filepath=path)

    assert error.spice_code == 'SPICE(CKINSUFFDATA)'
    assert error.filepath is path


def test_the_attributes_default_to_none() -> None:
    """Without the keywords, spice_code and filepath are None."""

    error = OopsSpiceError('no pointing')

    assert error.spice_code is None
    assert error.filepath is None


@pytest.mark.parametrize(('base', 'expected'), [(OopsException, True),
                                                (RuntimeError, True),
                                                (HostError, False),
                                                (HostFormatError, False),
                                                (ValueError, False)],
                         ids=lambda value: getattr(value, '__name__', str(value)))
def test_the_base_classes(base: type, expected: bool) -> None:
    """It is an OopsException and a RuntimeError, and not a host or value error."""

    assert issubclass(OopsSpiceError, base) is expected


def test_it_is_exported_from_oops() -> None:
    """The class is published by the oops package and listed in its __all__."""

    assert oops.OopsSpiceError is OopsSpiceError
    assert 'OopsSpiceError' in oops.__all__


@pytest.mark.parametrize(('time', 'filepath', 'expected'), [
    (TDB, FILEPATH, f'{DESCRIPTION} at {ISO} for {FILEPATH}: SPICE(CKINSUFFDATA)'),
    (None, FILEPATH, f'{DESCRIPTION} for {FILEPATH}: SPICE(CKINSUFFDATA)'),
    (TDB, None, f'{DESCRIPTION} at {ISO}: SPICE(CKINSUFFDATA)'),
    (None, None, f'{DESCRIPTION}: SPICE(CKINSUFFDATA)'),
], ids=['time_and_file', 'file_only', 'time_only', 'neither'])
def test_the_message_form(time: float | None, filepath: str | None,
                          expected: str) -> None:
    """The message is the description, then the time, the file and the SPICE code."""

    error = OopsSpiceError.from_spice_error(OSError(CKGP_MESSAGE),
                                            description=DESCRIPTION, time=time,
                                            filepath=filepath)

    assert str(error) == expected


@pytest.mark.parametrize(('exc', 'code'), [
    (OSError(CKGP_MESSAGE), 'SPICE(CKINSUFFDATA)'),
    (RuntimeError(PXFORM_MESSAGE), 'SPICE(NOFRAMECONNECT)'),
], ids=['ckgp', 'pxform'])
def test_the_code_is_taken_from_a_cspyce_message(exc: Exception, code: str) -> None:
    """The SPICE short error at the start of a cspyce message becomes spice_code."""

    error = OopsSpiceError.from_spice_error(exc, description=DESCRIPTION)

    assert error.spice_code == code


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
                                            description=DESCRIPTION, time=TDB,
                                            filepath=FILEPATH)

    assert error.spice_code is None
    assert str(error) == f'{DESCRIPTION} at {ISO} for {FILEPATH}'


def test_the_classmethod_sets_the_filepath() -> None:
    """The filepath given to from_spice_error is the filepath attribute."""

    path = pathlib.Path(FILEPATH)
    error = OopsSpiceError.from_spice_error(OSError(CKGP_MESSAGE),
                                            description=DESCRIPTION, filepath=path)

    assert error.filepath is path


def test_the_classmethod_builds_an_instance_of_its_class() -> None:
    """A subclass calling from_spice_error gets an instance of the subclass."""

    class _Subclass(OopsSpiceError):
        pass

    error = _Subclass.from_spice_error(OSError(CKGP_MESSAGE), description=DESCRIPTION)

    assert type(error) is _Subclass


def test_pickling_keeps_the_attributes() -> None:
    """A pickled and restored error keeps its message, spice_code and filepath."""

    error = OopsSpiceError.from_spice_error(OSError(CKGP_MESSAGE),
                                            description=DESCRIPTION, filepath=FILEPATH)

    restored = pickle.loads(pickle.dumps(error))

    assert str(restored) == str(error)
    assert restored.spice_code == 'SPICE(CKINSUFFDATA)'
    assert restored.filepath == FILEPATH

##########################################################################################

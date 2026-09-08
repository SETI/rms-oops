##########################################################################################
# tests/calibration/test_nullcalib.py
##########################################################################################

import pickle

from oops.calibration import FlatCalib, NullCalib


def test_nullcalib_leaves_values_unchanged() -> None:
    """Every conversion of a NullCalib is the identity."""

    cal = NullCalib('TEST')

    assert cal.extended_from_dn(5., (512, 512)) == 5.
    assert cal.dn_from_extended(5., (512, 512)) == 5.
    assert cal.point_from_dn(5., (512, 512)) == 5.
    assert cal.dn_from_point(5., (512, 512)) == 5.


def test_nullcalib_prescale_returns_a_flatcalib() -> None:
    """Pre-scaling cannot stay an identity, so it yields a FlatCalib."""

    cal = NullCalib('TEST').prescale(5., 1.)

    assert isinstance(cal, FlatCalib)
    assert cal.name == 'TEST'
    assert cal.extended_from_dn(3., (512, 512)) == 10.


def test_nullcalib_prescale_renames_when_asked() -> None:
    """A blank name preserves the old one; a given name replaces it."""

    assert NullCalib('TEST').prescale(5.).name == 'TEST'
    assert NullCalib('TEST').prescale(5., name='OTHER').name == 'OTHER'


def test_nullcalib_survives_a_pickle_round_trip() -> None:
    """__getstate__ returns a tuple, so the calibration can be pickled."""

    cal = pickle.loads(pickle.dumps(NullCalib('REFLECTIVITY')))

    assert cal.name == 'REFLECTIVITY'
    assert cal.extended_from_dn(5., (512, 512)) == 5.

##########################################################################################

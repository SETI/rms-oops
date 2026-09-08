##########################################################################################
# tests/calibration/test_calibration.py: the Calibration abstract base class
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath          import Pair, Scalar
from oops.calibration  import FlatCalib, NullCalib, Radiance, RawCounts
from oops.fov          import FlatFOV

UV = Pair([(1., 1.), (2., 2.)])
DN = Scalar([1., 2.])


def _flat() -> FlatCalib:
    """A calibration that simply doubles each DN."""

    return FlatCalib(name='IOF', factor=2.)


def test_a_calibration_names_the_quantity_it_produces() -> None:
    """The name identifies the physical quantity the DNs are converted to."""

    assert _flat().name == 'IOF'


def test_a_calibration_without_a_baseline() -> None:
    """A calibration with no offset reports that it has none."""

    calibration = _flat()

    assert not calibration.has_baseline
    assert calibration.baseline == Scalar(0.)


def test_a_calibration_with_a_baseline() -> None:
    """An offset is subtracted from each DN before the factor is applied."""

    calibration = FlatCalib(name='IOF', factor=2., baseline=1.)

    assert calibration.has_baseline
    assert calibration.extended_from_dn(Scalar([1., 2.]), UV) == Scalar([0., 2.])


def test_a_scalar_calibration_is_shapeless() -> None:
    """A single factor and baseline broadcast against any data array."""

    assert _flat().shape == ()


def test_a_calibration_without_an_fov() -> None:
    """A calibration that needs no field of view reports None."""

    assert _flat().fov is None


def test_extended_from_dn_applies_the_factor() -> None:
    """The extended-source value is the DN scaled by the factor."""

    assert _flat().extended_from_dn(DN, UV) == Scalar([2., 4.])


def test_dn_from_extended_inverts_extended_from_dn() -> None:
    """The two conversions are inverses of one another."""

    calibration = _flat()
    values = calibration.extended_from_dn(DN, UV)

    assert calibration.dn_from_extended(values, UV) == DN


def test_dn_from_point_inverts_point_from_dn() -> None:
    """The two point-source conversions are inverses of one another."""

    calibration = _flat()
    values = calibration.point_from_dn(DN, UV)

    assert calibration.dn_from_point(values, UV) == DN


def test_value_from_dn_is_the_extended_source_value() -> None:
    """The deprecated name gives the extended-source calibration."""

    calibration = _flat()

    assert calibration.value_from_dn(DN, UV) == calibration.extended_from_dn(DN, UV)


def test_dn_from_value_is_the_extended_source_inverse() -> None:
    """The deprecated name gives the extended-source inverse."""

    calibration = _flat()
    values = calibration.extended_from_dn(DN, UV)

    assert calibration.dn_from_value(values, UV) == calibration.dn_from_extended(values,
                                                                                 UV)


def test_prescale_folds_a_scale_factor_into_the_calibration() -> None:
    """The new object applies the extra factor before its own."""

    rescaled = _flat().prescale(3.)

    assert rescaled.extended_from_dn(DN, UV) == Scalar([6., 12.])


def test_prescale_folds_in_a_baseline() -> None:
    """The baseline is subtracted from each DN before the new factor is applied."""

    rescaled = _flat().prescale(3., 1.)

    assert rescaled.extended_from_dn(Scalar([1., 2.]), UV) == Scalar([0., 6.])


def test_prescale_preserves_the_name_by_default() -> None:
    """A blank name leaves the existing one in place."""

    assert _flat().prescale(3.).name == 'IOF'


def test_prescale_can_rename_the_quantity() -> None:
    """A new name replaces the existing one."""

    assert _flat().prescale(3., name='DN').name == 'DN'


def test_dn_and_uv_are_cast_to_a_common_shape() -> None:
    """A shapeless DN broadcasts against an array of pixel coordinates."""

    assert _flat().extended_from_dn(Scalar(1.), UV) == Scalar([2., 2.])


def test_area_factor_needs_a_field_of_view() -> None:
    """The relative pixel area comes from the calibration's FOV."""

    fov = FlatFOV((1.e-4, 1.e-4), (10, 10))
    calibration = Radiance(name='I/F', factor=2., fov=fov)

    factors = calibration.area_factor(UV)

    assert factors.shape == (2,)
    assert np.allclose(factors.vals, 1., atol=1.e-4)


def test_a_null_calibration_leaves_the_dn_alone() -> None:
    """A NullCalib passes the DN values through unchanged."""

    calibration = NullCalib(name='DN')

    assert calibration.extended_from_dn(DN, UV) == DN
    assert calibration.dn_from_extended(DN, UV) == DN

##########################################################################################
# Serialization
##########################################################################################

FOV = FlatFOV((1.e-4, 1.e-4), (64, 64))


def _calibrations() -> dict:
    """One instance of every Calibration subclass, keyed by name.

    Returns:
        dict: The calibrations to test.
    """

    return {
        'FlatCalib':  FlatCalib(name='IOF', factor=2., baseline=0.5),
        'NullCalib':  NullCalib(name='DN'),
        'Radiance':   Radiance(name='I/F', fov=FOV, factor=2., baseline=0.5),
        'RawCounts':  RawCounts(name='COUNTS', fov=FOV, factor=2., baseline=0.5),
    }


@pytest.mark.parametrize('name', sorted(_calibrations()))
def test_a_calibration_survives_a_round_trip_through_pickle(name: str) -> None:
    """Unpickling rebuilds the calibration and reproduces the conversion it defines."""

    calibration = _calibrations()[name]

    restored = pickle.loads(pickle.dumps(calibration))

    assert type(restored) is type(calibration)
    assert restored.name == calibration.name
    assert restored.extended_from_dn(DN, UV) == calibration.extended_from_dn(DN, UV)

##########################################################################################

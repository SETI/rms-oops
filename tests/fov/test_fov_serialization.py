##########################################################################################
# tests/fov/test_fov_serialization.py: pickling across the FOV subclasses
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath      import Pair
from oops          import mutable
from oops.config   import LOGGING
from oops.fov      import (BarrelFOV, FlatFOV, GapFOV, NullFOV, OffsetFOV, Platescale,
                           PolynomialFOV, SliceFOV, Subarray, SubsampledFOV, TDIFOV)
from oops.fov.fov_    import FOV
from oops.fov.wcsfov  import WCSFOV

from tests.fov.test_wcsfov import header2 as WCS_HEADER

FLAT = FlatFOV((1.e-4, 1.e-4), (64, 64))

# A mild barrel distortion, and a polynomial mapping with the same leading terms
BARREL = (1., 0., 1.e-3)
POLYNOMIAL = np.zeros((3, 3, 2))
POLYNOMIAL[1, 0, 0] = 1.e-4
POLYNOMIAL[0, 1, 1] = 1.e-4
POLYNOMIAL[2, 0, 0] = 1.e-9


def _fovs() -> dict[str, FOV]:
    """One instance of every pickleable FOV subclass, keyed by name.

    Returns:
        dict[str, FOV]: The FOVs to test.
    """

    return {
        'FlatFOV':       FLAT,
        'NullFOV':       NullFOV(),
        'OffsetFOV':     OffsetFOV(FLAT, uv_offset=(1., 2.)),
        'Platescale':    Platescale(2., FLAT),
        'SliceFOV':      SliceFOV(FLAT, (10, 20), (8, 8)),
        'Subarray':      Subarray(FLAT, (10., 20.), (8, 8)),
        'SubsampledFOV': SubsampledFOV(FLAT, 2.),
        'GapFOV':        GapFOV(FLAT, 0.5),
        'TDIFOV':        TDIFOV(FLAT, 100., 1., 'v'),
        'BarrelFOV':     BarrelFOV((1.e-4, 1.e-4), (64, 64),
                                   coefft_xy_from_uv=BARREL),
        'PolynomialFOV': PolynomialFOV((64, 64), coefft_xy_from_uv=POLYNOMIAL),
    }


# A point inside every one of these fields of view, and a time within the TDI readout
UV = Pair([(10.5, 20.5), (30.25, 40.75)])
TIME = 99.5


@pytest.mark.parametrize('name', sorted(_fovs()))
def test_an_fov_survives_a_round_trip_through_pickle(name: str) -> None:
    """Unpickling rebuilds the FOV and reproduces the mapping it defines."""

    fov = _fovs()[name]

    restored = pickle.loads(pickle.dumps(fov))

    assert type(restored) is type(fov)
    assert restored.uv_shape == fov.uv_shape
    assert restored.xy_from_uvt(UV, time=TIME) == fov.xy_from_uvt(UV, time=TIME)


@pytest.mark.parametrize('name', sorted(_fovs()))
def test_an_unpickled_fov_is_frozen(name: str) -> None:
    """Unpickling restores the values as they stood, so the result is not refittable."""

    fov = _fovs()[name]

    assert mutable.is_frozen(pickle.loads(pickle.dumps(fov)))


##########################################################################################
# Constructor validation
##########################################################################################

def test_a_tdi_fov_rejects_an_unknown_axis() -> None:
    """The TDI readout runs along u or v, in either direction."""

    with pytest.raises(ValueError, match="invalid tdi_axis value: 'w'"):
        TDIFOV(FLAT, 100., 1., 'w')


@pytest.mark.parametrize('class_, args', [(BarrelFOV, ((1.e-4, 1.e-4), (64, 64))),
                                          (PolynomialFOV, ((64, 64),))],
                         ids=['BarrelFOV', 'PolynomialFOV'])
def test_a_distortion_needs_coefficients_in_one_direction_at_least(class_, args) -> None:
    """A distorted FOV is defined by a mapping one way or the other."""

    with pytest.raises(ValueError, match='at least one of coefft_xy_from_uv'):
        class_(*args)


def test_a_barrel_fov_accepts_an_explicit_line_of_sight_and_area() -> None:
    """The optic axis and the pixel area can be given rather than derived."""

    fov = BarrelFOV((1.e-4, 1.e-4), (64, 64), coefft_xy_from_uv=BARREL,
                    uv_los=(20., 30.), uv_area=1.e-9)

    assert fov.uv_los == Pair((20., 30.))
    assert fov.uv_area == 1.e-9


def test_a_polynomial_fov_accepts_an_explicit_area() -> None:
    """The pixel area can be given rather than derived from the coefficients."""

    fov = PolynomialFOV((64, 64), coefft_xy_from_uv=POLYNOMIAL, uv_area=1.e-9)

    assert fov.uv_area == 1.e-9


##########################################################################################
# Remasking outside the field of view
##########################################################################################

# A point inside the 64x64 field and two beyond it
OUTSIDE_UV = Pair([(30., 20.), (100., 20.), (30., 90.)])


def test_a_flat_fov_masks_coordinates_outside_the_field() -> None:
    """remask=True masks the (u,v) coordinates that fall outside the field of view."""

    xy = FLAT.xy_from_uvt(OUTSIDE_UV)

    uv = FLAT.uv_from_xyt(xy, remask=True)

    assert list(uv.mask) == [False, True, True]


def test_a_polynomial_fov_masks_coordinates_outside_the_field() -> None:
    """The same holds in both directions through a polynomial distortion."""

    fov = PolynomialFOV((64, 64), coefft_xy_from_uv=POLYNOMIAL)

    xy = fov.xy_from_uvt(OUTSIDE_UV, remask=True)
    uv = fov.uv_from_xyt(fov.xy_from_uvt(OUTSIDE_UV), remask=True)

    assert list(xy.mask) == [False, True, True]
    assert list(uv.mask) == [False, True, True]


def test_a_gap_fov_masks_coordinates_that_fall_in_a_gap() -> None:
    """A point in the dead space between pixels is masked when remask is set."""

    fov = GapFOV(FLAT, 0.5)

    # Each pixel is sensitive over the first half of its cell, so u = 30.6 falls in the
    # dead space that follows pixel 30, while u = 30.25 is on the pixel itself
    uv = fov.uv_from_xyt(FLAT.xy_from_uvt(Pair([(30.25, 20.25), (30.6, 20.25)])),
                         remask=True)

    assert list(uv.mask) == [False, True]


def test_an_offset_fov_maps_xy_back_through_its_offset() -> None:
    """The (x,y) offset is added back before the wrapped FOV is consulted."""

    fov = OffsetFOV(FLAT, xy_offset=(1.e-4, 2.e-4))

    uv = fov.uv_from_xyt(FLAT.xy_from_uvt(UV))

    assert uv == UV + Pair((1., 2.))

##########################################################################################
# The iterative inversion of a distorted field of view
##########################################################################################

# Points inside the 64x64 field, one near the center and one near a corner where the
# distortion is largest
DISTORTED_UV = Pair([(10.5, 20.5), (60., 60.)])


def _barrel(**kwargs) -> BarrelFOV:
    """A BarrelFOV with a mild radial distortion.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        BarrelFOV: The field of view.
    """

    return BarrelFOV((1.e-4, 1.e-4), (64, 64), coefft_xy_from_uv=BARREL, **kwargs)


def _polynomial(**kwargs) -> PolynomialFOV:
    """A PolynomialFOV with a mild quadratic distortion.

    Parameters:
        kwargs: Overrides of the default constructor arguments.

    Returns:
        PolynomialFOV: The field of view.
    """

    return PolynomialFOV((64, 64), coefft_xy_from_uv=POLYNOMIAL, **kwargs)


@pytest.mark.parametrize('name', ['barrel', 'polynomial'])
def test_a_distorted_fov_inverts_its_own_mapping(name: str) -> None:
    """The iterative solution returns the pixel the forward mapping started from."""

    fov = _barrel() if name == 'barrel' else _polynomial()

    uv = fov.uv_from_xyt(fov.xy_from_uvt(DISTORTED_UV))

    assert uv.vals == pytest.approx(DISTORTED_UV.vals, abs=1.e-6)


@pytest.mark.parametrize('name', ['barrel', 'polynomial'])
def test_a_fully_masked_input_skips_the_solver(name: str) -> None:
    """With nothing to solve for, the result is masked without iterating."""

    fov = _barrel() if name == 'barrel' else _polynomial()
    xy = fov.xy_from_uvt(DISTORTED_UV).remask(True)

    uv = fov.uv_from_xyt(xy)

    assert np.all(uv.mask)


@pytest.mark.parametrize('name, marker',
                         [('barrel', 'BarrelFOV._solve_ratio'),
                          ('polynomial', 'PolynomialFOV._solve_polynomial')])
def test_the_iterations_of_the_solver_can_be_logged(
        name: str, marker: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of Newton's method reports the change it made."""

    fov = _barrel() if name == 'barrel' else _polynomial()
    xy = fov.xy_from_uvt(DISTORTED_UV)

    LOGGING.on()
    LOGGING.fov_iterations = True
    try:
        fov.uv_from_xyt(xy)
    finally:
        LOGGING.fov_iterations = False
        LOGGING.off()

    assert marker in capsys.readouterr().out

def test_a_wcs_fov_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the FOV from the FITS header it was constructed with."""

    fov = WCSFOV(WCS_HEADER)
    uv = Pair([(100.5, 200.5), (1500.25, 1000.75)])

    restored = pickle.loads(pickle.dumps(fov))

    assert isinstance(restored, WCSFOV)
    assert restored.uv_shape == fov.uv_shape
    assert restored.xy_from_uv(uv) == fov.xy_from_uv(uv)


# Distortions strong enough that two passes of Newton's method fall short of the
# precision the solver asks for
STRONG_BARREL = (1., 0., 1.)
STRONG_POLYNOMIAL = np.zeros((4, 4, 2))
STRONG_POLYNOMIAL[1, 0, 0] = 1.e-4
STRONG_POLYNOMIAL[0, 1, 1] = 1.e-4
STRONG_POLYNOMIAL[3, 0, 0] = 1.e-6
STRONG_POLYNOMIAL[0, 3, 1] = 1.e-6

# A point in the corner, where a radial distortion is largest
CORNER_UV = Pair([(10.5, 20.5), (63.5, 63.5)])


def test_a_barrel_solution_that_runs_out_of_iterations_is_reported(
        capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind.

    The solution is still returned, and is still close.
    """

    fov = BarrelFOV((1.e-3, 1.e-3), (64, 64), coefft_xy_from_uv=STRONG_BARREL, iters=2)
    xy = fov.xy_from_uvt(CORNER_UV)

    LOGGING.on()
    try:
        uv = fov.uv_from_xyt(xy)
    finally:
        LOGGING.off()

    assert 'BarrelFOV._solve_ratio did not converge' in capsys.readouterr().out
    assert uv.vals == pytest.approx(CORNER_UV.vals, abs=1.e-3)


def test_a_polynomial_solution_that_stops_improving_is_reported(
        capsys: pytest.CaptureFixture[str]) -> None:
    """A solution that stops converging is abandoned, and the failure is reported."""

    fov = PolynomialFOV((64, 64), coefft_xy_from_uv=STRONG_POLYNOMIAL, iters=2)
    xy = fov.xy_from_uvt(CORNER_UV)

    LOGGING.on()
    try:
        fov.uv_from_xyt(xy)
    finally:
        LOGGING.off()

    assert 'PolynomialFOV._solve_polynomial did not converge' \
           in capsys.readouterr().out

##########################################################################################

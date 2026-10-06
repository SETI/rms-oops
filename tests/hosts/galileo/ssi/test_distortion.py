##########################################################################################
# tests/hosts/galileo/ssi/test_distortion.py
##########################################################################################

import numpy as np
import pytest

import oops
import oops.hosts.galileo.ssi as ssi

# The IK distortion model: R - r = A * r**3, with R and r in pixels from the field center
A_PER_PIXEL2 = 6.58e-9


@pytest.fixture(scope='module')
def fov_full() -> oops.FOV:
    ssi.initialize()
    return ssi.SSI.fovs['FULL']


@pytest.mark.parametrize('r_ideal', [100., 283., 400., 400. * np.sqrt(2.)])
def test_radial_distortion_matches_ik_model(fov_full: oops.FOV, r_ideal: float) -> None:
    """A direction at ideal pixel radius r lands at r + A*r**3 along the diagonal."""

    center = fov_full.uv_los.vals
    ideal_uv = oops.Pair(center + r_ideal / np.sqrt(2.))
    xy = fov_full.flat_fov.xy_from_uv(ideal_uv)
    uv = fov_full.uv_from_xy(xy)

    r_actual = np.hypot(*(uv.vals - center))
    assert r_actual - r_ideal == pytest.approx(A_PER_PIXEL2 * r_ideal**3, rel=1.e-9)


def test_corner_displacement(fov_full: oops.FOV) -> None:
    """The field corner moves outward by about 1.19 pixels."""

    xy = fov_full.flat_fov.xy_from_uv(oops.Pair((800., 800.)))
    uv = fov_full.uv_from_xy(xy)
    assert uv.vals - 800. == pytest.approx(A_PER_PIXEL2 * 400.**3 * 2., rel=1.e-9)


def test_round_trip(fov_full: oops.FOV) -> None:
    """Mapping pixels to directions and back recovers the pixels."""

    uv = oops.Pair(np.array([[0., 0.], [800., 800.], [123.4, 567.8], [400., 400.]]))
    back = fov_full.uv_from_xy(fov_full.xy_from_uv(uv))
    assert np.abs(back.vals - uv.vals).max() < 1.e-9


def test_summed_mode_scales_with_full(fov_full: oops.FOV) -> None:
    """The 2x2 summed FOV carries the same distortion at half the pixel scale."""

    summed = ssi.SSI.fovs['HIS']
    xy = fov_full.flat_fov.xy_from_uv(oops.Pair((800., 800.)))
    assert summed.uv_from_xy(xy).vals == pytest.approx(fov_full.uv_from_xy(xy).vals / 2.,
                                                       rel=1.e-12)

##########################################################################################
# tests/fov/test_tdifov.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Pair
from oops.fov import FlatFOV, NullFOV, TDIFOV


def test_tdifov():
    np.random.seed(9816)
    from oops.fov.flatfov import FlatFOV

    ######################################################################################
    # 10 lines, TDI -v, 8 sec/shift, tstop=100
    ######################################################################################

    staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
    fov = TDIFOV(staticfov, 100, 8., '-v')

    uv = Pair.combos(np.arange(0,101,50), np.arange(11))
    xy0 = staticfov.xy_from_uvt(uv)

    assert fov.xy_from_uvt(uv, time=100) == xy0
    assert fov.xy_from_uvt(uv, time=92) == xy0
    assert fov.xy_from_uvt(uv, time=84)[:,:-1] == xy0[:,1:]
    assert fov.xy_from_uvt(uv, time=83)[:,:-2] == xy0[:,2:]
    assert fov.xy_from_uvt(uv, time=101)[:,1:] == xy0[:,:-1]

    assert fov.uv_from_xyt(xy0, time=100) == uv
    assert fov.uv_from_xyt(xy0, time=92) == uv
    assert fov.uv_from_xyt(xy0, time=84)[:,1:] == uv[:,:-1]
    assert fov.uv_from_xyt(xy0, time=83)[:,2:] == uv[:,:-2]
    assert fov.uv_from_xyt(xy0, time=101)[:,:-1] == uv[:,1:]

    # with derivs
    N = 100
    uv = Pair.combos(50 + 20. * np.random.randn(N),
                      5 +  3. * np.random.randn(N))
    time = Scalar(90 + 20 * np.random.randn(N))
    uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
    uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
    uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

    xy0 = staticfov.xy_from_uvt(uv, derivs=True)
    xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
    assert xy0.d_drs == xy.d_drs
    assert xy0.d_dq == xy.d_dq

    diffs = xy0.d_dt - xy.d_dt
    assert np.all(diffs.vals[...,0] == 0)
    assert np.all(abs(diffs.vals[...,1] - 1/2048./8.) < 1.e-14)

    ######################################################################################
    # 10 lines, TDI +v, 8 sec/shift, tstop=100
    ######################################################################################

    staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
    fov = TDIFOV(staticfov, 100, 8., '+v')

    uv = Pair.combos(np.arange(0,101,50), np.arange(11))
    xy0 = staticfov.xy_from_uvt(uv)

    assert fov.xy_from_uvt(uv, time=100) == xy0
    assert fov.xy_from_uvt(uv, time=92) == xy0
    assert fov.xy_from_uvt(uv, time=84)[:,1:] == xy0[:,:-1]
    assert fov.xy_from_uvt(uv, time=83)[:,2:] == xy0[:,:-2]
    assert fov.xy_from_uvt(uv, time=101)[:,:-1] == xy0[:,1:]

    assert fov.uv_from_xyt(xy0, time=100) == uv
    assert fov.uv_from_xyt(xy0, time=92) == uv
    assert fov.uv_from_xyt(xy0, time=84)[:,:-1] == uv[:,1:]
    assert fov.uv_from_xyt(xy0, time=83)[:,:-2] == uv[:,2:]
    assert fov.uv_from_xyt(xy0, time=101)[:,1:] == uv[:,:-1]

    # with derivs
    N = 100
    uv = Pair.combos(50 + 20. * np.random.randn(N),
                      5 +  3. * np.random.randn(N))
    time = Scalar(90 + 20 * np.random.randn(N))
    uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
    uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
    uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

    xy0 = staticfov.xy_from_uvt(uv, derivs=True)
    xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
    assert xy0.d_drs == xy.d_drs
    assert xy0.d_dq == xy.d_dq

    diffs = xy0.d_dt - xy.d_dt
    assert np.all(diffs.vals[...,0] == 0)
    assert np.all(abs(diffs.vals[...,1] + 1/2048./8.) < 1.e-14)

class _SharingFOV(FlatFOV):
    """An FOV whose `uv_from_xyt` returns the same object on every call.

    Every FOV subclass in this package builds a fresh Pair per call, so none of them
    exercises a wrapped FOV that hands back an object it still owns. NullFOV comes
    closest: it returns the global, readonly `Pair.ZEROS`.
    """

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        return self.shared_uv


def _sharing_fov(uv):
    """A `_SharingFOV` that hands back `uv` itself.

    Parameters:
        uv (Pair): The object every `uv_from_xyt` call will return.

    Returns:
        _SharingFOV: A 64 by 64 FOV returning `uv`.
    """

    fov = _SharingFOV((1.e-4, 1.e-4), (64, 64))
    fov.shared_uv = uv
    return fov


def test_uv_from_xyt_does_not_modify_the_wrapped_fovs_result() -> None:
    """The TDI line shift must not be written into an object the wrapped FOV owns."""

    uv = Pair([(3., 4.), (5., 6.)])
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    fov.uv_from_xyt(Pair([(0., 0.), (0., 0.)]), time=Scalar([50., 50.]))

    assert uv == Pair([(3., 4.), (5., 6.)])


def test_uv_from_xyt_returns_a_new_object() -> None:
    """The result is a copy, so a caller cannot reach the wrapped FOV's object."""

    uv = Pair([(3., 4.)])
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    assert fov.uv_from_xyt(Pair([(0., 0.)]), time=Scalar([50.])) is not uv


def test_uv_from_xyt_still_applies_the_line_shift() -> None:
    """Copying must not cost the shift: at t=50 with tstop=100 the -v shift is 6 lines."""

    uv = Pair([(3., 4.)])
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    assert fov.uv_from_xyt(Pair([(0., 0.)]), time=Scalar([50.])) == Pair([(3., -2.)])


def test_uv_from_xyt_accepts_a_readonly_result() -> None:
    """A wrapped FOV may return a readonly object; writing into one raises in polymath."""

    uv = Pair([(3., 4.)]).as_readonly()
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    assert fov.uv_from_xyt(Pair([(0., 0.)]), time=Scalar([50.])) == Pair([(3., -2.)])


def test_uv_from_xyt_does_not_modify_a_shared_derivative() -> None:
    """The TDI readout compensation must not be written into the caller's derivative."""

    uv = Pair([(3., 4.)])
    dt = Pair([(1., 1.)])
    uv.insert_deriv('t', dt)
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    fov.uv_from_xyt(Pair([(0., 0.)]), time=Scalar([50.]), derivs=True)

    assert dt == Pair([(1., 1.)])


def test_uv_from_xyt_still_compensates_the_time_derivative() -> None:
    """Copying the derivative dict must not cost the readout compensation."""

    uv = Pair([(3., 4.)])
    uv.insert_deriv('t', Pair([(1., 1.)]))
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    result = fov.uv_from_xyt(Pair([(0., 0.)]), time=Scalar([50.]), derivs=True)

    assert result.derivs['t'] == Pair([(1., 1. - 1./8.)])


def test_uv_from_xyt_leaves_the_null_fov_constant_alone() -> None:
    """NullFOV returns the global, readonly Pair.ZEROS; corrupting it would be library
    wide."""

    fov = TDIFOV(NullFOV(), 100., 8., '-v')
    fov.uv_from_xyt(Pair((0., 0.)), time=Scalar(50.))

    assert Pair.ZEROS == Pair((0., 0.))


@pytest.mark.parametrize('tdi_axis, line', [('-v', 1), ('+v', 1),
                                            ('-u', 0), ('+u', 0)],
                         ids=['-v', '+v', '-u', '+u'])
def test_xy_from_uvt_shifts_a_shapeless_pair(tdi_axis: str, line: int) -> None:
    """At t=50 with tstop=100 and 8 sec/shift, the shift is 6 stages along the TDI axis.

    A shapeless Pair must be shifted exactly as a shaped one is. The component of a
    shapeless Pair is a copy rather than a view, so an implementation that shifts one
    component in place silently leaves a shapeless input alone.
    """

    staticfov = FlatFOV((1., 1.), (64, 64))
    fov = TDIFOV(staticfov, 100., 8., tdi_axis)
    sign = -1 if '-' in tdi_axis else 1

    expected = [32., 32.]
    expected[line] -= sign * 6

    assert fov.xy_from_uvt(Pair((32., 32.)), time=Scalar(50.)) == \
           staticfov.xy_from_uvt(Pair(expected))


def test_xy_from_uvt_agrees_between_shapeless_and_shaped() -> None:
    """A shapeless input and a one-element shaped input must give the same result."""

    fov = TDIFOV(FlatFOV((1., 1.), (64, 64)), 100., 8., '-v')

    shapeless = fov.xy_from_uvt(Pair((32., 32.)), time=Scalar(50.))
    shaped = fov.xy_from_uvt(Pair([(32., 32.)]), time=Scalar([50.]))

    assert shapeless == shaped[0]


def test_uv_from_xyt_shifts_a_shapeless_pair() -> None:
    """The inverse must shift a shapeless Pair too, in the opposite direction."""

    uv = Pair((3., 4.))
    fov = TDIFOV(_sharing_fov(uv), 100., 8., '-v')

    assert fov.uv_from_xyt(Pair((0., 0.)), time=Scalar(50.)) == Pair((3., -2.))


def test_uv_from_xyt_agrees_between_shapeless_and_shaped() -> None:
    """A shapeless input and a one-element shaped input must give the same result."""

    fov = TDIFOV(FlatFOV((1., 1.), (64, 64)), 100., 8., '-v')
    xy = fov.xy_from_uvt(Pair((32., 32.)), time=Scalar(100.))

    shapeless = fov.uv_from_xyt(xy, time=Scalar(50.))
    shaped = fov.uv_from_xyt(Pair([xy.vals]), time=Scalar([50.]))

    assert shapeless == shaped[0]


def test_xy_from_uvt_leaves_a_shapeless_input_alone() -> None:
    """The caller's Pair must not be shifted in place, shapeless or not."""

    uv = Pair((32., 32.))
    fov = TDIFOV(FlatFOV((1., 1.), (64, 64)), 100., 8., '-v')

    fov.xy_from_uvt(uv, time=Scalar(50.))

    assert uv == Pair((32., 32.))


def test_xy_from_uvt_compensates_a_shapeless_time_derivative() -> None:
    """The TDI readout compensation applies to a shapeless input as well."""

    uv = Pair((32., 32.))
    uv.insert_deriv('t', Pair((1., 1.)))
    fov = TDIFOV(FlatFOV((1., 1.), (64, 64)), 100., 8., '-v')

    result = fov.xy_from_uvt(uv, time=Scalar(50.), derivs=True)

    assert result.derivs['t'] == Pair((1., 1. + 1./8.))

##########################################################################################

##########################################################################################
# oops/frame/frameshift.py: Subclass FrameShift of class Frame
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Frame, FrameShift, SpiceFrame
from oops.path  import SpicePath


def test_frameshift(core_kernels):
    np.random.seed(2865)

    DT = 10.
    _ = SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')

    shifted = FrameShift(DT, mars, frame_id='+')
    assert shifted.frame_id == 'IAU_MARS_SHIFT'
    assert shifted.dt == DT
    assert shifted.link is None
    assert shifted.reference == mars.reference

    # The shifted frame at time t matches the original frame at time t + dt
    time = Scalar(1.e8 + np.arange(10) * 1000.)
    assert (shifted.transform_at_time(time).matrix
            == mars.transform_at_time(time + DT).matrix)
    assert (shifted.transform_at_time(time).omega
            == mars.transform_at_time(time + DT).omega)

    # Mars turns steadily about its pole, so the angle a shift introduces grows in
    # proportion to the shift
    t0 = Scalar(1.e8)
    v0 = mars.transform_at_time(t0).rotate(Vector3.XAXIS)
    sep1 = shifted.transform_at_time(t0).rotate(Vector3.XAXIS).sep(v0)
    shifted2 = FrameShift(2. * DT, mars, frame_id='mars_shifted_x2')
    sep2 = shifted2.transform_at_time(t0).rotate(Vector3.XAXIS).sep(v0)
    assert sep1.vals > 0.
    assert (sep2 / sep1).vals == pytest.approx(2., abs=0.5e-6)

    # A zero shift leaves the frame alone
    unshifted = FrameShift(0., mars, frame_id='mars_unshifted')
    assert (unshifted.transform_at_time(time).matrix
            == mars.transform_at_time(time).matrix)

    # A linked FrameShift tracks the offset of the object it is linked to
    linked = FrameShift(shifted, mars, frame_id='mars_shifted_2')
    assert linked.link is shifted
    assert linked.dt == DT
    assert linked.params == (DT,)
    assert linked.nparams == 1

    shifted.set_params(np.array([2. * DT]))
    linked.refresh()
    assert linked.dt == 2. * DT
    assert (linked.transform_at_time(time).matrix
            == mars.transform_at_time(time + 2. * DT).matrix)

    # Freezing severs the link but preserves the offset
    frozen = FrameShift(linked, mars, frame_id='mars_shifted_3', freeze=True)
    assert frozen.is_frozen
    assert frozen.link is None
    assert frozen.dt == 2. * DT

    # Freezing an object also freezes the Fittable objects it was built from, so
    # the source can no longer be refit
    assert linked.is_frozen
    assert shifted.is_frozen
    with pytest.raises(ValueError, match='frozen'):
        shifted.set_params(np.array([3. * DT]))


def test_frameshift_rejects_an_unregistered_frame(core_kernels) -> None:
    """A frame ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        FrameShift(10., 'NOT_A_REGISTERED_FRAME')


def test_frameshift_is_fittable(core_kernels) -> None:
    """The time shift is the single fittable parameter."""

    SpicePath('MARS', 'SSB')
    shifted = FrameShift(10., SpiceFrame('IAU_MARS', 'J2000'))

    assert shifted.params == (10.,)
    assert shifted.nparams == 1
    assert not shifted.is_frozen


def test_frameshift_set_params_changes_the_shift(core_kernels) -> None:
    """Fitting the shift moves the frame in time."""

    SpicePath('MARS', 'SSB')
    shifted = FrameShift(10., SpiceFrame('IAU_MARS', 'J2000'))
    shifted.set_params((20.,))

    assert shifted.dt == 20.


def test_frameshift_freeze_blocks_fitting(core_kernels) -> None:
    """freeze=True returns an object that can no longer be fitted."""

    SpicePath('MARS', 'SSB')
    frozen = FrameShift(10., SpiceFrame('IAU_MARS', 'J2000'), freeze=True)

    assert frozen.is_frozen
    assert frozen.dt == 10.


def test_frameshift_of_zero_matches_the_original(core_kernels) -> None:
    """A zero shift leaves the frame where it was."""

    SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')
    shifted = FrameShift(0., mars)
    time = Scalar(1.e8)

    assert shifted.transform_at_time(time).matrix == mars.transform_at_time(time).matrix


def test_frameshift_tracks_a_linked_shift(core_kernels) -> None:
    """A linked FrameShift always matches the shift of the object it is linked to."""

    SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')
    source = FrameShift(10., mars, frame_id='linked_source')
    linked = FrameShift(source, mars, frame_id='linked_follower')

    assert linked.link is source
    assert linked.dt == 10.

    source.set_params((20.,))
    linked.refresh()
    assert linked.dt == 20.


def test_frameshift_pickle(core_kernels) -> None:
    """Pickling restores the shift and the underlying Frame."""

    SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')
    shifted = FrameShift(10., mars)
    restored = pickle.loads(pickle.dumps(shifted))
    time = Scalar(1.e8)

    assert isinstance(restored, FrameShift)
    assert restored.dt == shifted.dt
    assert restored.transform_at_time(time).matrix \
           == shifted.transform_at_time(time).matrix


def test_frameshift_getstate_roundtrip(core_kernels) -> None:
    """The state captured by __getstate__ fully restores the object."""

    SpicePath('MARS', 'SSB')
    shifted = FrameShift(10., SpiceFrame('IAU_MARS', 'J2000'))
    state = shifted.__getstate__()

    copied = Frame.__new__(FrameShift)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################

##########################################################################################
# oops/frame/quickframe.py: Subclass QuickFrame of class Frame
##########################################################################################

import numpy as np
import pytest

from polymath   import Matrix3, Quaternion, Scalar, Vector3
from oops.frame import (Frame, Cmatrix, Navigation, PosTargFrame, QuickFrame,
                        Rotation, SpiceFrame, SpinFrame, TwoVectorFrame)
from oops.path  import SpicePath


def test_quickframe(core_kernels):
    np.random.seed(4417)

    _ = SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')

    epoch = 1.e8
    time = Scalar(epoch + np.arange(0., 100., 0.01))

    ######################################################################################
    # Tabulating a Frame does not spawn a second, nested QuickFrame
    ######################################################################################

    # SpiceFrame quickens itself when handed an array of times, so the tabulation
    # inside QuickFrame must not re-enter that machinery. The span is short enough
    # that a QuickFrame of the tabulation times would otherwise be judged worthwhile.
    short_time = Scalar(epoch + np.arange(0., 0.5, 0.001))
    assert isinstance(mars.quick_frame(short_time, quick={}), QuickFrame)
    assert len(mars._quickframes) == 1

    # The same must hold when the frame being tabulated is a composite, which
    # requires LinkedFrame to forward `quick` to the frames it combines. The times
    # are well clear of the tabulation above, so a second QuickFrame of IAU_MARS
    # could not be mistaken for a re-use of the first.
    linked_time = short_time + 1000.
    twovector = TwoVectorFrame(mars, Vector3.XAXIS, 'X', Vector3.YAXIS, 'Y',
                               frame_id='nested_twovector')
    linked = twovector.wrt(Frame.J2000)
    assert isinstance(linked.quick_frame(linked_time, quick={}), QuickFrame)
    assert len(mars._quickframes) == 1

    ######################################################################################
    # A Frame whose transform is fixed in time is never tabulated
    ######################################################################################

    # These Frames return one Transform regardless of the times requested, so a
    # QuickFrame could not interpolate them and would gain nothing if it could
    fixed = [Cmatrix(Matrix3.IDENTITY, reference=mars, frame_id='fixed_cmatrix'),
             PosTargFrame(1.e-5, 2.e-5, mars, frame_id='fixed_postarg'),
             Rotation(0.3, 2, mars, frame_id='fixed_rotation'),
             TwoVectorFrame(mars, Vector3.XAXIS, 'X', Vector3.YAXIS, 'Y',
                            frame_id='fixed_twovector')]

    for frame in fixed:
        assert not frame._USE_QUICKFRAMES
        assert frame.quick_frame(time, quick={}) is frame

        # ...but the composite with a time-dependent reference still is tabulated
        linked = frame.wrt(Frame.J2000)
        assert linked._USE_QUICKFRAMES
        assert isinstance(linked.quick_frame(time, quick={}), QuickFrame)

    ######################################################################################
    # Tabulating a fixed Frame raises a meaningful error
    ######################################################################################

    cmatrix = fixed[0]
    with pytest.raises(ValueError):
        QuickFrame(cmatrix, epoch, epoch + 100.)

    ######################################################################################
    # Quaternions q and -q describe the same rotation, so the tabulated values can
    # reverse sign where the rotation angle passes pi; the splines require them to
    # be continuous
    ######################################################################################

    # 110 seconds at 0.1 rad/s sweeps through pi more than three times
    spin = SpinFrame(0., 0.1, epoch, 2, mars, frame_id='fast_spin')
    spin_wrt_j2000 = spin.wrt(Frame.J2000)

    quick = spin_wrt_j2000.quick_frame(time, quick={})
    assert isinstance(quick, QuickFrame)

    exact = spin_wrt_j2000.transform_at_time(time, quick=False)
    interpolated = quick.transform_at_time(time)
    error = np.max(np.abs(interpolated.matrix.vals - exact.matrix.vals))
    assert error < 1.e-8

    # The tabulation itself must be free of sign reversals
    quats = Quaternion.as_quaternion(quick._xforms.matrix).vals
    unwrapped = QuickFrame._unwrap_quaternions(quats)
    assert np.any(np.sum(quats[:-1] * quats[1:], axis=-1) < 0.)
    assert np.all(np.sum(unwrapped[:-1] * unwrapped[1:], axis=-1) > 0.)

    # Unwrapping preserves the rotations it describes
    before = Matrix3.as_matrix3(Quaternion(quats)).vals
    after = Matrix3.as_matrix3(Quaternion(unwrapped)).vals
    assert np.max(np.abs(after - before)) < 1.e-14

    # A tabulation without sign reversals is returned unchanged
    assert QuickFrame._unwrap_quaternions(unwrapped) is unwrapped

    ######################################################################################
    # A tabulation of a fittable Frame is redone after that Frame is re-fit
    ######################################################################################

    # The Cmatrix contributes no time dependence, but the SpiceFrame underneath it
    # does, so the composite is worth tabulating
    cmatrix = Cmatrix([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], mars,
                      frame_id='fitted_cmatrix')
    nav = Navigation((1.e-3, 2.e-3), cmatrix, frame_id='fitted_nav')
    nav_wrt_j2000 = nav.wrt(Frame.J2000)
    assert nav_wrt_j2000._USE_QUICKFRAMES

    quick = nav_wrt_j2000.quick_frame(time, quick={})
    assert isinstance(quick, QuickFrame)

    nav.set_params(np.array([0.5, 0.5]))
    exact = nav_wrt_j2000.transform_at_time(time, quick=False)

    # The same QuickFrame is handed back, but tabulated afresh
    reused = nav_wrt_j2000.quick_frame(time, quick={})
    assert reused is quick
    error = np.max(np.abs(reused.transform_at_time(time).matrix.vals
                          - exact.matrix.vals))
    assert error < 1.e-8


##########################################################################################
# QuickFrame.for_frame: creation, re-use, and extension
##########################################################################################

_EPOCH = 1.e8


def _dense_times(start: float, stop: float) -> Scalar:
    """Enough closely-spaced times that a QuickFrame is worth building."""

    return Scalar(_EPOCH + np.arange(start, stop, 0.01))


def _mars_frame() -> SpiceFrame:
    """The IAU_MARS body-fixed frame, with the Mars path registered alongside it."""

    SpicePath('MARS', 'SSB')

    return SpiceFrame('IAU_MARS', 'J2000')


def test_for_frame_builds_a_quickframe_when_it_is_worthwhile(core_kernels) -> None:
    """A dense set of times justifies the overhead of tabulating the frame."""

    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})

    assert isinstance(quick, QuickFrame)


def test_for_frame_returns_the_frame_when_quick_is_false(core_kernels) -> None:
    """quick=False creates no QuickFrame and returns the frame itself."""

    mars = _mars_frame()

    assert QuickFrame.for_frame(mars, _dense_times(0., 100.), quick=False) is mars


def test_for_frame_saves_the_quickframe_on_the_frame(core_kernels) -> None:
    """A QuickFrame is saved in the list inside frame._quickframes."""

    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})

    assert quick in mars._quickframes


def test_for_frame_reuses_a_covering_quickframe(core_kernels) -> None:
    """A pre-existing QuickFrame that covers the range is returned as it is."""

    mars = _mars_frame()
    first = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})
    second = QuickFrame.for_frame(mars, _dense_times(20., 80.), quick={})

    assert second is first
    assert len(mars._quickframes) == 1


def test_for_frame_extends_a_partially_covering_quickframe(core_kernels) -> None:
    """A QuickFrame covering part of the range is extended rather than duplicated."""

    mars = _mars_frame()
    first = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})
    original_end = first._times[-1]

    second = QuickFrame.for_frame(mars, _dense_times(50., 200.), quick={})

    assert second is first
    assert len(mars._quickframes) == 1
    assert second._times[-1] > original_end


def test_for_frame_builds_a_second_quickframe_for_a_distant_time(core_kernels) -> None:
    """A range nowhere near the first tabulation gets a QuickFrame of its own."""

    mars = _mars_frame()
    first = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})
    second = QuickFrame.for_frame(mars, Scalar(5.e8 + np.arange(0., 100., 0.01)),
                                  quick={})

    assert second is not first
    assert len(mars._quickframes) == 2


def test_quickframe_matches_the_frame_it_emulates(core_kernels) -> None:
    """Interpolation reproduces the rotation of the underlying frame."""

    np.random.seed(2288)
    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})

    times = Scalar(_EPOCH + np.random.rand(20) * 100.)
    vectors = Vector3(np.random.randn(20, 3))
    rotated = quick.transform_at_time(times).rotate(vectors)
    expected = mars.transform_at_time(times).rotate(vectors)

    assert abs(rotated - expected).max() < 1.e-9


def test_quickframe_reproduces_the_rotation_vector(core_kernels) -> None:
    """The tabulated omega matches that of the underlying frame."""

    np.random.seed(4471)
    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})

    times = Scalar(_EPOCH + np.random.rand(20) * 100.)
    error = abs(quick.transform_at_time(times).omega
                - mars.transform_at_time(times).omega).max()

    assert error < 1.e-12


def test_extend_widens_the_tabulated_interval(core_kernels) -> None:
    """extend() re-tabulates the frame over the wider interval."""

    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})

    quick.extend(_EPOCH - 200., _EPOCH + 300.)

    assert quick._times[0] <= _EPOCH - 200.
    assert quick._times[-1] >= _EPOCH + 300.


def test_extend_to_a_narrower_interval_changes_nothing(core_kernels) -> None:
    """An interval already covered leaves the tabulation alone."""

    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})
    before = (quick._times[0], quick._times[-1])

    quick.extend(_EPOCH + 20., _EPOCH + 80.)

    assert (quick._times[0], quick._times[-1]) == before


def test_extended_quickframe_is_still_accurate(core_kernels) -> None:
    """The frame is still reproduced accurately over the extended interval."""

    np.random.seed(6003)
    mars = _mars_frame()
    quick = QuickFrame.for_frame(mars, _dense_times(0., 100.), quick={})
    quick.extend(_EPOCH - 200., _EPOCH + 300.)

    times = Scalar(_EPOCH + np.random.rand(20) * 500. - 200.)
    vectors = Vector3(np.random.randn(20, 3))
    error = abs(quick.transform_at_time(times).rotate(vectors)
                - mars.transform_at_time(times).rotate(vectors)).max()

    assert error < 1.e-9


def test_unwrap_quaternions_flips_a_sign_reversal() -> None:
    """A sample in the opposite hemisphere is flipped to match its predecessor."""

    vals = np.array([[1., 0., 0., 0.],
                     [0.9, 0.1, 0., 0.],
                     [-0.8, -0.2, 0., 0.]])
    unwrapped = QuickFrame._unwrap_quaternions(vals)

    assert unwrapped[2, 0] == pytest.approx(0.8)
    assert unwrapped[2, 1] == pytest.approx(0.2)


def test_unwrap_quaternions_leaves_the_earlier_samples_alone() -> None:
    """Only the samples after a reversal are flipped."""

    vals = np.array([[1., 0., 0., 0.],
                     [0.9, 0.1, 0., 0.],
                     [-0.8, -0.2, 0., 0.]])
    unwrapped = QuickFrame._unwrap_quaternions(vals)

    assert unwrapped[0].tolist() == [1., 0., 0., 0.]
    assert unwrapped[1].tolist() == [0.9, 0.1, 0., 0.]


def test_unwrap_quaternions_returns_its_input_when_there_is_no_reversal() -> None:
    """An array with no sign reversals is returned as it is."""

    vals = np.array([[1., 0., 0., 0.], [0.9, 0.1, 0., 0.]])

    assert QuickFrame._unwrap_quaternions(vals) is vals

##########################################################################################

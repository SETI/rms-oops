##########################################################################################
# tests/frame/test_spiceframe.py
##########################################################################################

import numpy as np
import pytest

import cspyce

from polymath       import Scalar, Vector3
from oops.config    import QUICK
from oops.constants import DPR
from oops.event     import Event
from oops.frame     import Frame, QuickFrame, Rotation, SpiceFrame
from oops.path      import Path
from oops.path.spicepath import SpicePath
from programs.gold_master.test_support import TEST_SPICE_PREFIX


# Nothing here needs QuickPath or QuickFrame suppressed. The comparisons against
# cspyce below would not survive an interpolator being substituted for the exact
# frame, but nothing substitutes one: the calls that could -- quick_path() and
# quick_frame() -- are never reached, and every transform_at_time() call in the
# QuickFrame section passes quick=False explicitly.
def test_spiceframe(core_kernels):
    np.random.seed(6242)

    _ = SpicePath('EARTH', 'SSB')

    _ = SpiceFrame('IAU_EARTH', 'J2000')
    time  = Scalar(np.random.rand(3,4,2) * 1.e8)
    posvel = np.random.rand(3,4,2,6,1)
    event = Event(time, (posvel[...,0:3,0],posvel[...,3:6,0]), 'SSB', 'J2000')
    rotated = event.wrt_frame('IAU_EARTH')

    for i,t in np.ndenumerate(time.vals):
        matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', t)
        spiceval = np.matmul(matrix6, posvel[i])[..., np.newaxis]

        dpos = rotated.pos[i].vals[...,np.newaxis] - spiceval[0:3,0]
        dvel = rotated.vel[i].vals[...,np.newaxis] - spiceval[3:6,0]

        assert np.all(np.abs(dpos) < 1.e-15)
        assert np.all(np.abs(dvel) < 1.e-15)

    # Tests of combined frames
    Path._reset_caches()
    Frame._reset_caches()

    _ = SpicePath('EARTH', 'SSB')
    _ = SpicePath('VENUS', 'EARTH')
    _ = SpicePath('MARS', 'VENUS')
    _ = SpicePath('MOON', 'VENUS')

    _ = SpiceFrame('IAU_EARTH', 'J2000')
    _ = SpiceFrame('B1950', 'IAU_EARTH')
    _ = SpiceFrame('IAU_VENUS', 'B1950')
    _ = SpiceFrame('IAU_MARS', 'J2000')
    _ = SpiceFrame('IAU_MOON', 'B1950')

    times = Scalar(np.arange(-3.e8, 3.01e8, 0.5e7))

    frame = Frame.as_frame('IAU_EARTH').wrt('J2000')
    transform = frame.transform_at_time(times)
    for i in range(times.vals.size):
        matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', times[i].vals)
        (matrix, omega) = cspyce.xf2rav(matrix6)

        dmatrix = transform.matrix[i].vals - matrix
        domega  = transform.omega[i].vals  - omega

        assert np.all(np.abs(dmatrix) < 1.e-14)
        assert np.all(np.abs(domega)  < 1.e-14)

    frame = Frame.as_frame('J2000').wrt('IAU_EARTH')
    transform = frame.transform_at_time(times)
    for i in range(times.vals.size):
        matrix6 = cspyce.sxform('IAU_EARTH', 'J2000', times[i].vals)
        (matrix, omega) = cspyce.xf2rav(matrix6)

        dmatrix = transform.matrix[i].vals - matrix
        domega  = transform.omega[i].vals  - omega

        assert np.all(np.abs(dmatrix) < 1.e-14)
        assert np.all(np.abs(domega)  < 1.e-14)

    frame = Frame.as_frame('B1950').wrt('J2000')
    transform = frame.transform_at_time(times)
    for i in range(times.vals.size):
        matrix6 = cspyce.sxform('J2000', 'B1950', times[i].vals)
        (matrix, omega) = cspyce.xf2rav(matrix6)

        dmatrix = transform.matrix[i].vals - matrix
        domega  = transform.omega[i].vals  - omega

        assert np.all(np.abs(dmatrix) < 1.e-14)
        assert np.all(np.abs(domega)  < 1.e-14)

    frame = Frame.as_frame('J2000').wrt('B1950')
    transform = frame.transform_at_time(times)
    for i in range(times.vals.size):
        matrix6 = cspyce.sxform('B1950', 'J2000', times[i].vals)
        (matrix, omega) = cspyce.xf2rav(matrix6)

        dmatrix = transform.matrix[i].vals - matrix
        domega  = transform.omega[i].vals  - omega

        assert np.all(np.abs(dmatrix) < 1.e-14)
        assert np.all(np.abs(domega)  < 1.e-14)

    Path._reset_caches()
    Frame._reset_caches()

    ######################################################################################
    # Test for a Cassini C kernel
    ######################################################################################

    # Load all the required kernels for Cassini ISS on 2007-312
    paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls',
                                        'cas00149.tsc',
                                        'cas_v40.tf',
                                        'cas_status_v04.tf',
                                        'cas_iss_v10.ti',
                                        'pck00010.tpc',
                                        'cpck14Oct2011.tpc',
                                        'de421.bsp',
                                        'sat052.bsp',
                                        'sat083.bsp',
                                        'sat125.bsp',
                                        'sat128.bsp',
                                        'sat164.bsp',
                                        '07312_07317ra.bc',
                                        '080123R_SCPSE_07309_07329.bsp'])
    for path in paths:
        cspyce.furnsh(path)

    _ = SpicePath('CASSINI', 'SSB')
    _ = SpiceFrame('CASSINI_ISS_NAC')
    _ = SpiceFrame('CASSINI_ISS_WAC')

    # Look up N1573186009_1.IMG from COISS_2039/data/1573186009_1573197826/
    timestring = '2007-312T03:34:16.391'
    TDB = cspyce.str2et(timestring)

    nacframe = Frame.J2000.wrt('CASSINI_ISS_NAC')
    matrix = nacframe.transform_at_time(TDB).matrix
    optic_axis = (matrix * Vector3((0,0,1))).vals

    test_ra  = (np.arctan2(optic_axis[1], optic_axis[0]) * DPR) % 360
    test_dec = np.arcsin(optic_axis[2]) * DPR

    right_ascension = 194.30861     # from the index table
    declination = 3.142808

    assert np.all(np.abs(test_ra - right_ascension) < 0.5)
    assert np.all(np.abs(test_dec - declination) < 0.5)

    ######################################################################################
    # Test of various omega methods
    ######################################################################################

    wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated', frame_id='wac1')
    wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical', frame_id='wac2')
    wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero',      frame_id='wac3')

    # Test a single time
    xform1 = wac1.transform_at_time(TDB, quick=False)
    xform2 = wac2.transform_at_time(TDB, quick=False)
    xform3 = wac3.transform_at_time(TDB, quick=False)
    assert xform1.matrix == xform2.matrix
    assert xform1.matrix == xform3.matrix
    assert xform3.omega == Vector3.ZERO

    # A Transform defines omega in the reference frame, so a vector perpendicular to it
    # is a reference-frame vector; seen from the rotating frame it sweeps through
    # |omega| * dt. Hence matrix.rotate(), which carries a vector from the reference
    # frame into this one. Each ratio below falls short of one by the O(DT^2) error of a
    # chord standing in for an arc.
    DT = 0.2
    xform2a = wac2.transform_at_time(TDB-DT, quick=False)
    xform2b = wac2.transform_at_time(TDB+DT, quick=False)

    axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
    for i in range(3):
        v = xform2.omega.ucross(axes[i])
        rotated_time0 = xform2a.matrix.rotate(v)
        rotated_time1 = xform2b.matrix.rotate(v)
        angle = rotated_time1.sep(rotated_time0)
        ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
        assert ratio == pytest.approx(1., abs=0.5e-9)

    # Test an array of times
    times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

    xform1 = wac1.transform_at_time(times, quick=False)
    xform2 = wac2.transform_at_time(times, quick=False)
    xform3 = wac3.transform_at_time(times, quick=False)
    assert xform1.matrix == xform2.matrix
    assert xform1.matrix == xform3.matrix
    assert xform3.omega == Vector3.ZERO

    DT = 1.
    xform2a = wac2.transform_at_time(times-DT, quick=False)
    xform2b = wac2.transform_at_time(times+DT, quick=False)

    axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
    for i in range(3):
        v = xform2.omega.ucross(axes[i])
        rotated_time0 = xform2a.matrix.rotate(v)
        rotated_time1 = xform2b.matrix.rotate(v)
        angle = rotated_time1.sep(rotated_time0)
        ratio = angle/(2.*DT) / xform2.omega.norm()
        mask = (angle.vals != 0.)
        assert abs(ratio[mask]).max() - 1. < 1.e-9

    # Test a single value using transform_at_time_if_possible
    xform1 = wac1.transform_at_time_if_possible(TDB, quick=False)[1]
    xform2 = wac2.transform_at_time_if_possible(TDB, quick=False)[1]
    assert xform1.matrix == xform2.matrix

    DT = 0.2
    xform2a = wac2.transform_at_time_if_possible(TDB-DT, quick=False)[1]
    xform2b = wac2.transform_at_time_if_possible(TDB+DT, quick=False)[1]

    axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
    for i in range(3):
        v = xform2.omega.ucross(axes[i])
        rotated_time0 = xform2a.matrix.rotate(v)
        rotated_time1 = xform2b.matrix.rotate(v)
        angle = rotated_time1.sep(rotated_time0)
        ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
        assert ratio == pytest.approx(1., abs=0.5e-9)

    # Test an array of times using transform_at_time_if_possible
    times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

    xform1 = wac1.transform_at_time_if_possible(times, quick=False)[1]
    xform2 = wac2.transform_at_time_if_possible(times, quick=False)[1]
    xform3 = wac3.transform_at_time_if_possible(times, quick=False)[1]
    assert xform1.matrix == xform2.matrix
    assert xform1.matrix == xform3.matrix
    assert xform3.omega == Vector3.ZERO

    DT = 1.
    xform2a = wac2.transform_at_time_if_possible(times-DT, quick=False)[1]
    xform2b = wac2.transform_at_time_if_possible(times+DT, quick=False)[1]

    axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
    for i in range(3):
        v = xform2.omega.ucross(axes[i])
        rotated_time0 = xform2a.matrix.rotate(v)
        rotated_time1 = xform2b.matrix.rotate(v)
        angle = rotated_time1.sep(rotated_time0)
        ratio = angle/(2.*DT) / xform2.omega.norm()
        mask = (angle.vals != 0.)
        assert abs(ratio[mask]).max() - 1. < 1.e-9

    # Test an array of times using transform_at_time_if_possible
    # In this run, several times will fail.
    times = Scalar((TDB-5., TDB-1.e20, TDB+2., TDB-9.5, TDB+7.7, TDB+1.e10))

    xform1 = wac1.transform_at_time_if_possible(times, quick=False)[1]
    xform2 = wac2.transform_at_time_if_possible(times, quick=False)[1]
    xform3 = wac3.transform_at_time_if_possible(times, quick=False)[1]
    assert xform1.matrix == xform2.matrix
    assert xform1.matrix == xform3.matrix
    assert xform3.omega == Vector3.ZERO

    assert xform1.shape == (4,)    # two of the times are invalid
    assert xform2.shape == (4,)

    DT = 1.
    xform2a = wac2.transform_at_time_if_possible(times-DT, quick=False)[1]
    xform2b = wac2.transform_at_time_if_possible(times+DT, quick=False)[1]

    axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
    for i in range(3):
        v = xform2.omega.ucross(axes[i])
        rotated_time0 = xform2a.matrix.rotate(v)
        rotated_time1 = xform2b.matrix.rotate(v)
        angle = rotated_time1.sep(rotated_time0)
        ratio = angle/(2.*DT) / xform2.omega.norm()
        mask = (angle.vals != 0.)
        assert abs(ratio[mask]).max() - 1. < 1.e-9

    ######################################################################################
    # Tests of QuickFrame interpolation
    ######################################################################################

    wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated',)
    wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical',
                                         omega_dt=0.1)
    wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero')

    quickdict = QUICK.dictionary.copy()
    quickdict['quickframe_numerical_omega'] = False
    quickdict['frame_time_step'] = 0.01
    wac1a = QuickFrame(wac1, TDB-100., TDB+100., quick=quickdict)

    quickdict['quickframe_numerical_omega'] = True
    wac1b = QuickFrame(wac1, TDB-100., TDB+100., quick=quickdict)

    _ = QuickFrame(wac3, TDB-100., TDB+100., quick=quickdict)

    # Test a single time
    time = TDB - 44.
    xform1 = wac1a.transform_at_time(time, quick=False)
    xform2 = wac1b.transform_at_time(time, quick=False)
    xform3 = wac2.transform_at_time(time, quick=False)
    xform4 = wac3.transform_at_time(time, quick=False)
    assert xform4.omega == Vector3.ZERO

    assert xform1.matrix == xform2.matrix

    diff = xform3.matrix.vals - xform1.matrix.vals
    assert np.max(abs(diff)) < 1.e-11

    diff = xform4.matrix.vals - xform1.matrix.vals
    assert np.max(abs(diff)) < 1.e-11

    diff = (xform3.omega - xform2.omega).norm()
    assert diff.vals < 1.e-7

    diff = (xform3.omega - xform2.omega).norm() / xform2.omega.norm()
    assert diff.vals < 1.e-3

    # Test the linear interpolation limit where delta-time < 1 sec
    time = Scalar((TDB - 41.0123, TDB - 41.1357, TDB - 41.6543))
    xform2 = wac1b.transform_at_time(time, quick=False)
    xform3 = wac2.transform_at_time(time, quick=False)
    xform4 = wac3.transform_at_time(time, quick=False)
    assert xform4.omega == Vector3.ZERO

    diff = xform3.matrix.vals - xform2.matrix.vals
    assert np.max(abs(diff)) < 3.e-9

    diff = xform4.matrix.vals - xform2.matrix.vals
    assert np.max(abs(diff)) < 3.e-9

    diff = (xform3.omega - xform2.omega).norm()
    assert diff.max() < 1.e-7

    # Test the linear interpolation limit where delta-time > 1 sec
    time = Scalar((TDB - 40.0123, TDB + 41.1357, TDB - 1.6543))
    xform2 = wac1b.transform_at_time(time, quick=False)
    xform3 = wac2.transform_at_time(time, quick=False)
    xform4 = wac3.transform_at_time(time, quick=False)
    assert xform4.omega == Vector3.ZERO

    diff = xform3.matrix.vals - xform2.matrix.vals
    assert np.max(abs(diff)) < 1.e-9

    diff = xform4.matrix.vals - xform2.matrix.vals
    assert np.max(abs(diff)) < 1.e-9

    diff = (xform3.omega - xform2.omega).norm()
    assert diff.max() < 1.e-7


def test_get_reuses_a_cached_frame(core_kernels) -> None:
    """`get` returns the frame it built before, rather than building another."""

    frame = SpiceFrame.get('IAU_MARS')

    assert SpiceFrame.get('IAU_MARS') is frame
    assert SpiceFrame.get('IAU_MARS', omega_type='tabulated') is frame

    # An unconstrained time step matches the frame built with the default step
    assert SpiceFrame.get('IAU_MARS', omega_dt=None) is frame


def test_get_distinguishes_the_omega_options(core_kernels) -> None:
    """Frames differing only in how omega is computed are cached separately."""

    tabulated = SpiceFrame.get('IAU_MARS')
    numerical = SpiceFrame.get('IAU_MARS', omega_type='numerical', omega_dt=2.)

    assert numerical is not tabulated
    assert numerical._omega_type == 'numerical'
    assert numerical._omega_dt == 2.

    assert SpiceFrame.get('IAU_MARS', omega_type='numerical', omega_dt=2.) is numerical

    # A different time step is a different frame
    assert (SpiceFrame.get('IAU_MARS', omega_type='numerical', omega_dt=5.)
            is not numerical)


def test_an_inertial_frame_is_shared_across_the_omega_options(core_kernels) -> None:
    """A frame inertial with respect to its reference has zero omega either way.

    The constructor forces `omega_type` to "zero" for such a frame, so one frame answers
    every request.
    """

    frame = SpiceFrame.get('B1950')
    assert frame._omega_type == 'zero'

    assert SpiceFrame.get('B1950', omega_type='numerical') is frame
    assert SpiceFrame.get('B1950', omega_type='zero') is frame


def test_resetting_the_caches_empties_the_frame_lookup(core_kernels) -> None:
    """The lookup holds frames the registry no longer knows, so a reset clears it."""

    SpiceFrame.get('IAU_MARS')
    assert SpiceFrame._FRAME_LOOKUP
    assert SpiceFrame._FOR_NAME

    Frame._reset_caches()

    assert not SpiceFrame._FRAME_LOOKUP
    assert not SpiceFrame._FOR_NAME


##########################################################################################
# Constructing a SpiceFrame
##########################################################################################

def test_a_frame_can_be_named(core_kernels) -> None:
    """A SPICE frame name identifies the frame."""

    SpicePath('MARS', 'SSB')

    assert SpiceFrame('IAU_MARS', 'J2000').frame_id == 'IAU_MARS'


def test_a_frame_can_be_given_by_its_code(core_kernels) -> None:
    """A SPICE frame code names the same frame as its string form."""

    SpicePath('MARS', 'SSB')
    by_name = SpiceFrame('IAU_MARS', 'J2000')
    by_code = SpiceFrame(cspyce.namfrm('IAU_MARS'), 'J2000', frame_id='mars_by_code')

    time = Scalar(1.e8)
    assert by_code.transform_at_time(time).matrix == by_name.transform_at_time(time).matrix


def test_a_body_name_selects_its_rotation_frame(core_kernels) -> None:
    """An argument naming a body gives the frame associated with that body."""

    SpicePath('MARS', 'SSB')
    by_body = SpiceFrame('MARS', 'J2000', frame_id='mars_by_body')
    by_frame = SpiceFrame('IAU_MARS', 'J2000')

    time = Scalar(1.e8)
    assert by_body.transform_at_time(time).matrix \
           == by_frame.transform_at_time(time).matrix


def test_an_unrecognized_frame_code_raises(core_kernels) -> None:
    """An integer that names no frame or body raises IndexError."""

    with pytest.raises(IndexError):
        SpiceFrame(999999999, frame_id='bad_code')


def test_an_unrecognized_frame_name_raises(core_kernels) -> None:
    """A string that names no frame or body raises KeyError."""

    with pytest.raises(KeyError):
        SpiceFrame('NOT_A_SPICE_FRAME', frame_id='bad_name')


def test_a_frame_must_be_an_integer_or_a_string(core_kernels) -> None:
    """Anything else raises TypeError."""

    with pytest.raises(TypeError):
        SpiceFrame(3.14159, frame_id='bad_type')


def test_the_reference_must_be_a_spice_frame(core_kernels) -> None:
    """A reference that is neither a SpiceFrame nor J2000 raises ValueError."""

    SpicePath('MARS', 'SSB')
    not_spice = Rotation(0.1, 'z', Frame.J2000, frame_id='not_a_spice_frame')

    with pytest.raises(ValueError, match='must be a SpiceFrame or J2000'):
        SpiceFrame('IAU_MARS', not_spice, frame_id='bad_reference')


def test_an_unrecognized_omega_type_raises(core_kernels) -> None:
    """omega_type must be "tabulated", "numerical", or "zero"."""

    SpicePath('MARS', 'SSB')

    with pytest.raises(ValueError, match='omega_type'):
        SpiceFrame('IAU_MARS', 'J2000', omega_type='bogus', frame_id='bad_omega')


def test_a_spiceframe_is_always_registered(core_kernels) -> None:
    """SpiceFrames register themselves under the SPICE name by default."""

    SpicePath('MARS', 'SSB')
    frame = SpiceFrame('IAU_MARS', 'J2000')

    assert frame.is_registered
    assert Frame.frame_id_exists('IAU_MARS')


##########################################################################################
# The rotation vector
##########################################################################################

def test_the_tabulated_omega_matches_the_rotation_rate(core_kernels) -> None:
    """Mars turns once every 24h 37m, so its rotation rate is about 7.09e-5 rad/s."""

    SpicePath('MARS', 'SSB')
    frame = SpiceFrame('IAU_MARS', 'J2000')

    omega = frame.transform_at_time(Scalar(1.e8)).omega

    assert abs(omega).vals == pytest.approx(7.088e-5, rel=1.e-3)


def test_a_zero_omega_frame_does_not_rotate(core_kernels) -> None:
    """omega_type="zero" ignores the rotation vector."""

    SpicePath('MARS', 'SSB')
    frame = SpiceFrame('IAU_MARS', 'J2000', omega_type='zero', frame_id='mars_zero')

    assert frame.transform_at_time(Scalar(1.e8)).omega == Vector3.ZERO


def test_a_zero_omega_frame_still_rotates_positions(core_kernels) -> None:
    """Only the velocity term is dropped; the pointing is unchanged."""

    SpicePath('MARS', 'SSB')
    tabulated = SpiceFrame('IAU_MARS', 'J2000')
    zeroed = SpiceFrame('IAU_MARS', 'J2000', omega_type='zero', frame_id='mars_zero_2')

    time = Scalar(1.e8)
    assert zeroed.transform_at_time(time).matrix == tabulated.transform_at_time(time).matrix


def test_the_numerical_omega_has_the_tabulated_magnitude(core_kernels) -> None:
    """Deriving omega by finite differences gives the same rotation rate."""

    SpicePath('MARS', 'SSB')
    tabulated = SpiceFrame('IAU_MARS', 'J2000')
    numerical = SpiceFrame('IAU_MARS', 'J2000', omega_type='numerical',
                           frame_id='mars_numerical')

    time = Scalar(1.e8)
    assert abs(numerical.transform_at_time(time).omega).vals \
           == pytest.approx(abs(tabulated.transform_at_time(time).omega).vals, rel=1.e-6)


def test_the_numerical_omega_matches_the_tabulated_one(core_kernels) -> None:
    """Both routes to omega should give the same vector in the reference frame."""

    SpicePath('MARS', 'SSB')
    tabulated = SpiceFrame('IAU_MARS', 'J2000')
    numerical = SpiceFrame('IAU_MARS', 'J2000', omega_type='numerical',
                           frame_id='mars_numerical_2')

    time = Scalar(1.e8)
    separation = tabulated.transform_at_time(time).omega.sep(
                        numerical.transform_at_time(time).omega)

    assert separation.vals == pytest.approx(0., abs=1.e-6)


def test_the_numerical_omega_is_in_the_reference_frame(core_kernels) -> None:
    """A Transform defines omega in the reference frame, not in the target frame."""

    SpicePath('MARS', 'SSB')
    tabulated = SpiceFrame('IAU_MARS', 'J2000')
    numerical = SpiceFrame('IAU_MARS', 'J2000', omega_type='numerical',
                           frame_id='mars_numerical_3')

    time = Scalar(1.e8)
    transform = tabulated.transform_at_time(time)
    omega = numerical.transform_at_time(time).omega

    # Mars's pole is nowhere near the J2000 pole, so a vector mistakenly left in the
    # target frame would lie along its z-axis instead
    assert transform.omega.sep(omega).vals == pytest.approx(0., abs=1.e-6)
    assert omega.sep(transform.matrix.rotate(omega)).vals > 0.1


def test_the_numerical_omega_is_shaped_like_its_times(core_kernels) -> None:
    """An array of times takes the multi-time branch, which derives omega the same way."""

    SpicePath('MARS', 'SSB')
    tabulated = SpiceFrame('IAU_MARS', 'J2000')
    numerical = SpiceFrame('IAU_MARS', 'J2000', omega_type='numerical',
                           frame_id='mars_numerical_4')

    times = Scalar([1.e8, 1.e8 + 1000., 1.e8 + 2000.])
    omega = numerical.transform_at_time(times, quick=False).omega
    expected = tabulated.transform_at_time(times, quick=False).omega

    assert omega.shape == (3,)
    assert omega.sep(expected).vals == pytest.approx([0., 0., 0.], abs=1.e-6)

def test_a_body_code_selects_its_rotation_frame(core_kernels) -> None:
    """An integer that is a body code, not a frame code, names that body's frame."""

    assert SpiceFrame._frame_code_and_name(399)[1] == 'IAU_EARTH'


def test_a_body_name_with_no_pole_raises(core_kernels) -> None:
    """A body whose pole the planetary constants do not give has no frame.

    New Horizons is a body name the Toolkit knows, but no test here furnishes a frame
    kernel for it, so it has no rotation frame.
    """

    with pytest.raises(KeyError, match='frame for body "NEW HORIZONS" is undefined'):
        SpiceFrame._frame_code_and_name('NEW HORIZONS')


def test_the_tolerant_transform_builds_a_quickframe_by_default(core_kernels) -> None:
    """With `quick` left at its default, an interpolator is built for a span of times."""

    SpicePath('EARTH', 'SSB')
    frame = SpiceFrame('IAU_EARTH', 'J2000')
    time = Scalar(np.arange(0., 100., 0.01))

    (valid, xform) = frame.transform_at_time_if_possible(time)

    assert valid.shape == time.shape
    assert xform.matrix.vals == pytest.approx(
        frame.transform_at_time(time, quick=False).matrix.vals, abs=1.e-9)


def test_the_tolerant_transform_of_a_numerical_omega_frame_limits_the_time_step(
        core_kernels) -> None:
    """A numerically differentiated omega needs the interpolation step it was built for.
    """

    SpicePath('EARTH', 'SSB')
    frame = SpiceFrame('IAU_EARTH', 'J2000', omega_type='numerical',
                       frame_id='TEST_NUMERICAL_OMEGA')
    time = Scalar(np.arange(0., 100., 0.01))

    (valid, xform) = frame.transform_at_time_if_possible(time)

    assert valid.shape == time.shape
    assert xform.omega.shape == time.shape

##########################################################################################

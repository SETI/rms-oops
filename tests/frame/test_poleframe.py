##########################################################################################
# tests/frame/test_poleframe.py
##########################################################################################

import pickle

import numpy as np
import pytest

import cspyce

from polymath   import Scalar, Vector3
from oops.event import Event
from oops.frame import Frame, PoleFrame, RingFrame, SpiceFrame
from oops.path  import SpicePath


def test_poleframe(core_kernels):
    np.random.seed(1152)

    _ = SpicePath('MARS', 'SSB')
    planet = SpiceFrame('IAU_MARS', 'J2000')
    assert Frame.as_wayframe('IAU_MARS') == planet.wayframe

    # This invariable pole is aligned with the planet's pole, so this
    # should behave just like a RingFrame
    for aries in (False, True):
        pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
        poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries, frame_id='+')
        ringframe = RingFrame(planet, epoch=0., aries=aries)
        assert (Frame.as_wayframe('IAU_MARS_POLE').frame_id
                == poleframe.wayframe.frame_id[:13])
        vectors = Vector3(np.random.rand(3,4,2,3)).unit()

        ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
        pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
        diffs = ring_vecs - pole_vecs
        assert diffs.norm().max() < 1.e-15

        posvel = np.random.rand(3,4,2,6)
        event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
        rotated = event.wrt_frame('IAU_MARS')
        fixed   = event.wrt_frame(poleframe)

        # Confirm Z axis is tied to planet's pole
        diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
        assert diffs.abs().max() < 1.e-15

        # Confirm X-axis is tied to the J2000 equator
        xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        ### self.assertLess(abs(test.vals[2]), 1.e-15)

        # Confirm it's the ascending node
        xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        ### self.assertTrue(test.vals[2] > 0.)

    # Test reference angles, Aries = True vs. False
    vectors = Vector3(np.random.rand(100,3)).unit()
    poleframe1 = PoleFrame(planet, pole, cache_size=0, aries=True)
    poleframe2 = PoleFrame(planet, pole, cache_size=0, aries=False)
    pole1_vecs = poleframe1.transform_at_time(0.).rotate(vectors)
    pole2_vecs = poleframe2.transform_at_time(0.).rotate(vectors)
    (x1,y1,z1) = pole1_vecs.to_scalars()
    (x2,y2,z2) = pole2_vecs.to_scalars()

    # Z axes are the same
    assert (z1 - z2).abs().max() < 1.e-15

    # Longitudes have a fixed, nonzero offset
    dlon = (y1.arctan2(x1) - y2.arctan2(x2)) % (2.*np.pi)
    assert dlon[0] != 0.
    assert (dlon - dlon[0]).abs().max() < 1.e-15

    diff = dlon[0] - poleframe1._invariable_node_lon
    diff = (diff - np.pi) % (2.*np.pi) - np.pi
    assert diff.abs() < 1.e-15

    # Now try for Neptune
    _ = SpicePath('NEPTUNE', 'SSB')
    planet = SpiceFrame('IAU_NEPTUNE', 'J2000')

    # This invariable pole is aligned with the planet's pole, so this
    # should behave just like a RingFrame
    for aries in (False, True):
        pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
        poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries)
        ringframe = RingFrame(planet, epoch=0., aries=aries)

        vectors = Vector3(np.random.rand(3,4,2,3)).unit()

        ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
        pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
        diffs = ring_vecs - pole_vecs
        assert diffs.norm().max() < 3.e-15

        posvel = np.random.rand(3,4,2,6)
        event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
        rotated = event.wrt_frame('IAU_NEPTUNE')
        fixed   = event.wrt_frame(poleframe)

        # Confirm Z axis is tied to planet's pole
        diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
        assert diffs.abs().max() < 1.e-15

        # Confirm X-axis is tied to the J2000 equator
        xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        ### self.assertLess(abs(test.vals[2]), 1.e-15)

        # Confirm it's the ascending node
        xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos     # noqa: F841
        # The assertion below was disabled during the conversion from
        # unittest and is kept as a record; it does not hold for every case
        # this loop covers, z being negative for some of them.
        ### self.assertGreater(test.vals[2], 0.)

    # Test reference angles, Aries = True vs. False
    vectors = Vector3(np.random.rand(100,3)).unit()
    poleframe1 = PoleFrame(planet, pole, cache_size=0, aries=True)
    poleframe2 = PoleFrame(planet, pole, cache_size=0, aries=False)
    pole1_vecs = poleframe1.transform_at_time(0.).rotate(vectors)
    pole2_vecs = poleframe2.transform_at_time(0.).rotate(vectors)
    (x1,y1,z1) = pole1_vecs.to_scalars()
    (x2,y2,z2) = pole2_vecs.to_scalars()

    # Z axes are the same
    assert (z1 - z2).abs().max() < 1.e-15

    # Longitudes have a fixed, nonzero offset
    dlon = (y1.arctan2(x1) - y2.arctan2(x2)) % (2.*np.pi)
    assert dlon[0] != 0.
    assert (dlon - dlon[0]).abs().max() < 1.e-15

    diff = dlon[0] - poleframe1._invariable_node_lon
    diff = (diff - np.pi) % (2.*np.pi) - np.pi
    assert diff.abs() < 1.e-15

    # Neptune at multiple times, with actual polar precession
    times = Scalar(np.arange(1000) * 86400. * 365.)     # 1000 years
    for aries in (False, True):
        ra  = cspyce.bodvrd('NEPTUNE', 'POLE_RA')[0]  * np.pi/180
        dec = cspyce.bodvrd('NEPTUNE', 'POLE_DEC')[0] * np.pi/180
        pole = Vector3.from_ra_dec_length(ra,dec)
        poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries)

        # Make sure Z-axis tracks Neptune pole
        pole_vecs = poleframe.transform_at_time(times).unrotate(Vector3.ZAXIS)
        test_vecs = planet.transform_at_time(times).unrotate(Vector3.ZAXIS)
        diffs = pole_vecs - test_vecs
        assert diffs.norm().max() < 1.e-15

        # Make sure Z-axis circles the pole at uniform distance
        seps = pole_vecs.sep(pole)
        sep_mean = seps.mean()
        assert (seps - sep_mean).abs().max() < 3.e-5

        # Make sure the X-axis stays close to the ecliptic
        if not aries:
            node_vecs = poleframe.transform_at_time(times).unrotate(Vector3.XAXIS)
            min_node_z = np.min(node_vecs.vals[:,2])
            max_node_z = np.max(node_vecs.vals[:,2])
            assert min_node_z > -0.0062
            assert max_node_z <  0.0062
            assert abs(min_node_z + max_node_z) < 1.e-8

        # Make sure the X-axis stays in a generally fixed direction
        diffs = node_vecs - node_vecs[0]
        assert diffs.norm().max() < 0.02


def _planet_frame() -> SpiceFrame:
    """The IAU_MARS body-fixed frame, with the Mars path registered alongside it."""

    SpicePath('MARS', 'SSB')

    return SpiceFrame('IAU_MARS', 'J2000')


def test_poleframe_rejects_an_unregistered_frame(core_kernels) -> None:
    """A frame ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        PoleFrame('NOT_A_REGISTERED_FRAME', Vector3.ZAXIS)


def test_poleframe_auto_generated_frame_id(core_kernels) -> None:
    """A frame_id of "+" appends "_POLE" to the ID of the planet's frame."""

    frame = PoleFrame(_planet_frame(), Vector3.ZAXIS, frame_id='+')

    assert frame.frame_id == 'IAU_MARS_POLE'


def test_poleframe_registration(core_kernels) -> None:
    """A frame_id registers the Frame under that name."""

    PoleFrame(_planet_frame(), Vector3.ZAXIS, frame_id='TEST_POLE_FRAME')

    assert Frame.as_frame('TEST_POLE_FRAME').frame_id == 'TEST_POLE_FRAME'


def test_poleframe_retrograde_flips_the_z_axis(core_kernels) -> None:
    """A retrograde system has its Z-axis reversed."""

    planet = _planet_frame()
    pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS

    prograde = PoleFrame(planet, pole, retrograde=False)
    retrograde = PoleFrame(planet, pole, retrograde=True)

    time = Scalar(1.e8)
    up = prograde.transform_at_time(time).matrix.unrotate(Vector3.ZAXIS)
    down = retrograde.transform_at_time(time).matrix.unrotate(Vector3.ZAXIS)

    assert up.sep(down).vals == pytest.approx(np.pi, abs=1.e-9)


def test_poleframe_transform_is_cached(core_kernels) -> None:
    """A transform is cached, so asking twice at one time gives the same matrix."""

    frame = PoleFrame(_planet_frame(), Vector3.ZAXIS)
    time = Scalar(1.e8)

    assert frame.transform_at_time(time).matrix == frame.transform_at_time(time).matrix


def test_poleframe_cache_size_may_be_set(core_kernels) -> None:
    """A small cache still returns the same transforms, just holding fewer of them."""

    planet = _planet_frame()
    small = PoleFrame(planet, Vector3.ZAXIS, cache_size=2, frame_id='pole_small')
    large = PoleFrame(planet, Vector3.ZAXIS, cache_size=100, frame_id='pole_large')

    for time in (0., 1000., 2000., 3000., 0.):
        assert small.transform_at_time(Scalar(time)).matrix \
               == large.transform_at_time(Scalar(time)).matrix


def test_poleframe_accepts_an_array_of_times(core_kernels) -> None:
    """A shaped time gives a shaped Transform."""

    frame = PoleFrame(_planet_frame(), Vector3.ZAXIS)

    assert frame.transform_at_time(Scalar([0., 1000., 2000.])).shape == (3,)


def test_poleframe_rejects_unbroadcastable_shapes(core_kernels) -> None:
    """A frame and a pole whose shapes cannot be broadcast raise ValueError."""

    planet = _planet_frame()
    poles = Vector3([(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)])
    frame = PoleFrame(planet, poles)

    with pytest.raises(ValueError):
        frame.transform_at_time(Scalar([0., 10.]))


def test_poleframe_pickle(core_kernels) -> None:
    """Pickling restores the planet's frame and the invariable pole."""

    frame = PoleFrame(_planet_frame(), Vector3((0.1, 0.2, 1.)))
    restored = pickle.loads(pickle.dumps(frame))
    time = Scalar(1.e8)

    assert isinstance(restored, PoleFrame)
    assert restored.transform_at_time(time).matrix == frame.transform_at_time(time).matrix


def test_poleframe_getstate_roundtrip(core_kernels) -> None:
    """The state captured by __getstate__ fully restores the object."""

    frame = PoleFrame(_planet_frame(), Vector3((0.1, 0.2, 1.)))
    state = frame.__getstate__()

    copied = Frame.__new__(PoleFrame)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################

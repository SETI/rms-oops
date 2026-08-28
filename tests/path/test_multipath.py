################################################################################
# tests/path/test_multipath.py
################################################################################

from oops.path import MultiPath, SpicePath


def test_multipath(core_kernels):
    sun   = SpicePath("SUN", "SSB")
    earth = SpicePath("EARTH", "SSB")
    moon  = SpicePath("MOON", "EARTH")

    test = MultiPath([sun,earth,moon], "SSB", path_id='+')

    # behavior changed
    # self.assertEqual(test.path_id, "SUN+EARTH+MOON")
    assert test.shape == (3,)

    # Single time
    event0 = test.event_at_time(0.)
    assert event0.shape == (3,)

    # Triple of times, shape = [3]
    event012 = test.event_at_time((0., 1.e5, 2.e5))
    assert event012.shape == (3,)

    assert event012.pos[0] == event0.pos[0]
    assert event012.vel[0] == event0.vel[0]
    assert event012.pos[1] != event0.pos[1]
    assert event012.vel[1] != event0.vel[1]
    assert event012.pos[2] != event0.pos[2]
    assert event012.vel[2] != event0.vel[2]

    # Times shaped [2,1]
    event01x = test.event_at_time([[0.], [1.e5]])
    assert event01x.shape == (2,3)

    assert event01x.pos[0,0] == event0.pos[0]
    assert event01x.vel[0,0] == event0.vel[0]
    assert event01x.pos[0,1] == event0.pos[1]
    assert event01x.vel[0,1] == event0.vel[1]
    assert event01x.pos[0,2] == event0.pos[2]
    assert event01x.vel[0,2] == event0.vel[2]

    assert event01x.pos[1,1] == event012.pos[1]
    assert event01x.vel[1,1] == event012.vel[1]
    assert event01x.pos[1,2] != event012.pos[2]
    assert event01x.vel[1,2] != event012.pos[2]

    # Triple of times, at all times, shape [3,1]
    event012a = test.event_at_time([[0.], [1.e5], [2.e5]])
    assert event012a.shape == (3,3)

    assert event012a.pos[0,:] == event0.pos
    assert event012a.vel[0,:] == event0.vel

    assert event012a.pos[0,0] == event012.pos[0]
    assert event012a.vel[0,0] == event012.vel[0]
    assert event012a.pos[1,1] == event012.pos[1]
    assert event012a.vel[1,1] == event012.vel[1]
    assert event012a.pos[2,2] == event012.pos[2]
    assert event012a.vel[2,2] == event012.vel[2]

    assert event012a.pos[0:2] == event01x.pos
    assert event012a.vel[0:2] == event01x.vel
################################################################################

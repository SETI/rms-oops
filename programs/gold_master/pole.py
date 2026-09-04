##########################################################################################
# programs/gold_master/pole.py
##########################################################################################
"""Gold master tests of the pole backplanes.

These backplanes describe the projected direction of a body's north pole in the sky
plane.
"""

from programs.gold_master import register_test_suite

def pole_test_suite(bpt):
    """Test the pole clock angle and pole position angle of every body and ring.

    Both angles are compared to their gold masters, and their sum is confirmed to be zero
    modulo 360 degrees, because the two angles are measured in opposite senses from
    perpendicular reference directions.

    Parameters:
        bpt (BackplaneTest): The gold master test object, which provides the Backplane to
            evaluate, the lists of surface names to test, and the gmtest() and compare()
            methods that log each result.
    """

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:

        clock = bp.pole_clock_angle(name)
        position = bp.pole_position_angle(name)
        bpt.gmtest(clock,
                   name + ' pole clock angle (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.gmtest(position,
                   name + ' pole position angle (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.compare(clock + position, 0.,
                    name + ' pole clock plus position angle (deg)',
                    method='mod360', limit=1.e-13, radius=1)

register_test_suite('pole', pole_test_suite)

##########################################################################################

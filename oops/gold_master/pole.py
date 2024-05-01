################################################################################
# oops/gold_master/pole.py
################################################################################

from oops.gold_master import register_test_suite

def pole_test_suite(bpt):

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

################################################################################

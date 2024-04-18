################################################################################
# oops/gold_master/orbit.py
################################################################################

from oops.gold_master import register_test_suite

def orbit_test_suite(bpt):

    bp = bpt.backplane
    for (_, name) in bpt.planet_moon_pairs:
        bpt.gmtest(bp.orbit_longitude(name, reference='obs'),
                   name + ' orbit longitude wrt observer (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(bp.orbit_longitude(name, reference='oha'),
                   name + ' orbit longitude wrt OHA (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(bp.orbit_longitude(name, reference='sun'),
                   name + ' orbit longitude wrt Sun (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(bp.orbit_longitude(name, reference='sha'),
                   name + ' orbit longitude wrt SHA (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(bp.orbit_longitude(name, reference='aries'),
                   name + ' orbit longitude wrt Aries (deg)',
                   method='mod360', limit=0.001)
        bpt.gmtest(bp.orbit_longitude(name, reference='node'),
                   name + ' orbit longitude wrt node (deg)',
                   method='mod360', limit=0.001)

register_test_suite('orbit', orbit_test_suite)

################################################################################

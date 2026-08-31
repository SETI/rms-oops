##########################################################################################
# programs/gold_master/orbit.py
##########################################################################################

"""Gold master tests of the orbit longitude backplane, which describes a moon's position
within its own orbit about its primary.
"""

from programs.gold_master import register_test_suite

def orbit_test_suite(bpt):
    """Test the orbit longitude of every moon against its gold master.

    Longitudes are measured relative to each of the six supported references: the
    observer, the observer hour angle, the Sun, the solar hour angle, the First Point of
    Aries, and the ascending node. All are compared modulo 360 degrees.

    Parameters:
        bpt (BackplaneTest): The gold master test object, which provides the Backplane to
            evaluate, the list of planet/moon pairs to test, and the gmtest() method that
            logs each result.
    """

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

##########################################################################################

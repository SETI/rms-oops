################################################################################
# oops/gold_master/where.py
################################################################################

from polymath         import Scalar
from oops.gold_master import register_test_suite

def where_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names:
        intercepted = bp.where_intercepted(name)
        sunward     = bp.where_sunward(name, tvl=True)
        antisunward = bp.where_antisunward(name, tvl=True)

        bpt.gmtest(intercepted,
                   name + ' where intercepted',
                   radius=1.5)
        bpt.gmtest(sunward,
                   name + ' where sunward',
                   radius=1.5)
        bpt.gmtest(antisunward,
                   name + ' where anti-sunward',
                   radius=1.5)
        bpt.compare(intercepted == (sunward.vals | antisunward.vals),
                    True,
                    name + ' mask eq sunward|antisunward')
        bpt.compare(sunward.tvl_eq(bp.where_below(('incidence_angle', name),
                                                  Scalar.HALFPI, tvl=True)),
                    True,
                    name + ' where sunward eq incidence below 90 deg')
        bpt.compare(antisunward.tvl_eq(bp.where_above(('incidence_angle', name),
                                                      Scalar.HALFPI, tvl=True)),
                    True,
                    name + ' where antisunward eq incidence above 90 deg')
        bpt.compare(bp.where_above(('phase_angle', name), Scalar.PI, tvl=False),
                    False,
                    name + ' where phase angle below 180 deg')
        bpt.compare(intercepted == bp.evaluate(('where_intercepted', name)),
                    True,
                    name + ' mask eq via evaluate')

    for (planet, ring) in bpt.planet_ring_pairs:

        # Planet first
        intercepted = bp.where_intercepted(planet)
        in_front = bp.where_in_front(planet, ring, tvl=True)
        in_back  = bp.where_in_back(planet, ring, tvl=True)
        bpt.gmtest(in_front,
                   planet + ' where in front of ' + ring,
                   radius=1.5)
        bpt.gmtest(in_back,
                   planet + ' where behind ' + ring,
                   radius=1.5)
        bpt.compare(intercepted == (in_front.vals | in_back.vals),
                    True,
                    planet + ' mask eq in front|behind ' + ring)

        inside  = bp.where_inside_shadow(planet, ring, tvl=True)
        outside = bp.where_outside_shadow(planet, ring, tvl=True)
        bpt.gmtest(inside,
                   planet + ' where shadowed by ' + ring,
                   radius=1.5)
        bpt.gmtest(outside,
                   planet + ' where un-shadowed by ' + ring,
                   radius=1.5)
        bpt.compare(intercepted == (inside.vals | outside.vals),
                    True,
                    planet + ' mask eq inside|outside shadow of ' + ring)

        # Ring first
        intercepted = bp.where_intercepted(ring)
        in_front = bp.where_in_front(ring, planet, tvl=True)
        in_back  = bp.where_in_back(ring, planet, tvl=True)
        bpt.gmtest(in_front,
                   ring + ' where in front of ' + planet,
                   radius=1.5)
        bpt.gmtest(in_back,
                   ring + ' where behind ' + planet,
                   radius=1.5)
        bpt.compare(intercepted == (in_front.vals | in_back.vals),
                    True,
                    ring + ' mask eq in front|behind ' + planet)

        inside  = bp.where_inside_shadow(ring, planet, tvl=True)
        outside = bp.where_outside_shadow(ring, planet, tvl=True)
        bpt.gmtest(inside,
                   ring + ' where inside shadow of ' + planet,
                   radius=1.5)
        bpt.gmtest(outside,
                   ring + ' where outside shadow of ' + planet,
                   radius=1.5)
        bpt.compare(intercepted == (inside.vals | outside.vals),
                    True,
                    ring + ' mask eq inside|outside shadow of ' + planet)

        # Ring inside planet test
        if ':' in ring:             # just consider the unmasked ring
            interior = bp.where_inside(ring, planet, tvl=True)
            bpt.gmtest(interior,
                       ring + ' where inside ' + planet,
                       radius=1.5)

    for name in bpt.ring_names:
        intercepted = bp.where_intercepted(name)
        sunward     = bp.where_sunward(name, tvl=True)
        antisunward = bp.where_antisunward(name, tvl=True)

        bpt.gmtest(intercepted,
                   name + ' where intercepted',
                   radius=1.5)
        bpt.gmtest(sunward,
                   name + ' where sunward',
                   radius=1.5)
        bpt.gmtest(antisunward,
                   name + ' where anti-sunward',
                   radius=1.5)

        bpt.compare(bp.where_below(('ring_radius', ring), 0.),
                    False,
                    name + ' where radius is negative')

register_test_suite('where', where_test_suite)

################################################################################

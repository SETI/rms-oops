################################################################################
# oops/gold_master/border.py
################################################################################

from oops.gold_master import register_test_suite

def border_test_suite(bpt):

    bp = bpt.backplane

    # Test border of each body intercepted mask
    for name in bpt.body_names:
        mask    = bp.where_intercepted(name)
        inside  = bp.border_inside(mask)
        outside = bp.border_outside(mask)

        bpt.gmtest(inside,
                   name + ' interior border',
                   method='border', radius=1)
        bpt.gmtest(outside,
                   name + ' exterior border',
                   method='border', radius=1)

        # ... additional tests
        bpt.compare(mask[inside], True,
                    name + ' where interior border overlaps mask')
        bpt.compare(mask[outside], False,
                    name + ' where exterior border overlaps mask')

    # Test ring boundaries
    for name in bpt.ring_names:
        radius = bp.ring_radius(name)
        below  = bp.border_below(('ring_radius', name), 100.e3)
        above  = bp.border_above(('ring_radius', name), 100.e3)
        atop   = bp.border_atop (('ring_radius', name), 100.e3)

        bpt.gmtest(below,
                   name + ' border below radius 100 kkm',
                   method='border', radius=1)
        bpt.gmtest(above,
                   name + ' border above radius 100 kkm',
                   method='border', radius=1)
        bpt.gmtest(atop ,
                   name + ' border atop radius 100 kkm',
                   method='border', radius=1)

        # ... additional tests
        bpt.compare(radius[below], 100.e3,
                    name + ' radii of border below 100 kkm',
                    operator='<=')
        bpt.compare(radius[above], 100.e3,
                    name + ' radii of border above 100 kkm',
                    operator='>=')
        bpt.compare((above | below)[atop], True,
                    name + ' border atop 100 kkm overlaps above|below',
                    radius=1)

register_test_suite('border', border_test_suite)

################################################################################

##########################################################################################
# programs/gold_master/border.py
##########################################################################################

"""Gold master tests of the border backplanes, which identify the pixels along the edge
of a mask or along a boundary in the values of another backplane.
"""

from programs.gold_master import register_test_suite

def border_test_suite(bpt):
    """Test the border backplanes of every body and ring.

    For each body, the interior and exterior borders of the intercept mask are compared
    to their gold masters, and the interior border is confirmed to fall inside that mask
    while the exterior border falls outside it. For each ring, the borders below, above,
    and atop a radius of 100,000 km are compared to their gold masters; radii along the
    lower and upper borders are confirmed to fall on the correct side of 100,000 km, and
    the border atop that radius is confirmed to overlap the union of the other two.

    Border comparisons use method='border', for which the radius value is expressed in
    undersampled rather than original pixels.

    Parameters:
        bpt (BackplaneTest): The gold master test object, which provides the Backplane to
            evaluate, the lists of surface names to test, and the gmtest() and compare()
            methods that log each result.
    """

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

##########################################################################################

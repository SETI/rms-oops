##########################################################################################
# programs/gold_master/resolution.py
##########################################################################################

"""Gold master tests of the surface resolution backplanes, which describe the projected
size of a pixel at the surface.
"""

from programs.gold_master import register_test_suite

def resolution_test_suite(bpt):
    """Test the resolution backplanes of every body, limb, ring, and ansa.

    The resolution and center resolution along the u and v axes are compared to their
    gold masters, as are the finest and coarsest resolutions. Because the finest and
    coarsest values diverge where the emission angle approaches 90 degrees, those two
    tests exclude the pixels whose cosine of the emission angle falls below 0.1.

    Parameters:
        bpt (BackplaneTest): The gold master test object, which provides the Backplane to
            evaluate, the lists of surface names to test, and the gmtest() method that
            logs each result.
    """

    bp = bpt.backplane
    for name in (bpt.body_names + bpt.limb_names +
                 bpt.ring_names + bpt.ansa_names):

        bpt.gmtest(bp.resolution(name, 'u'),
                   name + ' resolution along u axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.resolution(name, 'v'),
                   name + ' resolution along v axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.center_resolution(name, 'u'),
                   name + ' center resolution along u axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.center_resolution(name, 'v'),
                   name + ' center resolution along v axis (km)',
                   limit=0.01, radius=1.5)

        # Because finest/coarsest resolution values diverge for emission angles
        # near 90, we need to apply an extra mask
        mu = bp.emission_angle(name).cos().abs()
        mask = mu.tvl_lt(0.1).as_mask_where_nonzero_or_masked()

        bpt.gmtest(bp.finest_resolution(name),
                   name + ' finest resolution (km)',
                   limit=0.01, radius=1.5, mask=mask)
        bpt.gmtest(bp.coarsest_resolution(name),
                   name + ' coarsest resolution (km)',
                   limit=0.1, radius=1.5, mask=mask)

register_test_suite('resolution', resolution_test_suite)

##########################################################################################

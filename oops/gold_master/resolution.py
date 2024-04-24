################################################################################
# oops/gold_master/resolution.py
################################################################################

from oops.gold_master import register_test_suite

def resolution_test_suite(bpt):

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

################################################################################

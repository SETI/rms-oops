################################################################################
# oops/gold_master/distance.py
################################################################################

import numpy as np

from oops.gold_master import register_test_suite

def distance_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:

        # Observer distance and light time
        bpt.gmtest(bp.distance(name),
                   name + ' distance to observer (km)',
                   limit=1., radius=1)
        bpt.gmtest(bp.center_distance(name),
                   name + ' center distance to observer (km)',
                   limit=1.)

        lt = bp.light_time(name)
        clt = bp.center_light_time(name)
        bpt.gmtest(lt,
                   name + ' light time to observer (s)',
                   limit=3.e-6, radius=1)
        bpt.gmtest(clt,
                   name + ' center light time to observer (km)',
                   limit=3.e-6)

        # Sun distance and light time
        bpt.gmtest(bp.distance(name, direction='arr'),
                   name + ' distance from Sun (km)',
                   limit=1., radius=1)
        bpt.gmtest(bp.center_distance(name, direction='arr'),
                    name + ' center distance from Sun (km)',
                   limit=1.)

        bpt.gmtest(bp.light_time(name, direction='arr'),
                   name + ' light time from Sun (km)',
                   limit=3.e-6, radius=1)
        bpt.gmtest(bp.center_light_time(name, direction='arr'),
                   name + ' center light time from Sun (km)',
                   limit=3.e-6, radius=1)

        # Event time
        bpt.gmtest(bp.event_time(name),
                   name + ' event time (TDB)',
                   limit=0.01, radius=1)

    for (planet, ring) in bpt.planet_ring_pairs:
        bpt.compare(bp.center_distance(planet) - bp.center_distance(ring),
                    0.,
                    planet + ' center minus ' + ring
                           + ' center to observer (km)',
                    limit=1.e-6)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.body_names + bpt.ring_names:

        km_per_los_radian = bp.distance(name) / bp.mu(name)
        (ulimit, vlimit) = km_per_los_radian.median() * pixel_duv * 1.e-4

        dist = bp.distance(name)
        ddist_duv = dist.d_dlos.chain(bp.dlos_duv)
        (ddist_du, ddist_dv) = ddist_duv.extract_denoms()

        ddist = bp_u1.distance(name) - bp_u0.distance(name)
        if not np.all(ddist.mask):
            bpt.compare((ddist.wod/bpt.duv - ddist_du).abs().median(), 0.,
                        name + ' distance d/du self-check (km/pix)',
                        limit=ulimit)

        ddist = bp_v1.distance(name) - bp_v0.distance(name)
        if not np.all(ddist.mask):
            bpt.compare((ddist.wod/bpt.duv - ddist_dv).abs().median(), 0.,
                        name + ' distance d/dv self-check (km/pix)',
                        limit=vlimit)

register_test_suite('distance', distance_test_suite)

################################################################################

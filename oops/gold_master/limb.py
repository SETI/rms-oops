################################################################################
# oops/gold_master/limb_backplanes.py
################################################################################

import numpy as np

from oops.gold_master import register_test_suite

def limb_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.limb_names:

        altitude = bp.limb_altitude(name)
        bpt.gmtest(altitude,
                   name + ' altitude (km)',
                   limit=0.1, radius=1)
        bpt.compare(bp.limb_longitude(name) - bp.longitude(name), 0.,
                   name + ' longitude, limb minus generic (deg)',
                   method='mod360', limit=1.e-13)
        bpt.compare(bp.limb_latitude(name) - bp.latitude(name), 0.,
                   name + ' latitude, limb minus generic (deg)',
                   method='degrees', limit=1.e-13)

        bpt.gmtest(bp.limb_clock_angle(name),
                   name + ' clock angle (deg)',
                   limit=0.001, radius=1, method='mod360')

        # Test a masked version
        key = ('limb_altitude', name, 0., 80000.)
        limited = bp.evaluate(key)
        mask = limited.expand_mask().mask

        bpt.gmtest(limited,
                   name + ' altitude masked above 80 kkm',
                   limit=0.1, radius=1)
        bpt.compare(limited - altitude, 0.,
                    name + ' altitude masked above 80 kkm minus unmasked')
        bpt.compare(limited - 80000., 0.,
                    name + ' altitude masked above 80 kkm minus 80,000',
                    operator='<=')

        # Test lat/lon derived from masked altitude
        bpt.compare(bp.limb_longitude(key).mask == mask,
                    True,
                    name + ' longitude mask eq altitude mask')
        bpt.compare(bp.limb_latitude(key).mask == mask,
                    True,
                    name + ' latitude mask eq altitude mask')

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.limb_names:

        ulimit = bp.center_distance(name).max() * pixel_duv[0] * 1.e-3
        vlimit = bp.center_distance(name).max() * pixel_duv[1] * 1.e-3

        # limb_altitude
        alt = bp.limb_altitude(name)
        dalt_duv = alt.d_dlos.chain(bp.dlos_duv)
        (dalt_du, dalt_dv) = dalt_duv.extract_denoms()

        dalt = bp_u1.limb_altitude(name) - bp_u0.limb_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_du,
                    name + ' altitude d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        dalt = bp_v1.limb_altitude(name) - bp_v0.limb_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_dv,
                    name + ' altitude d/dv self-check (km/pix)',
                    limit=vlimit, radius=1)

        # limb_clock_angle
        clock = bp.limb_clock_angle(name)
        dclock_duv = clock.d_dlos.chain(bp.dlos_duv)
        (dclock_du, dclock_dv) = dclock_duv.extract_denoms()

        dclock = bp_u1.limb_clock_angle(name) - bp_u0.limb_clock_angle(name)
        bpt.compare(dclock.wod/bpt.duv, dclock_du,
                    name + ' clock angle d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        dclock = bp_v1.limb_clock_angle(name) - bp_v0.limb_clock_angle(name)
        bpt.compare(dclock.wod/bpt.duv, dclock_dv,
                    name + ' clock angle d/dv self-check (km/pix)',
                    limit=vlimit, radius=1, method='mod360')

register_test_suite('limb', limb_test_suite)

################################################################################

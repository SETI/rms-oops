################################################################################
# oops/gold_master/ansa_backplanes.py
################################################################################

import numpy as np

from polymath         import Scalar
from oops.gold_master import register_test_suite

def ansa_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.ansa_names:
        bpt.gmtest(bp.ansa_radius(name),
                   name + ' radius (km)',
                   limit=0.1, radius=1)
        bpt.gmtest(bp.ansa_altitude(name),
                   name + ' altitude (km)',
                   limit=0.1, radius=1)
        bpt.gmtest(bp.ansa_radial_resolution(name),
                   name + ' radial resolution (km)',
                   limit=0.003, radius=1.5)
        bpt.gmtest(bp.ansa_vertical_resolution(name),
                   name + ' vertical resolution (km)',
                   limit=0.003, radius=1.5)

        bpt.gmtest(bp.ansa_longitude(name, 'node'),
                   name + ' longitude wrt node (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.gmtest(bp.ansa_longitude(name, 'aries'),
                   name + ' longitude wrt Aries (deg)',
                   method='mod360', limit=0.001, radius=1)

        longitude = bp.ansa_longitude(name, 'obs')
        bpt.gmtest(longitude,
                   name + ' longitude wrt observer (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.compare(longitude - bp.ansa_longitude(name, 'oha'),
                    Scalar.PI,
                    name + ' longitude wrt observer minus wrt OHA (deg)',
                    method='mod360', limit=1.e-13)

        longitude = bp.ansa_longitude(name, 'sun')
        bpt.gmtest(longitude,
                   name + ' longitude wrt Sun (deg)',
                   method = 'mod360', limit = 0.001, radius = 1)
        bpt.compare(longitude - bp.ansa_longitude(name, 'sha'),
                    Scalar.PI,
                    name + ' longitude wrt Sun minus wrt SHA (deg)',
                    method='mod360', limit=1.e-13)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.ansa_names:

        ulimit = bp.center_distance(name).max() * pixel_duv[0] * 1.e-3
        vlimit = bp.center_distance(name).max() * pixel_duv[1] * 1.e-3

        # ansa_radius
        rad = bp.ansa_radius(name)
        drad_duv = rad.d_dlos.chain(bp.dlos_duv)
        (drad_du, drad_dv) = drad_duv.extract_denoms()

        drad = bp_u1.ansa_radius(name) - bp_u0.ansa_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_du,
                    name + ' radius d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        drad = bp_v1.ansa_radius(name) - bp_v0.ansa_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_dv,
                    name + ' radius d/dv self-check (km/pix)',
                    limit=vlimit, radius=1)

        # ansa_altitude
        alt = bp.ansa_altitude(name)
        dalt_duv = alt.d_dlos.chain(bp.dlos_duv)
        (dalt_du, dalt_dv) = dalt_duv.extract_denoms()

        dalt = bp_u1.ansa_altitude(name) - bp_u0.ansa_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_du,
                    name + ' altitude d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        dalt = bp_v1.ansa_altitude(name) - bp_v0.ansa_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_dv,
                    name + ' altitude d/dv self-check (km/pix)',
                    limit=vlimit, radius=1)

register_test_suite('ansa', ansa_test_suite)

################################################################################

################################################################################
# oops/gold_master/sky.py: Sky plane (celestial coordinates) backplanes
################################################################################

import numpy as np

from polymath         import Scalar
from oops.constants   import DPR
from oops.gold_master import register_test_suite

def sky_test_suite(bpt):

    bp = bpt.backplane

    # Right ascension
    cos_dec = bp.declination().cos().mean(builtins=True)
    actual = bp.right_ascension(apparent=False)
    apparent = bp.right_ascension(apparent=True)
    bpt.gmtest(actual,
               'Right ascension (deg, actual)',
               limit=1.e-6/cos_dec, method='mod360', radius=1.)
    bpt.gmtest(apparent,
               'Right ascension (deg, apparent)',
               limit=1.e-6/cos_dec, method='mod360', radius=1.)
    bpt.compare(actual - apparent, 0.,
                'Right ascension, actual minus apparent (deg)',
                limit=0.1/cos_dec, method='mod360')

    # Declination
    actual = bp.declination(apparent=False)
    apparent = bp.declination(apparent=True)
    bpt.gmtest(actual,
               'Declination (deg, actual)',
               limit=1.e-6, method='degrees', radius=1.)
    bpt.gmtest(apparent,
               'Declination (deg, apparent)',
               limit=1.e-6, method='degrees', radius=1.)
    bpt.compare(actual - apparent, 0.,
                'Declination, actual minus apparent (deg)',
                limit=0.1/cos_dec, method='mod360')

    # Sky angles
    north = bp.celestial_north_angle()
    east  = bp.celestial_east_angle()
    bpt.gmtest(north,
               'Celestial north angle (deg)',
               method='mod360', limit=0.001)
    bpt.gmtest(east,
               'Celestial east angle (deg)',
               method='mod360', limit=0.001)
    bpt.compare(north - east, Scalar.HALFPI,
                'Celestial north minus east angles (deg)',
                method='mod360', limit=2.)

    for name in bpt.body_names:

        # Right ascension
        cos_dec = bp.center_declination(name).cos().mean(builtins=True)
        actual = bp.center_right_ascension(name, apparent=False)
        apparent = bp.center_right_ascension(name, apparent=True)
        bpt.gmtest(actual,
                   name + ' center right ascension (deg, actual)',
                   limit=1.e-6/cos_dec, method='mod360')
        bpt.gmtest(apparent,
                   name + ' center right ascension (deg, apparent)',
                   limit=1.e-6/cos_dec, method='mod360')
        bpt.compare(actual - apparent, 0.,
                    name + ' center right ascension, actual minus apparent (deg)',
                    limit=0.1/cos_dec, method='mod360')

        # Declination
        actual = bp.center_declination(name, apparent=False)
        apparent = bp.center_declination(name, apparent=True)
        bpt.gmtest(actual,
                   name + ' center declination (deg, actual)',
                   limit=1.e-6, method='degrees')
        bpt.gmtest(apparent,
                   name + ' center declination (deg, apparent)',
                   limit=1.e-6, method='degrees')
        bpt.compare(actual - apparent, 0.,
                    name + ' center declination, actual minus apparent (deg)',
                    limit=0.1, method='degrees')

    # Derivative tests
    if bpt.derivs:
        (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
        pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)
        cos_dec = bp.declination().cos().mean(builtins=True)
        (ulimit, vlimit) = DPR * pixel_duv * 1.e-4

        # right_ascension
        ra = bp.right_ascension()
        dra_duv = ra.d_dlos.chain(bp.dlos_duv)
        (dra_du, dra_dv) = dra_duv.extract_denoms()

        dra = bp_u1.right_ascension() - bp_u0.right_ascension()
        dra = Scalar.PI - (dra.wod - Scalar.PI).abs()
        bpt.compare(dra/bpt.duv, dra_du,
                    'Right ascension d/du self-check (deg/pix)',
                    limit=ulimit/cos_dec, radius=1, method='degrees')

        dra = bp_v1.right_ascension() - bp_v0.right_ascension()
        dra = Scalar.PI - (dra.wod - Scalar.PI).abs()
        bpt.compare(dra/bpt.duv, dra_dv,
                    'Right ascension d/dv self-check (deg/pix)',
                    limit=vlimit/cos_dec, radius=1, method='degrees')

        # declination
        dec = bp.declination()
        ddec_duv = dec.d_dlos.chain(bp.dlos_duv)
        (ddec_du, ddec_dv) = ddec_duv.extract_denoms()

        ddec = bp_u1.declination() - bp_u0.declination()
        bpt.compare(ddec.wod/bpt.duv, ddec_du,
                    'Declination d/du self-check (deg/pix)',
                    limit=ulimit, radius=1, method='degrees')

        ddec = bp_v1.declination() - bp_v0.declination()
        bpt.compare(ddec.wod/bpt.duv, ddec_dv,
                    'Declination d/dv self-check (deg/pix)',
                    limit=vlimit, radius=1, method='degrees')

register_test_suite('sky', sky_test_suite)

################################################################################

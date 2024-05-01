################################################################################
# oops/gold_master/lighting.py
################################################################################

import numpy as np

from polymath         import Scalar
from oops.body        import Body
from oops.constants   import DPR
from oops.gold_master import register_test_suite

def lighting_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:

        apparent = bp.phase_angle(name, apparent=True)
        actual   = bp.phase_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' phase angle, apparent (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.gmtest(actual,
                   name + ' phase angle, actual (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.compare(bp.phase_angle(name) + bp.scattering_angle(name),
                    Scalar.PI,
                    name + ' phase plus scattering angle (deg)',
                    limit=1.e-14, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' phase angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

        apparent = bp.center_phase_angle(name, apparent=True)
        actual   = bp.center_phase_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' center phase angle, apparent (deg)',
                   limit=0.001, method='degrees')
        bpt.gmtest(actual,
                   name + ' center phase angle, actual (deg)',
                   limit=0.001, method='degrees')
        bpt.compare(bp.center_phase_angle(name)
                    + bp.center_scattering_angle(name),
                    Scalar.PI,
                    name + ' center phase plus scattering angle (deg)',
                    limit=1.e-14, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' center phase angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

        apparent = bp.incidence_angle(name, apparent=True)
        actual   = bp.incidence_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' incidence angle, apparent (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.gmtest(actual,
                   name + ' incidence angle, actual (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' incidence angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

        apparent = bp.emission_angle(name, apparent=True)
        actual   = bp.emission_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' emission angle, apparent (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.gmtest(actual,
                   name + ' emission angle, actual (deg)',
                   limit=0.001, radius=1.5, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' emission angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

    for name in bpt.ring_names:
        apparent = bp.center_incidence_angle(name, apparent=True)
        actual   = bp.center_incidence_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' center incidence angle, apparent (deg)',
                   limit=0.001, method='degrees')
        bpt.gmtest(actual,
                   name + ' center incidence angle, actual (deg)',
                   limit=0.001, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' center incidence angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

        apparent = bp.center_emission_angle(name, apparent=True)
        actual   = bp.center_emission_angle(name, apparent=False)
        bpt.gmtest(apparent,
                   name + ' center emission angle, apparent (deg)',
                   limit=0.001, method='degrees')
        bpt.gmtest(actual,
                   name + ' center emission angle, actual (deg)',
                   limit=0.001, method='degrees')
        bpt.compare(apparent - actual,
                    0.,
                    name + ' center emission angle, apparent minus actual (deg)',
                    limit=0.1, method='degrees')

    # Surface laws
    for name in bpt.body_names:
        bpt.gmtest(bp.lambert_law(name),
                   name + ' as a Lambert law',
                   limit=0.001, radius=1)
        bpt.gmtest(bp.minnaert_law(name, 0.5),
                   name + ' as a Minnaert law (k=0.7)',
                   limit=0.001, radius=1)
        bpt.gmtest(bp.lommel_seeliger_law(name),
                   name + ' as a Lommel-Seeliger law',
                   limit=0.001, radius=1)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      # incidence and emission, bodies
      for name in bpt.body_names:

        # Get approximate projected surface scale in degrees per pixel
        km_per_fov_radian = bp.distance(name) / bp.mu(name)
        rad_per_fov_radian = km_per_fov_radian / Body.lookup(name).radius
        deg_per_fov_radian = rad_per_fov_radian * DPR

        # Select the 95th percentile as a large but representative value
        # This is needed because km_per_fov_radian diverges near limb
        values = deg_per_fov_radian[deg_per_fov_radian.antimask].vals
        if values.size == 0:
            continue
        values = np.sort(values, axis=None)
        cutoff = (values.size * 95) // 100
        deg_per_fov_radian = values[cutoff]

        (ulimit, vlimit) = deg_per_fov_radian * pixel_duv * 1.e-4

        # incidence_angle
        inc = bp.incidence_angle(name)
        dinc_duv = inc.d_dlos.chain(bp.dlos_duv)
        (dinc_du, dinc_dv) = dinc_duv.extract_denoms()

        utest = ((bp_u1.incidence_angle(name) - bp_u0.incidence_angle(name))
                 / bpt.duv)
        vtest = ((bp_v1.incidence_angle(name) - bp_v0.incidence_angle(name))
                 / bpt.duv)
        if not np.all(utest.mask):
            bpt.compare((utest - dinc_du).abs().median(), 0.,
                        name + ' incidence angle d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')
            bpt.compare((vtest - dinc_dv).abs().median(), 0.,
                        name + ' incidence angle d/dv self-check (deg/pix)',
                        limit=vlimit, method='degrees')

        # emission_angle
        em = bp.emission_angle(name)
        dem_duv = em.d_dlos.chain(bp.dlos_duv)
        (dem_du, dem_dv) = dem_duv.extract_denoms()

        utest = ((bp_u1.emission_angle(name) - bp_u0.emission_angle(name))
                 / bpt.duv)
        vtest = ((bp_v1.emission_angle(name) - bp_v0.emission_angle(name))
                 / bpt.duv)
        if not np.all(utest.mask):
            bpt.compare((utest - dem_du).abs().median(), 0.,
                        name + ' emission angle d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')
            bpt.compare((vtest - dem_dv).abs().median(), 0.,
                        name + ' emission angle d/dv self-check (deg/pix)',
                        limit=vlimit, method='degrees')

      # incidence and emission, rings
      for name in bpt.ring_names:

        # incidence_angle
        inc = bp.incidence_angle(name)
        dinc_duv = inc.d_dlos.chain(bp.dlos_duv)
        (dinc_du, dinc_dv) = dinc_duv.extract_denoms()

        deg_per_fov = DPR * (inc.max() - inc.min())
        (ulimit, vlimit) = deg_per_fov / np.array(bp.obs.uv_shape)
            # Because variations in ring incidence are so small, this numerical
            # derivative is not very accurate; limit is one pixel.

        utest = ((bp_u1.incidence_angle(name) - bp_u0.incidence_angle(name))
                 / bpt.duv)
        vtest = ((bp_v1.incidence_angle(name) - bp_v0.incidence_angle(name))
                 / bpt.duv)
        if not np.all(utest.mask):
            bpt.compare((utest - dinc_du).abs().median(), 0.,
                        name + ' incidence angle d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')
            bpt.compare((utest - dinc_du).abs().median(), 0.,
                        name + ' incidence angle d/dv self-check (deg/pix)',
                        limit=vlimit, method='degrees')

        # emission_angle
        em = bp.emission_angle(name)
        dem_duv = em.d_dlos.chain(bp.dlos_duv)
        (dem_du, dem_dv) = dem_duv.extract_denoms()

        deg_per_fov = DPR * (em.max() - em.min())
        (ulimit, vlimit) = deg_per_fov / np.array(bp.obs.uv_shape) * 0.01

        utest = ((bp_u1.emission_angle(name) - bp_u0.emission_angle(name))
                 / bpt.duv)
        vtest = ((bp_v1.emission_angle(name) - bp_v0.emission_angle(name))
                 / bpt.duv)
        if not np.all(utest.mask):
            bpt.compare((utest - dem_du).abs().median(), 0.,
                        name + ' emission angle d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')
            bpt.compare((vtest - dem_dv).abs().median(), 0.,
                        name + ' emission angle d/dv self-check (deg/pix)',
                        limit=ulimit, method='degrees')

      # phase angle
      for name in bpt.body_names + bpt.ring_names:

        # phase_angle
        ph = bp.phase_angle(name)
        dph_duv = ph.d_dlos.chain(bp.dlos_duv)
        (dph_du, dph_dv) = dph_duv.extract_denoms()

        deg_per_pixel = DPR * pixel_duv
        (ulimit, vlimit) = deg_per_pixel * 0.01

        utest = (bp_u1.phase_angle(name) - bp_u0.phase_angle(name)) / bpt.duv
        vtest = (bp_v1.phase_angle(name) - bp_v0.phase_angle(name)) / bpt.duv
        if not np.all(utest.mask):
            bpt.compare((utest - dph_du).abs().median(), 0.,
                        name + ' phase angle d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')
            bpt.compare((vtest - dph_dv).abs().median(), 0.,
                        name + ' phase angle d/dv self-check (deg/pix)',
                        limit=ulimit, method='degrees')

register_test_suite('lighting', lighting_test_suite)

################################################################################

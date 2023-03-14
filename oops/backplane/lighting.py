################################################################################
# oops/backplanes/lighting.py: Lighting geometry backplanes
################################################################################

from polymath       import Boolean, Scalar
from oops.backplane import Backplane

def incidence_angle(self, event_key, apparent=True):
    """Incidence angle of the arriving photons at the local surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('incidence_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, arrivals=True)
    incidence = event.incidence_angle(apparent=apparent, derivs=self.ALL_DERIVS)

    # Ring incidence angles should always be 0 to pi/2
    if event.surface.COORDINATE_TYPE == 'polar':

        # Save this as the "prograde" ring incidence angle
        ring_key = ('ring_incidence_angle', event_key, 'prograde', apparent)
        self.register_backplane(ring_key, incidence)

        # _ring_flip is True wherever incidence angle has to be replaced by
        # PI - incidence. This is needed by the emission_angle backplane.
        flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
        flip_key = ('_ring_flip', event_key)
        self.register_backplane(flip_key, flip)

        # Now flip incidence angles where necessary
        if flip.any():
            incidence = Scalar.PI * flip + (1 - 2*flip) * incidence

    return self.register_backplane(key, incidence)

#===============================================================================
def emission_angle(self, event_key, apparent=True):
    """Emission angle of the departing photons at the local surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('emission_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key)
    emission = event.emission_angle(apparent=apparent, derivs=self.ALL_DERIVS)

    # Ring emission angles are always measured from the lit side normal
    if event.surface.COORDINATE_TYPE == 'polar':

        # Save this as the "prograde" ring incidence angle
        ring_key = ('ring_emission_angle', event_key, 'prograde', apparent)
        self.register_backplane(ring_key, emission)

        # Get the "ring flip" flag
        _ = self.incidence_angle(event_key)
        flip_key = ('_ring_flip', event_key)
        flip = self.get_backplane(flip_key)

        # Now flip emission angles where necessary
        if flip.any():
            emission = Scalar.PI * flip + (1 - 2*flip) * emission

    return self.register_backplane(key, emission)

#===============================================================================
def phase_angle(self, event_key, apparent=True):
    """Phase angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('phase_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, arrivals=True)
    phase = event.phase_angle(apparent=apparent, derivs=self.ALL_DERIVS)
    return self.register_backplane(key, phase)

#===============================================================================
def scattering_angle(self, event_key, apparent=True):
    """Scattering angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('scattering_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    phase = self.phase_angle(event_key, apparent=apparent)
    return self.register_backplane(key, Scalar.PI - phase)

#===============================================================================
def center_incidence_angle(self, event_key, apparent=True):
    """Gridless incidence angle of the arriving photons at the body's central
    path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.incidence_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_emission_angle(self, event_key, apparent=True):
    """Gridless emission angle of the departing photons at the body's central
    path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.emission_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_phase_angle(self, event_key, apparent=True):
    """Gridless phase angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.phase_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_scattering_angle(self, event_key, apparent=True):
    """Gridless scattering angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.scattering_angle(gridless_key, apparent=apparent)

#===============================================================================
def mu0(self, event_key, apparent=True):
    """Cosine of the incidence angle of the arriving photons at the surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('mu0', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    incidence = self.incidence_angle(event_key, apparent=apparent)
    return self.register_backplane(key, incidence.cos())

#===============================================================================
def mu(self, event_key, apparent=True):
    """Cosine of the emission angle of the photons departing from the surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('mu', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    emission = self.emission_angle(event_key, apparent=apparent)
    return self.register_backplane(key, emission.cos())

#===============================================================================
def lambert_law(self, event_key):
    """Lambert law model cos(incidence_angle) for the surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lambert_law', event_key)
    if key in self.backplanes:
        return self.get_backplane(key)

    lambert_law = self.mu0(event_key, apparent=True)
    lambert_law = lambert_law.mask_where(lambert_law.vals <= 0., 0.)
    return self.register_backplane(key, lambert_law)

#===============================================================================
def minnaert_law(self, event_key, k, k2=None, clip=0.2):
    """Minnaert law model for the surface.

    Input:
        event_key       key defining the surface event.
        k               The Minnaert exponent (for cos(i)).
        k2              Optional second Minnaert exponent (for cos(e)).
                        Defaults to k-1.
        clip            lower limit on cos(e). Needed because otherwise the
                        Minnaert law diverges near the limb. Default 0.2.
    """

    event_key = self.standardize_event_key(event_key)

    if k2 is None:
        k2 = k - 1
    key = ('minnaert_law', event_key, k, k2, clip)

    if key in self.backplanes:
        return self.get_backplane(key)

    mu0 = self.mu0(event_key, apparent=True)
    mu  = self.mu( event_key, apparent=True)
    mu = mu.clip(clip, None)
    minnaert_law = (mu0 ** k) * (mu ** k2)
    return self.register_backplane(key, minnaert_law)

#===============================================================================
def lommel_seeliger_law(self, event_key):
    """Lommel-Seeliger law model for the surface.

    Returns mu0 / (mu + mu0)

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lommel_seeliger_law', event_key)
    if key in self.backplanes:
        return self.get_backplane(key)

    mu0 = self.mu0(event_key, apparent=True)
    mu  = self.mu( event_key, apparent=True)

    lommel_seeliger_law = mu0 / (mu + mu0)
    lommel_seeliger_law = lommel_seeliger_law.mask_where(mu0 <= 0., 0.)
    return self.register_backplane(key, lommel_seeliger_law)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

import numpy as np
from oops.body import Body
from oops.backplane.gold_master import register_test_suite
from oops.constants import DPR

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
        values = deg_per_fov_radian.vals[deg_per_fov_radian.antimask]
        if values.size == 0:
            continue
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

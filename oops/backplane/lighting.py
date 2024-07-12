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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    gridless_key = Backplane.gridless_event_key(event_key)
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

    gridless_key = Backplane.gridless_event_key(event_key)
    return self.emission_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_phase_angle(self, event_key, apparent=True):
    """Gridless phase angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = Backplane.gridless_event_key(event_key)
    return self.phase_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_scattering_angle(self, event_key, apparent=True):
    """Gridless scattering angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = Backplane.gridless_event_key(event_key)
    return self.scattering_angle(gridless_key, apparent=apparent)

#===============================================================================
def mu0(self, event_key, apparent=True):
    """Cosine of the incidence angle of the arriving photons at the surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)

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

    event_key = Backplane.standardize_event_key(event_key)
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

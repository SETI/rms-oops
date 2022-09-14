################################################################################
# oops/backplanes/where.py: Boolean backplanes
################################################################################

from __future__ import print_function

from oops.constants import HALFPI

from . import Backplane

############################################################################
# Boolean Masks
############################################################################

def where_intercepted(self, event_key):
    """A Boolean array where the surface was intercepted."""

    event_key  = self.standardize_event_key(event_key)
    key = ('where_intercepted', event_key)
    if key not in self.backplanes:
        event = self.get_surface_event(event_key)
        mask = self.mask_as_boolean(event.antimask)
        self.register_backplane(key, mask)

    return self.backplanes[key]

#===========================================================================
def where_inside_shadow(self, event_key, shadow_body):
    """A mask where the surface is in the shadow of a second body."""

    event_key = self.standardize_event_key(event_key)
    shadow_body = self.standardize_event_key(shadow_body)

    key = ('where_inside_shadow', event_key, shadow_body[0])
    if key not in self.backplanes:
        shadow_event = self.get_surface_event(shadow_body + event_key)
        mask = self.mask_as_boolean(shadow_event.antimask)
        self.register_backplane(key, mask)

    return self.backplanes[key]

#===========================================================================
def where_outside_shadow(self, event_key, shadow_body):
    """A mask where the surface is outside the shadow of a second body."""

    event_key  = self.standardize_event_key(event_key)
    shadow_body = self.standardize_event_key(shadow_body)

    key = ('where_outside_shadow', event_key, shadow_body[0])
    if key not in self.backplanes:
        shadow_event = self.get_surface_event(shadow_body + event_key)
        mask = shadow_event.mask & self.where_intercepted(event_key).values
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_in_front(self, event_key, back_body):
    """A mask where the first surface is in not obscured by the second
    surface.

    This is where the back_body is either further away than the front body
    or not intercepted at all.
    """

    event_key = self.standardize_event_key(event_key)
    back_body  = self.standardize_event_key(back_body)

    key = ('where_in_front', event_key, back_body[0])
    if key not in self.backplanes:
        front_distance = self.distance(event_key)
        back_distance  = self.distance(back_body)
        mask = front_distance.values < back_distance.values
        mask |= back_distance.mask
        mask &= front_distance.antimask
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_in_back(self, event_key, front_body):
    """A mask where the first surface is behind (obscured by) the second
    surface.
    """

    event_key = self.standardize_event_key(event_key)
    front_body = self.standardize_event_key(front_body)

    key = ('where_in_back', event_key, front_body[0])
    if key not in self.backplanes:
        mask = (self.distance(event_key) > self.distance(front_body))
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_sunward(self, event_key):
    """A mask where the surface of a body is facing toward the Sun."""

    event_key = self.standardize_event_key(event_key)
    key = ('where_sunward',) + event_key
    if key not in self.backplanes:
        mask = (self.incidence_angle(event_key) < HALFPI)
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_antisunward(self, event_key):
    """A mask where the surface of a body is facing away fron the Sun."""

    event_key = self.standardize_event_key(event_key)
    key = ('where_antisunward',) + event_key
    if key not in self.backplanes:
        mask = (self.incidence_angle(event_key) > HALFPI)
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

############################################################################
# Masks derived from backplanes
############################################################################

def where_below(self, backplane_key, value):
    """A mask where the backplane is <= the specified value."""

    backplane_key = Backplane.standardize_backplane_key(backplane_key)
    key = ('where_below', backplane_key, value)
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        mask = (backplane <= value)
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_above(self, backplane_key, value):
    """A mask where the backplane is >= the specified value."""

    backplane_key = Backplane.standardize_backplane_key(backplane_key)
    key = ('where_above', backplane_key, value)
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        mask = (backplane >= value)
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_between(self, backplane_key, low, high):
    """A mask where the backplane is between the given values, inclusive."""

    backplane_key = Backplane.standardize_backplane_key(backplane_key)
    key = ('where_between', backplane_key, low, high)
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        mask = (backplane >= low) & (backplane <= high)
        self.register_backplane(key, self.mask_as_boolean(mask))

    return self.backplanes[key]

#===========================================================================
def where_not(self, backplane_key):
    """A mask where the value of the given backplane is False, zero, or
    masked."""

    backplane_key = Backplane.standardize_backplane_key(backplane_key)
    key = ('where_not', backplane_key)
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        self.register_backplane(key, ~backplane)

    return self.backplanes[key]

#===========================================================================
def where_any(self, *backplane_keys):
    """A mask where any of the given backplanes is True."""

    key = ('where_any',) + backplane_keys
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_keys[0]).copy()
        for next_mask in backplane_keys[1:]:
            backplane |= self.evaluate(next_mask)

        self.register_backplane(key, backplane)

    return self.backplanes[key]

#===========================================================================
def where_all(self, *backplane_keys):
    """A mask where all of the given backplanes are True."""

    key = ('where_all',) + backplane_keys
    if key not in self.backplanes:
        backplane = self.evaluate(backplane_keys[0]).copy()
        for next_mask in backplane_keys[1:]:
            backplane &= self.evaluate(next_mask)

        self.register_backplane(key, backplane)

    return self.backplanes[key]

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################

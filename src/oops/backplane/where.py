##########################################################################################
# oops/backplane/where.py
##########################################################################################

from polymath       import Boolean, Scalar
from oops.backplane import Backplane

##########################################################################################
# Boolean Masks
##########################################################################################

def where_intercepted(self, event_key):
    """A Boolean array that is True where the surface was intercepted.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
    """

    self.refresh()
    event_key  = Backplane.standardize_event_key(event_key)
    key = ('where_intercepted', event_key)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key)
    intercepted = Boolean(event.dep.expand_mask().antimask)
    return self.register_backplane(key, intercepted)


def where_inside_shadow(self, event_key, surface_key, tvl=False):
    """A mask where the surface is in the shadow of a second body.

    If tvl is True, this uses three-valued logic, where locations outside the surface are
    masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the shadowing body, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the surface remain masked; False to return False there.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    return self._where_inside_or_outside_shadow(event_key, surface_key, tvl=tvl,
                                                inside=True)


def where_outside_shadow(self, event_key, surface_key, tvl=False):
    """A mask where the surface is outside the shadow of a second body.

    If tvl is True, this uses three-valued logic, where locations outside the
    surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the shadowing body, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the surface remain masked; False to return False there.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    return self._where_inside_or_outside_shadow(event_key, surface_key, tvl=tvl,
                                                inside=False)


def _where_inside_or_outside_shadow(self, event_key, surface_key, tvl, inside):
    """Internal method implementing where_inside_shadow and where_outside_shadow.

    Both the three-valued and the two-valued results are cached on the first call.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the shadowing body, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool): True to use three-valued logic, in which locations outside the
            surface remain masked; False to return False there.
        inside (bool): True for the region inside the shadow; False for the region
            outside it.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)
    if len(event_key) != 2:
        raise ValueError('invalid event key for shadowing: ' + repr(event_key))

    surface_key = surface_key.upper()
    if inside:
        key = ('where_inside_shadow', event_key, surface_key, tvl)
    else:
        key = ('where_outside_shadow', event_key, surface_key, tvl)

    if key not in self.backplanes:

        # First body is un-shadowed if its incoming photons do not intercept the shadow
        # body. The shadow event will inherit the first event's mask.
        shadow_event_key = event_key[:1] + (surface_key,) + event_key[1:]
        shadow_event = self.get_surface_event(shadow_event_key)
        event = self.get_surface_event(event_key)

        if inside:
            result_vals = shadow_event.antimask
        else:
            result_vals = shadow_event.mask

        # Exclude where the event is inside the shadower's surface
        surface = Backplane.get_surface(surface_key)
        if surface.HAS_INTERIOR:
            where_inside = self.where_inside(event_key, surface_key)
            result_vals = result_vals | where_inside.vals

        # Set the internal values to False at every masked location
        tvl_result = Boolean(result_vals & event.antimask, event.mask)

        # Save both TVL and non-TVL results at the same time
        self.register_backplane(key[:-1] + (True,),  tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_in_front(self, event_key, surface_key, tvl=False):
    """A mask where the first surface is not obscured by the second surface.

    This is where the second surface is either further away than the first or not
    intercepted at all.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the first surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        surface_key (str): A registered body ID for the second surface, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the first surface remain masked; False to return False there.
    """

    return self._where_in_front_or_in_back(event_key, surface_key, tvl=tvl, in_front=True)


def where_in_back(self, event_key, surface_key, tvl=False):
    """A mask where the first surface is behind (obscured by) the second surface.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the first surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        surface_key (str): A registered body ID for the second surface, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the first surface remain masked; False to return False there.
    """

    return self._where_in_front_or_in_back(event_key, surface_key, tvl=tvl,
                                           in_front=False)


def _where_in_front_or_in_back(self, event_key, surface_key, tvl, in_front):
    """Internal method implementing where_in_front and where_in_back.

    Both the three-valued and the two-valued results are cached on the first call.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        surface_key (str): A registered body ID for the second surface, optionally
            modified with ":ANSA", ":RING" or ":LIMB" to select an associated surface.
        tvl (bool): True to use three-valued logic, in which locations outside the first
            surface remain masked; False to return False there.
        in_front (bool): True where the first surface is the closer of the two; False
            where it is the farther.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)

    surface_key = surface_key.upper()
    if in_front:
        key = ('where_in_front', event_key, surface_key, tvl)
    else:
        key = ('where_in_back', event_key, surface_key, tvl)

    if key not in self.backplanes:

        # First body is in front if it is closer than the second. Both bodies
        # must be intercepted.
        surface_event_key = event_key[:1] + (surface_key,) + event_key[2:]

        distance1 = self.distance(event_key)
        distance2 = self.distance(surface_event_key)

        if in_front:
            rejected = distance1.tvl_gt(distance2).vals | distance1.mask
            tvl_result = Boolean(rejected, distance1.mask).logical_not()
        else:
            tvl_result = distance1.tvl_gt(distance2)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        # We save both TVL and non-TVL results at the same time
        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_sunward(self, event_key, tvl=False):
    """A mask where the surface of a body is facing toward the Sun.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the surface remain masked; False to return False there.
    """

    return self._where_sunward_or_antisunward(event_key, tvl=tvl, sunward=True)


def where_antisunward(self, event_key, tvl=False):
    """A mask where the surface of a body is facing away from the Sun.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the surface remain masked; False to return False there.
    """

    return self._where_sunward_or_antisunward(event_key, tvl=tvl, sunward=False)


def _where_sunward_or_antisunward(self, event_key, tvl, sunward):
    """Internal method implementing where_sunward and where_antisunward.

    The incidence angle is measured against the observed pole for a ring and against the
    surface normal otherwise. Both the three-valued and the two-valued results are cached
    on the first call.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        tvl (bool): True to use three-valued logic, in which locations outside the
            surface remain masked; False to return False there.
        sunward (bool): True where the incidence angle is at most 90 degrees; False where
            it exceeds 90 degrees.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)

    if sunward:
        key = ('where_sunward', event_key, tvl)
    else:
        key = ('where_antisunward', event_key, tvl)

    if key not in self.backplanes:

        # This is slightly different for rings vs. planets.
        surface = Backplane.get_surface(event_key[-1])
        if surface.COORDINATE_TYPE == 'polar':
            incidence = self.ring_incidence_angle(event_key, pole='observed')
        else:
            incidence = self.incidence_angle(event_key)

        if sunward:
            tvl_result = incidence.tvl_le(Scalar.HALFPI)
        else:
            tvl_result = incidence.tvl_gt(Scalar.HALFPI)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        # We save both TVL and non-TVL results at the same time
        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_inside(self, event_key, surface_key, tvl=False):
    """A mask where the first surface is interior to the second surface.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the first surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the surface whose interior is
            tested, optionally modified with ":ANSA", ":RING" or ":LIMB" to select an
            associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the first surface remain masked; False to return False there.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    return self._where_inside_or_outside(event_key, surface_key, tvl=tvl,
                                         inside=True)


def where_outside(self, event_key, surface_key, tvl=False):
    """A mask where the first surface is exterior to the second surface.

    If tvl is True, this mask uses three-valued logic, where locations outside
    the surface are masked; otherwise, they are False.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the surface whose interior is
            tested, optionally modified with ":ANSA", ":RING" or ":LIMB" to select an
            associated surface.
        tvl (bool, optional): True to use three-valued logic, in which locations
            outside the first surface remain masked; False to return False there.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    return self._where_inside_or_outside(event_key, surface_key, tvl=tvl,
                                         inside=False)


def _where_inside_or_outside(self, event_key, surface_key, tvl, inside):
    """Internal method implementing where_inside and where_outside.

    A surface with no interior is treated as containing nothing. Both the three-valued
    and the two-valued results are cached on the first call.

    Parameters:
        event_key (str or tuple): Key defining the surface event. Once standardized, it
            must contain exactly two items, the light source and one body.
        surface_key (str): A registered body ID for the surface whose interior is
            tested, optionally modified with ":ANSA", ":RING" or ":LIMB" to select an
            associated surface.
        tvl (bool): True to use three-valued logic, in which locations outside the first
            surface remain masked; False to return False there.
        inside (bool): True for the region inside the second surface; False for the
            region outside it.

    Raises:
        ValueError: If the standardized event key does not contain exactly two items.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)
    if len(event_key) != 2:
        raise ValueError('invalid event key for inside/outside calculations: '
                         + repr(event_key))

    surface_key = surface_key.upper()
    if inside:
        key = ('where_inside', event_key, surface_key, tvl)
    else:
        key = ('where_outside', event_key, surface_key, tvl)

    if key not in self.backplanes:

        # Check positions with respect to the surface interior
        surface = Backplane.get_surface(surface_key)
        event = self.get_surface_event(event_key)
        if surface.HAS_INTERIOR:
            surface_pos = event.wrt(surface.origin, surface.frame).pos
            is_inside = surface.position_is_inside(surface_pos, obs=self.obs,
                                                                time=self.time)
            result = (is_inside == inside)
        else:
            result = Boolean(not inside).broadcast_to(event.shape)

        # Apply the event mask; set internal values to False at masked locations
        tvl_result = Boolean(result.vals & event.antimask, event.mask)

        # Save both TVL and non-TVL results at the same time
        self.register_backplane(key[:-1] + (True,),  tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)

##########################################################################################
# Masks derived from backplanes
##########################################################################################

def where_below(self, backplane_key, value, tvl=False):
    """A mask where the backplane is <= the specified value.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        backplane_key (str or tuple): Key defining the backplane to evaluate.
        value (float or Scalar): The upper limit on the backplane value.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    backplane_key = self.standardize_backplane_key(backplane_key)

    value = value.vals if isinstance(value, Scalar) else value
    key = ('where_below', backplane_key, value, tvl)

    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        tvl_result = backplane.tvl_le(value)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & backplane.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_above(self, backplane_key, value, tvl=False):
    """A mask where the backplane is >= the specified value.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        backplane_key (str or tuple): Key defining the backplane to evaluate.
        value (float or Scalar): The lower limit on the backplane value.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    backplane_key = self.standardize_backplane_key(backplane_key)

    value = value.vals if isinstance(value, Scalar) else value
    key = ('where_above', backplane_key, value, tvl)

    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        tvl_result = backplane.tvl_ge(value)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & backplane.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_between(self, backplane_key, low, high, tvl=False):
    """A mask where the backplane is between the given values, inclusive.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        backplane_key (str or tuple): Key defining the backplane to evaluate.
        low (float or Scalar): The lower limit on the backplane value.
        high (float or Scalar): The upper limit on the backplane value.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    backplane_key = self.standardize_backplane_key(backplane_key)

    low  = low.vals  if isinstance(low,  Scalar) else low
    high = high.vals if isinstance(high, Scalar) else high
    key = ('where_between', backplane_key, low, high, tvl)

    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        tvl_result = backplane.tvl_ge(low) & backplane.tvl_le(high)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_not(self, backplane_key, tvl=False):
    """A mask where the value of the given backplane is False or zero.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        backplane_key (str or tuple): Key defining the backplane to evaluate.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    backplane_key = self.standardize_backplane_key(backplane_key)
    key = ('where_not', backplane_key, tvl)

    if key not in self.backplanes:
        backplane = self.evaluate(backplane_key)
        tvl_result = backplane.logical_not()

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_any(self, *backplane_keys, tvl=False):
    """A mask where any of the given backplanes is True or nonzero.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        *backplane_keys (str or tuple): Keys defining the backplanes to combine.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    key = ('where_any',) + backplane_keys + (tvl,)
    if key not in self.backplanes:
        tvl_result = self.evaluate(backplane_keys[0]).copy()
        for next_mask in backplane_keys[1:]:
            tvl_result |= self.evaluate(next_mask)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)


def where_all(self, *backplane_keys, tvl=False):
    """A mask where all of the given backplanes are True or nonzero.

    If tvl is True, this uses three-valued logic, where masked backplane values remain
    masked; otherwise, they are False.

    Parameters:
        *backplane_keys (str or tuple): Keys defining the backplanes to combine.
        tvl (bool, optional): True to use three-valued logic, in which masked
            backplane values remain masked; False to return False there.
    """

    self.refresh()
    key = ('where_all',) + backplane_keys + (tvl,)
    if key not in self.backplanes:
        tvl_result = self.evaluate(backplane_keys[0]).copy()
        for next_mask in backplane_keys[1:]:
            tvl_result &= self.evaluate(next_mask)

        # Set the internal values to False at every masked location
        tvl_result = Boolean(tvl_result.vals & tvl_result.antimask,
                             tvl_result.mask)

        self.register_backplane(key[:-1] + (True,), tvl_result)
        self.register_backplane(key[:-1] + (False,), Boolean(tvl_result.vals))

    return self.get_backplane(key)

##########################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

##########################################################################################

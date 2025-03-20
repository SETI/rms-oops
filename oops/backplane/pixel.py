################################################################################
# oops/backplanes/pixel.py: pixel coordinate backplanes
################################################################################
import numpy as np

from oops.constants import C
from oops.backplane import Backplane
from oops.body import Body

#===============================================================================
def body_diameter_in_pixels(self, event_key, radius=0, axis="max"):
    """Gridless approximate apparent diameter of the body in pixels.

    Input:
        event_key       key defining the event on the body's path.
        radius          If nonzero, override the radius of the body referred to
                        in the event key.
        axis            "u"   : horizontal pixel direction.
                        "v"   : vertical pixel direction.
                        "min" : direction of largest diameter.
                        "max" : direction of smallest diameter.
    """
    if not self.obs.INVENTORY_IMPLEMENTED:
        raise NotImplementedError('body_diameter_in_pixels not defined for '
                                    + type(self.obs).__name__)

    if axis not in {'u', 'v', 'min', 'max'}:
        raise ValueError('invalid axis: ' + repr(axis))

    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('body_diameter_in_pixels', gridless_key, radius, axis)
    if key in self.backplanes:
        return self.get_backplane(key)

    # compute apparent distance
    event = self.get_surface_event(gridless_key, arrivals=True)
    distance = event._dep_lt_*C

    # compute apparent enclosing radius
    (body, mod) = Backplane.get_body_and_modifier(gridless_key[1])
    if radius==0:
        radius = body.radius

    radii_in_pixels = radius/distance / self.obs.fov.uv_scale.vals

    if axis == 'u':
        radius_in_pixels = radii_in_pixels[0]
    elif axis == 'v':
        radius_in_pixels = radii_in_pixels[1]
    elif axis == 'max':
        radius_in_pixels = radii_in_pixels.max()
    else:
        radius_in_pixels = radii_in_pixels.min()

    return self.register_backplane(key, 2*radius_in_pixels)

#===============================================================================
def center_coordinate(self, event_key, axis="u"):
    """Gridless coordinate of the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        axis            "u" (horizontal pixel direction) or "v" (vertical
                        pixel direction).
    """
    if axis not in {'u', 'v'}:
        raise ValueError('invalid axis: ' + repr(axis))

    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('center_coordinate', gridless_key, axis)
    if key in self.backplanes:
        return self.get_backplane(key)

    body = Body.lookup(gridless_key[1])
    uv = self.obs.uv_from_path(body.path)

    index = 0 if axis == "u" else 1
    return self.register_backplane(key, uv.to_scalars()[index])

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################

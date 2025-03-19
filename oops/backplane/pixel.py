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
        axis            uv axis to evaluate on, or "min" or "max".
    """
    if not self.obs.INVENTORY_IMPLEMENTED:
        raise NotImplementedError('body_diameter_in_pixels not defined for '
                                    + type(self.obs).__name__)

    gridless_key = Backplane.gridless_event_key(event_key)

    key = ('body_diameter_in_pixels', gridless_key, radius, axis)
    if key in self.backplanes:
        return self.get_backplane(key)

    index = None
    # interpret axis input
    try:
        index = {'u':0, 'v':1}[axis]
    except KeyError:
        if axis.lower() not in {"min", "max"}:
            raise ValueError('invalid axis: ' + repr(axis))

    # compute apparent distance
    event = self.get_surface_event(gridless_key, arrivals=True)
    distance = event._dep_lt_*C

    # compute apparent enclosing radius
    (body, mod) = Backplane.get_body_and_modifier(gridless_key[1])
    if radius==0:
        radius = body.radius

    theta = radius/distance
    radii_in_pixels = np.divide(theta, self.obs.fov.uv_scale.values)
    if index is not None:
        radius_in_pixels = radii_in_pixels[index].values
    else:
        radius_in_pixels = getattr(radii_in_pixels, axis)()

    return 2*self.register_backplane(key, radius_in_pixels)

#===============================================================================
def center_coordinate(self, event_key, axis="u"):
    """Gridless coordinate of the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        axis            "u" (horizonatal pixel direction) or "v" (vertical
                        pixel direction).
    """
    if axis not in {'u', 'v'}:
        raise ValueError('Invalid axis: ' + repr(axis))

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

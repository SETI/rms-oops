##########################################################################################
# oops/surface_/spice_shape.py: For bodies with shapes defined in SPICE.
##########################################################################################

import cspyce

from oops.frame.spiceframe  import SpiceFrame
from oops.path.spicepath    import SpicePath
from oops.surface.ellipsoid import Ellipsoid
from oops.surface.spheroid  import Spheroid


def spice_shape(spice_id, frame=None, default_radii=None):
    """Construct a Spheroid or Ellipsoid defining the shape and orientation of a body
    defined in the SPICE toolkit.

    Parameters:
        spice_id (str or int): The SPICE body name or integer code.
        frame (Frame, optional): The rotation Frame of the body. By default, this is
            inferred from the `spice_id`.
        default_radii (tuple, optional): Three radius values to use if the PCK radius
            values are not found.

    Returns:
        (Spheroid or Ellipsoid): The surface of the Body.

    Raises:
        KeyError: If the radius values are missing from the SPICE kernel pool and default
            values are not provided.
        LookupError: If the `spice_id` is not recognized.
    """

    spice_body_code = SpicePath._body_code_and_name(spice_id)[0]
    path = SpicePath.get(spice_body_code)
    frame = frame or SpiceFrame.get(spice_body_code)

    try:
        radii = cspyce.bodvcd(spice_body_code, 'RADII')
    except (RuntimeError, KeyError) as e:
        if default_radii is None:
            raise e
        radii = default_radii

    if radii[0] == radii[1]:
        return Spheroid(path, frame, (radii[0], radii[2]))
    else:
        return Ellipsoid(path, frame, radii)

##########################################################################################

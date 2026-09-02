##########################################################################################
# oops/surface/spice_shape.py: For bodies with shapes defined in SPICE.
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
        default_radii (tuple[float, float, float], optional): Three radius values to
            use if the PCK radius values are not found.

    Returns:
        Spheroid or Ellipsoid: The surface of the body.

    Raises:
        IndexError: If `spice_id` is an integer but is not a recognized body code.
        KeyError: If `spice_id` is a string but is not a recognized body name.
        KeyError: If the body's radius values are missing from the SPICE kernel pool and
            `default_radii` is not provided.
        TypeError: If `spice_id` is not an integer or string.
    """

    spice_body_code, spice_body_name = SpicePath._body_code_and_name(spice_id)
    path = SpicePath.get(spice_body_code)
    frame = frame or SpiceFrame.get(spice_body_code)

    try:
        radii = cspyce.bodvcd(spice_body_code, 'RADII')
    except (RuntimeError, KeyError):
        if default_radii is None:
            raise KeyError('radii are not available for SPICE body '
                           f'{spice_body_name}') from None  # suppress SPICE traceback
        radii = default_radii

    if radii[0] == radii[1]:
        return Spheroid(path, frame, (radii[0], radii[2]))
    else:
        return Ellipsoid(path, frame, radii)

##########################################################################################

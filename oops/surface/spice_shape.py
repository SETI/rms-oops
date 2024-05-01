################################################################################
# oops/surface_/spice_shape.py: For bodies with shapes defined in SPICE.
################################################################################

import cspyce

from oops.surface.ellipsoid import Ellipsoid
from oops.surface.spheroid  import Spheroid
import oops.spice_support as spice

def spice_shape(spice_id, frame_id=None, default_radii=None):
    """Construct a Spheroid or Ellipsoid defining the path, orientation and
    shape of a body defined in the SPICE toolkit.

    Input:
        spice_id        the name or ID of the body as defined in the SPICE
                        toolkit.
        frame_id        the ID of the body's frame if already defined; otherwise
                        None. This typically allows a Synchronous frame to be
                        used if the SPICE frame is missing.
        default_radii   three radii values to use if PCK values are not found;
                        None to raise a LookupError on missing radii.
    """

    spice_body_id = spice.body_id_and_name(spice_id)[0]
    origin_id = spice.PATH_TRANSLATION[spice_body_id]

    if frame_id is None:
        try:
            spice_frame_name = spice.frame_id_and_name(spice_id)[1]
        except LookupError:     # moons with unknown spin inherit from the planet
            planet_id = 100 * int(str(spice_id)[0]) + 99
            spice_frame_name = spice.frame_id_and_name(planet_id)[1]

        frame_id = spice.FRAME_TRANSLATION[spice_frame_name]

    try:
        radii = cspyce.bodvcd(spice_body_id, "RADII")
    except (RuntimeError, KeyError) as e:
        if default_radii is None:
            raise e
        radii = default_radii

    if radii[0] == radii[1]:
        return Spheroid(origin_id, frame_id, (radii[0], radii[2]))
    else:
        return Ellipsoid(origin_id, frame_id, radii)

################################################################################

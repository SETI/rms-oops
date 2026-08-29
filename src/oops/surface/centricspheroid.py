##########################################################################################
# oops/surface/centricspheroid.py: CentricSpheroid subclass of class Surface.
##########################################################################################

from polymath                      import Scalar
from oops.surface.centricellipsoid import CentricEllipsoid
from oops.surface.ellipsoid        import Ellipsoid
from oops.surface.spheroid         import Spheroid

class CentricSpheroid(Spheroid):
    """A variant of Spheroid in which latitudes are planetocentric."""

    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the surface, relative to this surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (lon, lat) or all three (lon, lat, z) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Scalar, optional): Optionally, the value of the coefficient p such that
                ground + p * normal(ground) = pos; ignored if the value is None (the
                default) or True. groundtrack True to return the intercept on the surface
                along with the coordinates.

        Returns:
            (tuple): Two to four items:

            * `lon` (Scalar): Longitude at the surface in radians.
            * `lat` (Scalar): Latitude at the surface in radians.
            * `z` (Scalar): Vertical altitude in km normal to the surface; included if
              axes == 3.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if input groundtrack is True.
        """

        return CentricEllipsoid.coords_from_vector3(self, pos, axes=axes,
                                                    derivs=derivs, hints=hints,
                                                    groundtrack=groundtrack)

    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                          groundtrack=False):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple): Two or three Scalars defining coordinates at or near this
                surface. These can have different shapes, but must be broadcastable to a
                common shape. lon     longitude at the surface in radians. lat
                latitude at the surface in radians. z       vertical altitude in km normal
                to the body surface.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives inside the
                coordinates and obs into the returned position vectors. groundtrack True
                to include the associated groundtrack points on the body surface in the
                returned result.

        Returns:
            Pos or (pos, track), where, where:

            * `pos` (Vector3): Points defined by the coordinates, relative to this
              surface's origin and frame.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if input groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lat = Spheroid.lat_from_centric(self, lat, derivs=derivs)
        new_coords = (lon, squashed_lat,) + coords[2:]

        return Ellipsoid.vector3_from_coords(self, new_coords, derivs=derivs,
                                                   groundtrack=groundtrack)

    ######################################################################################
    # Latitude conversions
    ######################################################################################

    def lat_to_centric(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Parameters:
            Return (Scalar): Planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    def lat_from_centric(self, lat, lon=None, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Parameters:
            Return (Scalar): Planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    def lat_to_graphic(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Parameters:
            Return (Scalar): Planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.unsquash_z_sq).arctan()

    def lat_from_graphic(self, lat, lon=None, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Parameters:
            Return (Scalar): Planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.squash_z_sq).arctan()

##########################################################################################

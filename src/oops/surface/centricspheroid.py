##########################################################################################
# oops/surface/centricspheroid.py: CentricSpheroid subclass of class Surface
##########################################################################################

from polymath                      import Scalar
from oops.surface.centricellipsoid import CentricEllipsoid
from oops.surface.ellipsoid        import Ellipsoid
from oops.surface.spheroid         import Spheroid


class CentricSpheroid(Spheroid):
    """A variant of Spheroid in which latitudes are planetocentric."""

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2,
                            derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (lon, lat) or all three (lon, lat, z) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Scalar, optional): Optionally, the value of the coefficient p such that
                ground + p * normal(ground) = pos; ignored if the value is None (the
                default) or True. If it is not None, the converged value of `p` is
                appended to the returned tuple; use `hints=True` if you lack an initial
                value but require the new value to be returned.
            groundtrack (bool, optional): True to append the intercept point on the
                surface to the returned tuple.

        Returns:
            tuple: Two to five items:

            * `lon` (Scalar): Longitude at the surface in radians.
            * `lat` (Scalar): Latitude at the surface in radians.
            * `z` (Scalar): Vertical altitude in km normal to the surface; included if
              axes == 3.
            * `p` (Scalar): The converged coefficient; included if the input value of
              `hints` is not None.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if `groundtrack` is True.
        """

        return CentricEllipsoid.coords_from_vector3(self, pos, obs=obs, time=time,
                                                    axes=axes, derivs=derivs, hints=hints,
                                                    groundtrack=groundtrack)

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None, groundtrack=False):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at
                or near this surface. These can have different shapes, but must be
                broadcastable to a common shape.

                * `lon` (rad): Longitude at the surface.
                * `lat` (rad): Latitude at the surface.
                * `z` (km, optional): Vertical altitude normal to the body surface.

            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives inside the
                coordinates and obs into the returned position vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.
            groundtrack (bool, optional): True to append the associated groundtrack points
                on the body surface to the returned result.

        Returns:
            Vector3 or tuple: `pos` or `(pos[, hints][, track])`, where:

            * `pos` (Vector3): Points defined by the coordinates, relative to this
              surface's origin and frame.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if `groundtrack` is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lat = Spheroid.lat_from_centric(self, lat, derivs=derivs)
        new_coords = (lon, squashed_lat,) + coords[2:]

        return Ellipsoid.vector3_from_coords(self, new_coords, obs=obs, time=time,
                                                   derivs=derivs, hints=hints,
                                                   groundtrack=groundtrack)

    ######################################################################################
    # Latitude conversions
    ######################################################################################

    def lat_to_centric(self, lat, lon=None, *, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    def lat_from_centric(self, lat, lon=None, *, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    def lat_to_graphic(self, lat, lon=None, *, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.unsquash_z_sq).arctan()

    def lat_from_graphic(self, lat, lon=None, *, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.squash_z_sq).arctan()

##########################################################################################

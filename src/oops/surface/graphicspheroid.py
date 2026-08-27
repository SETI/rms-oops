################################################################################
# oops/surface/graphicspheroid.py: GraphicSpheroid subclass of class Surface
################################################################################

from polymath import Scalar

from oops.surface.spheroid         import Spheroid
from oops.surface.graphicellipsoid import GraphicEllipsoid

class GraphicSpheroid(Spheroid):
    """A variant of Spheroid in which latitudes are planetographic."""

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (lon, lat) or all three (lon, lat, z) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       optionally, the value of the coefficient p such that
                            ground + p * normal(ground) = pos;
                        for the ground point on the body surface. Ignored if the
                        value is None (the default) or True.
            groundtrack True to return the intercept on the surface along with
                        the coordinates.

        Return:         a tuple of two to four items:
            lon         longitude at the surface in radians.
            lat         latitude at the surface in radians.
            z           vertical altitude in km normal to the surface; included
                        if axes == 3.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        return GraphicEllipsoid.coords_from_vector3(self, pos, axes=axes,
                                                    derivs=derivs, hints=hints,
                                                    groundtrack=groundtrack)

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                          groundtrack=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
                lon     longitude at the surface in radians.
                lat     latitude at the surface in radians.
                z       vertical altitude in km normal to the body surface.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame; ignored for this Surface
                        subclass.
            time        a Scalar time at which to evaluate the surface; ignored
                        for this Surface subclass.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.
            groundtrack True to include the associated groundtrack points on the
                        body surface in the returned result.

        Return:         pos or (pos, groundtrack), where
            pos         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lat = Spheroid.lat_from_graphic(self, lat, derivs=derivs)
        new_coords = (lon, squashed_lat) + coords[2:]

        return Spheroid.vector3_from_coords(self, new_coords, derivs=derivs,
                                                  groundtrack=groundtrack)

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Input:
            lat         planetographic latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.squash_z_sq).arctan()

    #===========================================================================
    def lat_from_centric(self, lat, lon=None, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Input:
            lat         planetocentric latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self.unsquash_z_sq).arctan()

    #===========================================================================
    def lat_to_graphic(self, lat, lon=None, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Input:
            lat         planetographic latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    #===========================================================================
    def lat_from_graphic(self, lat, lon=None, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Input:
            lat         planetographic latitide, radians.
            lon         ignored, included for compatibility with Ellipsoids.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

################################################################################

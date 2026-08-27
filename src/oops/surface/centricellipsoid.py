################################################################################
# oops/surface/centricellipsoid.py: CentricEllipsoid subclass of class Surface
################################################################################

import numpy as np

from polymath               import Scalar, Vector3
from oops.surface.ellipsoid import Ellipsoid

class CentricEllipsoid(Ellipsoid):
    """A variant of Ellipsoid in which latitudes and longitudes are
    planetocentric.
    """

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
                        ignored if the value is None (the default) or True.
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

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Use the quick solution for the body points if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            (track, p) = self.intercept_normal_to(pos, guess=True)
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self.unsquash_sq
            track = pos.element_div(denom)

        # Derive the coordinates
        (x,y,z) = track.to_scalars()
        lat = (z/track.norm()).arcsin()
        lon = y.arctan2(x) % Scalar.TWOPI

        results = (lon, lat)

        if axes == 3:
            r = (pos - track).norm() * p.sign()
            results += (r,)

        if groundtrack:
            results += (track,)

        return results

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

        Return:         pos or (pos, track), where
            pos         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
            track       intercept point on the surface (where z == 0); included
                        if input groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_centric(self, lat, squashed_lon,
                                                        derivs=derivs)
        new_coords = (squashed_lon, squashed_lat,) + coords[2:]

        return Ellipsoid.vector3_from_coords(self, new_coords, derivs=derivs,
                                                   groundtrack=groundtrack)

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    #===========================================================================
    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.unsquash_y_sq).arctan2(lon.cos())

    #===========================================================================
    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.squash_y_sq).arctan2(lon.cos())

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Input:
            lat         squashed latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    #===========================================================================
    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Input:
            lat         planetocentric latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    #===========================================================================
    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Input:
            lat         squashed latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic latitude.
        """

        # This could be done more efficiently I'm sure
        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_centric(self, lat, squashed_lon,
                                                        derivs=derivs)

        return Ellipsoid.lat_to_graphic(self, squashed_lat, squashed_lon,
                                              derivs=derivs)

    #===========================================================================
    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Input:
            lat         planetographic latitide, radians.
            lon         planetocentric longitude, radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric latitude.
        """

        squashed_lon = Ellipsoid.lon_from_centric(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_graphic(self, lat, squashed_lon,
                                                        derivs=derivs)

        return Ellipsoid.lat_to_centric(self, squashed_lat, squashed_lon,
                                              derivs=derivs)

################################################################################

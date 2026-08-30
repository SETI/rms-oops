##########################################################################################
# oops/surface/graphicellipsoid.py: GraphicEllipsoid subclass of class Surface
##########################################################################################

import numpy as np

from polymath               import Scalar, Vector3
from oops.surface.ellipsoid import Ellipsoid


class GraphicEllipsoid(Ellipsoid):
    """A variant of Ellipsoid in which latitudes and longitudes are planetographic,
    meaning that their direction is defined by the local surface normal.

    Note that planetographic longitude differs from conventional
    (planetocentric) longitude for triaxial ellipsoids, and is an unconventional
    choice. Use method lon_to_centric() if you wish to convert it to centric
    longitude.
    """

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None, groundtrack=False):
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
              `axes` == 3.
            * `p` (Scalar): The converged coefficient; included if the input value of
              `hints` is not None.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if `groundtrack` is True.
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
        normal = track.element_mul(self.unsquash_sq)
        (x,y,z) = normal.to_scalars()
        lat = (z/normal.norm()).arcsin()
        lon = y.arctan2(x) % Scalar.TWOPI

        results = (lon, lat)

        if axes == 3:
            r = (pos - track).norm() * p.sign()
            results += (r,)

        if hints is not None:
            results += (p,)

        if groundtrack:
            results += (track,)

        return results

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
            * `hints` (Any): The input value of `hints`, included if it is not None.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if `groundtrack` is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (lon, lat) = coords[:2]
        squashed_lon = Ellipsoid.lon_from_graphic(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_graphic(self, lat, squashed_lon, derivs=derivs)
        new_coords = (squashed_lon, squashed_lat,) + coords[2:]

        return Ellipsoid.vector3_from_coords(self, new_coords, obs=obs, time=time,
                                                   derivs=derivs, hints=hints,
                                                   groundtrack=groundtrack)

    ######################################################################################
    # Longitude conversions
    ######################################################################################

    def lon_to_centric(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetocentric longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.squash_y_sq).arctan2(lon.cos())

    def lon_from_centric(self, lon, *, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, recursive=derivs)
        return (lon.sin() * self.unsquash_y_sq).arctan2(lon.cos())

    def lon_to_graphic(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    def lon_from_graphic(self, lon, *, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    ######################################################################################
    # Latitude conversions
    ######################################################################################

    def lat_to_centric(self, lat, lon, *, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        # This could be done more efficiently I'm sure
        squashed_lon = Ellipsoid.lon_from_graphic(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_graphic(self, lat, squashed_lon, derivs=derivs)
        return Ellipsoid.lat_to_centric(self, squashed_lat, squashed_lon, derivs=derivs)

    def lat_from_centric(self, lat, lon, *, derivs=False):
        """Convert planetocentric latitude to internal coordinates.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        squashed_lon = Ellipsoid.lon_from_graphic(self, lon, derivs=derivs)
        squashed_lat = Ellipsoid.lat_from_centric(self, lat, squashed_lon, derivs=derivs)
        return Ellipsoid.lat_to_graphic(self, squashed_lat, squashed_lon, derivs=derivs)

    def lat_to_graphic(self, lat, lon, *, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

    def lat_from_graphic(self, lat, lon, *, derivs=False):
        """Convert a planetographic latitude to internal coordinates.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar): The longitude in radians, which this conversion requires
                because the surface is triaxial.
            derivs (bool, optional): True to propagate any derivatives of `lat` and
                `lon` into the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        return Scalar.as_scalar(lat, recursive=derivs)

##########################################################################################

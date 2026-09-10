##########################################################################################
# oops/surface/spheroid.py
##########################################################################################

import numpy as np

from polymath import Scalar, Vector3
from oops.config import SURFACE_PHOTONS
from oops._convergence import limited_step, solve
from oops.surface.ellipsoid import Ellipsoid


class Spheroid(Ellipsoid):
    """A spheroidal surface centered on a path and fixed within a frame.

    The short radius of the spheroid is oriented along the *z*-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based on the
    assumption that a spherical body has been "squashed" along the *z*-axis. The latitude
    defined in this manner is neither planetocentric nor planetographic; functions are
    provided to perform the conversion to either choice. Longitudes are measured in a
    right-handed manner, increasing toward the east; values range from 0 to 2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then subtracting off
    the equatorial radius of the body. Thus, the surface is defined as the locus of points
    where elevation equals zero. However, note that with this definition, the gradient of
    the elevation value is not exactly normal to the surface.
    """

    def __init__(self, origin, frame, radii):
        """Constructor for a Spheroid surface.

        Parameters:
            origin (Path | str): The Path or the ID of the Path defining the center of
                the spheroid.
            frame (Frame | str): The Frame or the ID of the Frame in which the
                spheroid is fixed, with the short radius along the *z*-axis.
            radii (tuple[float, ...]): `(a, c)` or `(a, a, c)`, the long and short radii
                of the spheroid, in km.
        """

        # Allow either two or three radius values
        if len(radii) == 2:
            radii = (radii[0], radii[0], radii[1])

        Ellipsoid.__init__(self, origin, frame, radii=radii)

    def intercept_normal_to(self, pos, *, obs=None, time=None, direction='dep',
                            derivs=False, guess=None, hints=None):
        """Intercept point whose normal vector passes through a given position.

        This is a bit faster and more reliable than the default method for Ellipsoids,
        because the polynomial is only fourth-order instead of sixth-order.

        Parameters:
            pos (Vector3Like): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3Like, optional): Observer position relative to this Surface's
                origin and frame; ignored for this Surface subclass.
            time (ScalarLike, optional): Time at which to evaluate the Surface; ignored
                for this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate derivatives in pos into the
                returned intercepts.
            guess (ScalarLike, optional): Optional initial guess at coefficient `p` such
                that ``intercept + p * normal(intercept) = pos``. Use `guess=True` for
                the converged value of `p` to be returned even if an initial guess is
                unavailable. The guess is used only where it is closer to a solution
                than the default estimate.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 | tuple: `intercept` or `(intercept[, p][, hints])`, where:

            * `intercept` (Vector3): Surface intercept points relative to this surface's
              origin and frame, in km. Each position is solved on its own; where no
              intercept exists, or the solution does not converge, values are masked.
            * `p` (Scalar): The converged solution such that
              ``intercept + p * normal(intercept) = pos``; included if the input value of
              `guess` is not None.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)
        pos = self._apply_exclusion(pos)

        # If we work in the plane defined by the position and the Z-axis, this
        # becomes a 2-D problem, with pos = (pos_x, pos_z).

        (pos_x3d, pos_y3d, pos_z) = pos.to_scalars()
        pos_x = (pos_x3d**2 + pos_y3d**2).sqrt()

        # We can always recover the 3D (x,y) coordinates via
        #   pos_x3d = pos_x * cos_lon
        #   pos_y3d = pos_y * sin_lon
        # where:

        cos_lon = pos_x3d / pos_x
        sin_lon = pos_y3d / pos_x

        # We need to solve for p such that:
        #   cept + p * normal(cept) = pos
        # where
        #   normal(cept) = cept.element_mul(self._unsquash_sq)
        #
        # This is subject to the constraint that cept is the intercept point on
        # the surface, where
        #   cept_unsquashed = cept.element_mul(self._unsquash)
        # and
        #   cept_unsquashed.dot(cept_unsquashed) = self._req_sq
        #
        # Let:

        C = self._unsquash_z_sq
        R = self._req_sq

        # Three equations with three unknowns:
        # cept_x + p * cept_x = pos_x
        # cept_z + p * cept_z * C = pos_z
        # cept_x**2 + cept_z**2 * C = R
        #
        # Let:

        X = pos_x**2
        Z = pos_z**2 * C

        # Plug the first two into the third and rearrange:
        #
        # F(p) = X / (1 + p)**2 + Z / (1 + C*p)**2 - R = 0
        #
        # F has poles at p = -1 and -1/C. Above the larger pole it decreases from
        # +infinity to -R and is convex, so it has exactly one root there, and that root
        # is the nearest surface point: p > 0 for a position outside the surface and
        # p < 0 for one inside. Newton's method from any start in that interval converges
        # to the root as long as no step crosses the pole. (The equivalent fourth-order
        # polynomial has further roots below the pole, which are surface points whose
        # normals pass through the position from the far side.)

        lower = -1. / max(1., C)

        def newton_step(p):
            """One step of Newton's method on F(p) = 0 from the given value of p.

            The step is limited to half the distance to the pole below.
            """

            d1 = 1. + p
            dC = 1. + C * p
            f = X / d1**2 + Z / dC**2 - R
            df_dp = -2. * (X / d1**3 + C * Z / dC**3)
            return limited_step(f / df_dp, p, lower)

        # Make an initial estimate of p

        # Unsquash into coordinates where the surface is a sphere
        pos_unsq = pos.wod.element_mul(self._unsquash)   # without derivs!

        # Estimate the intercept point as on a straight line to the origin
        # (Note that this estimate is exact for points at the surface.)
        cept_guess_unsq = pos_unsq.with_norm(self._req)

        # Make a guess at the normal vector in unsquashed coordinates
        normal_guess_unsq = cept_guess_unsq.element_mul(self._unsquash_sq)

        # Estimate p
        p = ((pos_unsq.norm() - cept_guess_unsq.norm()) / normal_guess_unsq.norm())

        # Iterate from the supplied guess where there is one and it lies above the pole.
        # The precision of p should match the default geometric accuracy defined by
        # SURFACE_PHOTONS.km_precision; set our precision goal on p accordingly. Each
        # position converges on its own.
        p = Scalar.maximum(p, 0.5 * lower)
        if isinstance(guess, (type(None), bool, np.bool_)):
            start = None
        else:
            start = Scalar.as_scalar(guess).wod
            start = start.remask_or(start.vals <= lower)

        km_scale = self._req
        precision = SURFACE_PHOTONS.km_precision / km_scale
        p = solve(newton_step, p, start, precision, SURFACE_PHOTONS.max_iterations + 5,
                  name=f'{type(self).__name__}.intercept_normal_to', km_scale=km_scale,
                  debug=Ellipsoid._DEBUG)

        cept_x = pos_x / (1. + p)
        cept_z = pos_z / (1. + C * p)
        cept = Vector3.from_scalars(cos_lon * cept_x, sin_lon * cept_x, cept_z)

        results = (cept,)
        if guess is not None:
            results += (p,)

        if hints is not None:
            results += (hints,)

        if len(results) == 1:
            return cept

        return results

    ######################################################################################
    # Longitude conversions
    ######################################################################################

    def lon_to_centric(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Parameters:
            lon (ScalarLike): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    def lon_from_centric(self, lon, *, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Parameters:
            lon (ScalarLike): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    def lon_to_graphic(self, lon, *, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Parameters:
            lon (ScalarLike): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetographic longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    def lon_from_graphic(self, lon, *, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Parameters:
            lon (ScalarLike): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    ######################################################################################
    # Latitude conversions
    ######################################################################################

    def lat_to_centric(self, lat, lon=None, *, derivs=False):
        """Convert latitude in internal coordinates to planetocentric.

        Parameters:
            lat (ScalarLike): The latitude in radians.
            lon (ScalarLike, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetocentric latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan(recursive=derivs) * self._squash_z).arctan()

    def lat_from_centric(self, lat, lon=None, *, derivs=False):
        """Convert planetocentric latitude to internal spheroid coordinates.

        Parameters:
            lat (ScalarLike): The latitude in radians.
            lon (ScalarLike, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Squashed latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self._unsquash_z).arctan()

    def lat_to_graphic(self, lat, lon=None, *, derivs=False):
        """Convert latitude in internal coordinates to planetographic.

        Parameters:
            lat (ScalarLike): The latitude in radians.
            lon (ScalarLike, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Planetographic latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self._unsquash_z).arctan()

    def lat_from_graphic(self, lat, lon=None, *, derivs=False):
        """Convert a planetographic latitude to internal spheroid latitude.

        Parameters:
            lat (ScalarLike): The latitude in radians.
            lon (ScalarLike, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Squashed latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self._squash_z).arctan()

##########################################################################################

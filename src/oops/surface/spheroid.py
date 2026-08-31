##########################################################################################
# oops/surface/spheroid.py: Spheroid subclass of class Surface
##########################################################################################

import numpy as np

from polymath               import Scalar, Vector3
from oops.config            import SURFACE_PHOTONS, LOGGING
from oops.surface.ellipsoid import Ellipsoid


class Spheroid(Ellipsoid):
    """A spheroidal surface centered on the given path and fixed with respect to the given
    frame. The short radius of the spheroid is oriented along the Z-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based on the
    assumption that a spherical body has been "squashed" along the Z-axis. The latitude
    defined in this manner is neither planetocentric nor planetographic; functions are
    provided to perform the conversion to either choice. Longitudes are measured in a
    right-handed manner, increasing toward the east; values range from 0 to 2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then subtracting off
    the equatorial radius of the body. Thus, the surface is defined as the locus of points
    where elevation equals zero. However, note that with this definition, the gradient of
    the elevation value is not exactly normal to the surface.
    """

    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for a Spheroid surface.

        Parameters:
            origin (Path or str): The Path or the ID of the Path defining the center of
                the spheroid.
            frame (Frame or str): The Frame or the ID of the Frame in which the
                spheroid is fixed, with the short radius along the Z-axis.
            radii (tuple[float, ...]): `(a, c)` or `(a, a, c)`, the long and short radii
                of the spheroid, in km.
            exclusion (float, optional): The fraction of the polar radius within which
                calculations of intercept_normal_to() are suppressed. Values of less than
                0.9 are not recommended because the problem becomes numerically unstable.
        """

        # Allow either two or three radius values
        if len(radii) == 2:
            radii = (radii[0], radii[0], radii[1])

        Ellipsoid.__init__(self, origin, frame, radii=radii, exclusion=exclusion)

    def intercept_normal_to(self, pos, *, obs=None, time=None, direction='dep',
                            derivs=False, guess=None, hints=None):
        """Intercept point whose normal vector passes through a given position.

        This is a bit faster and more reliable than the default method for Ellipsoids,
        because the polynomial is only fourth-order instead of sixth-order.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate derivatives in pos into the
                returned intercepts.
            guess (Scalar, optional): Optional initial guess at coefficient `p` such
                that::

                    intercept + p * normal(intercept) = pos.

                Use `guess=True` for the converged value of `p` to be returned even if an
                initial guess is unavailable.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple: `intercept` or `(intercept[, p][, hints])`, where:

            * `intercept` (Vector3): Surface intercept points relative to this surface's
              origin and frame, in km. Where no intercept exists, values are masked.
            * `p` (Scalar): The converged solution such that
              `intercept + p * normal(intercept) = pos`; included if the input value of
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
        # f(p) = (  X * ((1 + C*p))**2
        #         + Z * ((1 + p))**2
        #         - R * ((1 + p) * (1 + C*p))**2)
        #
        # This is a fourth-order polynomial, which we need to solve for f(p) = 0.
        #
        # Using SymPy, this expands to:
        #
        # f(p) = (-C**2*R) * p**4
        #        + (-2*C**2*R - 2*C*R) * p**3
        #        + (-C**2*R + C**2*X - 4*C*R - R + Z) * p**2
        #        + (-2*C*R + 2*C*X - 2*R + 2*Z) * p
        #        + (- R + X + Z)
        #
        # Let f(p) = (((f4*p + f3)*p + f2)*p + f1)*p + f0

        C2 = C**2

        # For efficiency, we segregate all the array ops (involving X and Z)
        f4 = -C2 * R
        f3 = -2 * R * (C2 + C)
        f2 = C2 * X + Z - R * (C2 + 4*C + 1)
        f1 = (2*C) * X + 2 * Z - 2 * R * (C + 1)
        f0 = X + Z - R

        g3 = 4 * f4
        g2 = 3 * f3
        g1 = 2 * f2
        g0 = f1

        # Make an initial guess at p if necessary
        if isinstance(guess, (type(None), bool, np.bool_)):

            # Unsquash into coordinates where the surface is a sphere
            pos_unsq = pos.wod.element_mul(self._unsquash)   # without derivs!

            # Estimate the intercept point as on a straight line to the origin
            # (Note that this estimate is exact for points at the surface.)
            cept_guess_unsq = pos_unsq.with_norm(self._req)

            # Make a guess at the normal vector in unsquashed coordinates
            normal_guess_unsq = cept_guess_unsq.element_mul(self._unsquash_sq)

            # Estimate p
            p = ((pos_unsq.norm() - cept_guess_unsq.norm()) / normal_guess_unsq.norm())

        else:
            p = guess.wod.copy()

        # The precision of p should match the default geometric accuracy defined
        # by SURFACE_PHOTONS.km_precision. Set our precision goal on p
        # accordingly.
        km_scale = self._req
        precision = SURFACE_PHOTONS.km_precision / km_scale

        # Iterate until convergence stops
        max_dp = 1.e99
        converged = False

        # We typically need a few extra iterations to reach desired precision
        for count in range(SURFACE_PHOTONS.max_iterations + 5):

            # Calculate f and df/dp
            f = (((f4*p + f3)*p + f2)*p + f1)*p + f0
            df_dp = ((g3*p + g2)*p + g1)*p + g0

            # One step of Newton's method
            dp = f / df_dp
            p -= dp

            prev_max_dp = max_dp
            max_dp = dp.abs().max(builtins=True, masked=-1.)

            if LOGGING.surface_iterations or Ellipsoid._DEBUG:
                LOGGING.convergence(f'{type(self).__name__}.intercept_normal_to: '
                                    f'iter={count+1}; change[km]={max_dp*km_scale:.6g}')

            if max_dp <= precision:
                converged = True
                break

            if max_dp >= prev_max_dp:
                break

        if not converged:
            LOGGING.warn(f'{type(self).__name__}.intercept_normal_to did not converge: '
                         f'iter={count+1}; change[km]={max_dp*km_scale:.6g}')

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
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Planetocentric longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

    def lon_from_centric(self, lon, *, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Parameters:
            lon (Scalar): The longitude in radians.
            derivs (bool, optional): True to propagate any derivatives of `lon` into
                the returned longitude.

        Returns:
            Scalar: Squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

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
            Scalar: Squashed longitude.
        """

        return Scalar.as_scalar(lon, recursive=derivs)

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

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan(recursive=derivs) * self._squash_z).arctan()

    def lat_from_centric(self, lat, lon=None, *, derivs=False):
        """Convert planetocentric latitude to internal spheroid coordinates.

        Parameters:
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
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
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
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
            lat (Scalar): The latitude in radians.
            lon (Scalar, optional): The longitude in radians; ignored, because
                this conversion is independent of longitude for a surface of revolution.
            derivs (bool, optional): True to propagate any derivatives of `lat` into
                the returned latitude.

        Returns:
            Scalar: Squashed latitude.
        """

        lat = Scalar.as_scalar(lat, recursive=derivs)
        return (lat.tan() * self._squash_z).arctan()

##########################################################################################

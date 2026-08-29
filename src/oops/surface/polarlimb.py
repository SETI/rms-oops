##########################################################################################
# oops/surface/polarlimb.py: PolarLimb subclass of class Surface
##########################################################################################

import numpy as np

from polymath          import Scalar, Vector3
from oops.surface.limb import Limb

class PolarLimb(Limb):
    """The locus of points where a surface normal from a spheroid or ellipsoid is
    perpendicular to the line of sight.

    This provides a convenient coordinate system for describing cloud features on the limb
    of a body. The coordinates of PolarLimb are (z, clock, d), where:

    * z (Scalar): The vertical distance in km normal to the limb of the body surface.
    * clock (Scalar): The angle of the normal vector on the sky, measured clockwise from
      the projected direction of the north pole.
    * d (Scalar): An offset distance beyond the virtual limb plane along the line of
      sight; usually zero.
    """

    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None, groundtrack=False):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the surface, relative to this surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this surface's origin
                and frame.
            time (Scalar, optional): Time at which to evaluate the surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (z, clock) or all three (z, clock, dist) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Scalar, optional): Optionally, the value of the coefficient p such that
                ground + p * normal(ground) = pos; for the ground point on the body
                surface. Ignored if the value is None (the default) or True. groundtrack
                True to return the intercept on the surface along with the coordinates.

        Returns:
            (tuple): Two to four values, where:

            * `z` (Scalar): The vertical distance in km normal to the limb of the body
              surface.
            * `clock` (Scalar): The angle in radians of the normal vector on the sky,
              measured clockwise from the projected direction of the north pole.
            * `dist`: Optional offset distance in km beyond the virtual limb plane
              along the line of sight, included if axes == 3.
            * `track` (Vector3): Intercept point on the surface (where z == 0); included
              if input groundtrack is True.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)

        # There's a quick solution for the surface point if hints are provided
        if isinstance(hints, (type(None), bool, np.bool_)):
            los = pos - obs
            (cept, _, p, track) = self.intercept(obs, los, derivs=derivs,
                                                 hints=True, groundtrack=True)
                # The returned value of p speeds up the next calculation
        else:
            p = Scalar.as_scalar(hints, recursive=derivs)
            denom = Vector3.ONES + p * self.ground.unsquash_sq
            track = pos.element_div(denom)
            cept = pos

        results = self.z_clock_from_intercept(cept, obs, derivs=derivs, hints=p)

        if axes == 3:
            d = los.dot(pos - cept) / los.norm()
            results += (d,)

        if groundtrack:
            results += (track,)

        return results

    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False,
                                  groundtrack=False):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple): Two or three Scalars defining coordinates at or near this
                surface. These can have different shapes, but must be broadcastable to a
                common shape. z       the vertical distance in km normal to the limb of
                the body surface. clock   the angle in radians of the normal vector on the
                sky, measured clockwise from the projected direction of the north pole.
                dist    optional offset distance in km beyond the virtual limb plane along
                the line of sight.
            obs (Vector3, optional): Observer positions relative to this surface's origin
                and frame.
            time (Scalar, optional): Time at which to evaluate the surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to include the partial derivatives of the
                intercept point with respect to observer and to the coordinates.
                groundtrack True to include the associated groundtrack points on the body
                surface in the returned result.

        Returns:
            Pos or (pos, track), where:

            * `pos` (Vector3): Points defined by the coordinates, relative to this
              surface's origin and frame.
            * `track` (Vector3): Associated points on the body surface; included if input
              groundtrack is True.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        (z, clock) = coords[:2]
        (cept, track) = self.intercept_from_z_clock(z, clock, obs,
                                                    derivs=derivs,
                                                    groundtrack=True)

        if len(coords) > 2:
            d = Scalar.as_scalar(clock, recursive=derivs)
            los = cept - obs
            cept += (d / los.norm()) * los

        if groundtrack:
            return (cept, track)
        else:
            return cept

##########################################################################################

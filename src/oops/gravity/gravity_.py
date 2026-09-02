##########################################################################################
# oops/gravity/gravity_.py
##########################################################################################

from oops.oops import Oops


class Gravity(Oops):
    """An abstract class describing the gravity field of a body.

    A Gravity object provides the orbital frequencies of a test particle: the mean motion
    `omega`, the radial (epicyclic) oscillation frequency `kappa`, and the vertical
    oscillation frequency `nu`, along with their radial derivatives and the linear
    combinations of them that describe apsidal and nodal precession. Distances are in km,
    times are in seconds, and angles are in radians throughout.

    Attributes:
        GRAVITY_REGISTRY (dict): Global dictionary of the pre-defined Gravity objects,
            keyed by the upper-case name of the body. It is populated by the
            `oops.gravity.oblategravity` module, which also defines each entry as an
            attribute of this class, replacing "+" and " " in the key with "_".
    """

    GRAVITY_REGISTRY = {}           # global dictionary of gravity objects
                                    # Defined in OblateGravity

    ######################################################################################
    # Methods to be defined for each Gravity subclass
    ######################################################################################

    def potential(self, a):
        """The potential energy per unit mass in the equatorial plane.

        Parameters:
            a (float or array): Radius in km.

        Returns:
            float or array: Potential energy per unit mass in km^2/s^2. The value is
            negative and approaches zero as `a` increases.
        """

        raise NotImplementedError(f'{type(self).__name__}.potential is not implemented')

    def omega(self, a, *, e=0., sin_i=0.):
        """The mean motion at a given semimajor axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Mean motion in radians/s.
        """

        raise NotImplementedError(f'{type(self).__name__}.omega is not implemented')

    def kappa(self, a, *, e=0., sin_i=0.):
        """The radial oscillation frequency at a given semimajor axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Radial oscillation frequency in radians/s.
        """

        raise NotImplementedError(f'{type(self).__name__}.kappa is not implemented')

    def nu(self, a, *, e=0., sin_i=0.):
        """The vertical oscillation frequency at a given semimajor axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Vertical oscillation frequency in radians/s.
        """

        raise NotImplementedError(f'{type(self).__name__}.nu is not implemented')

    def domega_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the mean motion at a given semimajor axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the mean motion in radians/s/km.
        """

        raise NotImplementedError(f'{type(self).__name__}.domega_da is not implemented')

    def dkappa_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the radial oscillation frequency at a given semimajor
        axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the radial oscillation frequency in
            radians/s/km.
        """

        raise NotImplementedError(f'{type(self).__name__}.dkappa_da is not implemented')

    def dnu_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the vertical oscillation frequency at a given
        semimajor axis.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the vertical oscillation frequency in
            radians/s/km.
        """

        raise NotImplementedError(f'{type(self).__name__}.dnu_da is not implemented')

    def combo(self, a, factors, *, e=0., sin_i=0.):
        """A linear combination of the orbital frequencies.

        The value returned is
        `factors[0] * omega(a) + factors[1] * kappa(a) + factors[2] * nu(a)`. Full
        numeric precision is preserved in the limit of first- or second-order
        cancellation of the coefficients.

        Parameters:
            a (float or array): Semimajor axis in km.
            factors (tuple): Three coefficients, applied to the mean motion, the radial
                oscillation frequency, and the vertical oscillation frequency in that
                order.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: The frequency combination in radians/s.
        """

        raise NotImplementedError(f'{type(self).__name__}.combo is not implemented')

    def dcombo_da(self, a, factors, *, e=0., sin_i=0.):
        """The radial derivative of a linear combination of the orbital frequencies.

        Unlike `combo`, this method does not guarantee full precision if the coefficients
        cancel to first or second order.

        Parameters:
            a (float or array): Semimajor axis in km.
            factors (tuple): Three coefficients, applied to the mean motion, the radial
                oscillation frequency, and the vertical oscillation frequency in that
                order.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: The derivative of the frequency combination in radians/s/km.
        """

        raise NotImplementedError(f'{type(self).__name__}.dcombo_da is not implemented')

    def solve_a(self, freq, factors=(1,0,0), *, e=0., sin_i=0.):
        """The semimajor axis at which a combination of the orbital frequencies takes a
        given value.

        Parameters:
            freq (float or array): The desired value of the frequency combination, in
                radians/s.
            factors (tuple, optional): Three coefficients, applied to the mean motion, the
                radial oscillation frequency, and the vertical oscillation frequency in
                that order; default (1,0,0), meaning the mean motion alone.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Semimajor axis in km, such that
            `combo(a, factors, e=e, sin_i=sin_i)` equals `freq`.
        """

        raise NotImplementedError(f'{type(self).__name__}.solve_a is not implemented')

    ######################################################################################
    # Useful alternative names...
    ######################################################################################

    def n(self, a, *, e=0., sin_i=0.):
        """The mean motion at semimajor axis `a`. Identical to `omega(a)`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Mean motion in radians/s.
        """

        return self.omega(a, e=e, sin_i=sin_i)

    def dmean_dt(self, a, *, e=0., sin_i=0.):
        """The mean motion at semimajor axis `a`. Identical to `omega(a)`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Mean motion in radians/s.
        """

        return self.omega(a, e=e, sin_i=sin_i)

    def dperi_dt(self, a, *, e=0., sin_i=0.):
        """The pericenter precession rate at semimajor axis `a`. Identical to
        `combo(a, (1,-1,0))`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Pericenter precession rate in radians/s, positive for a
            prograde orbit about an oblate body.
        """

        return self.combo(a, (1,-1,0), e=e, sin_i=sin_i)

    def dnode_dt(self, a, *, e=0., sin_i=0.):
        """The nodal regression rate (negative) at semimajor axis `a`. Identical to
        `combo(a, (1,0,-1))`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Nodal regression rate in radians/s, negative for a prograde
            orbit about an oblate body.
        """

        return self.combo(a, (1,0,-1), e=e, sin_i=sin_i)

    def d_dmean_dt_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the mean motion at semimajor axis `a`. Identical to
        `domega_da(a)`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the mean motion in radians/s/km.
        """

        return self.domega_da(a, e=e, sin_i=sin_i)

    def d_dperi_dt_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the pericenter precession rate at semimajor axis `a`.
        Identical to `dcombo_da(a, (1,-1,0))`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the pericenter precession rate in radians/s/km.
        """

        return self.dcombo_da(a, (1,-1,0), e=e, sin_i=sin_i)

    def d_dnode_dt_da(self, a, *, e=0., sin_i=0.):
        """The radial derivative of the nodal regression rate (negative) at semimajor axis
        `a`. Identical to `dcombo_da(a, (1,0,-1))`.

        Parameters:
            a (float or array): Semimajor axis in km.
            e (float or array, optional): Orbital eccentricity; default 0.
            sin_i (float or array, optional): Sine of the orbital inclination; default 0.

        Returns:
            float or array: Derivative of the nodal regression rate in radians/s/km.
        """

        return self.dcombo_da(a, (1,0,-1), e=e, sin_i=sin_i)

    def ilr_pattern(self, n, m, *, p=1):
        """The pattern speed of the `m:m-p` inner Lindblad resonance, given the mean
        motion `n` of the perturber.

        The value returned is `n + kappa(a) * p/m`, where `a` is the semimajor axis at
        which the mean motion equals `n`. An inner Lindblad resonance always has a pattern
        speed faster than `n`.

        Parameters:
            n (float or array): Mean motion of the perturber in radians/s.
            m (int): The first index of the resonance, for which the resonance is named.
            p (int, optional): The order of the resonance; default 1.

        Returns:
            float or array: The pattern speed in radians/s, always greater than `n`.
        """

        a = self.solve_a(n, (1,0,0))
        return n + self.kappa(a) * p/m

    def olr_pattern(self, n, m, *, p=1):
        """The pattern speed of the `m:m+p` outer Lindblad resonance, given the mean
        motion `n` of the perturber.

        The value returned is `n - kappa(a) * p/(m+p)`, where `a` is the semimajor axis at
        which the mean motion equals `n`. An outer Lindblad resonance always has a pattern
        speed slower than `n`.

        Parameters:
            n (float or array): Mean motion of the perturber in radians/s.
            m (int): The first index of the resonance, for which the resonance is named.
            p (int, optional): The order of the resonance; default 1.

        Returns:
            float or array: The pattern speed in radians/s, always less than `n`.
        """

        a = self.solve_a(n, (1,0,0))
        return n - self.kappa(a) * p/(m+p)

    ######################################################################################
    # Gravity registry
    ######################################################################################

    @staticmethod
    def lookup(key):
        """A gravity field from the registry given its name.

        Parameters:
            key (str): The name of the body, case-insensitive.

        Returns:
            Gravity: The gravity field registered under this name.

        Raises:
            KeyError: If the name is not in the registry.
        """

        return Gravity.GRAVITY_REGISTRY[key.upper()]

    @staticmethod
    def exists(key):
        """True if the body's name exists in the gravity registry.

        Parameters:
            key (str): The name of the body, case-insensitive.

        Returns:
            bool: True if a gravity field is registered under this name.
        """

        return key.upper() in Gravity.GRAVITY_REGISTRY

##########################################################################################

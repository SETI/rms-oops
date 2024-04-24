################################################################################
# oops/gravity/gravity_.py: Abstract Gravity class
################################################################################

class Gravity(object):
    """An abstract class describing the gravity field of a body."""

    # To avoid circular imports...
    BODY_CLASS = None               # filled in by oops/__init__.py

    GRAVITY_REGISTRY = {}           # global dictionary of gravity objects
                                    # Defined in OblateGravity

    def potential(self, a, e=0., sin_i=0.):
        """The potential energy at radius a, in the equatorial plane."""
        pass

    def omega(self, a, e=0., sin_i=0.):
        """The mean motion (radians/s) at semimajor axis a."""
        pass

    def kappa(self, a, e=0., sin_i=0.):
        """The radial oscillation frequency (radians/s) at semimajor axis a.
        """
        pass

    def nu(self, a, e=0., sin_i=0.):
        """The vertical oscillation frequency (radians/s) at semimajor axis a.
        """
        pass

    def domega_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the mean motion (radians/s/km) at semimajor
        axis a.
        """
        pass

    def dkappa_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the radial oscillation frequency
        (radians/s/km) at semimajor axis a.
        """
        pass

    def dnu_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the vertical oscillation frequency
        (radians/s/km) at semimajor axis a.
        """
        pass

    def combo(self, a, factors, e=0., sin_i=0.):
        """A frequency combination, based on given coefficients for omega,
        kappa and nu. Full numeric precision is preserved in the limit of first-
        or second-order cancellation of the coefficients.
        """
        pass

    def dcombo_da(self, a, factors, e=0., sin_i=0.):
        """The radial derivative of a frequency combination, based on given
        coefficients for omega, kappa and nu. Unlike method combo(), this one
        does not guarantee full precision if the coefficients cancel to first
        or second order.
        """
        pass

    def solve_a(self, freq, factors=(1,0,0), e=0., sin_i=0.):
        """Solve for the semimajor axis at which the frequency is equal to the
        given combination of factors on omega, kappa and nu.
        """
        pass

    ############################################################################
    # Useful alternative names...
    ############################################################################

    def n(self, a, e=0., sin_i=0.):
        """The mean motion at semimajor axis a. Identical to omega(a)."""
        return self.omega(a)

    def dmean_dt(self, a, e=0., sin_i=0.):
        """The mean motion at semimajor axis a. Identical to omega(a)."""
        return self.omega(a)

    def dperi_dt(self, a, e=0., sin_i=0.):
        """The pericenter precession rate at semimajor axis a. Identical to
        combo(a, (1,-1,0)).
        """
        return self.combo(a, (1,-1,0))

    def dnode_dt(self, a, e=0., sin_i=0.):
        """The nodal regression rate (negative) at semimajor axis a. Identical
        to combo(a, (1,0,-1)).
        """
        return self.combo(a, (1,0,-1))

    def d_dmean_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the mean motion at semimajor axis a.
        Identical to domega_da(a).
        """
        return self.domega_da(a)

    def d_dperi_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the pericenter precession rate at semimajor
        axis a. Identical to dcombo_da(a, (1,-1,0)).
        """
        return self.dcombo_da(a, (1,-1,0))

    def d_dnode_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the nodal regression rate (negative) at
        semimajor axis a. Identical to dcombo_da(a, (1,0,-1)).
        """
        return self.dcombo_da(a, (1,0,-1))

    def ilr_pattern(self, n, m, p=1):
        """The pattern speed of the m:m-p inner Lindblad resonance, given the
        mean motion n of the perturber.
        """
        a = self.solve_a(n, (1,0,0))
        return (n + self.kappa(a) * p/m)

    def olr_pattern(self, n, m, p=1):
        """The pattern speed of the m:m+p outer Lindblad resonance, given the
        mean motion n of the perturber.
        """
        a = self.solve_a(n, (1,0,0))
        return (n - self.kappa(a) * p/(m+p))

    ############################################################################
    # Gravity registry
    ############################################################################

    @staticmethod
    def lookup(key):
        """A gravity filed from the registry given its name."""
        return Gravity.GRAVITY_REGISTRY[key.upper()]

    @staticmethod
    def exists(key):
        """True if the body's name exists in the gravity registry."""
        return key.upper() in Gravity.GRAVITY_REGISTRY

################################################################################

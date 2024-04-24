################################################################################
# oops/gravity/oblategravity.py: OblateGravity subclass of class Gravity
################################################################################

import numpy as np
import warnings

from oops.config    import LOGGING
from oops.constants import TWOPI
from oops.gravity   import Gravity

class OblateGravity(Gravity):
    """A class describing the gravity field of an oblate planet using an
    expansion in gravity moments J2, J4, etc.
    """

    #===========================================================================
    def __init__(self, gm, jlist=[], radius=1.):
        """The constructor for a OblateGravity object.

        Input:
            gm          The body's GM in units of km^3/s^2
            jlist       optional list of even gravity harmonics: [jJ2, J4, ...].
            radius      body radius for associated J-values.
        """

        self.gm = gm
        self.jn = jlist
        self.rp = radius
        self.r2 = radius * radius

        # Evaluate coefficients for frequencies
        n = 0
        pn_zero = 1.
        potential_jn = []
        omega_jn = []
        kappa_jn = []
        nu_jn    = []
        domega_jn = []
        dkappa_jn = []
        dnu_jn    = []
        for i in range(len(jlist)):
            n += 2          # i == 0 corresponds to J2; i == 1 to J4; etc.
            pn_zero = -(n-1.)/n * pn_zero

            potential_jn.append(pn_zero * jlist[i])

            omega_jn.append(      -(n+1) * pn_zero * jlist[i])
            kappa_jn.append( (n-1)*(n+1) * pn_zero * jlist[i])
            nu_jn.append(   -(n+1)*(n+1) * pn_zero * jlist[i])

            domega_jn.append(-(n+3) * omega_jn[i])
            dkappa_jn.append(-(n+3) * kappa_jn[i])
            dnu_jn.append(   -(n+3) * nu_jn[i])

        self.potential_jn = np.array(potential_jn)

        self.omega_jn  = np.array(omega_jn)
        self.kappa_jn  = np.array(kappa_jn)
        self.nu_jn     = np.array(nu_jn)
        self.domega_jn = np.array(domega_jn)
        self.dkappa_jn = np.array(dkappa_jn)
        self.dnu_jn    = np.array(dnu_jn)

    def __getstate__(self):
        return (self.gm, self.jn, self.rp)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def _jseries(coefficients, ratio2):
        """Internal method to evaluate a series of the form:
        coefficients[0] * ratio2 + coefficients[1] * ratio2^2 ...
        """

        return ratio2 * np.polyval(coefficients[::-1], ratio2)

    #===========================================================================
    def potential(self, a):
        """The potential energy at radius a, in the equatorial plane."""

        return -self.gm/a * (1. - OblateGravity._jseries(self.potential_jn,
                                                         self.r2/a**2))

    #===========================================================================
    def omega(self, a, e=0., sin_i=0.):
        """The mean motion (radians/s) at semimajor axis a.

        Corrections for e and sin(i) are accurate to second order.
        """

        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        omega2 = gm_a3 * (1. + OblateGravity._jseries(self.omega_jn, ratio2))
        omega1 = np.sqrt(omega2)

        if (e or sin_i) and self.jn:
            omega1 += np.sqrt(gm_a3) * ratio2 * self.jn[0] * \
                      (3. * e**2 - 12. * sin_i**2)

        return omega1

    #===========================================================================
    def kappa2(self, a):
        """The square of the radial oscillation frequency (radians/s) at
        semimajor axis a.
        """

        a2 = a * a
        kappa2 = self.gm/(a*a2) * (1. + OblateGravity._jseries(self.kappa_jn,
                                                               self.r2/a2))
        return kappa2

    #===========================================================================
    def kappa(self, a, e=0., sin_i=0.):
        """The radial oscillation frequency (radians/s) at semimajor axis a.
        axis a.
        """

        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        kappa2 = gm_a3 * (1. + OblateGravity._jseries(self.kappa_jn, ratio2))
        kappa1 = np.sqrt(kappa2)

        if (e or sin_i) and self.jn:
            kappa1 += np.sqrt(gm_a3) * ratio2 * self.jn[0] * (-9. * sin_i**2)

        return kappa1

    #===========================================================================
    def nu(self, a, e=0., sin_i=0.):
        """The vertical oscillation frequency (radians/s) at semimajor axis a.
        """

        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        nu2 = gm_a3 * (1. + OblateGravity._jseries(self.nu_jn, ratio2))
        nu1 = np.sqrt(nu2)

        if (e or sin_i) and self.jn:
            nu1 += np.sqrt(gm_a3) * ratio2 * self.jn[0] * \
                      (6. * e**2 - 12.75 * sin_i**2)

        return nu1

    #===========================================================================
    def domega_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the mean motion (radians/s/km) at semimajor
        axis a.
        """

        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        domega2 = gm_a4 * (-3. + OblateGravity._jseries(self.domega_jn, ratio2))
        domega1 = domega2 / (2. * self.omega(a))

        if (e or sin_i) and self.jn:
            domega1 -= 3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0] * \
                       (3. * e**2 - 12. * sin_i**2)

        return domega1

    #===========================================================================
    def dkappa_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the radial oscillation frequency
        (radians/s/km) at semimajor axis a.
        """

        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        dkappa2 = gm_a4 * (-3. + OblateGravity._jseries(self.dkappa_jn, ratio2))
        dkappa1 = dkappa2 / (2. * self.kappa(a))

        if (e or sin_i) and self.jn:
            dkappa1 -= 3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0] * \
                       (-9. * sin_i**2)

        return dkappa1

    #===========================================================================
    def dnu_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the vertical oscillation frequency
        (radians/s/km) at semimajor axis a.
        """

        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        dnu2 = gm_a4 * (-3. + OblateGravity._jseries(self.dnu_jn, ratio2))
        dnu1 = dnu2 / (2. * self.nu(a))

        if (e or sin_i) and self.jn:
            dnu1 -= 3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0] * \
                       (6. * e**2 - 12.75 * sin_i**2)

        return dnu1

    #===========================================================================
    def combo(self, a, factors, e=0., sin_i=0.):
        """A frequency combination, based on given coefficients for omega,
        kappa and nu. Full numeric precision is preserved in the limit of first-
        or second-order cancellation of the coefficients.
        """

        # Shortcut for nonzero e or i, to be refined later
        if e or sin_i:
            sum_values = 0.
            if factors[0]:
                sum_values = sum_values + factors[0] * self.omega(a, e, sin_i)
            if factors[1]:
                sum_values = sum_values + factors[1] * self.kappa(a, e, sin_i)
            if factors[2]:
                sum_values = sum_values + factors[2] * self.nu(a, e, sin_i)

            return sum_values

        a2 = a * a
        ratio2 = self.r2 / a2
        gm_over_a3 = self.gm / (a * a2)

        sum_factors = 0
        sum_values = 0.

        # omega term
        if factors[0] != 0:
            omega2_jsum = OblateGravity._jseries(self.omega_jn, ratio2)
            omega2 = gm_over_a3 * (1. + omega2_jsum)
            omega  = np.sqrt(omega2)

            sum_factors += factors[0]
            sum_values  += factors[0] * omega

        # kappa term
        if factors[1] != 0:
            kappa2_jsum = OblateGravity._jseries(self.kappa_jn, ratio2)
            kappa2 = gm_over_a3 * (1. + kappa2_jsum)
            kappa  = np.sqrt(kappa2)

            sum_factors += factors[1]
            sum_values  += factors[1] * kappa

        # nu term
        if factors[2] != 0:
            nu2_jsum = OblateGravity._jseries(self.nu_jn, ratio2)
            nu2 = gm_over_a3 * (1. + nu2_jsum)
            nu  = np.sqrt(nu2)

            sum_factors += factors[2]
            sum_values  += factors[2] * nu

        if sum_factors != 0:
            return sum_values

        # In the special cause where sum_factors = 0, we get cancellation to
        # leading order. We employ the following trick to improve accuracy.
        #
        # Because
        #   omega^2 - GM/a^3 = GM/a^3 * Jsum
        # we have
        #   [omega - sqrt(GM/a^3)] [omega + sqrt(GM/a^3)] = GM/a^3 * Jsum
        # or
        #   omega - sqrt(GM/a^3) = GM/a^3 * Jsum / [omega + sqrt(GM/a^3)]
        #
        # Similarly for kappa and nu. Our solution is to sum the quantities
        # (omega - sqrt(GM/a^3)), (kappa - sqrt(GM/a^3)) and (nu - sqrt(GM/a^3))
        # instead.

        sqrt_gm_over_a3 = np.sqrt(gm_over_a3)
        sum_values = 0.

        if factors[0] != 0:
            omega_diff = gm_over_a3 * omega2_jsum / (omega + sqrt_gm_over_a3)
            sum_values += factors[0] * omega_diff

        if factors[1] != 0:
            kappa_diff = gm_over_a3 * kappa2_jsum / (kappa + sqrt_gm_over_a3)
            sum_values += factors[1] * kappa_diff

        if factors[2] != 0:
            nu_diff = gm_over_a3 * nu2_jsum / (nu + sqrt_gm_over_a3)
            sum_values += factors[2] * nu_diff

        if factors[1] != factors[2]:
            return sum_values

        # In the final special case where
        #   factors[1] = factors[2] = -factors[0]/2
        # we get still higher-order cancellation. We employ another trick. The
        # expression becomes
        #   -factors[1] (2 omega - kappa - nu)
        #
        # Note that
        #   (2 omega - kappa - nu) (omega + kappa)
        #       = 2 omega^2 + omega kappa - omega nu - kappa^2 - kappa nu
        # Because
        #   2 omega^2 - kappa^2 = nu^2,
        # we get
        #   (2 omega - kappa - nu) (omega + kappa)
        #       = nu^2 + omega kappa - omega nu - kappa nu
        #       = (nu - omega) (nu - kappa)
        # Thus,
        #   2 omega - kappa - nu = (nu - omega) (nu - kappa) / (omega + kappa)

        if factors[1] == 0:
            return 0

        sum_values = -factors[1] * ((nu_diff - omega_diff)
                                 *  (nu_diff - kappa_diff)
                                 /  (omega + kappa))

        return sum_values

    #===========================================================================
    def dcombo_da(self, a, factors, e=0., sin_i=0.):
        """The radial derivative of a frequency combination, based on given
        coefficients for omega, kappa and nu. Unlike method combo(), this one
        does not guarantee full precision if the coefficients cancel to first
        or second order.
        """

        sum_values = 0.

        if factors[0]:
            sum_values += factors[0] * self.domega_da(a, e, sin_i)
        if factors[1]:
            sum_values += factors[1] * self.dkappa_da(a, e, sin_i)
        if factors[2]:
            sum_values += factors[2] * self.dnu_da(a, e, sin_i)

        return sum_values

    #===========================================================================
    def solve_a(self, freq, factors=(1,0,0), e=0., sin_i=0.):
        """Solve for the semimajor axis at which the frequency is equal to the
        given combination of factors on omega, kappa and nu. Solution is via
        Newton's method.
        """

        # Find an initial guess
        sum_factors = np.sum(factors)

        # No first-order cancellation:
        #   freq(a) ~ sum[factors] * sqrt(GM/a^3)
        #
        #   a^3 ~ GM * (sum[factors] / freq)^2

        if sum_factors != 0:
            a = (self.gm * (sum_factors/freq)**2)**(1./3.)

        # No second-order cancellation:
        #   freq(a) ~ 1/2 * sum[factor*term] * sqrt(GM/a^3) * Rp^2 / a^2
        #
        #   a^7 ~ GM * (sum[factor*term]/2 / freq)^2 Rp^4

        elif factors[1] != factors[2]:
            term = (factors[0] * self.omega_jn[0] +
                    factors[1] * self.kappa_jn[0] +
                    factors[2] * self.nu_jn[0]) / 2.
            a = (self.gm * (term * self.r2 / freq)**2)**(1/7.)

        # Second-order cancellation:
        #   freq(a) ~ -1/8 * sum[factor*term^2] * sqrt(GM/a^3) * Rp^4 / a^4
        #
        #   a^11 ~ GM * (-sum[factor*term^2]/8 / freq)^2 Rp^8

        else:
            term = (factors[0] * self.omega_jn[0]**2 +
                    factors[1] * self.kappa_jn[0]**2 +
                    factors[2] * self.nu_jn[0]**2) / (-8.)
            a = (self.gm * (term * self.r2 * self.r2 / freq)**2)**(1/11.)

        # Iterate using Newton's method
        da_prev_max = 1.e99
        for count in range(20):
            # a step in Newton's method: x(i+1) = x(i) - f(xi) / fp(xi)
            # our f(x) = self.combo() - freq
            #     fp(x) = self.dcombo()

            da = ((self.combo(a, factors, e, sin_i) - freq)
                   / self.dcombo_da(a, factors, e, sin_i))
            da_max = np.max(np.abs(da))
            if da_max == 0.:
                break

            a -= da

            # If Newton's method stops converging, return what we've got
            if count > 4 and da_max >= da_prev_max:
                break

            da_prev_max = da_max

        return a

    ############################################################################
    # Useful alternative names...
    ############################################################################

    def n(self, a, e=0., sin_i=0.):
        """The mean motion at semimajor axis a. Identical to omega(a)."""

        return self.omega(a, e, sin_i)

    #===========================================================================
    def dmean_dt(self, a, e=0., sin_i=0.):
        """The mean motion at semimajor axis a. Identical to omega(a)."""

        return self.omega(a, e, sin_i)

    #===========================================================================
    def dperi_dt(self, a, e=0., sin_i=0.):
        """The pericenter precession rate at semimajor axis a. Identical to
        combo(a, (1,-1,0)).
        """

        return self.combo(a, (1,-1,0), e, sin_i)

    #===========================================================================
    def dnode_dt(self, a, e=0., sin_i=0.):
        """The nodal regression rate (negative) at semimajor axis a. Identical
        to combo(a, (1,0,-1)).
        """

        return self.combo(a, (1,0,-1), e, sin_i)

    #===========================================================================
    def d_dmean_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the mean motion at semimajor axis a.
        Identical to domega_da(a).
        """

        return self.domega_da(a, e, sin_i)

    #===========================================================================
    def d_dperi_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the pericenter precession rate at semimajor
        axis a. Identical to dcombo_da(a, (1,-1,0)).
        """

        return self.dcombo_da(a, (1,-1,0), e, sin_i)

    #===========================================================================
    def d_dnode_dt_da(self, a, e=0., sin_i=0.):
        """The radial derivative of the nodal regression rate (negative) at
        semimajor axis a. Identical to dcombo_da(a, (1,0,-1)).
        """

        return self.dcombo_da(a, (1,0,-1), e, sin_i)

    #===========================================================================
    def ilr_pattern(self, n, m, p=1):
        """The pattern speed of the m:m-p inner Lindblad resonance, given the
        mean motion n of the perturber.
        """

        a = self.solve_a(n, (1,0,0))
        return (n + self.kappa(a) * p/m)

    #===========================================================================
    def olr_pattern(self, n, m, p=1):
        """The pattern speed of the m:m+p outer Lindblad resonance, given the
        mean motion n of the perturber.
        """

        a = self.solve_a(n, (1,0,0))
        return (n - self.kappa(a) * p/(m+p))

    ############################################################################
    # Orbital elements
    ############################################################################

    def state_from_osc(self, elements, body_gm=0.):
        """Position and velocity based on osculating orbital elements: (a, e, i,
        mean longitude, longitude of pericenter, longitude of ascending node).

        Routine adapted from SWIFT's orbel_el2xv.f by Rob French. Only works
        well for e < 0.18.
        """

        gm = self.gm + body_gm

        (a, e, inc, mean_lon, long_peri, long_node) = elements
        a = np.asfarray(a)
        e = np.asfarray(e)
        inc = np.asfarray(inc)
        mean_lon = np.asfarray(mean_lon)
        long_peri = np.asfarray(long_peri)
        long_node = np.asfarray(long_node)

        mean_anomaly = mean_lon - long_peri

        sp = np.sin(long_peri)
        cp = np.cos(long_peri)
        so = np.sin(long_node)
        co = np.cos(long_node)
        si = np.sin(inc)
        ci = np.cos(inc)
        d11 = cp*co - sp*so*ci
        d12 = cp*so + sp*co*ci
        d13 = sp*si
        d21 = -sp*co - cp*so*ci
        d22 = -sp*so + cp*co*ci
        d23 = cp*si

        sm = np.sin(mean_anomaly)
        cm = np.cos(mean_anomaly)

        x = mean_anomaly + e*sm*( 1. + e*( cm + e*( 1. - 1.5*sm*sm)))

        sx = np.sin(x)
        cx = np.cos(x)
        es = e*sx
        ec = e*cx
        f = x - es  - mean_anomaly
        fp = 1. - ec
        fpp = es
        fppp = ec
        dx = -f/fp
        dx = -f/(fp + dx*fpp/2.)
        dx = -f/(fp + dx*fpp/2. + dx*dx*fppp/6.)

        cape = x + dx

        scap = np.sin(cape)
        ccap = np.cos(cape)
        sqe = np.sqrt(1. -e*e)
        sqgma = np.sqrt(gm*a)
        xfac1 = a*(ccap - e)
        xfac2 = a*sqe*scap
        ri = 1./(a*(1. - e*ccap))
        vfac1 = -ri * sqgma * scap
        vfac2 = ri * sqgma * sqe * ccap

        x =  d11*xfac1 + d21*xfac2
        y =  d12*xfac1 + d22*xfac2
        z =  d13*xfac1 + d23*xfac2
        vx = d11*vfac1 + d21*vfac2
        vy = d12*vfac1 + d22*vfac2
        vz = d13*vfac1 + d23*vfac2

        # Broadcast to a common shape and create vectors
        (x,y,z,vx,vy,vz) = np.broadcast_arrays(x,y,z,vx,vy,vz)

        pos = np.stack([x, y, z], axis=-1)
        vel = np.stack([vx, vy, vz], axis=-1)

        return (pos,vel)

    ############################################################################
    # Orbital elements
    ############################################################################

    def osc_from_state(self, pos, vel, body_gm=0.):
        """Osculating orbital elements based on position and velocity.

        Routine adapted from SWIFT's orbel_vx2el.f by Rob French.
        """

        (pos, vel) = np.broadcast_arrays(pos, vel)
        pos = np.asfarray(pos)
        vel = np.asfarray(vel)

        x = pos[...,0]
        y = pos[...,1]
        z = pos[...,2]

        vx = vel[...,0]
        vy = vel[...,1]
        vz = vel[...,2]

        tiny = 1e-300

        # Warning: This only works with elliptical orbits!
        gmsum = self.gm + body_gm

        # Compute the angular momentum H, and thereby the inclination INC.
        hx = y*vz - z*vy
        hy = z*vx - x*vz
        hz = x*vy - y*vx
        h2 = hx*hx + hy*hy + hz*hz
        h  = np.sqrt(h2)
        inc = np.arccos(hz/h)

        # Compute longitude of ascending node long_node and the argument of
        # latitude u.
        fac = np.sqrt(hx**2 + hy**2)/h

        long_node = np.where(fac < tiny, np.zeros(x.shape),
                                         Gravity._pos_arctan2(hx,-hy))
        tmp = np.arctan2(y, x)
        tmp = np.where(np.abs(inc - np.pi) < 10.*tiny, -tmp, tmp)
        tmp = tmp % TWOPI

        sin_inc = np.sin(inc)
        if np.shape(sin_inc) == ():             # Avoid possible divide-by-zero
            if sin_inc == 0.:
                sin_inc = 1.
        else:
            sin_inc[sin_inc == 0.] = 1.

        u = np.where(fac < tiny, tmp, Gravity._pos_arctan2(z/sin_inc,
                                                           x*np.cos(long_node) +
                                                           y*np.sin(long_node)))

        # Compute the radius R and velocity squared V2, and the dot
        # product RDOTV, the energy per unit mass ENERGY.
        r = np.sqrt(x*x + y*y + z*z)
        v2 = vx*vx + vy*vy + vz*vz
        vdotr = x*vx + y*vy + z*vz
        energy = 0.5*v2 - gmsum/r

        a = -0.5*gmsum/energy

        fac = 1. - h2/(gmsum*a)
        e = np.where(fac > tiny, np.minimum(np.sqrt(fac), 1.), 0.) # XXX
        face = (a-r)/(a*e)
        face = np.minimum(face, 1.)
        face = np.maximum(face, -1.)
        cape = np.arccos(face)
        cape = np.where(vdotr < 0., 2.*np.pi-cape, cape)
        cape = np.where(fac > tiny, cape, u)
        cw = (np.cos(cape) - e)/(1. - e*np.cos(cape))
        sw = np.sqrt(1. - e*e)*np.sin(cape)/(1. - e*np.cos(cape))
        w = np.where(fac > 0., Gravity._pos_arctan2(sw,cw), u)

        mean_anomaly = (cape - e*np.sin(cape)) % TWOPI
        long_peri = (u - w) % TWOPI

        mean_lon = (mean_anomaly + long_peri) % TWOPI

        # Convert any shapeless arrays to scalars
        elements = []
        for element in (a, e, inc, mean_lon, long_peri, long_node):
            if isinstance(element, np.ndarray) and element.shape == ():
                elements.append(element[()])
            else:
                elements.append(element)

        return tuple(elements)

    #===========================================================================
    def state_from_geom(self, elements, body_gm=0.):
        """Position and velocity based on geometric orbital elements: (a, e, i,
        mean longitude, longitude of pericenter, longitude of ascending node).

        Adapted from Renner & Sicardy (2006) EQ 2-13 by Rob French.

        Take the geometric osculating elements and convert to X,Y,Z,VX,VY,VZ
        Returns x, y, z, vx, vy, vz
        From Renner & Sicardy (2006) EQ 2-13
        """

        (a, e, inc, mean_lon, long_peri, long_node) = elements
        a = np.asfarray(a)
        e = np.asfarray(e)
        inc = np.asfarray(inc)
        lam = np.asfarray(mean_lon)
        long_peri = np.asfarray(long_peri)
        long_node = np.asfarray(long_node)

        (n, kappa, nu, eta2, chi2,
         alpha1, alpha2, alphasq) = self._geom_to_freq(a, e, inc, body_gm)
        kappa2 = kappa**2
        n2 = n**2
        nu2 = nu**2

        # Convert to cylindrical
        r = a*(1. - e*np.cos(lam-long_peri) +
               e**2*(3./2. * eta2/kappa2 - 1. -
                      eta2/2./kappa2 * np.cos(2.*(lam-long_peri))) +
               inc**2*(3./4.*chi2/kappa2 - 1. +
                        chi2/4./alphasq * np.cos(2.*(lam-long_node))))

        L = (lam + 2.*e*n/kappa*np.sin(lam-long_peri) +
             e**2*(3./4. + nu2/2./kappa2)*n/kappa * np.sin(2.*(lam-long_peri)) -
             inc**2*chi2/4./alphasq*n/nu*np.sin(2.*(lam-long_node)))

        z = a * inc * (np.sin(lam-long_node) +
                    e*chi2/2./kappa/alpha1*np.sin(2.*lam-long_peri-long_node) -
                    e*3./2.*chi2/kappa/alpha2*np.sin(long_peri-long_node))

        rdot = a * kappa * (e*np.sin(lam-long_peri) +
                            e**2*eta2/kappa2*np.sin(2.*(lam-long_peri)) -
                            inc**2*chi2/2./alphasq*nu/kappa*
                            np.sin(2.*(lam-long_node)))

        Ldot = n*(1. + 2.*e*np.cos(lam-long_peri) +
                  e**2 * (7./2. - 3.*eta2/kappa2 - kappa2/2./n2 +
                           (3./2.+eta2/kappa2)*np.cos(2.*(lam-long_peri))) +
                  inc**2 * (2. - kappa2/2./n2 - 3./2.*chi2/kappa2 -
                             chi2/2./alphasq*np.cos(2.*(lam-long_node))))

        vz = a*inc*nu*(np.cos(lam-long_node) +
                       e*chi2*(kappa+nu)/2./kappa/alpha1/nu *
                       np.cos(2*lam-long_peri-long_node) +
           e*3./2.*chi2*(kappa-nu)/kappa/alpha2/nu*np.cos(long_peri-long_node))

        x = r*np.cos(L)
        y = r*np.sin(L)
        vx = rdot*np.cos(L) - r*Ldot*np.sin(L)
        vy = rdot*np.sin(L) + r*Ldot*np.cos(L)

        # Broadcast to a common shape and create vectors
        (x,y,z,vx,vy,vz) = np.broadcast_arrays(x,y,z,vx,vy,vz)

        pos = np.stack([x, y, z], axis=-1)
        vel = np.stack([vx, vy, vz], axis=-1)

        return (pos, vel)

    #===========================================================================
    def geom_from_state(self, pos, vel, body_gm=0., tol=1.e-6):
        """Geometric orbital elements based on position and velocity.

        Routine adapted from SWIFT's orbel_vx2el.f by Rob French.

        Given the state vector x,y,z,vx,vy,vz retrieve the geometric elements
        Returns: a, e, inc, long_peri, long_node, mean_anomaly
        From Renner and Sicardy (2006) EQ 22-47
        """

        (pos, vel) = np.broadcast_arrays(pos, vel)
        pos = np.asfarray(pos)
        vel = np.asfarray(vel)

        x = pos[...,0]
        y = pos[...,1]
        z = pos[...,2]

        vx = vel[...,0]
        vy = vel[...,1]
        vz = vel[...,2]

        # EQ 22-25
        r = np.sqrt(x**2 + y**2)
        L = Gravity._pos_arctan2(y, x)
        rdot = vx*np.cos(L) + vy*np.sin(L)
        Ldot = (vy*np.cos(L)-vx*np.sin(L))/r

        # Initial conditions
        a = r
        e = 0.
        inc = 0.
        rc = 0.
        Lc = 0.
        zc = 0.
        rdotc = 0.
        Ldotc = 0.
        zdotc = 0.

        old_diffmax = 1.e38
        old_diff = None
        idx_to_use = np.where(x!=-1e38,True,False) # All True
        announced = False
        while True:
            (n, kappa, nu, eta2, chi2,
             alpha1, alpha2, alphasq) = self._geom_to_freq(a, e, inc, body_gm)
            ret = Gravity._freq_to_geom(r, L, z, rdot, Ldot, vz, rc, Lc, zc,
                                   rdotc, Ldotc, zdotc, n, kappa, nu, eta2,
                                    chi2, alpha1, alpha2, alphasq)
            old_a = a
            (a, e, inc, long_peri, long_node, lam,
             rc, Lc, zc, rdotc, Ldotc, zdotc) = ret
            diff = np.abs(a-old_a)
            diffmax = np.max(diff[idx_to_use])
            if diffmax < tol:
                break
            if diffmax > old_diffmax:
                idx_to_use = np.where(diff > old_diff,False,True) & idx_to_use
                if not idx_to_use.any():
                    break
                if not announced:
                    LOGGING.warn('geom_from_state() started diverging! ' +
                                 'Tolerance met = %e' % diffmax)
                    announced = True

                diff_of_diff = diff - old_diff
                bad_idx = diff_of_diff.argmax()
                warnings.warn('Bad index ' + str(bad_idx) +
                              '; X = ' + str(x[bad_idx]) +
                              '; Y = ' + str(y[bad_idx]) +
                              '; Z =' + str(z[bad_idx]) +
                              '; VX = ' + str(vx[bad_idx]) +
                              '; VY = ' + str(vy[bad_idx]) +
                              '; VZ = ' + str(vz[bad_idx]))
            old_diffmax = diffmax
            old_diff = diff

        return (a, e, inc, lam, long_peri, long_node)

    ############################################################################
    # Internal methods
    ############################################################################

    def _geom_to_freq(self, a, e, inc, body_gm=0.):
        """Take the geometric osculating elements and create frequencies
        Returns n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq
        From Renner & Sicardy (2006)  EQ 14-21
        """

        gmsum = self.gm + body_gm
        j2 = 0.
        j4 = 0.
        if len(self.jn) > 0:
            j2 = self.jn[0] * self.r2/a**2
        if len(self.jn) > 1:
            j4 = self.jn[1] * self.r2**2/a**4

        gm_a3 = gmsum / a**3
        sqrt_gm_a3 = np.sqrt(gm_a3)

        n = sqrt_gm_a3 * (1. + 3./4.*j2 - 15./16.*j4 -
                               9./32.*j2**2 + 45./64.*j2*j4 +
                               27./128.*j2**3 +
                               3.*j2*e**2 - 12.*j2*inc**2)

        kappa = sqrt_gm_a3 * (1. - 3./4.*j2 + 45./16.*j4 -
                                   9./32.*j2**2 + 135./64.*j2*j4 -
                                   27./128.*j2**3 - 9.*j2*inc**2)

        nu = sqrt_gm_a3 * (1. + 9./4.*j2 - 75./16.*j4 -
                                81./32.*j2**2 + 675./64.*j2*j4 +
                                729./128.*j2**3 +
                                6.*j2*e**2 - 51./4.*j2*inc**2)

        eta2 = gm_a3 * (1. - 2.*j2 + 75./8.*j4)

        chi2 = gm_a3 * (1. + 15./2.*j2 - 175./8.*j4)

        alpha1 = 1./3. * (2.*nu + kappa)
        alpha2 = 2.*nu - kappa
        alphasq = alpha1 * alpha2

        return (n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq)

    #===========================================================================
    @staticmethod
    def _freq_to_geom(r, L, z, rdot, Ldot, zdot, rc, Lc, zc, rdotc, Ldotc,
                      zdotc, n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq):
        """Take the frequencies and convert them to cylindrical coordinates
        Returns a, e, inc, long_peri, long_node, lam, rc, Lc, zc, rdotc, Ldotc,
        zdotc.
        From Renner & Sicardy (2006) EQ 36-41
        """

        kappa2 = kappa**2
        n2 = n**2

        # EQ 42-47
        a = (r-rc) / (1.-(Ldot-Ldotc-n)/(2.*n))

        e = np.sqrt(((Ldot-Ldotc-n)/(2.*n))**2 + ((rdot-rdotc)/(a*kappa))**2)

        inc = np.sqrt(((z-zc)/a)**2 + ((zdot-zdotc)/(a*nu))**2)

        lam = L - Lc - 2.*n/kappa*(rdot-rdotc)/(a*kappa)

        long_peri = (lam - Gravity._pos_arctan2(rdot-rdotc,
                                                a*kappa*(1.-(r-rc)/a))) % TWOPI

        long_node = (lam - Gravity._pos_arctan2(nu*(z-zc), zdot-zdotc)) % TWOPI

        # EQ 36-41
        rc = (a * e**2 * (3./2.*eta2/kappa2 - 1. -
                           eta2/2./kappa2*np.cos(2.*(lam-long_peri))) +
              a * inc**2 * (3./4.*chi2/kappa2 - 1. +
                             chi2/4./alphasq*np.cos(2.*(lam-long_node))))

        Lc = (e**2*(3./4. + eta2/2./kappa2)*n/kappa*np.sin(2.*(lam-long_peri)) -
              inc**2*chi2/4./alphasq*n/nu*np.sin(2.*(lam-long_node)))

        zc = a*inc*e*(chi2/2./kappa/alpha1*np.sin(2*lam-long_peri-long_node) -
                      3./2.*chi2/kappa/alpha2*np.sin(long_peri-long_node))

        rdotc = (a*e**2*eta2/kappa*np.sin(2.*(lam-long_peri)) -
                 a*inc**2*chi2/2./alphasq*nu*np.sin(2.*(lam-long_node)))

        Ldotc = (e**2*n*(7./2. - 3.*eta2/kappa2 - kappa2/2./n2 +
                          (3./2. + eta2/kappa2)*np.cos(2.*(lam-long_peri))) +
                 inc**2*n*(2. - kappa2/2./n2 - 3./2.*chi2/kappa2 -
                            chi2/2./alphasq*np.cos(2.*(lam-long_node))))

        zdotc = a*inc*e*(chi2*(kappa+nu)/2./kappa/
                            alpha1*np.cos(2*lam-long_peri-long_node) +
                 3./2.*chi2*(kappa-nu)/kappa/alpha2*np.cos(long_peri-long_node))

        # EQ 30-35
    #    r = a*(1. - e*np.cos(lam-long_peri)) + rc
    #
    #    L = lam + 2*e*n/kappa*np.sin(lam-long_peri) + Lc
    #
    #    z = a*inc*np.sin(lam-long_node) + zc
    #
    #    rdot = a*e*kappa*np.sin(lam-long_peri) + rdotc
    #
    #    Ldot = n*(1. + 2.*e*np.cos(lam-long_peri)) + Ldotc
    #
    #    zdot = a*inc*nu*np.cos(lam-long_node) + zdotc

        return (a, e, inc, long_peri, long_node, lam,
                rc, Lc, zc, rdotc, Ldotc, zdotc)

    #===========================================================================
    # A nicer version of arctan2
    @staticmethod
    def _pos_arctan2(y, x):
        return np.arctan2(y, x) % TWOPI

################################################################################
# Planetary gravity fields defined...
################################################################################

# From http://ssd.jpl.nasa.gov/?planet_phys_par
G_MKS = 6.67428e-11     # m^3 kg^-1 s^-2
G_CGS = 6.67428e-08     # cm^3 g^-1 s^-2

G_PER_KG = G_MKS / 1.e9
G_PER_G  = G_CGS / 1.e15

# From http://ssd.jpl.nasa.gov/?planet_phys_par
SUN = OblateGravity(132712440018., [], 695500.)

# From http://ssd.jpl.nasa.gov/?planet_phys_par
MERCURY = OblateGravity(0.330104e24 * G_PER_KG, [], 2439.7 )
VENUS   = OblateGravity( 4.86732e24 * G_PER_KG, [], 6051.8 )
EARTH   = OblateGravity( 5.97219e24 * G_PER_KG, [], 6378.14)
MARS    = OblateGravity(0.641693e24 * G_PER_KG, [], 3396.19)

# Earlier values from http://ssd.jpl.nasa.gov/?gravity_fields_op
JUPITER_V1 = OblateGravity(126686535., [14696.43e-06, -587.14e-06, 34.25e-06], 71492.)
#SATURN  = OblateGravity( 37931208.,  [16290.71e-06, -935.83e-06, 86.14e-06], 60330.)
SATURN_V1  = OblateGravity( 37931207.7, [16290.71e-06, -936.83e-06, 86.14e-06, -10.e-06], 60330.)
URANUS_V1  = OblateGravity(  5793964., [ 3341.29e-06,  -30.44e-06           ], 26200.)
NEPTUNE_V1 = OblateGravity(  6835100., [ 3408.43e-06,  -33.40e-06           ], 25225.)

# Updated September 15, 2015 from http://ssd.jpl.nasa.gov/?gravity_fields_op
JUPITER = OblateGravity(126686536.1, [14695.62e-06, -591.31e-06, 20.78e-06], 71492.)
SATURN  = OblateGravity( 37931208. , [16290.71e-06, -935.83e-06, 86.14e-06,
                                                            -10.e-06], 60330.)
URANUS  = OblateGravity(  5793951.3, [ 3510.68e-06,  -34.17e-06           ], 25559.)
NEPTUNE = OblateGravity(  6835100. , [ 3408.43e-06,  -33.40e-06           ], 25225.)

# From http://arxiv.org/abs/0712.1261
PLUTO_ONLY = OblateGravity(869.6, [], 1151.)
PLUTO = PLUTO_ONLY

# From http://ssd.jpl.nasa.gov/?sat_phys_par
MOON      = OblateGravity(4902.801, [], 1737.5)

IO        = OblateGravity(5959.916, [], 1821.6)
EUROPA    = OblateGravity(3202.739, [], 1560.8)
GANYMEDE  = OblateGravity(9887.834, [], 2631.2)
CALLISTO  = OblateGravity(7179.289, [], 2410.3)

MIMAS     = OblateGravity(   2.5026, [],  198.20)
ENCELADUS = OblateGravity(   7.2027, [],  252.10)
TETHYS    = OblateGravity(  41.2067, [],  533.00)
DIONE     = OblateGravity(  73.1146, [],  561.70)
RHEA      = OblateGravity( 153.9426, [],  764.30)
TITAN     = OblateGravity(8978.1382, [], 2574.73)
HYPERION  = OblateGravity(   0.3727, [],  135.00)
IAPETUS   = OblateGravity( 120.5038, [],  735.60)
PHOEBE    = OblateGravity(   0.5532, [],  106.50)

MIRANDA   = OblateGravity(   4.4, [], 235.8)
ARIEL     = OblateGravity(  86.4, [], 578.9)
UMBRIEL   = OblateGravity(  81.5, [], 584.7)
TITANIA   = OblateGravity( 228.2, [], 788.9)
OBERON    = OblateGravity( 192.4, [], 761.4)

TRITON    = OblateGravity(1427.6, [], 1353.4)
NEREID    = OblateGravity(  2.06, [],  170.)

CHARON    = OblateGravity(105.9, [], 603.6)

# Sets with relatively large mass ratios
SUN_JUPITER = OblateGravity(SUN.gm + JUPITER.gm, [], SUN.rp)

JUPITER_GALS = OblateGravity(JUPITER.gm + IO.gm + EUROPA.gm + GANYMEDE.gm +
                       CALLISTO.gm, JUPITER.jn, JUPITER.rp)

SATURN_TITAN = OblateGravity(SATURN.gm + TITAN.gm, SATURN.jn, SATURN.rp)

PLUTO_CHARON_OLD = OblateGravity(PLUTO_ONLY.gm + CHARON.gm, [], PLUTO_ONLY.rp)

################################################################################
# Revised Pluto-Charon gravity
#
# Outside a ring of radius R, the gravity moments are -P2n(0).
#   J2 = 1/2; J4 = -3/8; J6 = 5/16; J8 = -35/128; J10 = 63/256
# We can stop there.
#
# The gravity potential in the equatorial plane for one body is:
#   phi(r) = -GM/r (1 - SUM[ J2n (R/r)^(2n) P_2n(0) ]
#          = -GM/r + (J2 GM R^2 P_2(0)) / r^3
#                  + (J4 GM R^4 P_4(0)) / r^5 + ...
#
# For two bodies with GM1, GM2, R1, R2, but the same J2n series...
#
#   phi(r) = -(GM1 + GM2) / r
#          +  (GM1 R1^2 + GM2 R2^2) (J2 P_2(0)) / r^3
#          +  (GM1 R1^4 + GM2 R2^4) (J4 P_4(0)) / r^5 ...
#
# Scaling everything to GM = GM1 + GM2; R = R2:
#   J2' = J2 (GM1 (R1/R2)^2 + GM2) / (GM1 + GM2)
#   J4' = J4 (GM1 (R1/R2)^4 + GM2) / (GM1 + GM2)
# etc.
################################################################################
PLUTO_A  = 19596. * CHARON.gm / (PLUTO.gm + CHARON.gm)
CHARON_A = 19596. - PLUTO_A
ratio2 = (PLUTO_A / CHARON_A)**2
gm1 = PLUTO_ONLY.gm
gm2 = CHARON.gm
PLUTO_CHARON_AS_RINGS = OblateGravity(gm1 + gm2,
        [ 1/2.    * (gm1 * ratio2    + gm2) / (gm1 + gm2),
         -3/8.    * (gm1 * ratio2**2 + gm2) / (gm1 + gm2),
          5/16.   * (gm1 * ratio2**3 + gm2) / (gm1 + gm2),
         -35/128. * (gm1 * ratio2**4 + gm2) / (gm1 + gm2),
          63/256. * (gm1 * ratio2**5 + gm2) / (gm1 + gm2)], CHARON_A)
PLUTO_CHARON = PLUTO_CHARON_AS_RINGS
################################################################################

Gravity.GRAVITY_REGISTRY["SUN"       ] = SUN
Gravity.GRAVITY_REGISTRY["MERCURY"   ] = MERCURY
Gravity.GRAVITY_REGISTRY["VENUS"     ] = VENUS
Gravity.GRAVITY_REGISTRY["EARTH"     ] = EARTH
Gravity.GRAVITY_REGISTRY["MARS"      ] = MARS
Gravity.GRAVITY_REGISTRY["JUPITER"   ] = JUPITER
Gravity.GRAVITY_REGISTRY["SATURN"    ] = SATURN
Gravity.GRAVITY_REGISTRY["URANUS"    ] = URANUS
Gravity.GRAVITY_REGISTRY["NEPTUNE"   ] = NEPTUNE
Gravity.GRAVITY_REGISTRY["PLUTO_ONLY"] = PLUTO_ONLY
Gravity.GRAVITY_REGISTRY["PLUTO"     ] = PLUTO_ONLY
Gravity.GRAVITY_REGISTRY["MOON"      ] = MOON
Gravity.GRAVITY_REGISTRY["IO"        ] = IO
Gravity.GRAVITY_REGISTRY["EUROPA"    ] = EUROPA
Gravity.GRAVITY_REGISTRY["GANYMEDE"  ] = GANYMEDE
Gravity.GRAVITY_REGISTRY["CALLISTO"  ] = CALLISTO
Gravity.GRAVITY_REGISTRY["MIMAS"     ] = MIMAS
Gravity.GRAVITY_REGISTRY["ENCELADUS" ] = ENCELADUS
Gravity.GRAVITY_REGISTRY["TETHYS"    ] = TETHYS
Gravity.GRAVITY_REGISTRY["DIONE"     ] = DIONE
Gravity.GRAVITY_REGISTRY["RHEA"      ] = RHEA
Gravity.GRAVITY_REGISTRY["TITAN"     ] = TITAN
Gravity.GRAVITY_REGISTRY["HYPERION"  ] = HYPERION
Gravity.GRAVITY_REGISTRY["IAPETUS"   ] = IAPETUS
Gravity.GRAVITY_REGISTRY["PHOEBE"    ] = PHOEBE
Gravity.GRAVITY_REGISTRY["MIRANDA"   ] = MIRANDA
Gravity.GRAVITY_REGISTRY["ARIEL"     ] = ARIEL
Gravity.GRAVITY_REGISTRY["UMBRIEL"   ] = UMBRIEL
Gravity.GRAVITY_REGISTRY["TITANIA"   ] = TITANIA
Gravity.GRAVITY_REGISTRY["OBERON"    ] = OBERON
Gravity.GRAVITY_REGISTRY["TRITON"    ] = TRITON
Gravity.GRAVITY_REGISTRY["NEREID"    ] = NEREID
Gravity.GRAVITY_REGISTRY["CHARON"    ] = CHARON

Gravity.GRAVITY_REGISTRY["SUN+JUPITER"      ] = SUN_JUPITER
Gravity.GRAVITY_REGISTRY["JUPITER+GALILEANS"] = JUPITER_GALS
Gravity.GRAVITY_REGISTRY["SATURN+TITAN"     ] = SATURN_TITAN
Gravity.GRAVITY_REGISTRY["PLUTO+CHARON"     ] = PLUTO_CHARON
Gravity.GRAVITY_REGISTRY["SSB"              ] = SUN_JUPITER

Gravity.GRAVITY_REGISTRY["JUPITER BARYCENTER"] = JUPITER_GALS
Gravity.GRAVITY_REGISTRY["SATURN BARYCENTER" ] = SATURN_TITAN
Gravity.GRAVITY_REGISTRY["URANUS BARYCENTER" ] = URANUS
Gravity.GRAVITY_REGISTRY["NEPTUNE BARYCENTER"] = NEPTUNE
Gravity.GRAVITY_REGISTRY["PLUTO BARYCENTER"  ] = PLUTO_CHARON

# Define these as attributes of the Gravity superclass
for name, value in Gravity.GRAVITY_REGISTRY.items():
    name = name.replace('+', '_').replace(' ', '_')
    setattr(Gravity, name, value)

################################################################################

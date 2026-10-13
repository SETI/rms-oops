##########################################################################################
# oops/lightsource/starsource.py
##########################################################################################
"""StarSource class, which defines a star as a fixed source of illumination."""

import numpy as np

from polymath         import Pair, Vector3
from oops._exceptions import OopsValueError
from oops.constants   import AU, RPD, RPS, SPD
from oops.lightsource import LightSource

_RADIANS_PER_MAS = RPS / 1000.
_SECONDS_PER_YEAR = 365.25 * SPD        # the Julian year of catalog proper motions


class StarSource(LightSource):
    """A star, or another distant object, as a source of illumination.

    A subclass of LightSource whose source is always a single direction in J2000
    coordinates, defined by a right ascension and declination, so `source_is_moving` is
    always False. The direction is barycentric and applies at the epoch;
    :meth:`photon_to_event` corrects it for the star's proper motion and for the parallax
    seen from the observer.

    Attributes:
        magnitude (float | None): The magnitude of the star; None if unknown.
        mag_type (str | None): The band of the magnitude, such as "V" for a visual
            magnitude; None if unknown.
        stellar_type (str | None): The stellar type or other descriptive type of the
            object, such as "A1Vm" or "Globular cluster"; None if unknown.
        parallax (float): The annual parallax of the star in milliarcseconds; zero for an
            object at infinite distance.
        proper_motion (Pair): The proper motion of the star in milliarcseconds per year,
            as (d(ra)/dt * cos(dec), d(dec)/dt).
        epoch (float): The time in seconds TDB at which the star is at the position
            given by its right ascension and declination.
    """

    def __init__(self, name, ra, dec, *, magnitude=None, mag_type=None, stellar_type=None,
                 parallax=0., proper_motion=None, epoch=0.):
        """Constructor for a StarSource.

        Parameters:
            name (str): Name of this source. It is registered in the Body dictionary
                under its key, as given by :meth:`~oops.lightsource.LightSource.as_key`.
            ra (float): The barycentric J2000 right ascension of the star at `epoch`, in
                degrees.
            dec (float): The barycentric J2000 declination of the star at `epoch`, in
                degrees.
            magnitude (float, optional): The magnitude of the star.
            mag_type (str, optional): The band of the magnitude, such as "V" for a visual
                magnitude.
            stellar_type (str, optional): The stellar type or other descriptive type of
                the object.
            parallax (float, optional): The annual parallax of the star in
                milliarcseconds; default zero.
            proper_motion (PairLike, optional): The proper motion of the star in
                milliarcseconds per year, as (d(ra)/dt * cos(dec), d(dec)/dt), the form in
                which catalogs give it; default zero.
            epoch (float, optional): The time in seconds TDB at which the star is at the
                position given by `ra` and `dec`; default zero, which is J2000.

        Raises:
            OopsTypeError: If `name` is not a string.
            OopsValueError: If `name` already names a Body, or if `ra` or `dec` is not a
                single value.
        """

        if np.shape(ra) != () or np.shape(dec) != ():
            raise OopsValueError('StarSource ra and dec must be single values, not '
                                 f'{ra!r} and {dec!r}')

        LightSource.__init__(self, name, (ra, dec))

        self.magnitude = magnitude
        self.mag_type = mag_type
        self.stellar_type = stellar_type
        self.parallax = parallax
        self.proper_motion = (Pair.ZEROS if proper_motion is None
                              else Pair.as_pair(proper_motion).as_readonly())
        self.epoch = epoch

        # The inputs in degrees are kept so that pickling can rebuild the star exactly
        self._ra_dec_degrees = (ra, dec)
        self._ra = ra * RPD
        self._dec = dec * RPD

        # The rates of (ra, dec) in radians per second, with the RA rate not scaled by
        # cos(dec), and the parallax in radians, for photon_to_event
        (pm_ra, pm_dec) = self.proper_motion.vals
        scale = _RADIANS_PER_MAS / _SECONDS_PER_YEAR
        self._ra_rate = pm_ra / np.cos(self._dec) * scale
        self._dec_rate = pm_dec * scale
        self._parallax = parallax * _RADIANS_PER_MAS

    def __getstate__(self):
        return (self.name,) + self._ra_dec_degrees + (self.magnitude, self.mag_type,
                                                      self.stellar_type, self.parallax,
                                                      self.proper_motion, self.epoch)

    def __setstate__(self, state):
        (name, ra, dec, magnitude, mag_type, stellar_type, parallax, proper_motion,
         epoch) = state
        self.__init__(name, ra, dec, magnitude=magnitude, mag_type=mag_type,
                      stellar_type=stellar_type, parallax=parallax,
                      proper_motion=proper_motion, epoch=epoch)

    def photon_to_event(self, event, *, derivs=False, guess=None, antimask=None,
                        quick=None, converge=None):
        """Solve for a photon arrival event from this star.

        The star's position is advanced by its proper motion from the epoch to the mean
        time of `event`, and the direction is then corrected for parallax using the
        position of each element of `event` relative to the Solar System Barycenter. The
        remaining parameters are accepted for compatibility with
        :meth:`LightSource.photon_to_event` and are unused.

        Parameters:
            event (Event): The event of the observation.
            derivs (bool, optional): True to retain the derivatives of `event` in the
                returned event. The time derivative is always retained. The arriving line
                of sight carries no derivatives.
            guess (ScalarLike, optional): Unused.
            antimask (numpy.ndarray | bool, optional): Unused.
            quick (dict | bool, optional): To override the configured default parameters
                for a :class:`~oops.path.QuickPath` or :class:`~oops.frame.QuickFrame`
                when locating the observer relative to the Solar System Barycenter; False
                to disable them.
            converge (dict, optional): Unused.

        Returns:
            tuple[None, Event]: (`departure_event`, `arrival_event`).

            * `departure_event`: Always None, because a star has no path.
            * `arrival_event`: A copy of the given `event`, carrying the arriving
              photon's line of sight.
        """

        if derivs:
            arrival = event.copy()
        else:
            arrival = event.wod.copy()

        # Proper motion, evaluated once at the mean time of the event
        if self._ra_rate or self._dec_rate:
            dt = arrival.time.wod.mean() - self.epoch
            los = Vector3.from_ra_dec_length(self._ra + self._ra_rate * dt,
                                             self._dec + self._dec_rate * dt,
                                             recursive=False)
        else:
            los = self.source

        # Parallax: the star sits at distance AU/tan(parallax) from the barycenter, so the
        # observer's offset from the barycenter, scaled by tan(parallax)/AU, shifts it
        if self._parallax:
            obs_pos = arrival.wrt_ssb(derivs=derivs, quick=quick).pos.wod
            los = (los - obs_pos * (np.tan(self._parallax) / AU)).unit()

        arrival.neg_arr_j2000 = los
        return (None, arrival)

##########################################################################################

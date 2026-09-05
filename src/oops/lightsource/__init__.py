##########################################################################################
# oops/lightsource/__init__.py
##########################################################################################
"""LightSource classes, which define a source of illumination."""

import numpy as np

from polymath       import Scalar, Pair, Vector3
from oops.body      import Body
from oops.constants import RPD
from oops.oops      import Oops
from oops.path      import Path


class LightSource(Oops):
    """A source of illumination, such as the Sun, a star, or a transmitter.
    """

    def __init__(self, name, source, weight=None):
        """Constructor for a LightSource.

        The source can have arbitrary shape, making it possible to describe an extended
        source via an array of nearby paths or directions.

        LightSource objects are stored by name in the Body registry, so they share the
        same name space as the Body class. This is necessary so they can be used as keys
        in Backplanes.

        Parameters:
            name (str): Name under which to register this source; it is converted to
                upper case.
            source (Path, str, Pair, or Vector3): The location of the source, as any of:

                * a Path or the ID of a registered Path, for a source that moves;
                * a Pair or any pair of values, interpreted as J2000 right ascension and
                  declination in degrees;
                * a Vector3 or any triple of values, defining a fixed direction in J2000
                  coordinates.

            weight (Scalar, optional): Relative weights along the given paths or
                directions, which must broadcast to the shape of `source`. This makes it
                possible to define an extended, non-uniform source of light and to
                retrieve a result that is integrated over the source. Default is uniform
                weighting.

        Raises:
            TypeError: If `name` is not a string.
            ValueError: If `name` is already the name of a Body, or if `source` is not
                one of the forms above.
        """

        # Check and validate the name
        if not isinstance(name, str):
            raise TypeError(f'LightSource name must be a string: {name!r}')

        self.name = name.upper()

        if Body.exists(self.name):
            thing = Body.lookup(self.name)
            if isinstance(thing, Body):
                raise ValueError(f'LightSource name is also a Body name: {self.name}')

        # Interpret the source. Each form is recognized explicitly rather than by trying
        # them in turn, so that an input matching none of them is rejected instead of
        # being forced into whichever interpretation happens to accept it first.
        if isinstance(source, (Path, str)):
            self.source = Path.as_primary_path(source)
            self.source_is_moving = True
        else:
            if isinstance(source, Pair):
                items = 2
            elif isinstance(source, Vector3):
                items = 3
            else:
                try:
                    values = np.asarray(source, dtype=np.float64)
                except (TypeError, ValueError):
                    values = np.array(0.)
                items = values.shape[-1] if values.ndim else 0

            if items == 2:
                (ra, dec) = Pair.as_pair(source).values * RPD
                self.source = Vector3.from_ra_dec_length(ra, dec, 1., recursive=False)
            elif items == 3:
                self.source = Vector3.as_vector3(source).unit()
            else:
                raise ValueError('LightSource source must be a Path, a path ID, an '
                                 f'(RA, dec) pair, or a line of sight: {source!r}')

            self.source_is_moving = False

        self.shape = self.source.shape

        # Interpret the weights
        if weight is not None:
            weight = Scalar.as_scalar(weight).broadcast_to(self.shape)
        else:
            weight = Scalar(1.).broadcast_to(self.shape)

        self.weight = weight.copy().mask_where(weight.mask, replace=0., remask=True)
        self.weight /= self.weight.sum()

        # Register as a Body
        Body.BODY_REGISTRY[self.name] = self

    def __getstate__(self):
        return (self.name, self.source, self.weight)

    def __setstate__(self, state):
        self.__init__(*state)

    def photon_to_event(self, event, derivs=False, guess=None, antimask=None, quick=None,
                        converge=None):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name, but for
        LightSources not identified with paths, the departure event is None.

        Parameters:
            event (Event): The event of the observation.
            derivs (bool, optional): True to propagate derivatives of the event position
                into the returned event. The time derivative is always retained.
            guess (Scalar, optional): An initial guess to use as the event time along the
                path; otherwise None. Should only be used if the event time was already
                returned from a similar calculation.
            antimask (numpy.ndarray or bool, optional): If not None, this is a boolean
                array to be applied to event times and positions. Only the indices where
                antimask=True will be used in the solution.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default :class:`~oops.path.QuickPath` and
                :class:`~oops.frame.QuickFrame` parameters. Use False to disable the use
                of QuickPaths and QuickFrames. The default quick dictionary is defined in
                config.py.
            converge (dict, optional): A dictionary of parameters to override the
                configured default path convergence parameters. The default configuration
                is defined in config.py. Convergence parameters are as follows:

                * `max_iterations` (int): The maximum number of iterations of Newton's
                  method to perform. It should almost never need to be > 6.
                * `dlt_precision` (float): Iteration stops when the largest change in
                  light travel time between one iteration and the next falls below this
                  threshold (in seconds).
                * `dlt_limit` (float): The maximum allowed absolute value of the change in
                  light travel time from the nominal range calculated initially. Changes
                  in light travel with absolute values larger than this limit are clipped.
                  This prevents the divergence of the solution in some cases.

        Returns:
            tuple[Event or None, Event]: (`departure_event`, `arrival_event`).

            * `departure_event`: The Event at the source that matches the light travel
              time to `event`. This is None for a LightSource defined by a fixed
              direction rather than by a path.
            * `arrival_event`: A copy of the given `event`, carrying the arriving
              photon's line of sight and light travel time.
        """

        if self.source_is_moving:
            return self.source.photon_to_event(event, derivs=derivs, guess=guess,
                                               antimask=antimask, quick=quick,
                                               converge=converge)

        if derivs:
            arrival = event.copy()
        else:
            arrival = event.wod.copy()

        arrival.neg_arr_j2000 = self.source
        return (None, arrival)

    def as_path(self):
        """This LightSource's path object if it has one; otherwise, None."""

        return (self.source if self.source_is_moving else None)


# Imported at the bottom because DiskSource subclasses LightSource, so this module must
# be fully defined before its subclass module can be imported.
from oops.lightsource.disksource import DiskSource

__all__ = ['LightSource', 'DiskSource']

##########################################################################################

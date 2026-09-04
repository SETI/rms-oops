##########################################################################################
# oops/lightsource.py
##########################################################################################

import numpy as np

from polymath       import Scalar, Pair, Vector3, Matrix3
from oops.body      import Body
from oops.constants import C, RPD, RPS
from oops.event     import Event
from oops.oops      import Oops
from oops.path      import Path


class LightSource(Oops):
    """Defines a source of illumination, such as the Sun, a star, or a radio transmitter
    on the Earth or a spacecraft.
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
            antimask (ndarray or bool, optional): If not None, this is a boolean
                array to be applied to event times and positions. Only the indices where
                antimask=True will be used in the solution.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in config.py.

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

##########################################################################################
##########################################################################################

class DiskSource(LightSource):
    """DiskSource is a subclass of LightSource that defines an extended, circular disk
    with uniform illumination.
    """

    def __init__(self, name, source, radius, size=11, compress=False):
        """Constructor for a DiskSource, a 2-D array representing a uniform, lit
        circular light source.

        Parameters:
            name (str): Name to register in the Body dictionary.
            source (Path, tuple, or Vector3): A Path, or a fixed direction in J2000
                coordinates given either as a (right ascension, declination) pair in
                degrees or as a single line of sight.
            radius (float): Radius of the source, in km for a path or in arcseconds for a
                fixed J2000 source.
            size (int, optional): Number of pixels along the side of the square array
                defining the lines of sight; default 11. Use an odd number so that the
                central pixel falls at the center of the source.
            compress (bool, optional): True to strip away the masked pixels outside the
                circle, which leaves about 20% fewer lines of sight to calculate and
                suits a case where the disk is only averaged over at the end. False, the
                default, keeps the source as a 2-D image for later use.

        Raises:
            ValueError: If `source` does not describe a single, unshaped direction.
        """

        # Start with the default LightSource
        lightsource = LightSource(name, source)

        # Make sure this one is un-shaped
        if lightsource.shape != ():
            del Body.BODY_REGISTRY[lightsource.name]
            raise ValueError('DiskSource source must have shape (), not ' +
                             str(lightsource.shape))

        # At this point, the LightSource internals are filled in and valid
        self.name = lightsource.name
        self.source = lightsource.source
        self.source_is_moving = lightsource.source_is_moving

        # Create a masked, circular array of vectors in the X/Y plane, inside
        # unit radius
        array = np.zeros((size,size,3))
        xy = (np.arange(size) - (size-1)/2.) / (size/2.)
        array[:,:,0] = xy[np.newaxis]
        array[:,:,1] = xy[:,np.newaxis]
        mask = (array[:,:,0]**2 + array[:,:,1]**2) > 1.

        if compress:
            self.xy_grid = Vector3(array[~mask,:])
            mask = np.zeros(self.xy_grid.shape, dtype='bool')
        else:
            self.xy_grid = Vector3(array, mask=mask)

        self.shape = self.xy_grid.shape
        self.radius = radius

        # For a fixed line of sight, rotate and scale the vectors now
        if not self.source_is_moving:
            self.radius *= RPS
            matrix = Matrix3.twovec(lightsource.source, 2, Vector3.YAXIS, 1)
            self.xy_grid = matrix * (Vector3.ZAXIS + self.radius * self.xy_grid)

        # Define the weights
        weight = 1. - np.asarray(mask, dtype=np.float64)
        self.weight = Scalar(weight / np.sum(weight))

        # Re-register as a Body
        Body.BODY_REGISTRY[self.name] = self

    def photon_to_event(self, event, derivs=False, guess=None, antimask=None, quick=None,
                        converge=None):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name. The disk
        spreads the arrival direction across an array of lines of sight, so no single
        departure event corresponds to them and the departure is always None.

        Parameters:
            event (Event): The event of the observation.
            derivs (bool, optional): True to propagate derivatives of the event position
                into the returned event. The time derivative is always retained.
            guess (Scalar, optional): An initial guess to use as the event time along the
                path; otherwise None. Should only be used if the event time was already
                returned from a similar calculation.
            antimask (ndarray or bool, optional): If not None, this is a boolean
                array to be applied to event times and positions. Only the indices where
                antimask=True will be used in the solution.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in config.py.

        Returns:
            tuple[None, Event]: (`departure_event`, `arrival_event`).

            * `departure_event`: Always None, because the arrival directions are spread
              across the disk rather than originating from a single point.
            * `arrival_event`: A copy of the given `event`, carrying the arriving
              photon's line of sight spread across the disk.
        """

        if self.source_is_moving:
            arrival = self.source.photon_to_event(event, derivs=derivs, guess=guess,
                                                  antimask=antimask, quick=quick,
                                                  converge=converge)[1]

            rad = self.radius / (C * arrival.arr_lt)
            los = arrival.neg_arr_ap_j2000
            matrix = Matrix3.twovec(los, 2, Vector3.YAXIS, 1)
            new_los = matrix * (Vector3.ZAXIS + rad * self.xy_grid)
            new_event = Event(event.time, event.state, event.origin, event.frame)
            new_event.neg_arr_ap_j2000 = new_los
            return (None, new_event)

        if derivs:
            new_event = event.copy()
        else:
            new_event = event.wod.copy()

        new_event.neg_arr_j2000 = self.xy_grid
        return (None, new_event)

##########################################################################################

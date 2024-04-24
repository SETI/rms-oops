################################################################################
# oops/lightsource.py: Classes LightSource and DiskSource
################################################################################

import numpy as np

from polymath       import Scalar, Pair, Vector, Vector3, Matrix3
from oops.body      import Body
from oops.path      import Path
from oops.constants import C, RPD, RPS
from oops.event     import Event

class LightSource(object):
    """Defines a source of illumination, such as the Sun, a star, or a radio
    transmitter on the Earth or a spacecraft.
    """

    #===========================================================================
    def __init__(self, name, source, weight=None):
        """Constructor for a LightSource. It can be specified as:
            - a path.
            - a Pair representing J2000 right ascension and declination values
              in degrees.
            - a Vector3 defining a fixed direction in J2000 coordinates.

        Note that the source can have arbitrary shape, making it possible to
        describe an extended source via an array of nearby paths or directions.

        Optionally, weight is a Scalar providing relative weights along the
        given paths or directions. If provided, it must be possible to broadcast
        the shape of the weight to match that of the source. This makes it
        possible to define an extended, non-uniform source of light and to
        retrieve a result that is integrated over the source.

        LightSource objects are stored by name in the Body registry, so they
        share the same name space as the Body class. This is necessary so they
        can be used as keys in Backplanes.
        """

        # Check and validate the name
        if not isinstance(name, str):
            raise TypeError('LightSource name must be a string: ' + str(name))

        self.name = name.upper()

        if Body.exists(self.name):
            thing = Body.lookup(self.name)
            if isinstance(thing, Body):
                raise ValueError('LightSource name is also a Body name: ' +
                                 self.name)

        # Interpret source as (ra,dec)
        self.source = None
        try:
            pair = Pair.as_pair(source)
        except (ValueError, TypeError):
            pass
        else:
            (ra, dec) = pair.values * RPD
            self.source = Vector3.from_ra_dec_length(ra, dec, 1.,
                                                     recursive=False)
            self.source_is_moving = False

        # Interpret the source as a Vector3
        if self.source is None:
            try:
                self.source = Vector3.as_vector3(source).unit()
                self.source_is_moving = False
            except (ValueError, TypeError):
                pass

        # Interpret the source as a path
        if self.source is None:
            self.source = Path.as_primary_path(source)
            self.source_is_moving = True

        self.shape = self.source.shape

        # Interpret the weights
        if weight:
            weight = Scalar.as_scalar(weight).broadcast_into_shape(self.shape)
        else:
            weight = Scalar(1.).broadcast_into_shape(self.shape)

        self.weight = weight.copy().mask_where(weight.mask, replace=0.,
                                               remask=True)
        self.weight /= self.weight.sum()

        # Register as a Body
        Body.BODY_REGISTRY[self.name] = self

    def __getstate__(self):
        return (self.name, self.source, self.weight)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def photon_to_event(self, event, derivs=False, guess=None,
                              antimask=None, quick={}, converge={}):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name, but
        for LightSources not identified with paths, the departure event is None.

        Input:
            event       the event of the observation.

            derivs      True to propagate derivatives of the event position into
                        the returned event. The time derivative is always
                        retained.

            guess       an initial guess to use as the event time along the
                        path; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            antimask    if not None, this is a boolean array to be applied to
                        event times and positions. Only the indices where
                        antimask=True will be used in the solution.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         arrival.

            arrival     a copy of the given event, with the photon arrival or
                        departure line of sight and light travel time filled in.

            These subfields and derivatives are defined:
                    arr         direction of the arriving photon at the path.
                    arr_lt      (negative) light travel time from the link
                                event.

        Convergence parameters are as follows:
            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. This prevents the
                        divergence of the solution in some cases.
        """

        if self.source_is_moving:
            return self.source.solve_photon(event, -1, derivs=derivs,
                                            guess=guess, antimask=antimask,
                                            quick=quick, converge=converge)

        if derivs:
            arrival = event.copy()
        else:
            arrival = event.wod.copy()

        arrival.neg_arr_j2000 = self.source
        return (None, arrival)

    #===========================================================================
    def as_path(self):
        """This LightSource's path object if it has one; otherwise, None."""

        return (self.source if self.source_is_moving else None)

################################################################################
################################################################################

class DiskSource(LightSource):
    """DiskSource is a subclass of LightSource that defines an extended,
    circular disk with uniform illumination.
    """

    def __init__(self, name, source, radius, size=11, compress=False):
        """Constructor for a DiskSource. This is a 2-D array respresenting a
        uniform, lit circular light source.

        Inputs:
            name        name to register in the Body dictionary.
            source      a Path object or a fixed direction in J2000 coordinates,
                        defined by a (right ascension, declination) pair in
                        degrees, or else by a single Vector3 line of sight.
            radius      radius of the source, in km for paths or in arcseconds
                        for a J2000 fixed source.
            size        number of pixels on the side of the square 2-D array
                        containing defining the lines of sight. Use an odd
                        number to ensure that the central pixel corresponds to
                        the center of the source.
            compress    if True, the masked pixels (outside the circle) of the
                        source are stripped away. This results in ~20% fewer
                        lines of sight to calculate, and is appropriate when one
                        is only going to average across the disk at the end. If
                        false, the source remains a 2-D image that can be used
                        later.
        """

        # Start with the default LightSource
        lightsource = LightSource(name, source)

        # Make sure this one is un-shaped
        if lightsource.shape != ():
            del Body.BODY_REGISTRY[lightsource.name]
            raise ValueError('DiskSource source must have shape (): ' +
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
            self.xy_grid = Vector3(array[mask,:])
            mask = np.zeros(self.xy_grid.shape, dtype='bool')
        else:
            self.xy_grid = Vector3(array, mask=mask)

        self.shape = self.xy_grid.shape
        self.radius = radius

        # For a fixed line of sight, rotate and scale the vectors now
        if not self.source_is_moving:
            self.radius *= RPS
            matrix = Matrix3.twovec(lightsource.source, 2, Vector3.YAXIS, 1)
            self.xy_grid = matrix * (Vector.ZAXIS + self.radius * self.xy_grid)

        # Define the weights
        self.weights = (1. - np.asfarray(mask))
        self.weights /= np.sum(self.weights)

        # Re-register as a Body
        Body.BODY_REGISTRY[self.name] = self

    #===========================================================================
    def photon_to_event(self, event, derivs=False, guess=None,
                              antimask=None, quick={}, converge={}):
        """Solve for a photon arrival event from this lightsource.

        Input parameters are identical to the Path method of the same name, but
        only the arrival event is returned.

        Input:
            event       the event of the observation.

            derivs      True to propagate derivatives of the event position into
                        the returned event. The time derivative is always
                        retained.

            guess       an initial guess to use as the event time along the
                        path; otherwise None. Should only be used if the event
                        time was already returned from a similar calculation.

            antimask    if not None, this is a boolean array to be applied to
                        event times and positions. Only the indices where
                        antimask=True will be used in the solution.

            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         arrival.

            arrival     a copy of the given event, with the photon arrival or
                        departure line of sight and light travel time filled in.

            These subfields and derivatives are defined:
                    arr         direction of the arriving photon at the path.
                    arr_lt      (negative) light travel time from the link
                                event.

        Convergence parameters are as follows:
            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. This prevents the
                        divergence of the solution in some cases.
        """

        if self.source_is_moving:
            arrival = self.source.solve_photon(event, -1, derivs, guess,
                                               antimask, quick, converge)[1]

            rad = self.radius / (C * self.arr_lt)
            los = arrival.neg_arr_ap_j2000
            matrix = Matrix3.twovec(los, 2, Vector3.YAXIS, 1)
            new_los = matrix * (Vector3.ZAXIS + rad * self.xy_grid)
            new_event = Event(event.time, event.state, event.path, event.frame)
            new_event.neg_arr_ap_j2000 = new_los
            return new_event

        if derivs:
            new_event = event.copy()
        else:
            new_event = event.wod.copy()

        new_event.neg_arr_j2000 = self.source
        return arrival

################################################################################

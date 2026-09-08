##########################################################################################
# oops/observation/insitu.py
##########################################################################################

from polymath             import Scalar
from oops.cadence         import Cadence
from oops.cadence.instant import Instant
from oops.fov.nullfov     import NullFOV
from oops.frame           import Frame
from oops.observation     import Observation
from oops.path            import Path


class InSitu(Observation):
    """An Observation with timing and path information but no pointing.

    A subclass of :class:`~oops.Observation` with no attributes related to pointing or to
    the direction of an incoming photon.

    It can be useful for describing in situ measurements. InSitu observations can also be
    used to evaluate gridless backplanes, which do not require directional information.
    """

    def __init__(self, cadence, path, **subfields):
        """Constructor for an InSitu observation.

        Parameters:
            cadence (Cadence): Object defining the time and duration of each
                "measurement". Note that the shape of the cadence defines the dimensions
                of the observation. As a special case, a Scalar value is converted to a
                Cadence of subclass Instant, making this observation suitable for
                evaluating any gridless backplane. The shape of this cadence defines the
                shape of the observation.
            path (Path): The path waypoint co-located with the observer.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed.

        Raises:
            TypeError: If `cadence` is neither a Cadence nor a Scalar.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.J2000

        # FOV
        self.fov = NullFOV()

        # Cadence
        if isinstance(cadence, Cadence):
            self.cadence = cadence
        elif isinstance(cadence, Scalar):
            self.cadence = Instant(cadence)
        else:
            raise TypeError('Invalid cadence class: ' + type(cadence).__name__)

        # Axes / Shape / Size
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False
        self.uv_shape = (1,1)
        self.shape = self.cadence.shape
        self.t_axis = list(range(len(self.shape)))

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        self.refresh()
        return (self.cadence, self.path, self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        return InSitu(self.cadence.time_shift(dtime), self.path, **self.subfields)

##########################################################################################

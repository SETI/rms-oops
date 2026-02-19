################################################################################
# oops/observation/insitu.py: Subclass InSitu of class Observation
################################################################################

import numpy as np

from polymath             import Scalar
from oops.cadence         import Cadence
from oops.cadence.instant import Instant
from oops.fov.nullfov     import NullFOV
from oops.frame           import Frame
from oops.observation     import Observation
from oops.path            import Path
import oops.mutable as mutable


class InSitu(Observation):
    """NOTE: This is still a work in progress. Not yet tested. Do not use.

    InSitu is a subclass of Observation that has timing and path information,
    but no attributes related to pointing or incoming photon direction. It can
    be useful for describing in situ measurements.

    InSitu Observations can also be used to evaluate gridless backplanes, which
    do not require directional information.
    """

    #===========================================================================
    def __init__(self, cadence, path, **subfields):
        """Constructor for an InSitu observation.

        Input:
            cadence     a Cadence object defining the time and duration of each
                        "measurement". Note that the shape of the cadence
                        defines the dimensions of the observation. As a special
                        case, a Scalar value is converted to a Cadence of
                        subclass Instant, making this observation suitable for
                        evaluating any gridless backplane. The shape of this
                        cadence defines the shape of the observation.

            path        the path waypoint co-located with the observer.

            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
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
        self.uv_shape = (1, 1)
        self.shape = self.cadence.shape
        self.t_axis = list(np.range(len(self.shape)))

        # Shape / Size
        self.shape = self.cadence.shape
        self.uv_shape = (1,1)

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        mutable.refresh(self)
        return (self.cadence, self.path, self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        mutable.freeze(self)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a shallow copy of the object with a new time.
        """

        return InSitu(self.cadence.time_shift(dtime), self.path,
                      **self.subfields)

################################################################################

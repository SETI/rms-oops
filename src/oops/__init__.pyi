##########################################################################################
# oops/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.

Each import uses the redundant `X as X` form, which is how a stub marks a name as
re-exported rather than merely imported for internal use. The subpackages are re-exported
the same way, because importing one binds it as an attribute of `oops` itself.
"""

import oops.backplane as backplane
import oops.body as body
import oops.cadence as cadence
import oops.calibration as calibration
import oops.config as config
import oops.constants as constants
import oops.event as event
import oops.fittable as fittable
import oops.fov as fov
import oops.frame as frame
import oops.gravity as gravity
import oops.meshgrid as meshgrid
import oops.mutable as mutable
import oops.observation as observation
import oops.observation as obs
import oops.path as path
import oops.spice_support as spice
import oops.surface as surface
import oops.transform as transform

from oops.backplane import Backplane as Backplane
from oops.body import Body as Body
from oops._cache import _Cache as _Cache
from oops.cadence import Cadence as Cadence
from oops.calibration import Calibration as Calibration
from oops.constants import (AU as AU, C as C, C_INVERSE as C_INVERSE, DPR as DPR,
                            HALFPI as HALFPI, PI as PI, RPD as RPD, RPS as RPS,
                            SPD as SPD, SPR as SPR, TWOPI as TWOPI)
from oops.event import Event as Event
from oops.fittable import Fittable as Fittable
from oops.fov import FOV as FOV
from oops.frame import Frame as Frame
from oops.gravity import Gravity as Gravity
from oops.meshgrid import Meshgrid as Meshgrid
from oops.observation import Observation as Observation
from oops.path import Path as Path
from oops.surface import Surface as Surface
from oops.transform import Transform as Transform
from polymath import (Boolean as Boolean, Matrix as Matrix, Matrix3 as Matrix3,
                      Pair as Pair, Quaternion as Quaternion, Qube as Qube,
                      Scalar as Scalar, Vector as Vector, Vector3 as Vector3)

__all__ = ['cadence', 'calibration', 'fov', 'gravity', 'frame', 'observation', 'path',
           'surface', 'obs', 'backplane', 'body', 'event', 'fittable',
           'meshgrid', 'transform', 'Cadence', 'Calibration', 'FOV', 'Gravity', 'Frame',
           'Observation', 'Path', 'Surface', 'Backplane', 'Body', 'Event',
           'Fittable', 'Meshgrid', 'Transform', 'constants', 'spice', 'config', 'mutable',
           'C', 'C_INVERSE', 'RPD', 'DPR', 'SPR', 'RPS', 'SPD', 'AU', 'PI', 'TWOPI',
           'HALFPI', 'Boolean', 'Matrix', 'Matrix3', 'Pair', 'Quaternion', 'Qube',
           'Scalar', 'Vector', 'Vector3']

##########################################################################################

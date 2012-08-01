################################################################################
# oops/__init__.py
#
# 3/8/12 Created by MRS.
# 6/28/12 MRS - Added "import cspice" to solve the conflict between cspice and
#   MKL functions inconveniently both named dpstrf_().
################################################################################

# Examples of import statements and how they work:
#
# >>> import oops
#   This imports the entire oops object tree (but not instruments), creating
#   classes such as oops.Scalar, ops.Event, oops.path.SpicePath, plus other
#   components such as oops.registry and oops.config. This is the recommended
#   form of import.
#
# >>> import oops as abc
#   Imports the ENTIRE oops object tree but with the prefix "abc" replacing
#   "oops".
#
# >>> from oops import *
#   This imports all the oops components without the "oops" prefix. It can fill
#   up the default name space but there is nothing wrong with it.
#
# >>> import oops.array
#   This statement imports the entire object tree, not just the requested
#   component. This may be counterintuitive, although importing only selected
#   components of oops would seem to be of little value.
#
# >>> import oops.array as arr
#   This statement imports Array and its subclasses with the given alias, e.g.,
#   arr.Array, arr.Scalar, etc. It does NOT import the rest of oops. This can
#   be a handy way to assign an alias to components of oops, but one should
#   also import the remaining components in a separate import statement.
#
# >>> from oops.array import *
#   This statement imports the selected component of oops with the "oops"
#   prefix, e.g., Array, Scalar, Empty. Does not import the remainder of oops.
#
# >>> from oops import surface, path
#   This statement imports the selected components of oops without the "oops"
#   prefix, e.g., surface.Spheroid and path.SpicePath. It does not import the
#   remainder of oops.

import cspice       # This is CRITICAL to avoid the MKL error in calls to
                    # dpstrf(). Somehow, this ensures that the cspice function
                    # overrides the MKL function of the same name.

from oops_.array.all import *

import oops_.cadence.all  as cadence
import oops_.calib.all    as calib
import oops_.cmodel.all   as cmodel
import oops_.format.all   as format
import oops_.fov.all      as fov
import oops_.frame.all    as frame
import oops_.obs.all      as obs
import oops_.path.all     as path
import oops_.surface.all  as surface

from oops_.backplane  import *
from oops_.body       import *
from oops_.constants  import *
from oops_.edelta     import *
from oops_.event      import *
from oops_.meshgrid   import *
from oops_.registry   import *
from oops_.transform  import *
from oops_.units      import *

import oops_.spice_support as spice
import oops_.config as config

################################################################################

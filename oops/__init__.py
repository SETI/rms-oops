################################################################################
# oops/__init__.py
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

# import cspice           # This is CRITICAL to avoid the MKL error in calls to
#                         # dpstrf(). Somehow, this ensures that the cspice
#                         # function overrides the MKL function of the same name.

import oops.cadence_.all as cadence
import oops.calib_.all   as calib
import oops.fov_.all     as fov
import oops.frame_.all   as frame
import oops.obs_.all     as obs
import oops.path_.all    as path
import oops.surface_.all as surface

from oops.backplane   import *
from oops.body        import *
from oops.constants   import *
from oops.event       import *
from oops.fittable    import *
from oops.meshgrid    import *
from oops.transform   import *

import oops.spice_support as spice
import oops.config        as config
import oops.utils as utils

################################################################################

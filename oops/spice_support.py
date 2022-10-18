################################################################################
# oops/spice_support.py
################################################################################

import numbers
import cspyce
import spicedb

from oops.path import Path

# Maintain dictionaries that translate names in the SPICE toolkit into their
# corresponding names in the Frame and Path registries.

FRAME_TRANSLATION = {'J2000':'J2000', cspyce.namfrm('J2000'):'J2000'}
PATH_TRANSLATION = {'SSB':'SSB', 0:'SSB', 'SOLAR SYSTEM BARYCENTER':'SSB'}

################################################################################
# Useful SPICE support utilities
################################################################################

LSK_LOADED = False

def load_leap_seconds():
    """Load the most recent leap seconds kernel if it was not already loaded.
    """

    global LSK_LOADED

    if LSK_LOADED:
        return

    # Furnish the LSK to the SPICE toolkit
    spicedb.open_db()
    _ = spicedb.furnish_lsk(fast=True)
    spicedb.close_db()

    LSK_LOADED = True

########################################

def body_id_and_name(arg):
    """Interpret the argument as the name or ID of a SPICE body."""

    # First see if the path is already registered
    try:
        path = Path.as_primary_path(PATH_TRANSLATION[arg])
        if path.path_id == 'SSB':
            return (0, 'SSB')

        if type(path).__name__ != 'SpicePath':
            raise TypeError('a SpicePath cannot originate from a ' +
                            type(path).__name__)

        return (path.spice_target_id, path.spice_target_name)
    except KeyError:
        pass

    # Interpret the argument given as a string
    if isinstance(arg, str):
        body_id = cspyce.bodn2c(arg)    # raises LookupError if not found
        name = cspyce.bodc2n(body_id)
        return (body_id, name)

    # Otherwise, interpret the argument given as an integer
    elif isinstance(arg, numbers.Integral):
        try:
            name = cspyce.bodc2n(arg)
        except LookupError:
            # In rare cases, a body has no name; use the ID instead
            name = str(arg)

        return (arg, name)

    else:
        raise LookupError('invalid SPICE body: %s' % str(arg))

########################################

def frame_id_and_name(arg):
    """The spice_id and spice_name of a name/ID/SPICE frame."""

    # Interpret the SPICE frame ID as an int
    if isinstance(arg, numbers.Integral):
        try:
            name = cspyce.frmnam(arg)   # does not raise an error; I may fix
        except ValueError:
            name = ''
        except KeyError:
            name = ''

        # If the int is recognized as a frame ID, return it
        if name != '':
            return (arg, name)

        # Make sure the body's frame is defined
        if not cspyce.bodfnd(arg, 'POLE_RA'):
            raise LookupError('frame for body %d is undefined' % arg)

        # Otherwise, perhaps it is a body ID
        return cspyce.cidfrm(arg)   # LookupError if not found

    # Interpret the argument given as a string
    if isinstance(arg, str):

        # Validate this as the name of a frame
        try:
            frame_id = cspyce.namfrm(arg)   # does not raise an error; I may fix
        except ValueError:
            frame_id = 0
        except KeyError:
            frame_id = 0

        # If a nonzero ID is found...
        if frame_id != 0:

            # Make sure the frame is defined
            body_id = cspyce.frinfo(frame_id)[0]
            if (body_id > 0) and not cspyce.bodfnd(body_id, 'POLE_RA'):
                raise LookupError('frame "%s" is undefined' % arg)

            # Return the official, capitalized name
            return (frame_id, cspyce.frmnam(frame_id))

        # See if this is the name of a body
        body_id = cspyce.bodn2c(arg)        # raises LookupError if not found

        # Make sure the body's frame is defined
        if not cspyce.bodfnd(body_id, 'POLE_RA'):
            raise LookupError('frame for body "%s" is undefined' % arg)

        # If this is a body, return the name of the associated frame
        return cspyce.cidfrm(body_id)

########################################

def initialize():
    global FRAME_TRANSLATION, PATH_TRANSLATION

    FRAME_TRANSLATION = {'J2000':'J2000', cspyce.namfrm('J2000'):'J2000'}
    PATH_TRANSLATION = {'SSB':'SSB', 0:'SSB', 'SOLAR SYSTEM BARYCENTER':'SSB'}

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_tools(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

################################################################################
# oops/spice_support.py
################################################################################

import spicedb, cspice, julian
import os

from oops.path_.path import Path

# Maintain dictionaries that translates names in SPICE toolkit into their
# corresponding names in the Frame and Path registries.

FRAME_TRANSLATION = {'J2000':'J2000', cspice.namfrm('J2000'):'J2000'}
PATH_TRANSLATION = {'SSB':'SSB', 0:'SSB', 'SOLAR SYSTEM BARYCENTER':'SSB'}

################################################################################
# Useful SPICE support utilities
################################################################################

LSK_LOADED = False

def load_leap_seconds():
    """Load the most recent leap seconds kernel if it was not already loaded.
    """

    global LSK_LOADED

    if LSK_LOADED: return

    # Furnish the LSK to the SPICE toolkit
    spicedb.open_db()
    lsk = spicedb.furnish_lsk(fast=True)
    spicedb.close_db()

    LSK_LOADED = True

########################################

def body_id_and_name(arg):
    """Intepret the argument as the name or ID of a SPICE body or SPICE body.
    """

    # First see if the path is already registered
    try:
        path = Path.as_primary_path(PATH_TRANSLATION[arg])
        if path.path_id == 'SSB': return (0, 'SSB')

        if type(path).__name__ != 'SpicePath':
            raise TypeError('a SpicePath cannot originate from a ' +
                            type(path).__name__)

        return (path.spice_target_id, path.spice_target_name)
    except KeyError: pass

    # Interpret the argument given as a string
    if type(arg) == str:
        id = cspice.bodn2c(arg)     # raises LookupError if not found
        name = cspice.bodc2n(id)
        return (id, name)

    # Otherwise, interpret the argument given as an integer
    elif type(arg) == int:
        try:
            name = cspice.bodc2n(arg)
        except LookupError:
            # In rare cases, a body has no name; use the ID instead
            name = str(arg)

        return (arg, name)

    else:
        raise LookupError('invalid SPICE body: %s' % str(arg))

########################################

def frame_id_and_name(arg):
    """Return the spice_id and spice_name of a name/ID/SPICE frame."""

    # Interpret the SPICE frame ID as an int
    if type(arg) == type(0):
        try:
            name = cspice.frmnam(arg)   # does not raise an error; I may fix
        except ValueError:
            name = ''
        except KeyError:
            name = ''

        # If the int is recognized as a frame ID, return it
        if name != '': return (arg, name)

        # Otherwise, perhaps it is a body ID
        return cspice.cidfrm(arg)       # raises LookupError if not found

    # Interpret the argument given as a string
    if type(arg) == type(""):

        # Validate this as the name of a frame
        try:
            id = cspice.namfrm(arg)     # does not raise an error; I may fix
        except ValueError:
            id = 0
        except KeyError:
            id = 0

        # If a nonzero ID is found, return the official, capitalized name
        if id != 0: return (id, cspice.frmnam(id))

        # See if this is the name of a body
        id = cspice.bodn2c(arg)         # raises LookupError if not found

        # If so, return the name of the associated frame
        return cspice.cidfrm(id)

########################################

def initialize():
    global FRAME_TRANSLATION, PATH_TRANSLATION

    FRAME_TRANSLATION = {'J2000':'J2000', cspice.namfrm('J2000'):'J2000'}
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

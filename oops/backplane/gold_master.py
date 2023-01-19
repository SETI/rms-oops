################################################################################
# oops/backplane/gold_master.py: Backplane gold master tester and support.
################################################################################
"""\
########################################################################
# How to use with the Python unittest module for a host or instrument...
########################################################################

import unittest
import oops.backplane.gold_master as gm

class Test_<your test name>(unittest.TestCase):

    def runTest(self):

        gm.execute_as_unittest(self,
                obspath = 'file path inside the test_data directory',
                module  = 'hosts.xxx.yyy',
                planet  = 'SATURN',             # for example
                moon    = 'ENCELADUS',          # for example
                ring    = 'SATURN_MAIN_RINGS',  # for example, optional
                kwargs  = {},                   # other from_file inputs
                <overrides of any default gold_master input arguments>)

########################################################################
# How to have a gold master tester program dedicated to an instrument...
########################################################################

import oops.backplane.gold_master as gm

# Define the default observation for testing; note that this can be
# overridden in the command line.

gm.set_default_obs(obspath = 'file path inside the test_data directory',
                   module = 'hosts.xxx.yyy',
                   planet = 'SATURN'            # for example
                   moon   = 'ENCELADUS'         # for example
                   ring   = 'SATURN_MAIN_RINGS' # for example, optional
                   kwargs = {})                 # other from_file inputs

# Define any additional default parameters

gm.set_default_args(arg = default_value, ...)

if __name__ == 'main':
    args = gm.parse_args()
    gm.run_tests(args)

"""

import argparse
import datetime
import functools
import importlib
import logging
import numbers
import numpy as np
import os
import pickle
import PIL
import sys
import unittest

from collections   import namedtuple
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.ndimage import zoom as zoom_image

from polymath         import Boolean, Scalar, Qube
from oops.backplane   import Backplane
from oops.body        import Body
from oops.config      import LOGGING
from oops.meshgrid    import Meshgrid
from oops.observation import Observation

from oops.unittester_support import (OOPS_TEST_DATA_PATH,
                                     OOPS_TEST_DATA_PATH_,
                                     OOPS_GOLD_MASTER_PATH,
                                     OOPS_BACKPLANE_OUTPUT_PATH)

################################################################################
# The "default defaults" are defined here. A call to set_default_args() can be
# used to replace them for some specific test.
#
# Use set_default_obs() to define the observation used for unit tests and as the
# default for a run from the command line.
################################################################################

DEFAULTS = {
    'planet'        : [],
    'moon'          : [],
    'ring'          : [],
    'task'          : 'compare',
    'tolerance'     : 1,
    'radius'        : 1,
    'ignore_missing': False,
    'suite'         : [],
    'du'            : 0.,
    'dv'            : 0.,
    'undersample'   : 16,
    'inventory'     : True,
    'border'        : 0,
    'browse'        : True,
    'zoom'          : 1,
    'browse_format' : 'png',
    'verbose'       : True,
    'log'           : True,
    'level'         : 'debug',
}

# Note that default values of output, info, convergence, diagnostics,
# performance, and platform cannot be overridden here.

# These are for the default unit test
DEFAULT_OBS = {
    'obspath'   : os.path.join(OOPS_TEST_DATA_PATH,
                               'cassini/ISS/W1573721822_1.IMG'),
    'module'    : 'hosts.cassini.iss',
    'planet'    : 'SATURN',
    'moon'      : 'EPIMETHEUS',
    'ring'      : 'SATURN_MAIN_RINGS',
    'kwargs'    : {},
}


def set_default_args(**options):
    """Set the default command-line arguments for a gold master test.

    Options:
        planet          name of a planet for which to generate backplane arrays,
                        or else a list of planet names.
        moon            name of a moon for which to generate backplane arrays,
                        or else a list of moon names.
        ring            name of a ring for which to generate backplane arrays,
                        or else a list of ring names. Backplane arrays are
                        always generated for the default ring of each planet.
        task            name of the default test to perform, one of "preview",
                        "compare", and "adopt"; default is "compare".
        tolerance       factor to apply to the defined error allowances for all
                        backplane arrays; default 1.
        radius          allowed radius in pixels for a possible spatial offset
                        between the gold master and the test array; default 1.
        ignore_missing  True to raise a warning on any missing gold masters;
                        False to raise an error. Default is False.
        suite           name or names of the default test suite(s) to run; use
                        [] (the default) to include all test suites.
        du, dv          pixel offsets to apply to the origin of the meshgrid,
                        for testing sensitivity to pointing offsets; default 0.
        undersample     undersample factor for backplane tests and browse
                        images; default 16.
        inventory       True to use an inventory when generating backplanes.
        border          Size of border for inventory; default 0.
        browse          True to save browse images; default True.
        zoom            zoom factor for browse images; default 1.
        browse_format   browse image format, one of "png", "jpg", or "tiff".
        output          default directory for saving test arrays and browse
                        products; default defined by environment variable
                        "OOPS_BACKPLANE_OUTPUT_PATH".
        verbose         True to print output to the terminal by default.
        log             True to save a log file by default.
    """

    global DEFAULTS

    for key, value in options:
        DEFAULTS[key] = value


def set_default_obs(obspath, module, planet, moon='', ring='', kwargs={}):
    """Set the details of the default observation to be used for the gold master
    test.

    These are the default observation file path and module to use if the they
    are not specified in the command line.

    The specified planet, moon, and ring are used as the defaults when the
    observation is unspecified, but can be overridden at the command line.

    Options:
        obspath         file path to the default data object to be used.
        module          name of the default module to import, e.g.,
                        "hosts.cassini.iss". This module must have a "from_file"
                        method.
        planet          name of the default planet, or list of planet names.
        moon            name of the default moon, if any, or list of moon names.
        ring            name of the default ring, if any, or list of ring names.
                        Backplane arrays are always generated for the full ring
                        plane of the specified planet.
        kwargs          an optional dictionary of keyword arguments to be passed
                        to from_file.
    """

    global DEFAULT_OBS

    DEFAULT_OBS['obspath'] = obspath
    DEFAULT_OBS['module']  = module
    DEFAULT_OBS['planet']  = planet
    DEFAULT_OBS['moon']    = moon
    DEFAULT_OBS['ring']    = ring
    DEFAULT_OBS['kwargs']  = kwargs


################################################################################
# Command line execution
################################################################################

def execute_as_command():
    """Parse command-line arguments for gold master testing of one or more
    backplanes and then run the tests.

    A "Namespace" object is returned, containing all of the command line
    attributes, plus:
        from_file       the "from_file" function of the selected module.
        abpaths         the list of absolute paths to the observations.
        backplane_tests the list of BackplaneTest objects.
    """

    # Define parser...
    parser = argparse.ArgumentParser(description='''Gold Master backplane
                                                    test utility.''')

    # Data objects
    gr = parser.add_argument_group('Data objects')
    gr.add_argument('obspath', type=str, nargs='*', metavar='filepath',
                    default=None,
                    help='''File path to the data object(s) to be used;
                            default is %s.'''
                            % repr(DEFAULT_OBS['obspath']))
    gr.add_argument('--module', type=str, default=None, metavar='hosts...',
                    help='''Name of the module containing the "from_file"
                            method for any file paths specified; default is
                            "%s".'''
                            % DEFAULT_OBS['module'])

    # Backplane targets
    gr = parser.add_argument_group('Backplane targets')
    gr.add_argument('-p', '--planet', type=str, nargs='*', metavar='name',
                    default=None,
                    help='''Name(s) of one or more planets for which to generate
                            backplane arrays; default is %s.'''
                         % repr(DEFAULT_OBS['planet']))
    gr.add_argument('-m', '--moon', type=str,nargs='*', metavar='name',
                    default=None,
                    help='''Name(s) of one or more moons for which to generate
                            backplane arrays; default is %s.'''
                         % repr(DEFAULT_OBS['moon']))
    gr.add_argument('-r', '--ring', type=str,nargs='*', metavar='name',
                    default=None,
                    help='''Name(s) of one or more rings for which to generate
                            backplane arrays. Arrays are always generated for
                            the default equatorial ring of the planet.
                            The default is %s.'''
                         % repr(DEFAULT_OBS['ring']))

    # Testing options
    gr = parser.add_argument_group('Testing options')
    gr.add_argument('--preview', dest='task', default='compare',
                    action='store_const', const='preview',
                    help='''Generate backplane arrays and browse images but do
                            not compare the arrays to the gold masters.''')
    gr.add_argument('-c', '--compare', dest='task',
                    action='store_const', const='compare',
                    help='''Generate backplane arrays and browse images and
                            compare the arrays to the gold masters.''')
    gr.add_argument('-a', '--adopt', dest='task',
                    action='store_const', const='adopt',
                    help='''Adopt these backplane arrays as the new gold
                            masters.''')
    gr.add_argument('--tolerance', metavar='TOL',
                    default=float(DEFAULTS['tolerance']),
                    help='''Factor to apply to backplane array error tolerances;
                            default %s.'''
                         % str(DEFAULTS['tolerance']))
    gr.add_argument('--radius', metavar='RAD',
                    default=float(DEFAULTS['radius']),
                    help='''Factor to apply to backplane array radial offset
                            limits; default %s.'''
                         % str(DEFAULTS['radius']))
    gr.add_argument('--ignore-missing', action='store_true',
                    default=float(DEFAULTS['ignore_missing']),
                    help='''Log a warning rather than an error if a gold
                            master backplane is missing.''')
    gr.add_argument('--suite', type=str, nargs='*', metavar='name',
                    default=[],
                    help='''Name(s) of the test suites to perform, e.g., "ring"
                            or "surface"; default is to perform all test suites.
                            ''')
    gr.add_argument('--du', type=float, default=0.,
                    help='''Offset to apply to the u-coordinate of the meshgrid
                            origin in units of pixels; default 0. This can be
                            useful for testing the tolerance to pointing
                            uncertainties.''')
    gr.add_argument('--dv', type=float, default=0.,
                    help='''Offset to apply to the v-coordinate of the meshgrid
                            origin in units of pixels; default 0. This can be
                            useful for testing the tolerance to pointing
                            uncertainties.''')

    # Backplane array options
    gr = parser.add_argument_group('Backplane array options')
    gr.add_argument('-u', '--undersample', type=int, metavar='N',
                    default=DEFAULTS['undersample'],
                    help='''Factor by which to undersample backplane arrays;
                            default %d.'''
                         % DEFAULTS['undersample'])
    gr.add_argument('--inventory', action='store_true',
                    default=DEFAULTS['inventory'],
                    help='Use a body inventory when generating backplane%s.'
                         % ' (default)' if DEFAULTS['inventory'] else '')
    gr.add_argument('--no-inventory', dest='inventory', action='store_false',
                    help='''Do not use a body inventory when generating
                            backplanes%s.'''
                         % '' if DEFAULTS['inventory'] else ' (default)')
    gr.add_argument('--border', type=int, metavar='N',
                    default=DEFAULTS['border'],
                    help='''Number of pixels by which to expand image borders
                            when constructing the inventory; default %d.'''
                         % DEFAULTS['border'])

    # Browse image options
    gr = parser.add_argument_group('Browse image options')
    gr.add_argument('--browse', action='store_true',
                    default=DEFAULTS['browse'],
                    help='Save browse images of the backplane arrays%s.'
                         % ' (default)' if DEFAULTS['browse'] else '')
    gr.add_argument('--no-browse', dest='browse', action='store_false',
                    default=DEFAULTS['browse'],
                    help='No not save browse images of backplane arrays%s.'
                         % '' if DEFAULTS['browse'] else ' (default)')
    gr.add_argument('--zoom', type=int, metavar='N', default=1,
                    help='Zoom factor to apply to browse images; default %d.'
                         % DEFAULTS['zoom'])
    gr.add_argument('--format', type=str, dest='browse_format', metavar='EXT',
                    default=DEFAULTS['browse_format'],
                    help='''Format for saving browse images, one of "png",
                            "jpg", or "tiff". Default is "%s".'''
                         % DEFAULTS['browse_format'])

    # Output options
    gr = parser.add_argument_group('Output options')
    gr.add_argument('-o', '--output', type=str, metavar='dir', default=None,
                    help='''Root directory for saved backplane arrays, browse
                            images, and logs; default is the value of the
                            environment variable OOPS_BACKPLANE_OUTPUT_PATH, if
                            defined, or else the current default directory.''')
    gr.add_argument('-v', '--verbose', action='store_true',
                    default=DEFAULTS['verbose'],
                    help='Write log information to the terminal%s.'
                         % ' (default)' if DEFAULTS['verbose'] else '')
    gr.add_argument('-q', '--quiet', dest='verbose', action='store_false',
                    help='Do not write log information to the terminal%s.'
                         % '' if DEFAULTS['verbose'] else ' (default)')
    gr.add_argument('--log', action='store_true',
                    default=DEFAULTS['log'],
                    help='Write a log file to the output directory%s.'
                         % ' (default)' if DEFAULTS['log'] else '')
    gr.add_argument('--no-log', dest='log', action='store_false', default=True,
                    help='Do not write a log file in the output directory%s.'
                         % '' if DEFAULTS['log'] else ' (default)')
    gr.add_argument('--level', type=str, metavar='LEVEL',
                    default=DEFAULTS['level'],
                    help='''Minimum level for messages to be logged: "debug",
                            "info", "warning", "error", or an integer; default
                            is %s.'''
                         % DEFAULTS['level'])
    gr.add_argument('--info', action='store_true', default=False,
                    help='Include array summary info in the log.')
    gr.add_argument('--convergence', action='store_true', default=False,
                    help='Show iterative convergence information in the log.')
    gr.add_argument('--diagnostics', action='store_true', default=False,
                    help='Include diagnostic information in the log.')
    gr.add_argument('--performance', action='store_true', default=False,
                    help='Include OOPS performance information in the log.')
    gr.add_argument('--platform', type=str, metavar='OS', default=None,
                    help='''Name of the OS, as a proxy for how to name output
                            files: "macos" for MacOS, "windows" for Windows,
                            "linux" for Linux; default is to derive the OS name
                            from the system where this program is running. Gold
                            master files always use Linux names.''')

    args = parser.parse_args()
    args.testcase = None
    args = _clean_up_args(args)
    run_tests(args)

################################################################################
# unittest module support
################################################################################

def execute_as_unittest(testcase, obspath, module, planet, moon=[], ring=[],
                        kwargs={}, **options):
    """Run the gold master test suites for one or more observations as a unit
    test.

    Inputs:
        testcase        the unittest TestCase object.
        obspath         file path to the default data object to be used.
        module          name of the default module to import, e.g.,
                        "hosts.cassini.iss". This module must have a "from_file"
                        method.
        planet          name of the default planet, or list of planet names.
        moon            name of the default moon, if any, or list of moon names.
        ring            name of the default ring, if any, or list of ring names.
                        Backplane arrays are always generated for the full ring
                        plane of the specified planet.
        kwargs          an optional dictionary of keyword arguments to be passed
                        to from_file.
        **options       overrides for any default gold_master input arguments.
    """

    global DEFAULTS

    # Set the default observation details
    set_default_obs(obspath, module, planet=planet, moon=moon, ring=ring,
                    kwargs={})

    # Initialize the command argument namespace
    args = argparse.Namespace()
    for key, value in DEFAULTS.items():
        setattr(args, key, value)

    # These values in the DEFAULTS dictionary are overridden
    args.browse = False
    args.log = False
    args.verbose = True

    # These have no entry in the DEFAULTS dictionary
    args.output = None
    args.info = False
    args.convergence = False
    args.diagnostics = False
    args.performance = False
    args.platform = None

    # Fill in any overrides
    for key, value in options.items():
        setattr(args, key, value)

    # These options are mandatory
    args.testcase = testcase
    args.task = 'compare'
    args.level = 'error'
    args.obspath = ''           # filled in by _clean_up_args

    # Clean up, also filling in observation, module, planet(s), moon(s), ring(s)
    args = _clean_up_args(args)
    run_tests(args)

#===============================================================================
def _clean_up_args(args):
    """Clean up arguments given in the command line."""

    global DEFAULTS, DEFAULT_OBS, TEST_SUITES

    # Use the default observation if necessary
    if not args.obspath:
        args.obspath = DEFAULT_OBS['obspath']
        args.module  = DEFAULT_OBS['module']
        if args.planet is None:
            args.planet = DEFAULT_OBS['planet']
        if args.moon is None:
            args.moon = DEFAULT_OBS['moon']
        if args.ring is None:
            args.ring = DEFAULT_OBS['ring']
    else:
        if isinstance(args.obspath, str):
            args.obspath = [args.obspath]
        if args.planet is None:
            args.planet = DEFAULTS['planet']
        if args.moon is None:
            args.moon = DEFAULTS['moon']
        if args.ring is None:
            args.ring = DEFAULTS['ring']

    # Clean up the inputs
    if isinstance(args.obspath, str):
        args.obspath = [args.obspath]
    if isinstance(args.planet, str):
        args.planet = [args.planet]
    if isinstance(args.moon, str):
        args.moon = [args.moon]
    if isinstance(args.ring, str):
        args.ring = [args.ring]

    if args.output is None:
        args.output = OOPS_BACKPLANE_OUTPUT_PATH

    if args.platform is None:
        args.platform = sys.platform
    else:
        args.platform = args.platform.lower()

    if args.browse_format not in ('png', 'jpg', 'tiff'):
        raise ValueError('unrecognized browse format: ' + args.browse_format)

    args.undersample = int(args.undersample)

    if not args.suite or args.task == 'adopt':  # use all suites for task adopt
        args.suite = list(TEST_SUITES.keys())
    elif isinstance(args.suite, str):
        args.suite = [args.suite]

    args.suite.sort()

    if args.task == 'adopt':    # required options for task adopt
        args.browse = True
        args.undersample = 1

    try:
        args.level = int(args.level)
    except ValueError:
        args.level = LOGGING.LEVELS[args.level.lower()]

    # Planet, moon and ring must be lists
    if args.planet:
        if isinstance(args.planet, str):
            args.planet = [args.planet]
    else:
        args.planet = []

    if args.moon:
        if isinstance(args.moon, str):
            args.moon = [args.moon]
    else:
        args.moon = []

    if args.ring:
        if isinstance(args.ring, str):
            args.ring = [args.ring]
    else:
        args.ring = []

    # Get the from_file method
    if not hasattr(args, 'module') or not args.module:
        args.module = DEFAULT_OBS['module']

    module = importlib.import_module(args.module)
    args.from_file = module.from_file

    # Get the absolute file paths
    args.abspaths = []
    for obspath in args.obspath:
        abspath = os.path.abspath(os.path.realpath(obspath))
        if args.task in ('compare', 'adopt'):
            if not OOPS_TEST_DATA_PATH:
                raise ValueError('Undefined environment variable: '
                                 + 'OOPS_TEST_DATA_PATH')
            if not abspath.startswith(OOPS_TEST_DATA_PATH_):
                raise ValueError('File is not in the test data directory: '
                                 + obspath)
        if not os.path.exists(abspath):
            raise FileNotFoundError('No such file: %s' % obspath)

        args.abspaths.append(abspath)

    # Create placeholders for the backplane surface names
    args.body_names = []
    args.limb_names = []
    args.ring_names = []
    args.ansa_names = []
    args.planet_moon_pairs = []     # list of (planet, moon) tuples
    args.planet_ring_pairs = []     # list of (planet, ring) tuples

    # Define the BackplaneTest objects
    args.backplane_tests = []
    for abspath in args.abspaths:
        result = args.from_file(abspath, **DEFAULT_OBS['kwargs'])
        if isinstance(result, Observation):
            bpt = BackplaneTest(result, args)
            args.backplane_tests.append(bpt)
        else:
            for k, obs in enumerate(result):
                bpt = BackplaneTest(obs, args, suffix='_' + str(k))
                args.backplane_tests.append(bpt)

    # Fill in all the backplane surface names
    for body in args.planet + args.moon:
        args.body_names.append(body)
        args.limb_names.append(body + ':LIMB')

        if Body.lookup(body).ring_body:
            args.ring_names.append(body + ':RING')
            args.ansa_names.append(body + ':ANSA')
            args.planet_ring_pairs.append((body, body + ':RING'))

    for moon in args.moon:
        planet = Body.lookup(moon).parent.name.upper()
        args.planet_moon_pairs.append((planet, moon))

    for ring in args.ring:
        args.ring_names.append(ring)
        args.ansa_names.append(ring + ':ANSA')

        planet = Body.lookup(ring).parent.name.upper()
        pair = (planet, ring)
        if pair not in args.planet_ring_pairs:
            args.planet_ring_pairs.append(pair)

    return args

#===============================================================================
def run_tests(args):
    """Run all the gold master tests."""

    logger = logging.Logger(__name__)
    if args.verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    LOGGING.set_logger(logger, level=args.level)
    LOGGING.set_stdout(False)
    LOGGING.set_stderr(False)
    LOGGING.set_file()
    LOGGING.all(args.convergence, category='convergence')
    LOGGING.all(args.diagnostics, category='diagnostics')
    LOGGING.all(args.performance, category='performance')

    LOGGING.reset()         # zero out error and warning counts
    try:
        for bpt in args.backplane_tests:
            bpt.run_tests()
    except Exception as e:
        LOGGING.exception(e)

    if LOGGING.errors > 0:
        if args.testcase is not None:
            args.testcase.assertTrue(False, 'gold_master tests FAILED')
        else:
            sys.exit(-1)

################################################################################
# Test suite management
#
# Each backplane module defines one or more functions that receive a
# BackplaneTest object as input, create various backplane arrays, and test them
# via calls to BackplaneTest.compare and BackplaneTest.gmtest.
#
# Once defined, they call register_test_suite(name, func) to register each test
# function within the Gold Master testing framework. Afterward, these tests will
# be included during unit testing.
################################################################################

TEST_SUITES = {}            # dictionary of all test suites

def register_test_suite(name, func):
    """Add the given function to the dictionary of exercise tests.

    This must be called for each defined test suite.
    """

    global TEST_SUITES

    TEST_SUITES[name] = func
    func.name = name            # add "name" attribute to the function itself

def get_test_suite(name):
    """Retrieve a test suite function given its name."""

    global TEST_SUITES

    return TEST_SUITES[name]

################################################################################
# BackplaneTest class
#
# This class manages information about the backplanes of a particular
# observation.
################################################################################

# namedtuples describing exercise results for a particular backplane array
_Comparison   = namedtuple('_Comparison', ['status', 'max_error', 'limit',
                                           'mask_errors', 'offset_errors',
                                           'size'])
_NoComparison = namedtuple('_NoComparison', ['status', 'pickle_path'])

TEST_SUITE = ''         # the name of the current test suite
LATEST_TITLE = ''       # used to track what was happening if an error occurs

SUMMARY_COMMENT = """\
# KEY:
#   Boolean backplanes:
#           (False values, True values, masked values, total values)
#   Floating-point backplanes:
#           (minimum, maximum, masked values, total values)
#       or, if all values are the same:
#           (value, masked values, total values)
#   Fully masked backplanes:
#           (None, masked values, total values)
"""

class BackplaneTest(object):
    """Class for managing information about the gold master tests of a specific
    observation.
    """

    def __init__(self, obs, args, suffix=''):
        """Construct a BackplaneTest for the given observation.

        Input:
            obs         Observation.
            args        A Namespace object containing the command line inputs.
            suffix      a suffix string used to distinguish between multiple
                        Observations all defined within the same data file; it
                        is appended to the array and browse directory names to
                        make them unique.
        """

        self.obs = obs
        self.args = args
        self.suffix = suffix

        self.upward = obs.fov.uv_scale.vals[1] < 0.     # direction of v-axis
        self.full_shape = obs.uv_shape[::-1] if obs.swap_uv else obs.uv_shape

        # Copy a few key args into self
        self.task         = args.task
        self.undersample  = args.undersample
        self.inventory    = args.inventory
        self.border       = args.border
        self.body_names   = args.body_names
        self.limb_names   = args.limb_names
        self.ring_names   = args.ring_names
        self.ansa_names   = args.ansa_names
        self.planet_moon_pairs = args.planet_moon_pairs
        self.planet_ring_pairs = args.planet_ring_pairs

        # Create backplane object
        self.meshgrid = obs.meshgrid(origin=(0.5 + self.args.du,
                                             0.5 + self.args.dv),
                                     undersample=int(self.undersample))
            # By setting origin to 0.5 and requiring undersampling to be
            # integral, we ensure that an undersampled meshgrid will always
            # sample the centers of pixels in the original (u,v) grid.

        if self.inventory:
            self.backplane = Backplane(obs, meshgrid=self.meshgrid,
                                            inventory={},
                                            inventory_border=self.border)
        else:
            self.backplane = Backplane(obs, meshgrid=self.meshgrid,
                                            inventory=None)

        # Determine file paths. Example:
        # filespec = $OOPS_TEST_DATA_PATH/cassini/ISS/N1460072401_1.IMG
        # masters: $OOPS_GOLD_MASTER_PATH/hosts.cassini.iss/ISS/N1460072401_1/
        # arrays: $OOPS_BACKPLANE_OUTPUT_PATH/N1460072401_1/arrays
        # browse: $OOPS_BACKPLANE_OUTPUT_PATH/N1460072401_1/browse

        self.abspath = os.path.abspath(os.path.realpath(obs.filespec))
        basename_prefix = os.path.splitext(os.path.basename(self.abspath))[0]

        self.gold_dir = os.path.join(OOPS_GOLD_MASTER_PATH,
                                     args.module, basename_prefix)
        self.gold_arrays = os.path.join(self.gold_dir, 'arrays' + self.suffix)
        self.gold_browse = os.path.join(self.gold_dir, 'browse' + self.suffix)

        self.output_dir = os.path.join(args.output, basename_prefix)
        self.output_arrays = os.path.join(self.output_dir,
                                          'arrays' + self.suffix)
        self.output_browse = os.path.join(self.output_dir,
                                          'browse' + self.suffix)

        # Initialize the comparison log
        self.gold_summary_ = None
        self.summary = {}
        self.results = {}

    ############################################################################
    # Test runner for one BackplaneTest
    ############################################################################

    def run_tests(self):
        """Run the complete suite of tests for this BackplaneTest."""

        global LATEST_TITLE, TEST_SUITE
        LATEST_TITLE = ''
        TEST_SUITE = ''

        # Set up diagnostics and performance logging
        Backplane.CONVERGENCE = self.args.convergence
        Backplane.DIAGNOSTICS = self.args.diagnostics
        Backplane.PERFORMANCE = self.args.performance

        # Re-initialize the comparison tracking
        self.gold_summary_ = None
        self.summary = {}
        self.results = {}

        # Set up the log handler; set aside any old log
        # Note that each BackplaneTest gets its own dedicated logger
        if self.args.log:
            log_path = os.path.join(self.output_dir, self.task + '.log')
            if os.path.exists(log_path):
                timestamp = os.path.getmtime(log_path)
                dt = datetime.datetime.fromtimestamp(timestamp)
                suffix = dt.strftime('-%Y-%m-%dT%H-%M-%S')
                dated_path = log_path[:-4] + suffix + '.log'
                os.rename(log_path, dated_path)

            handler = logging.FileHandler(log_path)
            LOGGING.logger.addHandler(handler)

        # Run the tests
        start = datetime.datetime.now()
        try:
            LOGGING.info('Beginning task ' + self.task)
            LOGGING.info('File: ' + self.abspath)
            if self.suffix:
                LOGGING.info('Suffix: ' + self.suffix)

            # Make sure test data and gold master files exist
            if self.task in ('compare', 'adopt'):
                if not OOPS_GOLD_MASTER_PATH:
                    LOGGING.info('Undefined environment variable: '
                                 + 'OOPS_GOLD_MASTER_PATH')
                    LOGGING.fatal('Undefined environment variable: '
                                  + 'OOPS_RESOURCES')
                    return

                if not OOPS_TEST_DATA_PATH:
                    LOGGING.info('Undefined environment variable: '
                                 + 'OOPS_TEST_DATA_PATH')
                    LOGGING.fatal('Undefined environment variable: '
                                 + 'OOPS_RESOURCES')
                    return

                if self.task == 'compare':
                    LOGGING.info('Reading masters from', self.gold_arrays)
                elif self.task == 'adopt':
                    LOGGING.info('Writing new masters to', self.gold_arrays)
                    LOGGING.info('Writing browse images to', self.gold_browse)

                os.makedirs(self.gold_arrays, exist_ok=True)
                os.makedirs(self.gold_browse, exist_ok=True)

            # Make sure directories exist; log their locations
            if self.task in ('preview', 'compare'):
                LOGGING.info('Writing arrays to', self.output_arrays)
                os.makedirs(self.output_arrays, exist_ok=True)

                if self.args.browse:
                    LOGGING.info('Writing browse images to', self.output_browse)
                    os.makedirs(self.output_browse, exist_ok=True)

                if not OOPS_BACKPLANE_OUTPUT_PATH:
                    LOGGING.info('   To change this destination, define '
                                 + 'OOPS_BACKPLANE_OUTPUT_PATH')

            # Run the tests...
            for key in self.args.suite:
                test_suite = get_test_suite(key)
                TEST_SUITE = key
                LATEST_TITLE = ''
                try:
                    test_suite(self)

                except Exception as e:
                    if LATEST_TITLE:
                        LOGGING.exception(e, 'Error in %s: %s'
                                             % (TEST_SUITE, LATEST_TITLE))
                    elif TEST_SUITE:
                        LOGGING.exception(e, 'Error in %s' % TEST_SUITE)
                    else:
                        LOGGING.exception(e)

            # Wrap up
            if self.task == 'preview':
                file_path = self.write_summary(self.output_dir)
                LOGGING.debug('Summary written: ' + file_path)
                if LOGGING.warnings:
                    LOGGING.info('Total warnings = ' + str(LOGGING.warnings))
                if LOGGING.errors:
                    LOGGING.info('Total errors = ' + str(LOGGING.errors))

            elif self.task == 'compare':
                file_path = self.write_summary(self.output_dir)
                LOGGING.debug('Summary written: ' + file_path)
                LOGGING.info('Total warnings = ' + str(LOGGING.warnings))
                LOGGING.info('Total errors = ' + str(LOGGING.errors))

            elif self.task == 'adopt':
                file_path = self.write_summary(self.gold_dir)
                LOGGING.debug('Summary written: ' + file_path)
                if LOGGING.warnings:
                    LOGGING.info('Total warnings = ' + str(LOGGING.warnings))
                if LOGGING.errors:
                    LOGGING.info('Total errors = ' + str(LOGGING.errors))

            # Diagnostics...
            if self.args.diagnostics:
                for i in (False, True):
                    LOGGING.diagnostic('\nSurface Events, derivs=%s' % i)
                    keys = list(self.backplane.surface_events[i].keys())
                    keys.sort()
                    for key in keys:
                        sum = np.sum(self.backplane.surface_events[i][key].mask)
                        LOGGING.diagnostic('   ', key, sum)

                for i in (False, True):
                    LOGGING.diagnostic('\nIntercepts, derivs=%s' % i)
                    keys = list(self.backplane.intercepts[i].keys())
                    keys.sort(key=BackplaneTest._sort_key)
                    for key in keys:
                        sum = np.sum(self.backplane.intercepts[i][key].mask)
                        LOGGING.diagnostic('   ', key, sum)

                LOGGING.diagnostic('\nGridless arrivals')
                keys = list(self.backplane.gridless_arrivals.keys())
                keys.sort(key=BackplaneTest._sort_key)
                for key in keys:
                    sum = np.sum(self.backplane.gridless_arrivals[key].mask)
                    LOGGING.diagnostic('   ', key, sum)

                LOGGING.diagnostic('\nBackplanes')
                keys = list(self.backplane.backplanes.keys())
                keys.sort(key=BackplaneTest._sort_key)
                for key in keys:
                    sum = np.sum(self.backplane.backplanes[key].mask)
                    LOGGING.diagnostic('   ', key, sum)

                LOGGING.diagnostic('\nAntimasks')
                keys = list(self.backplane.antimasks.keys())
                keys.sort()
                for key in keys:
                    antimask = self.backplane.antimasks[key]
                    info = ('array' if isinstance(antimask, np.ndarray)
                                    else str(antimask))
                    LOGGING.diagnostic('   ', key, '(%s)' % info)

                LOGGING.diagnostic()

            seconds = (datetime.datetime.now() - start).total_seconds()
            LOGGING.info('Elapsed time: %.3f s' % seconds)

        # Be sure to remove the BackplaneTest-specific file handler afterward
        finally:
            if self.args.log:
                LOGGING.logger.removeHandler(handler)

    @staticmethod
    def _sort_key(key):
        """Sort key function, needed to handle occurrences of Frames, Paths,
        and None is some dictionary keys.

        Also allow sorting among numbers, strings and tuples: numbers first,
        strings second, objects third, tuples fourth.
        """
        if isinstance(key, (tuple, list)):
            return 4, tuple([BackplaneTest._sort_key(item) for item in key])
        if isinstance(key, numbers.Real):
            return 1, key
        if isinstance(key, str):
            return 2, key
        return 3, str(key)

    ############################################################################
    # Test methods
    #
    # Results of a comparison are described by one of these named tuples:
    # _Comparison(status, max_error, limit, mask_errors, radius)
    # _NoComparison(status, pickle_path)
    #
    # The _Comparison status value is always one of these:
    #   "Success"               test passed
    #   "Value mismatch"        values differ
    #   "Mask mismatch"         masks differ
    #   "Value/mask mismatch"   both the values and the mask differ
    #
    # The _NoComparison status value is always one of these:
    #   "Success"               test passed
    #   "Shape mismatch"        shapes do not match
    #   "No gold master"        gold master info is missing
    #   "Invalid gold master"   gold master cannot be read
    ############################################################################

    _STATUS_IS_OK = {
        'Success'            : True,
        'Value mismatch'     : False,
        'Mask mismatch'      : False,
        'Value/mask mismatch': False,
        'Shape mismatch'     : False,
        'No gold master'     : False,   # See args.ignore_missing
        'Invalid gold master': False,
    }

    _MINMAX_VALUES = {
        'float': (sys.float_info.min, sys.float_info.max),
        'bool' : (False, True),
        'int'  : (-sys.maxsize - 1, sys.maxsize),
    }

    def compare(self, array, master, title, limit=0., method='', radius=0.):
        """Compare two backplane arrays and log the results.

        Note that the array can be a backplane that has been undersampled. The
        gold master array can be either full-resolution or undersampled.

        Inputs:
            array       backplane array to be compared.
            master      reference value or gold master array.
            title       title string describing the test; must be unique.
            limit       upper limit on the difference between the arrays.
            method      ''       for standard comparisons;
                        'mod360' for doing comparisons mod 360;
                        'border' for comparisons of border backplanes, in which
                                 case the radius value is interpreted in units
                                 of undersampled pixels rather than original
                                 pixels;
                        '>', '>=', '<', '<='
                                 for comparisons according to one of these
                                 operations, with any radius input ignored.
            radius      the radius of a circle, in units of pixels, by which to
                        check for a possible spatial shift for the values the
                        mask. This values is rounded down, so radius < 1
                        indicates no shift.
        """

        (array, title,
         limit, method, radius) = self._validate_inputs(array, title,
                                                        limit, method, radius)
        self._compare(array, master, title, limit, method, radius)

    def _compare(self, array, master, title, limit=0., method='', radius=0.):
        """Internal method that performs a comparison _after_ the inputs have
        been validated.
        """

        # Internal function to log the results
        def my_comparison(status, max_error=0, mask_errors=0, offset_errors=0,
                                                              pixels=0):
            if status == 'Shape mismatch':

                # Include the pickle path, if the file exists, in the message
                basename = self._basename(title, gold=True)
                pickle_path = os.path.join(self.gold_arrays,
                                           basename + '.pickle')
                if os.path.exists(pickle_path):
                    comparison = _NoComparison(status, pickle_path)
                else:
                    comparison = _NoComparison(status, '')

                self._log_comparison(comparison, title)
                return

            comparison = _Comparison(status, max_error, limit,
                                     mask_errors, offset_errors, pixels)
            self._log_comparison(comparison, title)

        if self.args.task != 'compare':
            LOGGING.debug('Summary:', title)
            return

        # Make objects compatible
        array = array.wod
        master = array.as_this_type(master, recursive=False, coerce=True)

        # Broadcast a shapeless object
        if array.shape and not master.shape:
            master = master.broadcast_to(array.shape)
            master = master.remask_or(array.mask)

        # Expand masks
        array = array.expand_mask()
        master = master.expand_mask()

        # A comparison with an undersampled border requires special handling.
        # In this case, the master array must be re-sampled in a way such that
        # a new pixel is True if any of the pixels from which it is derived are
        # True.
        if method == 'border' and self.undersample != 1:
            master_vals = master.vals.copy()
            master_vals[master.mask] = False
            new_vals = maximum_filter(master_vals, self.undersample,
                                      mode='constant', cval=False)
            if np.any(master.mask):
                new_mask = minimum_filter(master.mask, self.undersample,
                                          mode='constant', cval=False)
            else:
                new_mask = np.zeros(new_vals.shape, dtype='bool')

            master = Boolean(new_vals, mask=new_mask)

        # Re-sample master at the array's meshgrid if necessary
        if array.shape == master.shape:
            master_grid = master
            indx = slice(None)          # this index that generally does nothing
        elif self.undersample == 1:
            my_comparison('Shape mismatch')
            return
        else:
            grid = self.meshgrid.uv.int(top=self.full_shape,
                                        clip=True, shift=True).vals
            if self.obs.swap_uv:        # un-swap the uv indices
                grid = grid[..., ::-1]

            indx = (grid[...,0], grid[...,1])
            if method == 'border':      # For "border", down-sample the master
                master = master[indx]
                master_grid = master
                indx = slice(None)
            else:                       # Otherwise, master stays at full
                                        # resolution; master_grid is resampled.
                master_grid = master[indx]
                master_grid = master_grid.expand_mask()

        # Find the differences among unmasked pixels
        if method.isalnum() or method == '':
            comparison_op = False
            if method == 'mod360':
                diff = ((array - master_grid - 180) % 360 - 180).abs()
            else:
                diff = (array - master_grid).abs()
            max_diff = diff.max(builtins=True)

        else:
            comparison_op = True
            diff = array - master_grid
            if method[0] == '>':                # bad where diff is positive
                diff = -diff
            max_diff = diff.max(builtins=True)  # all good if max is <= 0

        # Compare masks
        mask_errors = np.sum(array.mask ^ master_grid.mask)
        comparison_args = (max_diff, mask_errors, mask_errors, array.size)

        # Handle fully masked results
        if isinstance(max_diff, Qube):
            zero = 0. if array.dtype() == 'float' else 0
            if np.all(array.mask) and np.all(master.mask):
                my_comparison('Success', *comparison_args)
            else:
                my_comparison('Value/mask mismatch', *comparison_args)
            return

        # Handle straightforward cases
        if comparison_op:
            if method[1:] == '=':
                diff_errors = diff.__gt__(0, builtins=False).sum(builtins=True)
            else:
                diff_errors = diff.__ge__(0, builtins=False).sum(builtins=True)
        else:
            diff_errors = diff.__gt__(limit, builtins=False).sum(builtins=True)

        if not diff_errors and not mask_errors:
            my_comparison('Success', *comparison_args)
            return

        if radius < 1 or array.shape == () or comparison_op:
            if diff_errors and not mask_errors:
                my_comparison('Value mismatch', *comparison_args)
            elif mask_errors and not diff_errors:
                my_comparison('Mask mismatch', *comparison_args)
            else:
                my_comparison('Value/mask mismatch', *comparison_args)
            return

        # See if the mask discrepancy is compatible with the radius...

        # The masks are compatible if...
        #   everywhere array.mask is True, so is master.mask expanded by radius
        #   everywhere array.mask is False, so is master.mask contracted
        #
        # Another way of saying this is that inside the region where master.mask
        # expanded equals master.mask contracted, array.mask must equal
        # master.mask.

        if method == 'border':
            fp = BackplaneTest._footprint(radius * self.undersample)
        else:
            fp = BackplaneTest._footprint(radius)

        master_mask_expanded   = maximum_filter(master.mask, footprint=fp,
                                                mode='constant', cval=False)
        master_mask_contracted = minimum_filter(master.mask, footprint=fp,
                                                mode='constant', cval=True)
        region = (master_mask_expanded == master_mask_contracted)

        # Apply the grid and compare
        region_grid = region[indx]
        mask_offset_errors = np.sum(array.mask[region_grid] !=
                                    master_grid.mask[region_grid])
        comparison_args = (max_diff, mask_errors, mask_offset_errors,
                                     array.size)
        if mask_offset_errors:
            if max_diff > limit:
                my_comparison('Value/mask mismatch', *comparison_args)
            else:
                my_comparison('Mask mismatch', *comparison_args)
            return

        # At this point, the masks are compatible.
        # If there and no bad values, we're done
        if max_diff <= limit:
            my_comparison('Success', *comparison_args)
            return

        # Determine if the radius can solve the value discrepancies...

        # There are two categories of discrepancies left
        # 1. The array is unmasked but master is masked.
        # 2. Both values are unmasked, but they differ by more than the limit.
        #
        # These are OK if the value of each discrepant pixel in the array is
        # within the range of the nearby unmasked pixels in the master array,
        # +/- the limit.

        # Determine the range of values in master adjacent to each value in array
        # Note that these are full-resolution arrays
        extremes = BackplaneTest._MINMAX_VALUES[Qube._dtype(array.vals)]
        master_vals = master.vals.copy()
        master_vals[master.mask] = extremes[1]
        min_master_vals = minimum_filter(master_vals, footprint=fp,
                                         mode='constant', cval=extremes[1])
        master_vals[master.mask] = extremes[0]
        max_master_vals = maximum_filter(master_vals, footprint=fp,
                                         mode='constant', cval=extremes[0])

        new_master_mask = (min_master_vals == extremes[1])

        # Resample the arrays
        min_master_vals = min_master_vals[indx]
        max_master_vals = max_master_vals[indx]
        new_master_mask = new_master_mask[indx]

        # Decide which pixels to check
        case1 = np.logical_not(array.mask) & master_grid.mask
        case2 = (diff > limit).as_mask_where_nonzero()
        cases = case1 | case2

        # Make sure every discrepant pixel is unmasked
        if np.any(new_master_mask[cases]):
            my_comparison('Value/mask mismatch', *comparison_args)

        # Select only the discrepant pixels
        array_vals = array.vals[cases]
        min_vals = min_master_vals[cases]
        max_vals = max_master_vals[cases]

        if array.is_bool():
            array_vals = array_vals.astype('int8')
            min_vals = min_vals.astype('int8')
            max_vals = max_vals.astype('int8')

        # Derive largest error among the discrepant pixels
        diffs_below = np.maximum(min_vals - array_vals, 0)
            # Positive where a pixel in the array is less than any of the nearby
            # pixels of the gold master; otherwise, 0.
        diffs_above = np.maximum(array_vals - max_vals, 0)
            # Positive where a pixel in the array is greater than any of the
            # nearby pixels of the gold master; otherwise, 0.
        case_diff = diffs_below + diffs_above
        max_case_diff = case_diff.max()

        comparison_args = (max_case_diff, mask_errors, mask_offset_errors,
                                          array.size)

        if max_case_diff <= limit:
            my_comparison('Success', *comparison_args)
        elif np.any(array.mask):
            my_comparison('Value/mask mismatch', *comparison_args)
        else:
            my_comparison('Value mismatch', *comparison_args)

        return

    #===========================================================================
    def gmtest(self, array, title, limit=0., method='', radius=0.):
        """Compare a backplane array against its gold master.

        Inputs:
            array       backplane array to be tested.
            title       title string describing the test; must be unique.
            limit       upper limit on the difference between the arrays.
            method      ''       for standard comparisons;
                        'mod360' for doing comparisons mod 360;
                        'border' for comparisons of border backplanes, in which
                                 case the radius value is interpreted in units
                                 of undersampled pixels rather than original
                                 pixels;
                        '>', '>=', '<', '<='
                                 for comparisons according to one of these
                                 operations.
            radius      the radius of a circle, in units of pixels, by which to
                        check for a possible spatial shift for the values the
                        mask. This values is rounded down, so radius < 1
                        indicates no shift.
        """

        # Validate inputs
        (array, title,
         limit, method, radius) = self._validate_inputs(array, title,
                                                        limit, method, radius)

        # Handle a 2-D array
        if array.shape:

            # Determine the storage precision
            if limit == 0.:
                array.set_pickle_digits('double', 'fpzip')
            else:
                # Could save at reduced precision, but better to use full...
                # digits = -np.log10(limit) + 1       # save one extra digit
                # array.set_pickle_digits(digits, 1.)
                array.set_pickle_digits('double', 'fpzip')

            # Write the pickle file
            if self.task == 'adopt':
                output_dir = self.gold_dir
                output_arrays = self.gold_arrays
                output_browse = self.gold_browse
                basename = self._basename(title, gold=True)
            else:
                output_dir = self.output_dir
                output_arrays = self.output_arrays
                output_browse = self.output_browse
                basename = self._basename(title, gold=False)

            pickle_path = os.path.join(output_arrays,
                                       basename + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(array, f)

            # Write the browse image
            if self.args.browse:
                browse_name = basename + '.' + self.args.browse_format
                browse_path = os.path.join(output_browse, browse_name)
                self.save_browse(array, browse_path)

            # For "compare"
            if self.task == 'compare':
                basename = self._basename(title, gold=True)
                pickle_path = os.path.join(self.gold_arrays,
                                           basename + '.pickle')

                # Handle a missing pickle file
                if not os.path.exists(pickle_path):
                    comparison = _NoComparison('No gold master', pickle_path)
                    self._log_comparison(comparison, title)

                else:
                    # Retrieve pickled backplane
                    try:
                        with open(pickle_path, 'rb') as f:
                            master = pickle.load(f)
                    except (ValueError, TypeError, OSError):
                        comparison = _NoComparison('Invalid gold master',
                                                   pickle_path)
                        self._log_comparison(comparison)

                    # Compare...
                    else:
                        self._compare(array, master, title, limit=limit,
                                      method=method, radius=radius)

            # For "preview" and "adopt"
            else:
                LOGGING.debug('Written:', os.path.basename(pickle_path))

        # Shapeless case
        else:

            # For "compare"
            if self.task == 'compare':
                if title not in self.gold_summary:
                    comparison = _NoComparison('No gold master', '')
                    self._log_comparison(comparison, title)

                else:
                    (min_val, max_val, masked,
                                       unmasked) = self.gold_summary[title]
                    # If gold master value is not shapeless...
                    if min_val != max_val or masked + unmasked > 1:
                        comparison = _NoComparison('Shape mismatch', '')
                        self._log_comparison(comparison, title)

                    else:
                        master = Scalar(min_val, masked > 0)
                        self._compare(array, master, title,
                                      limit=limit, method=method, radius=radius)

            # For "preview" and "adopt"
            else:
                LOGGING.debug('Summary:', title)

    #===========================================================================
    def _validate_inputs(self, array, title, limit, method, radius):
        """Initial steps for both compare() and gmtest()."""

        global LATEST_TITLE
        LATEST_TITLE = title

        # Validate method
        if method not in ('', 'mod360', 'border', '>', '>=', '<', '<='):
            raise ValueError('unknown comparison method: ' + repr(method))

        # Validate limit
        if isinstance(limit, Qube):
            if np.any(limit.mask):
                limit = 0.
            else:
                limit = limit.vals

        # Warn about duplicated titles
        if title in self.results:
            LOGGING.error('Duplicated title:', title)

        # Validate array
        if not isinstance(array, Qube):
            if (isinstance(array, (bool, np.bool_))
                or (isinstance(array, np.ndarray) and array.dtype.kind == 'b')):
                    array = Boolean(array)
            else:
                array = Scalar(array).as_float()

        # Save the summary info in the dictionary
        self._summarize(array, title, method=method)

        return (array, title, limit * self.args.tolerance, method,
                              radius * self.args.radius)

    #===========================================================================
    def _summarize(self, array, title, method=''):
        """Save the summary info for this backplane array.

        For boolean arrays, the saved tuple is
            (False count, True count, masked count, total pixels)
        For floats, the saved tuple is:
            (minimum value, maximum vale, masked count, total pixels)
        The case are distinguished by whether the first value is int of float.
        """

        def _save(minval, maxval, masked, total):
            self.summary[title] = (minval, maxval, masked, total)

            if self.args.info:
                message = ['  ', title, ': ']
                if isinstance(minval, numbers.Integral):   # if array is boolean
                    message += ['false,true = %d,%d' % (minval, maxval)]
                elif masked < total:
                    if minval == maxval:
                        message += ['value = ', self._valstr(minval)]
                    else:
                        message += ['min,max = ', self._valstr(minval), ', ',
                                                  self._valstr(maxval)]
                if masked:
                    percent = (100. * masked) / total
                    if 99.9 < percent < 100.:   # "100.0"% means every pixel
                        percent = 99.9
                    if message[-1] != ': ':
                        message += ['; ']
                    message += ['mask ', '%.1f' % percent, '%']

                LOGGING.literal(''.join(message))

        masked = array.count_masked()
        total = array.size

        if array.is_bool():
            trues  = np.sum(array.antimask & array.vals)
            falses = np.sum(array.antimask & np.logical_not(array.vals))
            _save(falses, trues, masked, total)
            return

        if masked == total:
            _save(array.default, array.default, masked, total)
            return

        if method != 'mod360':
            _save(array.min(builtins=True), array.max(builtins=True),
                  masked, total)
            return

        # Handle mod360 case
        if np.shape(array.mask):
            vals = array.vals[array.antimask]
        elif np.shape(array.vals):
            vals = array.vals.ravel()
        else:
            _save(array.vals, array.vals, masked, total)
            return

        sorted = np.sort(vals)
        sorted = np.hstack((sorted, [sorted[0] + 360]))
        diffs = np.diff(sorted)
        argmax = np.argmax(diffs)

        maxval = sorted[argmax]
        if argmax == len(diffs) - 1:
            minval = sorted[-1] - 360
        else:
            minval = sorted[argmax + 1]

        _save(minval, maxval, masked, total)

    #===========================================================================
    def _log_comparison(self, comparison, title):
        """Log this comparison info."""

        # Check the status
        status = comparison.status

        # Select error level
        if BackplaneTest._STATUS_IS_OK[status]:
            level = logging.INFO
        else:
            level = logging.ERROR

        if status == 'No gold master' and self.args.ignore_missing:
            level = logging.WARNING

        self.results[title] = comparison

        # Construct the message
        if isinstance(comparison, _Comparison):
            val = self._valstr(comparison.max_error)
            message = status + ': "%s"; diff=%s' % (title, val)
            if comparison.limit:
                message += '/' + self._valstr(comparison.limit)

            if comparison.offset_errors == comparison.mask_errors:
                message += '; pixels=%d/%d' % (comparison.mask_errors,
                                               comparison.size)
            else:
                message += '; pixels=%d/%d/%d' % (comparison.mask_errors,
                                                  comparison.offset_errors,
                                                  comparison.size)

        else:       # _NoComparison
            if comparison.pickle_path:
                basename = os.path.basename(comparison.pickle_path)
                message = status + ': ' + basename
            else:
                message = status + ': "%s"' % title

        LOGGING.print(message, level=level)

    #===========================================================================
    @staticmethod
    def _valstr(value):
        """value formatter, avoiding "0.000" and "1.000e-12"."""

        if isinstance(value, Qube):
            return '--'
        elif isinstance(value, numbers.Integral):
            return str(value)
        else:
            formatted = '%#.7g' % value
            parts = formatted.partition('e')
            return parts[0].rstrip('0') + parts[1] + parts[2]

    #===========================================================================
    @functools.lru_cache(maxsize=10)
    def _footprint(radius):
        """Circular footprint of the given pixel radius, for scip.ndarray
        filters.
        """

        rounded = int(radius // 1)      # rounded down
        size = 2 * rounded + 1
        x = np.arange(size) - size//2   # [-N to N] inclusive
        y = x[:, np.newaxis]
        return (x**2 + y**2) <= radius**2

    #===========================================================================
    def _basename(self, title, gold=False):
        """Convert a title to a file basename.

        if gold is True, the Linux shell-friendly basename is always returned,
        regardless of the platform. Otherwise, the platform-specific basename is
        returned.
        """

        if self.args.platform in ('darwin', 'macos') and not gold:
            title = title.replace(':', '-')
            title = title.replace('/', '|')
            title = title.replace('\\', '|')
            return title

        elif (self.args.platform in ('win32', 'cygwin', 'msys', 'windows')
              and not gold):
            translated = []
            for c in title:
                if c in '<>:"/\\|?*':
                    c = '-'
                translated.append(c)

            filename = ''.join(translated)
            while '--' in filename:
                filename.replace('--', '-')

            return filename

        else:
            title = title.replace(' (', '_')
            title = title.replace(') ', '_')
            title = title.replace(', ', '_')
            title = title.replace('. ', '_')
            title = title.rstrip(')')

            translated = []
            for c in title:
                if c == ' ':
                    c = '_'
                elif c.isalnum() or c in ('-_'):
                    pass
                else:
                    c = '-'
                translated.append(c)

            while not translated[0].isalnum():
                translated = translated[1:]
            while not translated[-1].isalnum():
                translated = translated

            filename = ''.join(translated)
            while '--' in filename:
                filename.replace('--', '-')
            while '__' in filename:
                filename.replace('__', '_')

            return filename

    ############################################################################
    # Browse image support
    ############################################################################

    def save_browse(self, array, browse_path):
        """Save a backplane as a PNG, JPG, or TIFF file."""

        # Get pixels and mask
        if isinstance(array, np.ndarray):
            image = array.copy()
            mask = False
        else:
            image = array.vals.copy()
            mask = array.mask

        image = np.asfarray(image)
        image = np.atleast_2d(image)

        # Handle the mask
        if np.all(mask):        # if fully masked
            image.fill(0.)
            minval = 0.
            maxval = 1.

        elif array.dtype() == 'bool':
            # masked -> -0.2 (black); False -> 0 (dark gray); True -> 1 (white)
            minval = -0.2
            maxval = 1.
            image = image.astype('int')
            image[mask] = minval

        elif np.any(mask):
            # masked is black; minimum unmasked value is very dark gray
            minval = np.min(image[np.logical_not(mask)])
            maxval = np.max(image[np.logical_not(mask)])
            new_minval = minval - 0.05 * (maxval - minval)
            image[mask] = new_minval
            minval = new_minval

        else:
            minval = np.min(image)
            maxval = np.max(image)

        # Create the scaled array of bytes
        if minval == maxval:
            scaled_bytes = np.zeros(image.shape, dtype=np.int8)
        else:
            scaled_floats = (image - minval) / float(maxval - minval)
            scaled_bytes = (255.999999 * scaled_floats).astype('uint8')

        # Apply zoom
        if self.args.zoom != 1:
            scaled_bytes = zoom_image(scaled_bytes, self.zoom, order=0)

        # Make sure y-axis increases downward
        if self.upward:
            scaled_bytes = scaled_bytes[::-1]

        shape = scaled_bytes.shape[::-1]
        im = PIL.Image.frombytes('L', shape, scaled_bytes)
        im.save(browse_path)

    #===========================================================================
    @staticmethod
    def read_browse(browse_path):
        """Read a PNG, JPG, or TIFF image file as a 2-D array of unsigned bytes.
        """

        with PIL.Image.open(browse_path, mode='r') as im:
            return np.array(im)

    ############################################################################
    # Result summary support
    #
    # The summary file is a text file containing the definition of a Python
    # dictionary. The dictionary is keyed by backplane title string and its
    # values are tuples (minimum value, maximum value, masked pixels, total
    # pixels). If the minimum and maximum are are equal, only one value is
    # listed; if the object is fully masked, the value is None.
    ############################################################################

    @property
    def gold_summary(self):

        if self.gold_summary_ is not None:
            return self.gold_summary_

        filepath = os.path.join(self.gold_dir, 'summary.py')
        if not os.path.exists(filepath):
            self.gold_summary_ = {}
        else:
            with open(filepath) as f:
                text = f.read()
            self.gold_summary_ = eval(text)

        # Expand tuples where value is None or min == max
        for key, value in self.gold_summary_.items():
            if len(value) == 3:
                if value[0] == None:
                    value = (0,0) + value[2:]
                else:
                    value = value[:1] + value[:1] + value[1:]
                self.gold_summary_[key] = value

        return self.gold_summary_

    #===========================================================================
    def write_summary(self, outdir):
        """Write the test summary as a Python dictionary; return its file path.
        """

        filepath = os.path.join(outdir, 'summary.py')

        if os.path.exists(filepath):

            # Append the latest modification date to any pre-existing file
            dt = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
            suffix = dt.strftime('-%Y-%m-%dT%H-%M-%S')
            dated_path = filepath[:-3] + suffix + '.py'
            os.rename(filepath, dated_path)

            LOGGING.info('Previous summary moved to: '
                         + os.path.basename(dated_path))

        titles = list(self.summary.keys())
        titles.sort(key=lambda key: key.lower())    # sort titles ignoring case

        # Write new file
        with open(filepath, 'w') as f:

            dt = datetime.datetime.now()
            f.write(dt.strftime('# gold_master summary %Y-%m-%dT%H-%M-%S\n'))
            f.write('#\n')
            f.write(SUMMARY_COMMENT)
            f.write('\n')

            f.write('{\n')
            for title in titles:
                f.write(('    "%s"' % title).ljust(63))
                f.write(': ')
                summary = self.summary[title]
                if isinstance(summary[0], numbers.Integral):  # array is boolean
                    pass
                elif summary[2] == summary[3]:                # fully masked
                    summary = (None,) + summary[2:]
                elif summary[0] == summary[1]:                # min == max
                    summary = summary[:1] + summary[2:]

                f.write(repr(summary))
                f.write(',\n')
            f.write('}\n')

        return filepath

################################################################################
# To run as a unit test, using the long-standing default Cassini image...
################################################################################

class Test_Backplane_Gold_Masters(unittest.TestCase):
    """This will run the gold master tests with default inputs as a unit test.

    All logging is off.
    """

    def test_backplane_gold_masters(self):

        args = get_args(task='compare', log=False, verbose=False, info=False,
                        convergence=False, diagnostics=False, performance=False,
                        unittest=True, testcase=self)
        run_tests(args)

################################################################################
# To handle gold master testing from the command line...
################################################################################

if __name__ == '__main__':

    import oops.backplane.gold_master as gm

    gm.execute_as_command()

################################################################################

################################################################################
# oops/backplane/gold_master.py: Backplane gold master tester and support.
################################################################################
"""\
########################################################################
# How to use with the Python unittest module for a host or instrument...
########################################################################
# Case 1: A single test
####################################

import unittest
import oops.backplane.gold_master as gm

class Test_<your test name>(unittest.TestCase):

    def runTest(self):

        # Define the default observation
        gm.define_default_obs(
                obspath = 'file path inside the test_data directory',
                index   = (index to apply to result of from_file, or None),
                planets = ['SATURN'],               # for example
                moons   = ['ENCELADUS'],            # for example
                rings   = ['SATURN_MAIN_RINGS'],    # for example, optional
                kwargs  = {})                       # other from_file inputs

        # Change any other default parameters, at least this one...
        gm.set_default_args(module='hosts.xxx.yyy', ...)

        gm.execute_as_unittest(self)

####################################
# Case 2: Multiple tests
####################################

import unittest
import oops.backplane.gold_master as gm

class Test_<your test name>(unittest.TestCase):

    def setUp():

        # Define the standard observations
        gm.define_standard_obs('obs1',
                obspath = 'file path inside the test_data directory',
                index   = (index to apply to result of from_file, or None),
                planets = ['SATURN'],               # for example
                moons   = ['ENCELADUS'],            # for example
                rings   = ]'SATURN_MAIN_RINGS'],    # for example, optional
                kwargs  = {})                       # other from_file inputs

        gm.define_standard_obs('obs2', ...)

        gm.define_standard_obs('obs3', ...)

        # Change any other default parameters, at least this one...
        gm.set_default_args(module='hosts.xxx.yyy', ...)

    def run_test1(self):
        gm.execute_as_unittest(self, 'obs1')

    def run_test2(self):
        gm.execute_as_unittest(self, 'obs2')

    def run_test3(self):
        gm.execute_as_unittest(self, 'obs3')

########################################################################
# How to have a gold master tester program dedicated to an instrument...
########################################################################

import os
import oops.backplane.gold_master as gm

# Define the default observation and any number of others for testing;
# note that the selection can be overridden on the command line.

gm.define_default_obs(
            obspath = 'file path inside the test_data directory',
            index   = (index to apply to result of from_file, or None),
            planets = ['SATURN'],               # for example
            moons   = ['ENCELADUS'],            # for example
            rings   = ['SATURN_MAIN_RINGS'],    # for example, optional
            kwargs  = {})                       # other from_file inputs
gm.define_standard_obs('test2', ...)
gm.define_standard_obs('test3', ...)

# Change any other default parameters, at least this one...
gm.set_default_args(module='hosts.xxx.yyy', ...)

if __name__ == '__main__':
    gm.execute_as_command()

########################################################################
# Log file format
########################################################################

A single record of the log file has this format:
    "<time> | oops.backplane.gold_master | <level> | <suite> | <message>"
where
    <time>  is the local time to the level of ms.
    <level> is one of "DEBUG", "INFO", "WARNING", "ERROR", "FATAL".
    <suite> is the name of the test suite, e.g., "ring".
    <message> is a descriptive message.

For comparison tests, the message has the following format:
    <status>: "<title>"; diff=<diff1>/<diff2>/<limit>; ...
                         offset=<offset>/<radius>; ...
                         pixels=<count1>/<count2>/<pixels>
where:
    <status> is one of:
        "Success"               if the test passed;
        "Value mismatch"        if the values disagree by more than the
                                limit, but the mask is in agreement;
        "Mask mismatch"         if the masks disagree, but the values are in
                                agreement;
        "Value/mask mismatch"   if both the values and the mask disagree.
    <title>   is the title of the test.
    <diff1>   is the maximum discrepancy among the unmasked values.
    <diff2>   is the maximum discrepancy after we have expanded the
              comparison to include neighboring pixels, as defined by the
              specified radius.
    <limit>   the specified discrepancy limit of the test.
    <offset>  the offset distance required to bring value discrepancies below
              the limit, or to resolve any mask discrepancies.
    <radius>  the specified upper limit on an offset.
    <count1>  the number of discrepant pixels before allowing for an offset.
    <count2>  the number of discrepant pixels that cannot be accommodated by an
              offset.
    <pixels>  the total number of pixels tested.

Note that <diff2> and <count2> are not listed if not offset is required. Also,
note that the offset values are not listed in this case.
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
import PIL.Image
import sys
import warnings

from collections   import defaultdict
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.ndimage import zoom as zoom_image

from polymath         import Boolean, Pair, Qube, Scalar
from oops.backplane   import Backplane
from oops.body        import Body
from oops.config      import LOGGING
from oops.constants   import DPR
from oops.observation import Observation

from oops.unittester_support import (OOPS_TEST_DATA_PATH,
                                     OOPS_GOLD_MASTER_PATH,
                                     OOPS_BACKPLANE_OUTPUT_PATH)

################################################################################
# Use set_default_obs() and set_standard_obs() to define the observation used
# for unit tests and as the default for a run from the command line.
################################################################################

# This is a dictionary test name -> key inputs for test
STANDARD_OBS_INFO = {}

def set_default_obs(obspath, index, planets, moons=[], rings=[], kwargs={}):
    """Set the details of the default observation to be used for the gold master
    test.

    These are the default observation file path and module to use if they are
    not specified in the command line.

    The specified planets, moons, and rings are used as the defaults when the
    observation is unspecified, but can be overridden at the command line.

    Options:
        obspath         file path to the default data object to be used.
        index           index to apply if from_file returns a list. If None,
                        backplanes will be generated for every Observation
                        returned by from_file.
        planets         name of the default planet, or list of planet names.
        moons           name of the default moon, if any, or list of moon names.
        rings           name of the default ring, if any, or list of ring names.
                        Backplane arrays are always generated for the full ring
                        plane of the specified planet.
        kwargs          an optional dictionary of keyword arguments to be passed
                        to from_file.
    """

    define_standard_obs('default', obspath=obspath, index=index,
                                   planets=planets, moons=moons, rings=rings,
                                   kwargs=kwargs)

def define_standard_obs(obsname, obspath, index, planets, moons=[], rings=[],
                        kwargs={}):
    """Set the details of a standard gold master test.

    These are the observation file path and module to use when a test is
    identified by name.

    The specified planets, moons, and rings are used as the defaults when the
    observation is unspecified, but can be overridden at the command line.

    Options:
        obsname         name given for this observation.
        obspath         file path to the default data object to be used.
        index           index to apply if from_file returns a list. If None,
                        backplanes will be generated for every Observation
                        returned by from_file.
        planets         name of the default planet, or list of planet names.
        moons           optional name of the default moon, if any, or list of
                        moon names.
        rings           optional name of the default ring, if any, or list of
                        ring names. Backplane arrays are always generated for
                        the full ring plane of the specified planet.
        kwargs          an optional dictionary of keyword arguments to be passed
                        to from_file.
    """

    global STANDARD_OBS_INFO

    planets = planets if isinstance(planets, (list,tuple)) else (planets,)
    moons   = moons   if isinstance(moons,   (list,tuple)) else (moons,)
    rings   = rings   if isinstance(rings,   (list,tuple)) else (rings,)

    STANDARD_OBS_INFO[obsname] = {}
    STANDARD_OBS_INFO[obsname]['obspath'] = obspath
    STANDARD_OBS_INFO[obsname]['index']   = index
    STANDARD_OBS_INFO[obsname]['planets'] = planets
    STANDARD_OBS_INFO[obsname]['moons']   = moons
    STANDARD_OBS_INFO[obsname]['rings']   = rings
    STANDARD_OBS_INFO[obsname]['kwargs']  = kwargs

################################################################################
# The "default defaults" are defined here. A call to set_default_args() can be
# used to replace them for some specific test. These defaults are required to be
# identical for all standard observations.
################################################################################

DEFAULTS = {
    'planets'       : [],           # used only if no standard obs is named
    'moons'         : [],           # ditto
    'rings'         : [],           # ditto
    'module'        : '',
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

# Note that default values of output, convergence, diagnostics, internals,
# performance, and platform cannot be overridden here.

def set_default_args(**options):
    """Set the default command-line arguments for a gold master test.

    Options:
        planets         name(s) of the planet(s) to use if if this run does not
                        employ a standard observation.
        moons           name(s) of the moon(s) to use if if this run does not
                        employ a standard observation.
        rings           name(s) of the ring(s) to use if if this run does not
                        employ a standard observation.
        module          Name of the module containing the "from_file" method,
                        e.g., "hosts.cassini.iss".
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
        derivs          True to include the unit tests of the derivatives, False
                        otherwise.
        undersample     undersample factor for backplane tests and browse
                        images; default 16.
        inventory       True to use an inventory when generating backplanes.
        border          Size of border for inventory; default 0.
        browse          True to save browse images; default True.
        zoom            zoom factor for browse images; default 1.
        browse_format   browse image format, one of "png", "jpg", or "tiff".
        verbose         True to print output to the terminal by default.
        log             True to save a log file by default.
        level           Minimum level for messages to be logged: "debug",
                        "info", "warning", "error", or an integer 1-30.
    """

    global DEFAULTS

    for key, value in options.items():
        DEFAULTS[key] = value

################################################################################
# Overrides of specific tests
#
# Sometimes we understand why certain comparison tests have values that exceed
# the hard-wired limit.
################################################################################

TEST_OVERRIDES = defaultdict(dict)

def override(title, value, names=None):
    """Override the hard-wired comparison values for specific tests.

    Input:
        title       the exact title of a test, e.g.,
                    "JUPITER:RING incidence angle, ring minus center (deg)".
        value       the revised comparison value, or None to suppress the test
                    entirely.
        names       name(s) of one or more standard observations; None to
                    apply to all standard observations.
    """

    global TEST_OVERRIDES

    if not names:
        obsnames = list(STANDARD_OBS_INFO.keys())
    elif isinstance(names, str):
        obsnames = [names]
    else:
        obsnames = names

    for obsname in obsnames:
        TEST_OVERRIDES[obsname][title] = value

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

    Inputs:
        **options       overrides for any default gold_master input arguments.
    """

    # Define parser...
    parser = argparse.ArgumentParser(
                    description='Gold Master backplane test utility%s.'
                                % ('' if not DEFAULTS['module'] else
                                   ' for module ' + DEFAULTS['module']))

    # Data objects
    gr = parser.add_argument_group('Data objects')
    gr.add_argument('obspath', type=str, nargs='*', metavar='filepath',
                    help='''File path to the data object(s) to be used in place
                            of a standard observation.''')
    gr.add_argument('--index', type=int, metavar='N',
                    help='''Index to use for a specified filepath; otherwise,
                            backplane arrays will be generated for each
                            observation in the file.''')
    gr.add_argument('--module', type=str, metavar='hosts...',
                    help='''Name of the module containing the "from_file"
                            method for the filepaths specified%s.'''
                         % ('' if DEFAULTS['module'] is None else
                            '; default is ' + DEFAULTS['module'] + '.'))
    gr.add_argument('--name', '-n', type=str, nargs='*',
                    default=list(STANDARD_OBS_INFO.keys()),
                    help='''Name(s) of the pre-defined standard observation to
                            use if a file path is not given explicitly. Default
                            is to use all of the standard observations.''')

    # Backplane targets
    gr = parser.add_argument_group('Backplane targets')
    gr.add_argument('-p', '--planet', type=str, nargs='*', metavar='name',
                    default=DEFAULTS['planets'], dest='planets',
                    help='''Name(s) of one or more planets for which to generate
                            backplane arrays. Defaults are defined by the named
                            standard observation(s)%s.'''
                         % ('' if DEFAULTS['planets'] is None else
                            '; otherwise ' + ', '.join(DEFAULTS['planets'])))
    gr.add_argument('-m', '--moon', type=str, nargs='*', metavar='name',
                    default=DEFAULTS['moons'], dest='moons',
                    help='''Name(s) of one or more moons for which to generate
                            backplane arrays. Defaults are defined by the named
                            standard observation(s)%s.'''
                         % ('' if DEFAULTS['moons'] is None else
                            '; otherwise ' + ', '.join(DEFAULTS['moons'])))
    gr.add_argument('-r', '--ring', type=str, nargs='*', metavar='name',
                    default=DEFAULTS['rings'], dest='rings',
                    help='''Name(s) of one or more rings for which to generate
                            backplane arrays. Arrays are always generated for
                            the default equatorial ring of the planet.'''
                         + ('' if DEFAULTS['rings'] is None else ' Default is '
                                + ', '.join(DEFAULTS['rings']) + '.'))

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
    gr.add_argument('--tolerance', type=float, metavar='TOL',
                    default=float(DEFAULTS['tolerance']),
                    help='''Factor to apply to backplane array error tolerances;
                            default %s.'''
                         % str(DEFAULTS['tolerance']))
    gr.add_argument('--radius', type=float, metavar='RAD',
                    default=float(DEFAULTS['radius']),
                    help='''Factor to apply to backplane array radial offset
                            limits; default %s.'''
                         % str(DEFAULTS['radius']))
    gr.add_argument('--ignore-missing', action='store_true',
                    default=DEFAULTS['ignore_missing'],
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
    gr.add_argument('--derivs', action='store_true', default=None,
                    help='''Perform tests of spatial derivatives of backplane
                            arrays. Default is to exclude the derivative tests
                            when undersampling=1, and to include them otherwise.
                            Note that tests can take several times longer with
                            this option enabled.''')
    gr.add_argument('--no-derivs', action='store_false', dest='derivs',
                    help='''Suppress tests of spatial derivatives of backplane
                            arrays.''')

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
    gr.add_argument('--save-sampled', '--ss', action='store_true',
                    default=False, dest='save_sampled',
                    help='''Save copies of the master arrays at the undersampled
                            grid points.''')

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
                    choices=('jpg', 'png', 'tiff'),
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
                    choices=['debug', 'warning', 'error']
                            + [str(k) for k in range(1,31)],
                    default=DEFAULTS['level'],
                    help='''Minimum level for messages to be logged: "debug",
                            "info", "warning", "error", or an integer 1-30;
                            default is %s.'''
                         % DEFAULTS['level'])
    gr.add_argument('--convergence', action='store_true', default=False,
                    help='Show iterative convergence information in the log.')
    gr.add_argument('--diagnostics', action='store_true', default=False,
                    help='Include diagnostic information in the log.')
    gr.add_argument('--internals', action='store_true', default=False,
                    help='''Include info about the Backplane internal state at
                            the end of the log.''')
    gr.add_argument('--performance', action='store_true', default=False,
                    help='Include OOPS performance information in the log.')
    gr.add_argument('--fullpaths', action='store_true', default=False,
                    help='''Include the full paths of all output files in the
                            log.''')
    gr.add_argument('--platform', type=str, metavar='OS', default=None,
                    choices=('macos', 'windows', 'linux'),
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

def execute_as_unittest(testcase, obsname='default'):
    """Run the gold master test suites for all of the defined standard
    observations.
    """

    try:
        # Initialize the command argument namespace
        args = argparse.Namespace()
        for key, value in DEFAULTS.items():
            setattr(args, key, value)

        # Set the default observation details
        args.name = [obsname]

        # These values in the DEFAULTS dictionary are overridden
        args.browse = False
        args.log = False
        args.verbose = True

        # These have no entry in the DEFAULTS dictionary
        args.obspath = None
        args.output = None
        args.convergence = False
        args.diagnostics = False
        args.internals = False
        args.performance = False
        args.fullpaths = False
        args.platform = None
        args.save_sampled = False

        # These options are mandatory
        args.testcase = testcase
        args.task = 'compare'
        args.level = 'error'
        args.verbose = True
        args.du = 0.
        args.dv = 0.
        args.derivs = True

        # Clean up, also filling in observation, module, planet(s), moon(s),
        # ring(s)
        args = _clean_up_args(args)

    except Exception as e:
        testcase.assertTrue(False, str(e))

    run_tests(args)

#===============================================================================
def _clean_up_args(args):
    """Clean up arguments given in the command line."""

    global DEFAULTS, TEST_SUITES

    # Define the module and observation if not a standard one
    args.module = args.module or DEFAULTS['module']

    # Given obspaths, define temporary "standard observations" and then use
    # their names.
    if args.obspath:
        if args.name is not None:
            raise ValueError('an observation filepath and a standard '
                             'observation name cannot be specified together.')

        if isinstance(args.obspath, str):
            args.obspath = [args.obspath]

        for obspath in args.obspaths:
            define_standard_obs(obspath, obspath, index=args.index,
                                planets=args.planets, moons=args.moons,
                                rings=args.rings)

        args.name = args.obspath        # using obspath as the temporary name

    # --output
    if args.output is None:
        args.output = OOPS_BACKPLANE_OUTPUT_PATH

    # --platform
    if args.platform is None:
        args.platform = sys.platform
    else:
        args.platform = args.platform.lower()

    # --format
    if args.browse_format not in ('png', 'jpg', 'tiff'):
        raise ValueError('unrecognized browse format: ' + args.browse_format)

    # --level
    if (args.convergence or args.diagnostics or args.performance):
            args.level = 'debug'
    try:
        args.level = int(args.level)
    except ValueError:
        args.level = LOGGING.LEVELS[args.level.lower()]

    # --suite
    if not args.suite or args.task == 'adopt':  # use all suites for task adopt
        args.suite = list(TEST_SUITES.keys())
    elif isinstance(args.suite, str):
        args.suite = [args.suite]

    args.suite.sort()

    # --derivs
    if args.derivs is None:
        args.derivs = (args.undersample > 1)

    # Special requirements for task --adopt
    if args.task == 'adopt':    # required options for task adopt
        args.browse = True
        args.undersample = 1
        args.du = 0.
        args.dv = 0.
        args.derivs = False
        args.save_sampled = False

    # Get the from_file method
    module = importlib.import_module(args.module)
    args.from_file = module.from_file

    # Define the BackplaneTest objects
    args.backplane_tests = []
    for obsname in args.name:
        obspath = STANDARD_OBS_INFO[obsname]['obspath']
        index   = STANDARD_OBS_INFO[obsname]['index']
        kwargs  = STANDARD_OBS_INFO[obsname]['kwargs']

        abspath = os.path.abspath(os.path.realpath(obspath))
        if args.task in ('compare', 'adopt'):
            if not OOPS_TEST_DATA_PATH:
                raise ValueError('Undefined environment variable: '
                                 'OOPS_TEST_DATA_PATH')
            test_data_path_ = os.path.realpath(OOPS_TEST_DATA_PATH) + '/'
            if not abspath.startswith(test_data_path_):
                warnings.warn('File is not in the test data directory: '
                              + obspath + '; ' + OOPS_TEST_DATA_PATH)
        if not os.path.exists(abspath):
            raise FileNotFoundError('No such file: ' + obspath)

        # Allow overrides of bodies
        planets = args.planets or STANDARD_OBS_INFO[obsname]['planets']
        moons   = args.moons   or STANDARD_OBS_INFO[obsname]['moons']
        rings   = args.rings   or STANDARD_OBS_INFO[obsname]['rings']

        result = args.from_file(abspath, **kwargs)
        if index is not None:
            result = result[index]

        overrides = TEST_OVERRIDES[obsname]
        if isinstance(result, Observation):
            bpt = BackplaneTest(result, planets, moons, rings, overrides, args)
            args.backplane_tests.append(bpt)
        else:
            for k, obs in enumerate(result):
                bpt = BackplaneTest(obs, planets, moons, rings, overrides, args,
                                    suffix='_' + str(k))
                args.backplane_tests.append(bpt)

    # Set the status of "No gold master"
    _BackplaneComparison.set_no_gold_master_status(args.ignore_missing)

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
# Internal _BackplaneComparison class
################################################################################

class _BackplaneComparison(object):
    """Class for managing information about a single BackplaneTest
    comparison.
    """

    STATUS_LEVEL = {
        'Success'            : 'INFO',
        'Value mismatch'     : 'ERROR',
        'Mask mismatch'      : 'ERROR',
        'Value/mask mismatch': 'ERROR',
        'Shape mismatch'     : 'ERROR',
        'No gold master'     : 'ERROR',     # Defined by args.ignore_missing
        'Invalid gold master': 'ERROR',
    }

    def __init__(self, **kwargs):
        """Container for comparison info.

        status is one of:
            "Success"               test passed;
            "Value mismatch"        values differ;
            "Mask mismatch"         masks differ;
            "Value/mask mismatch"   both the values and the mask differ;
            "Shape mismatch"        shapes do not match;
            "No gold master"        gold master info is missing;
            "Invalid gold master"   gold master cannot be read.

        title        = title of the test.
        suite        = name of test suite.
        limit        = maximum allowed difference.
        method       = name of the comparison method, e.g., 'mod360.
        operator     = comparison operator.
        radius       = allowed offset distance in pixels.
        mask         = optional mask to exclude pixels from comparison.
        pickle_path  = path to the pickle file, if any.

        max_diff1    = the largest difference between unmasked pixels of array
                       and master, initially.
        diff_errors1 = the initial number of value discrepancies.
        mask_errors1 = the number of mask discrepancies.

        distance     = the largest offset adequate to eliminate a value
                       discrepancy; zero if no offset is adequate.

        max_diff2    = the final largest difference.
        diff_errors2 = the final number of value discrepancies.
        mask_errors2 = the final number of mask discrepancies.

        pixels       = the total number of pixels.
        """

        self.title        = ''
        self.suite        = ''
        self.limit        = 0.
        self.method       = ''
        self.operator     = '='
        self.radius       = 0.
        self.mask         = False
        self.pickle_path  = ''

        self.status       = ''
        self.max_diff1    = 0.
        self.diff_errors1 = 0
        self.mask_errors1 = 0
        self.distance     = 0.
        self.max_diff2    = 0.
        self.diff_errors2 = 0
        self.mask_errors2 = 0
        self.pixels       = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def logging_level(self):
        """Logging level for this test."""

        return _BackplaneComparison.STATUS_LEVEL[self.status]

    @staticmethod
    def set_no_gold_master_status(is_ok=False):
        """Set the success value of status "No gold master".

        This is defined globally by input argument "ignore_missing".
        """

        level = 'WARNING' if is_ok else 'ERROR'
        _BackplaneComparison.STATUS_LEVEL['No gold master'] = level

################################################################################
# BackplaneTest class
#
# This class manages information about the backplanes of a particular
# observation.
################################################################################

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

    def __init__(self, obs, planets, moons, rings, overrides, args, suffix=''):
        """Construct a BackplaneTest for the given observation.

        Input:
            obs         Observation.
            planets     list of one or more planet names to use in backplanes.
            moons       list of one or more moon names to use in backplanes.
            rings       list of one or more ring names to use in backplanes.
            overrides   dictionary of test overrides for this observation.
            args        A Namespace object containing the command line inputs.
            suffix      a suffix string used to distinguish between multiple
                        Observations all defined within the same data file; it
                        is appended to the array and browse directory names to
                        make them unique.
        """

        self.obs = obs
        self.overrides = overrides
        self.args = args
        self.suffix = suffix

        self.upward = obs.fov.uv_scale.vals[1] < 0.     # direction of v-axis
        self.full_shape = (obs.uv_shape[::-1] if obs.swap_uv else obs.uv_shape)

        # Copy a few key args into self
        self.task        = args.task
        self.derivs      = args.derivs
        self.undersample = args.undersample
        self.inventory   = args.inventory
        self.border      = args.border

        # Identify the planet, body and ring names
        self.body_names = []
        self.limb_names = []
        self.ring_names = []
        self.ansa_names = []
        self.planet_moon_pairs = []
        self.planet_ring_pairs = []

        # Fill in all the backplane surface names
        for body in planets + moons:
            if body:
                self.body_names.append(body)
                self.limb_names.append(body + ':LIMB')

                if Body.lookup(body).ring_body:
                    self.ring_names.append(body + ':RING')
                    self.ansa_names.append(body + ':ANSA')
                    self.planet_ring_pairs.append((body, body + ':RING'))

        for moon in moons:
            if moon:
                planet = Body.lookup(moon).parent.name.upper()
                self.planet_moon_pairs.append((planet, moon))

        for ring in rings:
            if ring:
                self.ring_names.append(ring)
                self.ansa_names.append(ring + ':ANSA')

                planet = Body.lookup(ring).parent.name.upper()
                pair = (planet, ring)
                if pair not in self.planet_ring_pairs:
                    self.planet_ring_pairs.append(pair)

        # Create backplane object plus four with offset meshgrids
        EPS = 1.e-5
        self.origins = [(0.5 + self.args.du      , 0.5 + self.args.dv      ),
                        (0.5 + self.args.du - EPS, 0.5 + self.args.dv      ),
                        (0.5 + self.args.du + EPS, 0.5 + self.args.dv      ),
                        (0.5 + self.args.du      , 0.5 + self.args.dv - EPS),
                        (0.5 + self.args.du      , 0.5 + self.args.dv + EPS)]
            # By setting origin to 0.5 and requiring undersampling to be
            # integral, we ensure that an undersampled meshgrid will always
            # sample the centers of pixels in the original (u,v) grid.
        self.duv = 2 * EPS

        self.meshgrids = []
        for origin in self.origins:
            meshgrid = obs.meshgrid(origin=origin,
                                    undersample=self.undersample,
                                    center_uv=np.array(obs.uv_shape)/2.)
            self.meshgrids.append(meshgrid)

        if self.inventory:
            inventory_dict = {'inventory': {}, 'inventory_border': self.border}
        else:
            inventory_dict = {'inventory': None}

        self.backplanes = []
        for meshgrid in self.meshgrids:
            backplane = Backplane(obs, meshgrid=meshgrid, **inventory_dict)
            self.backplanes.append(backplane)

        # Select the primary meshgrid and backplane
        self.meshgrid = self.meshgrids[0]
        self.backplane = self.backplanes[0]
        self.backplane.ALL_DERIVS = True

        # Determine file paths. Example:
        # filespec = $OOPS_TEST_DATA_PATH/cassini/ISS/N1460072401_1.IMG
        # masters: $OOPS_GOLD_MASTER_PATH/hosts.cassini.iss/ISS/N1460072401_1/
        # arrays: $OOPS_BACKPLANE_OUTPUT_PATH/N1460072401_1/arrays
        # browse: $OOPS_BACKPLANE_OUTPUT_PATH/N1460072401_1/browse
        # gold masters sampled at the undersampling grid:
        #         $OOPS_BACKPLANE_OUTPUT_PATH/N1460072401_1/sampled_gold

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
        self.sampled_gold = os.path.join(self.output_dir,
                                          'sampled_gold' + self.suffix)

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

        # Make sure the output directory exits
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up the log handler; set aside any old log
        # Note that each BackplaneTest gets its own dedicated logger
        LOGGING.push()
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

                if self.args.save_sampled:
                    LOGGING.info('Writing sampled masters to', self.sampled_gold)
                    os.makedirs(self.sampled_gold, exist_ok=True)

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
                        LOGGING.exception(e, '%s | Fatal error in %s'
                                             % (TEST_SUITE, LATEST_TITLE))
                    else:
                        LOGGING.exception(e, '%s | Fatal error' % TEST_SUITE)

            # Wrap up
            if self.task in ('preview', 'compare'):
                file_path = self.write_summary(self.output_dir)
                LOGGING.debug('Summary written: ' + file_path)

            else:
                file_path = self.write_summary(self.gold_dir)
                LOGGING.info('Summary written: ' + file_path)

            # Internals...
            if self.args.internals:
                LOGGING.push()
                LOGGING.set_logger_level('DEBUG')
                bp = self.backplane

                for i in (False, True):
                    LOGGING.diagnostic('\nSurface Events, derivs=%s' % i)
                    keys = list(bp.surface_events[i].keys())
                    keys.sort()
                    for key in keys:
                        sum = np.sum(bp.surface_events[i][key].mask)
                        LOGGING.diagnostic('   ', key, sum)

                for i in (False, True):
                    LOGGING.diagnostic('\nIntercepts, derivs=%s' % i)
                    keys = list(bp.intercepts[i].keys())
                    keys.sort(key=BackplaneTest._sort_key)
                    for key in keys:
                        sum = np.sum(bp.intercepts[i][key].mask)
                        LOGGING.diagnostic('   ', key, sum)

                LOGGING.diagnostic('\nGridless arrivals')
                keys = list(bp.gridless_arrivals.keys())
                keys.sort(key=BackplaneTest._sort_key)
                for key in keys:
                    sum = np.sum(bp.gridless_arrivals[key].mask)
                    LOGGING.diagnostic('   ', key, sum)

                LOGGING.diagnostic('\nBackplanes')
                keys = list(bp.backplanes.keys())
                keys.sort(key=BackplaneTest._sort_key)
                for key in keys:
                    sum = np.sum(bp.backplanes[key].mask)
                    if key in bp.backplanes_with_derivs:
                        derivs = bp.backplanes_with_derivs[key].derivs
                        flag = ' '
                        if 't' in derivs:
                            if 'los' in derivs:
                                flag = '*'
                            else:
                                flag = 't'
                        elif 'los' in derivs:
                            flag = 'l'
                    else:
                        flag = ' '
                    LOGGING.diagnostic('   %s%s' % (flag, key), sum)

                LOGGING.diagnostic('\nAntimasks')
                keys = list(bp.antimasks.keys())
                keys.sort()
                for key in keys:
                    antimask = bp.antimasks[key]
                    info = ('array' if isinstance(antimask, np.ndarray)
                                    else str(antimask))
                    LOGGING.diagnostic('   ', key, '(%s)' % info)

                LOGGING.diagnostic()
                LOGGING.pop()

            seconds = (datetime.datetime.now() - start).total_seconds()
            LOGGING.info('Elapsed time: %.3f s' % seconds)

        # Be sure to remove the BackplaneTest-specific file handler afterward
        finally:
            if LOGGING.warnings:
                LOGGING.debug('Total warnings = ' + str(LOGGING.warnings))
            if LOGGING.errors:
                LOGGING.info('Total errors = ' + str(LOGGING.errors),
                             force=True)

            if self.args.log:
                LOGGING.logger.removeHandler(handler)
                handler.close()

            LOGGING.pop()

    @staticmethod
    def _sort_key(key):
        """Key function for the sort operation, needed to handle occurrences of
        Frames, Paths, and None in some dictionary keys.

        Also allow sorting among numbers, strings and tuples: numbers first,
        strings second, objects third, tuples fourth.
        """

        if isinstance(key, (tuple, list)):
            return 4, tuple(BackplaneTest._sort_key(item) for item in key)
        if isinstance(key, numbers.Real):
            return 1, key
        if isinstance(key, str):
            return 2, key
        return 3, str(key)

    _MINMAX_VALUES = {
        'float': (sys.float_info.min, sys.float_info.max),
        'bool' : (False, True),
        'int'  : (-sys.maxsize - 1, sys.maxsize),
    }

    #===========================================================================
    def compare(self, array, master, title, limit=0., method='', operator='=',
                                            radius=0., mask=False):
        """Compare two backplane arrays and log the results.

        Note that the array can be a backplane that has been undersampled. The
        gold master array can be either full-resolution or undersampled.

        Inputs:
            array       backplane array to be compared.
            master      reference value or gold master array.
            title       title string describing the test; must be unique.
            limit       upper limit on the difference between the arrays.
            method      ''        for standard comparisons;
                        'mod360'  for doing comparisons in degrees mod 360;
                        'degrees' for doing comparisons in degrees;
                        'border'  for comparisons of border backplanes, in which
                                  case the radius value is interpreted in units
                                  of undersampled pixels rather than original
                                  pixels;
            operator    the operator to use for the comparison, one of '=', '>',
                        '>=', '<', or '<='.
            radius      the radius of a circle, in units of pixels, by which to
                        check for a possible spatial shift for the values the
                        mask. This values is rounded down, so radius < 1
                        indicates no shift.
            mask        optional mask to apply. Mask areas are not included in
                        the comparison.
        """

        global TEST_SUITE, LATEST_TITLE

        LATEST_TITLE = title
        (array, comparison) = self._validate_inputs(array, title, limit, method,
                                                    operator, radius, mask)
        comparison.suite = TEST_SUITE

        if self.args.task == 'compare':
            if comparison.limit is None:
                LOGGING.debug(comparison.suite, f'| Canceled: "{title}"')
                return

        else:               # adopt or preview
            LOGGING.debug(comparison.suite, f'| Summary: "{title}";',
                          comparison.text)
            return

        if method in ('mod360', 'degrees'):
            master = master * DPR
            # array is already converted to degrees by _validate_inputs

        self._compare(array, master, comparison)
        LATEST_TITLE = ''

    #===========================================================================
    def gmtest(self, array, title, limit=0., method='', operator='=',
                                   radius=0., mask=False):
        """Compare a backplane array against its gold master. Save the array,
        browse image, and sampled master.

        Inputs:
            array       backplane array to be tested.
            title       title string describing the test; must be unique.
            limit       upper limit on the difference between the arrays.
            method      ''        for standard comparisons;
                        'mod360'  for doing comparisons in degrees mod 360;
                        'degrees' for doing comparisons in degrees;
                        'border'  for comparisons of border backplanes, in which
                                  case the radius value is interpreted in units
                                  of undersampled pixels rather than original
                                  pixels;
            operator    the operator to use for the comparison, one of '=', '>',
                        '>=', '<', or '<='.
            radius      the radius of a circle, in units of pixels, by which to
                        check for a possible spatial shift for the values the
                        mask. This values is rounded down, so radius < 1
                        indicates no shift.
            mask        optional mask to apply. Mask areas are not included in
                        the comparison.
        """

        global TEST_SUITE, LATEST_TITLE

        # Validate inputs
        LATEST_TITLE = title
        (array, comparison) = self._validate_inputs(array, title, limit, method,
                                                    operator, radius, mask)
        comparison.suite = TEST_SUITE

        # Handle a 2-D array
        if array.shape:

            # Determine the storage precision
            # Gradients are saved at single precision
            if limit == 0.:
                array.set_pickle_digits(('double', 'single'), 'fpzip')
            else:
                # Could save at reduced precision, but better to use full...
                # digits = -np.log10(limit) + 1       # save one extra digit
                # array.set_pickle_digits((digits, 'single'), (1., 'fpzip'))
                array.set_pickle_digits(('double', 'single'), 'fpzip')

            # Write the pickle file
            if self.task == 'adopt':
                output_arrays = self.gold_arrays
                output_browse = self.gold_browse
                basename = self._basename(title, gold=True)
            else:
                output_arrays = self.output_arrays
                output_browse = self.output_browse
                basename = self._basename(title, gold=False)

            output_pickle_path = os.path.join(output_arrays,
                                              basename + '.pickle')
            comparison.output_pickle_path = output_pickle_path
            with open(output_pickle_path, 'wb') as f:
                pickle.dump(array, f)

            # Write the browse image
            if self.args.browse:
                browse_name = basename + '.' + self.args.browse_format
                browse_path = os.path.join(output_browse, browse_name)
                comparison.browse_path = browse_path
                self.save_browse(array, browse_path)

            # For "compare"
            if self.task == 'compare':
                basename = self._basename(title, gold=True)
                gold_pickle_path = os.path.join(self.gold_arrays,
                                                basename + '.pickle')
                comparison.gold_pickle_path = gold_pickle_path

                # Handle a missing pickle file
                if not os.path.exists(gold_pickle_path):
                    self._log_comparison(comparison, 'No gold master')

                else:
                    # Retrieve pickled backplane
                    try:
                        with open(gold_pickle_path, 'rb') as f:
                            master = pickle.load(f)
                    except (ValueError, TypeError, OSError):
                        self._log_comparison(comparison, 'Invalid gold master')

                    # Compare...
                    else:
                        if self.args.save_sampled:
                            basename = self._basename(comparison.title,
                                                      gold=False)
                            comparison.sampled_gold_path = os.path.join(
                                                           self.sampled_gold,
                                                           basename + '.pickle')

                        self._compare(array, master, comparison)

            # For "preview" and "adopt"
            else:
                LOGGING.debug(comparison.suite, '| Written:',
                              os.path.basename(output_pickle_path) + ';',
                              comparison.text)

        # Shapeless case
        else:

            # For "compare"
            if self.task == 'compare':
                if title not in self.gold_summary:
                    self._log_comparison(comparison, 'No gold master')

                else:
                    (min_val, max_val, masked,
                                       unmasked) = self.gold_summary[title]
                    # If gold master value is not shapeless...
                    if min_val != max_val or masked + unmasked > 1:
                        self._log_comparison(comparison, 'Shape mismatch')

                    else:
                        master = Scalar(min_val, masked > 0)
                        self._compare(array, master, comparison)

            # For "preview" and "adopt"
            else:
                LOGGING.debug(comparison.suite, f'| Summary: "{title}";',
                              comparison.text)

        LATEST_TITLE = ''

    #===========================================================================
    def _compare(self, array, master, comparison):
        """Internal method that performs a comparison _after_ the inputs have
        been validated. Radians must already be converted to degrees.
        """

        # Make objects suitable and compatible
        array_dtype = array.dtype()
        if array_dtype == 'int':        # convert ints to floats
            array = array.as_float()

        master = array.as_this_type(master, recursive=True, coerce=True)

        # Broadcast a shapeless master object
        if array.shape and not master.shape:
            master = master.broadcast_to(array.shape)
            master = master.remask_or(array.mask)

        # Expand masks
        array = array.expand_mask()
        master = master.expand_mask()

        # Set aside derivs, if any, in case needed
        array_derivs = array.derivs
        array = array.wod

        master_derivs = master.derivs
        master = master.wod

        # Convert a Boolean array to integers, internally int8
        if array_dtype == 'bool':
            array = Scalar(array.vals.astype('int8'), array.mask)
            master = Scalar(master.vals.astype('int8'), master.mask)

        # A comparison with an undersampled border requires special handling.
        # In this case, the master array must be re-sampled in a way such that
        # a new pixel is True if any of the pixels from which it is derived are
        # True.
        if comparison.method == 'border' and self.undersample != 1:
            master_vals = master.vals.copy()
            master_vals[master.mask] = 0
            new_vals = maximum_filter(master_vals, self.undersample,
                                      mode='constant', cval=0)
            if np.any(master.mask):
                new_mask = minimum_filter(master.mask, self.undersample,
                                          mode='constant', cval=1)
            else:
                new_mask = np.zeros(new_vals.shape, dtype='bool')

            master = Scalar(new_vals, mask=new_mask)

        # Re-sample master at the array's meshgrid if necessary
        if array.shape == master.shape:
            master_grid = master
            indx = slice(None)          # this index does nothing
        elif self.undersample == 1:
            self._log_comparison(comparison, 'Shape mismatch')
            return
        else:
            if self.obs.swap_uv:
                indx = (self.meshgrid.uv.vals[...,1].astype('int'),
                        self.meshgrid.uv.vals[...,0].astype('int'))
            else:
                indx = (self.meshgrid.uv.vals[...,0].astype('int'),
                        self.meshgrid.uv.vals[...,1].astype('int'))

            # For "border", down-sample the master
            if comparison.method == 'border':
                master = master[indx]
                master_grid = master
                indx = slice(None)
            else:                       # Otherwise, master stays at full
                                        # resolution; master_grid is resampled.
                master_grid = master[indx]
                master_grid = master_grid.expand_mask()

        # The "_grid" suffix indicates the master array when sampled at the
        # meshgrid of the array.

        # Saved the sampled array if necessary
        if hasattr(comparison, 'sampled_gold_path'):
            with open(comparison.sampled_gold_path, 'wb') as f:
                pickle.dump(master_grid, f)

        # Find the differences among unmasked pixels
        diff = array - master_grid
        if comparison.method == 'mod360':
            diff = (diff - 180.) % 360. - 180.

        if comparison.operator == '=':
            diff = diff.abs()
        elif comparison.operator[0] == '>':
            diff = -diff

        max_diff = diff[comparison.antimask].max(builtins=True, masked=-1)
        comparison.max_diff1 = max_diff
        comparison.max_diff2 = max_diff

        # Compare masks
        mask_diff_mask = (array.mask ^ master_grid.mask) & comparison.antimask
        mask_errors = np.sum(mask_diff_mask)
        comparison.mask_errors1 = mask_errors
        comparison.mask_errors2 = mask_errors

        # Handle a fully masked result
        zero = 0. if array.dtype() == 'float' else 0
        if max_diff < 0:
            comparison.max_diff1 = zero
            comparison.max_diff2 = zero

            # diff_errors2 is zero by default inside the comparison, so success
            # or failure will depend on the mask_errors2
            self._log_comparison(comparison)
            return

        # Compare values
        if comparison.operator[-1] == '=':
            tvl_error_mask = diff.tvl_gt(comparison.limit)
        else:
            tvl_error_mask = diff.tvl_ge(comparison.limit)

        diff_error_mask = (tvl_error_mask.as_mask_where_nonzero()
                           & comparison.antimask)
            # mask where array and master are both unmasked, and their
            # difference exceeds the limit
        diff_errors = np.sum(diff_error_mask)
        comparison.diff_errors1 = diff_errors
        comparison.diff_errors2 = diff_errors

        # If no errors, we're done
        if diff_errors + mask_errors == 0:
            self._log_comparison(comparison)
            return

        # If we have no flexibility, we're done
        if (comparison.radius == 0. or comparison.operator != '='
                                    or not array.shape):
            self._log_comparison(comparison)
            return

        # Get the gradient, if any, trying various sources
        d_dlos = None
        grad = None
        if 'los' not in array_derivs and hasattr(array, 'backplane_key'):
            array_derivs = self.backplane.evaluate(array.backplane_key,
                                                   derivs=True).derivs

        if 'los' in array_derivs:
            d_dlos = array_derivs['los']

        if d_dlos is None and 'los' in master_derivs:
            d_dlos = master_derivs['los'][indx]

        if d_dlos is not None:
            d_duv = d_dlos.chain(self.backplane.dlos_duv)
            grad = d_duv.join_items(Pair).norm()

        # If a gradient is available, use it to determine the magnitude of any
        # pointing offset. If this is below the radius limit, success.
        footprint = None        # footprint map
        if comparison.operator == '=' and diff_errors and grad is not None:

            grad_vals = grad.vals.copy()
            grad_vals[grad.mask] = 1.e-99
                # small but > 0., which forces offset to be large
            distance = None

            for k in range(2):
                # If the first pass failed, maybe the issue is that we're at the
                # edge of a mask, where a gradient value is missing. Handle this
                # by using the radius limit to expand the number of pixels where
                # the gradient is defined, and then re-calculating the offset.
                if k == 1:
                    if comparison.radius < 1.:
                        break

                    footprint = BackplaneTest._footprint(comparison.radius)
                    grad_vals = maximum_filter(grad_vals, footprint=footprint,
                                               mode='constant', cval=1.e-99)
                    distance = comparison.radius

                # Require all offsets to be <= radius and all diffs <= limit
                offset_to_zero = diff / grad_vals
                clipped_offset = offset_to_zero.clip(-comparison.radius,
                                                      comparison.radius)
                improved_diff = diff - grad_vals * clipped_offset
                new_invalid_diff_mask = (improved_diff.abs() > comparison.limit)

                if not new_invalid_diff_mask.any():
                    diff_errors = 0
                    max_diff = improved_diff.abs().max(builtins=True)

                    if distance is None:
                        selected_offsets = clipped_offset[diff_errors]
                        distance = selected_offsets.abs().max(builtins=True,
                                                              masked=0.)

                    comparison.max_diff2 = max_diff
                    comparison.diff_errors2 = diff_errors
                    comparison.distance = distance

                    if mask_errors:
                        break

                    self._log_comparison(comparison)
                    return

        # At this point, our only hope of resolving discrepancies involves
        # invoking the radius limit. If this radius limit is less than 1, then
        # the footprint for re-sampling is a single pixel, and therefore won't
        # accomplish anything.

        if comparison.radius < 1:
            self._log_comparison(comparison)
            return

        comparison.distance = comparison.radius

        # See if any mask discrepancy is compatible with the radius...
        #
        # The masks are compatible if...
        # - everywhere array.mask is True, so is master.mask once expanded by
        #   radius;
        # - everywhere array.mask is False, so is master.mask contracted by
        #   radius.
        #
        # Another way of saying this is that inside the region where master.mask
        # expanded equals master.mask contracted, array.mask must equal
        # master.mask. Elsewhere, a discrepancy is OK.

        if mask_errors:
            if comparison.method == 'border':
                footprint = BackplaneTest._footprint(comparison.radius
                                                     * self.undersample)
            elif footprint is None:
                footprint = BackplaneTest._footprint(comparison.radius)

            comparison.distance = comparison.radius
            master_mask_contracted = minimum_filter(master.mask,
                                                    footprint=footprint,
                                                    mode='constant', cval=True)
            master_mask_expanded   = maximum_filter(master.mask,
                                                    footprint=footprint,
                                                    mode='constant', cval=False)
            region_mask = (master_mask_expanded == master_mask_contracted)

            # Apply the grid and compare; re-calculate mask errors
            region_grid = region_mask[indx]
                # region_grid has the shape of master, so we have to re-sample
                # it at the meshgrid to match the array.
            temp_mask = region_grid & comparison.antimask
                # ...and then apply the comparison antimask

            comparison.mask_errors2 = np.sum(array.mask[temp_mask] !=
                                             master_grid.mask[temp_mask])

        # Determine if the full radius can solve any value discrepancies...
        #
        # These are OK if the value of each discrepant pixel in the array is
        # within the range of the nearby unmasked pixels in the master array,
        # +/- the limit.

        if diff_errors:

            # Determine the range of values in master adjacent to each value in
            # the array. Note that these are full-resolution arrays.
            if footprint is None:
                footprint = BackplaneTest._footprint(comparison.radius)

            extremes = BackplaneTest._MINMAX_VALUES[array_dtype]
            master_vals = master.vals.copy()
            master_vals[master.mask] = extremes[1]
            min_master_vals = minimum_filter(master_vals, footprint=footprint,
                                             mode='constant', cval=extremes[1])
            master_vals[master.mask] = extremes[0]
            max_master_vals = maximum_filter(master_vals, footprint=footprint,
                                             mode='constant', cval=extremes[0])

            # Resample these versions of the master to match the array.
            min_master_grid = min_master_vals[indx]
            max_master_grid = max_master_vals[indx]

            # Then select the discrepant pixels
            array_vals = array.vals[diff_error_mask]
            min_vals = min_master_grid[diff_error_mask]
            max_vals = max_master_grid[diff_error_mask]

            # Locate and check the revised discrepancies
            diffs_below = min_vals - array_vals
            diffs_above = array_vals - max_vals
            comparison.max_diff2 = max(0, np.max(diffs_below),
                                          np.max(diffs_above))

            if comparison.operator[-1] == '=':
                diff_error_mask_below = diffs_below > comparison.limit
                diff_error_mask_above = diffs_above > comparison.limit
            else:
                diff_error_mask_below = diffs_below >= comparison.limit
                diff_error_mask_above = diffs_above >= comparison.limit

            comparison.diff_errors2 = (np.sum(diff_error_mask_below) +
                                       np.sum(diff_error_mask_above))

        self._log_comparison(comparison)
        return

    #===========================================================================
    def _validate_inputs(self, array, title, limit, method, operator, radius,
                               mask):
        """Initial steps for both compare() and gmtest()."""

        global TEST_SUITE

        # Validate comparison options
        if method not in ('', 'mod360', 'degrees', 'border'):
            raise ValueError('unknown comparison method: ' + repr(method))

        if operator not in ('=', '>', '>=', '<', '<='):
            raise ValueError('unknown operator: ' + repr(operator))

        if operator != '=' and radius != 0:
            raise ValueError('operator "%s" '  % operator
                             + ' is incompatible with nonzero radius')

        if operator != '=' and method == 'border':
            raise ValueError('operator "%s" ' % operator
                             + 'is incompatible with method "border"')

        # Validate limit
        if title in self.overrides:
            limit = self.overrides[title]
        elif isinstance(limit, Qube):
            if np.any(limit.mask):
                limit = 0.
            else:
                limit = abs(limit.vals * self.args.tolerance)
        else:
                limit = abs(limit * self.args.tolerance)

        # Warn about duplicated titles
        if title in self.results:
            LOGGING.error(TEST_SUITE, f'| Duplicated title: "{title}";',
                          title)

        # Validate array
        if not isinstance(array, Qube):
            if Qube._dtype(array) == 'bool':
                array = Boolean(array)
            else:
                array = Scalar(array)

        # Convert to degrees if necessary
        if method in ('mod360', 'degrees'):
            new_array = array * DPR
            if hasattr(array, 'backplane_key'):
                new_array.backplane_key = array.backplane_key
            array = new_array

        # Strip any derivatives other than d_dlos
        if tuple(array.derivs.keys()) not in ((), ('los',)):
            new_array = array.without_derivs(preserve='los')
            if hasattr(array, 'backplane_key'):
                new_array.backplane_key = array.backplane_key
            array = new_array

        # Check mask
        if np.shape(mask) not in ((), array.shape):
            raise ValueError('mask shape does not match array')

        # Construct the comparison object
        comparison = _BackplaneComparison(title = title,
                                          limit = limit,
                                          method = method,
                                          operator = operator,
                                          radius = radius * self.args.radius,
                                          mask = mask,
                                          pixels = array.size)

        # Extra attributes
        comparison.antimask = np.logical_not(mask)
        comparison.text = self._summarize(array, title, method=method)

        return (array, comparison)

    #===========================================================================
    def _summarize(self, array, title, method):
        """Save the summary info for this backplane array.

        For boolean arrays, the saved tuple is
            (False count, True count, masked count, total pixels)
        For floats, the saved tuple is:
            (minimum value, maximum vale, masked count, total pixels)
        The cases are distinguished by whether the first value is int of float.
        """

        def _summary_text(minval, maxval, masked, total):
            """Save the summary info and return a formatted text string."""

            self.summary[title] = (minval, maxval, masked, total)

            message = []
            if isinstance(minval, numbers.Integral):   # if array is boolean
                message += ['false,true=%d,%d' % (minval, maxval)]
            elif masked < total:
                if minval == maxval:
                    message += ['value=', self._valstr(minval)]
                else:
                    message += ['min,max=', self._valstr(minval), ',',
                                            self._valstr(maxval)]
            if masked:
                percent = (100. * masked) / total
                if 99.9 < percent < 100.:   # "100.0"% means every pixel
                    percent = 99.9
                if message:
                    message += [', ']
                message += ['mask=', '%.1f' % percent, '%']

            return ''.join(message)

        masked = array.count_masked()
        total = array.size

        if array.is_bool():
            trues  = np.sum(array.antimask & array.vals)
            falses = np.sum(array.antimask & np.logical_not(array.vals))
            return _summary_text(falses, trues, masked, total)

        if masked == total:
            return _summary_text(array.default, array.default, masked, total)

        if method != 'mod360':
            return _summary_text(array.min(builtins=True),
                                 array.max(builtins=True),
                                 masked, total)

        # Handle mod360 case

        # Convert to a 1-D array of values
        if np.shape(array.mask):
            vals = array.vals[array.antimask]
        elif np.shape(array.vals):
            vals = array.vals.ravel()
        else:
            return _summary_text(array.vals, array.vals, masked, total)

        # Sort mod 360
        sorted = np.sort(vals)
        sorted = np.hstack((sorted, [sorted[0] + 360]))
                                        # duplicate first item at end
        diffs = np.diff(sorted)         # find successive difference
        argmax = np.argmax(diffs)       # locate largest gap
        maxval = sorted[argmax]
        if argmax == len(diffs) - 1:    # min is angle after largest gap
            minval = sorted[-1] - 360
        else:
            minval = sorted[argmax + 1]

        return _summary_text(minval, maxval, masked, total)

    #===========================================================================
    def _log_comparison(self, comparison, status=''):
        """Log this comparison info.

        A single record of the log file has this format:
          "<time> | oops.backplane.gold_master | <level> | <suite> | <message>"
        where
          <time>  is the local time to the level of ms.
          <level> is one of "DEBUG", "INFO", "WARNING", "ERROR", "FATAL".
          <suite> is the name of the test suite, e.g., "ring".
          <message> is a descriptive message.

        For comparisons, the message has the following format:
          <status>: "<title>"; diff=<diff1>/<diff2>/<limit>; ...
                               offset=<offset>/<radius>; ...
                               pixels=<count1>/<count2>/<pixels>
        where:
          <status> is one of:
            "Success"               if the test passed;
            "Value mismatch"        if the values disagree by more than the
                                    limit, but the mask is in agreement;
            "Mask mismatch"         if the masks disagree, but the values are in
                                    agreement;
            "Value/mask mismatch"   if both the values and the mask disagree.
          <title>   is the title of the test.
          <diff1>   is the maximum discrepancy among the unmasked values.
          <diff2>   is the maximum discrepancy after we have expanded the
                    comparison to include neighboring pixels, as defined by the
                    specified radius.
          <limit>   the specified discrepancy limit of the test.
          <offset>  the offset distance required to bring value discrepancies
                    below the limit, or to resolve any mask discrepancies.
          <radius>  the specified upper limit on an offset.
          <count1>  the number of discrepant pixels before allowing for an
                    offset.
          <count2>  the number of discrepant pixels that cannot be accommodated
                    by an offset.
          <pixels>  the total number of pixels tested.

        Note that <diff2> and <count2> are not listed if not offset is required.
        Also, note that the offset values are not listed in this case.
        """

        global TEST_SUITE

        # Fill in the status if necessary
        if status:
            comparison.status = status

        errors1 = comparison.diff_errors1 + comparison.mask_errors1
        errors2 = comparison.diff_errors2 + comparison.mask_errors2
        if comparison.status == '':         # if not yet filled in
            if comparison.diff_errors2:
                if comparison.mask_errors2:
                    comparison.status = 'Value/mask mismatch'
                else:
                    comparison.status = 'Value mismatch'
            else:
                if comparison.mask_errors2:
                    comparison.status = 'Mask mismatch'
                else:
                    comparison.status = 'Success'

        # Construct the message
        message = [comparison.suite, ' | ', comparison.status, ': "',
                   comparison.title, '"; ', comparison.text]

        # Handle failed comparison cases
        if comparison.status == 'Success' or errors2 > 0:
            message += ['; diff=', self._valstr(comparison.max_diff1)]

            if comparison.max_diff2 != comparison.max_diff1:
                if comparison.max_diff2 <= comparison.limit * 1.e-8:
                    if isinstance(comparison.max_diff2, numbers.Integral):
                        message += ['/0']
                    else:
                        message += ['/0.']
                else:
                    message += ['/', self._valstr(comparison.max_diff2)]

            message += ['/']
            if (isinstance(comparison.max_diff2, numbers.Integral)
                    and comparison.limit == 0):
                message += ['0']
            else:
                message += [self._valstr(comparison.limit)]

            if comparison.distance:
                message += [', offset=', self._valstr(comparison.distance),
                            '/', self._valstr(comparison.radius)]

            message += [', pixels=', str(errors1)]

            if errors2 != errors1:
                message += ['/', str(errors2)]

            message += ['/', str(comparison.pixels)]

        LOGGING.print(''.join(message), level=comparison.logging_level)

        # Save the results
        self.results[comparison.title] = comparison

        # Log the file paths
        ATTRS = ('gold_pickle_path', 'output_pickle_path', 'browse_path',
                 'sampled_gold_path')
        LABELS = ('master:', 'array:', 'browse:', 'sampled master:')

        if self.args.fullpaths:
            path_logged = False
            for (attr, label) in zip(ATTRS, LABELS):
                if hasattr(comparison, attr):
                    path = getattr(comparison, attr)
                    LOGGING.info(TEST_SUITE, '|', label, path, force=True)
                    path_logged = True

            if path_logged:
                LOGGING.literal()

    #===========================================================================
    @staticmethod
    def _valstr(value):
        """value formatter, avoiding "0.000" and "1.000e-12"."""

        if isinstance(value, Qube):
            if value.mask:
                return '--'
            value = value.vals

        if isinstance(value, numbers.Integral):
            return str(value)
        else:
            formatted = '%#.4g' % value
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
    # pixels). If the minimum and maximum are equal, only one value is listed;
    # if the object is fully masked, the value is None.
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
                if value[0] is None:
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
# To handle gold master testing from the command line...
################################################################################

if __name__ == '__main__':

    import oops.backplane.gold_master as gm

    # Define the default observation
    gm.set_default_obs(
            obspath = os.path.join(OOPS_TEST_DATA_PATH,
                                   'cassini/ISS/W1573721822_1.IMG'),
            index   = None,
            planets = ['SATURN'],
            moons   = ['EPIMETHEUS'],
            rings   = ['SATURN_MAIN_RINGS'])

    gm.set_default_args(module='hosts.cassini.iss')

    # The d/dv numerical ring derivatives are extra-uncertain due to the high
    # foreshortening in the vertical direction.

    gm.override('SATURN longitude d/du self-check (deg/pix)', 0.3)
    gm.override('SATURN longitude d/dv self-check (deg/pix)', 0.05)
    gm.override('SATURN_MAIN_RINGS azimuth d/dv self-check (deg/pix)', 1.)
    gm.override('SATURN_MAIN_RINGS distance d/dv self-check (km/pix)', 0.3)
    gm.override('SATURN_MAIN_RINGS longitude d/dv self-check (deg/pix)', 1.)
    gm.override('SATURN:RING azimuth d/dv self-check (deg/pix)', 0.1)
    gm.override('SATURN:RING distance d/dv self-check (km/pix)', 0.3)
    gm.override('SATURN:RING longitude d/dv self-check (deg/pix)', 0.1)

    gm.execute_as_command()

################################################################################

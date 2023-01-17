################################################################################
# oops/backplane/unittester_support.py
################################################################################

import os
import numbers
import numpy as np

from polymath import Scalar
from PIL      import Image

#===============================================================================
def _read_image(filename):
    """Save an image file of a 2-D array.

    Input:        filename    the name of the input file, which should end with
                              the type, e.g., '.png' or '.jpg'
    """
    try:
        im = Image.open(filename, mode='r')
    except FileNotFoundError:
        return None

    return Scalar.as_scalar((np.array(im)))

#===============================================================================
def _save_image(image, filename):
    """Read an image file of a 2-D array.

    Input:
        image       a 2-D byte array.
        filename    the name of the output file, which should end with the
                    type, e.g., '.png' or '.jpg'
    """

    shape = image.shape[::-1]
    if len(shape) == 1:
        shape = (shape[0],1)

    im = Image.frombytes('L', shape, image)
    im.save(filename)

#===============================================================================
def _scale_image(array, minval, maxval):
    """Rescale an image and convert to byte."""

    image = array.vals.copy()
    image[array.mask] = minval - 0.05 * (maxval - minval)

    image = np.asfarray(image)

    lo = image.min()
    hi = image.max()

    if hi == lo:
        bytes = np.zeros(image.shape, dtype=np.int8)
    else:
        scaled = (image[::-1] - lo) / float(hi - lo)
        bytes = (256.*scaled).clip(0,255).astype('uint8')

    return bytes

#===============================================================================
def _compare_backplanes(array, reference, margin=0.05):
    """Compare a backplane array to the reference array."""

    array = Scalar.as_scalar(array)
    reference = Scalar.as_scalar(reference)

    diff = abs(array - reference)
    assert diff.max() <= reference.max()*margin

#===============================================================================
def _convert_filename(filename):
    """Convert file-system-unfriendly characters in a backplane filename."""

    filename = filename.replace(':','_')
    filename = filename.replace('/','_')
    filename = filename.replace(' ','_')
    filename = filename.replace('(','_')
    filename = filename.replace(')','_')
    filename = filename.replace('[','_')
    filename = filename.replace(']','_')
    filename = filename.replace('&','_')
    filename = filename.replace(',','_')
    filename = filename.replace('-','_')
    filename = filename.replace('__','_')
    filename = filename.replace('__','_')
    filename = filename.replace('_.','.')
    filename = filename.replace("'",'')
    return filename.lower().rstrip('_')

#===============================================================================
def _construct_filename(bp, array, title, dir):
    """Construct a backplane filename."""

    # Construct base filename from title
    filename = _convert_filename('backplane-' +
                                 bp.obs.basename.split('.')[0] + '-' + title)

    # Ensure unique filename by using the backplane key, if it exists
    # NOTE: if no backplane key exists, a non-unique filename could result
    if hasattr(array, 'key'):
        key = array.key
        id = key[0]
        for item in key[1:]:
            if item != ():
                id = id + '_' + str(item)
        id = _convert_filename(id)
#        filename = filename + '[' + id + ']'
        filename = filename + '--' + id

    filename = filename + '.png'

    # Add path
    filename = os.path.join(dir, filename)
    return filename

#===============================================================================
def _print(*x, printing=True):
    """Print contignent upon verbosity."""

    if not printing:
        return

    print(*x)

#===============================================================================
def show_info(bp, title, array, **options):
    """Internal method to print summary information and display images."""

    #Get options
    printing =  options.get('printing', True)
    saving =    options.get('saving', False)
    dir =       options.get('dir', './')
    refdir =    options.get('refdir', None)

    # No action if no input arary or no actions requested
    if array is None:
        return

    if not printing and not saving and refdir is None:
        return

    _print(title, printing=printing)

    # Scalar summary
    if isinstance(array, numbers.Number):
        _print('  ', array, printing=printing)

    # Mask summary
    elif (type(array.vals) == bool or
            (isinstance(array.vals, np.ndarray) and
             array.vals.dtype == np.dtype('bool'))):
        count = np.sum(array.vals)
        total = np.size(array.vals)
        percent = int(count / float(total) * 100. + 0.5)
        _print('  ', (count, total-count),
               (percent, 100-percent), '(True, False pixels)', printing=printing)
        minval = 0.
        maxval = 1.

    # Unmasked backplane summary
    elif array.mask is False:
        minval = np.min(array.vals)
        maxval = np.max(array.vals)
        if minval == maxval:
            _print('  ', minval, printing=printing)
        else:
            _print('  ', (minval, maxval), '(min, max)', printing=printing)

    # Masked backplane summary
    else:
        _print('  ', (array.min().as_builtin(),
               array.max().as_builtin()), '(masked min, max)', printing=printing)
        total = np.size(array.mask)
        masked = np.sum(array.mask)
        percent = int(masked / float(total) * 100. + 0.5)
        _print('  ', (masked, total-masked),
               (percent, 100-percent), '(masked, unmasked pixels)', printing=printing)

        if total == masked:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
        else:
            minval = array.min().as_builtin()
            maxval = array.max().as_builtin()

    # The rest of the method applies only to arrays
    if array.shape == ():
        return

    # scale image
    if minval == maxval:
        maxval += 1.
    image = _scale_image(array, minval, maxval)

    # Save backplane array if directed
    if saving:
        os.makedirs(dir, exist_ok=True)

        filename = _construct_filename(bp, array, title, dir)
        _save_image(image, filename)

    # Compare with reference array if refdir is known
    if refdir is not None:
        assert os.path.exists(refdir), f'Reference directory not found: {refdir}'

        filename = _construct_filename(bp, array, title, refdir)
        reference = _read_image(filename)
        assert reference is not None, f'Reference file is missing: {filename}'

        _compare_backplanes(image, reference)


#*******************************************************************************
class Backplane_Settings(object):
    """Class for storing command-line preferences."""

    # Static default values
    ARGS = None
    GENERIC_ARGS = []
    EXERCISES_ONLY = False
    NO_EXERCISES = False
    NO_COMPARE = False
    SAVING = True
    LOGGING = False
    PRINTING = False
    UNDERSAMPLE = 16
    REF = False

    # TBD attributes
#    DIFF = []
#    PLANET_KEY = None
#    MOON_KEY = None
#    RING_KEY = None

    # Attributes that depend on the input file.  Defaults are determined
    # in exercise_backplanes_settings() for each input file.
    OUTPUT = None
    REFERENCE = None


#===============================================================================
def backplane_unittester_args():
    """Parse command-line arguments for backplane unit tests.

    This function must be called by main() in order for these arguments to
    to be parsed.  If not, the above default values will apply."""
    import argparse
    import sys

    ## Define arguments ##
    parser = argparse.ArgumentParser(description='Backplane unit tester.')

    # Generic arguments
    parser.add_argument('--args', nargs='*', metavar='arg', default=None,
                        help="""Generic arguments to pass to the test modules.
                                Must occur last in the argument list.""")

    # Basic controls
    parser.add_argument('--verbose', action='store_true', default=None,
                        help='Print output to the terminal.')

#    parser.add_argument('--diff', nargs=2, metavar=('old', 'new'), default=None,
#                        help='Compare new and old backplane logs.')
#
    parser.add_argument('--exercises-only', action='store_true', default=None,
                        help='Execute only the backplane exercises.')

    parser.add_argument('--no-exercises', action='store_true', default=None,
                        help='Execute all tests except the backplane exercises.')

    parser.add_argument('--no-compare', action='store_true', default=None,
                        help='Do not compare backplanes with references.')

    parser.add_argument('--output', nargs=1, metavar='dir', default=None,
                        help="""Directory in which to save backplane PNG images.
                                Default is $OOPS_BACKPLANE_OUTPUT_PATH/[data dir].
                                If the directory does not exist, it is created.""")

    parser.add_argument('--no-output', action='store_true', default=None,
                        help='Disable saving of backplane PNG files.')

    parser.add_argument('--log', action='store_true', default=None,
                        help='Enable the internal oops logging.')

    parser.add_argument('--undersample', nargs=1, type=int, metavar='N', default=None,
                        help='Amount by which to undersample backplanes.  Default is 16.')

    parser.add_argument('--reference', action='store_true', default=None,
                        help='Generate reference backplanes and exit.')

### TODO: currently only works for 80-char width
    parser.add_argument('--test-level', nargs=1, type=int, metavar='N', default=None,
                        help="""Selects among pre-set parameter combinations:
                                -test-level 1: no printing, no saving, undersample 32.
                                -test-level 2: printing, no saving, undersample 16.
                                -test-level 3: printing, saving, no undersampling.
                                These behaviors are overridden by other arguments.""")

    ## Parse arguments, leaving unknown args for some other parser ##
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1]+left
    Backplane_Settings.ARGS = args                # Save command args

    ## Implement argments ##
    if args.test_level is not None:
        test_level = args.test_level[0]
        if test_level == 1:
            Backplane_Settings.PRINTING = False
            Backplane_Settings.SAVING = False
            Backplane_Settings.UNDERSAMPLE = 32

        if test_level == 2:
            Backplane_Settings.PRINTING = True
            Backplane_Settings.SAVING = False
            Backplane_Settings.UNDERSAMPLE = 16

        if test_level == 3:
            Backplane_Settings.PRINTING = True
            Backplane_Settings.SAVING = True
            Backplane_Settings.UNDERSAMPLE = 1

    if args.args is not None:
        Backplane_Settings.GENERIC_ARGS = args.args

    if args.verbose is not None:
        Backplane_Settings.PRINTING = args.verbose

#   TBD
#    if args.diff is not None:
#        _diff_logs(args.diff[0], args.diff[1], verbose=Backplane_Settings.PRINTING)
#        exit()

    if args.exercises_only is not None:
        Backplane_Settings.EXERCISES_ONLY = args.exercises_only

    if args.no_exercises is not None:
        Backplane_Settings.NO_EXERCISES = args.no_exercises

    if args.no_compare is not None:
        Backplane_Settings.NO_COMPARE = args.no_compare

    if args.output is not None:
        Backplane_Settings.OUTPUT = args.output[0]

    if args.no_output is not None:
        Backplane_Settings.SAVING = not args.no_output

    if args.log is not None:
        Backplane_Settings.LOGGING = args.log

    if args.undersample is not None:
        Backplane_Settings.UNDERSAMPLE = args.undersample[0]

    if args.reference is not None:
        Backplane_Settings.EXERCISES_ONLY = True
        Backplane_Settings.NO_COMPARE = True
        Backplane_Settings.SAVING = True
        Backplane_Settings.REF = True

#   TBD
#    # Body keywords
#    if '--planet' in sys.argv:
#        k = sys.argv.index('--planet')
#        Backplane_Settings.PLANET_KEY = sys.argv[k+1]
#        del sys.argv[k:k+2]

#    if '--moon' in sys.argv:
#        k = sys.argv.index('--moon')
#        Backplane_Settings.PLANET_KEY = sys.argv[k+1]
#        del sys.argv[k:k+2]

#    if '--ring' in sys.argv:
#        k = sys.argv.index('--ring')
#        Backplane_Settings.PLANET_KEY = sys.argv[k+1]
#        del sys.argv[k:k+2]

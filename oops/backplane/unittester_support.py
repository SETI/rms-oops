################################################################################
# oops/backplane/unittester_support.py
################################################################################

import unittest
import re
import os
import sys
import numpy as np

from polymath   import Scalar
from PIL        import Image


#===============================================================================
def _read_image(filename):
    """Save an image file of a 2-D array.

    Input:        filename    the name of the input file, which should end with
                              the type, e.g., '.png' or '.jpg'
    """
    try:
        im = Image.open(filename, mode="r")
    except:
        return(None)

    return(Scalar.as_scalar((np.array(im))))



#===============================================================================
def _save_image(image, filename):
    """Read an image file of a 2-D array.

    Input:
        image       a 2-D byte array.
        filename    the name of the output file, which should end with the
                    type, e.g., '.png' or '.jpg'
    """

    im = Image.frombytes('L', (image.shape[1], image.shape[0]), image)
    im.save(filename)



#===============================================================================
def _scale_image(array, minval, maxval):
    """Rescales an image and converts to to byte.
    """

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

    return(bytes)



#===============================================================================
def _compare_backplanes(array, reference, margin=0.05):
    """
    Compare a backplane array to the reference array.
    """

    array = Scalar.as_scalar(array)
    reference = Scalar.as_scalar(reference)

    diff = abs(array - reference)
    assert(diff.max() <= reference.max()*margin)



#===============================================================================
def _convert_filename(filename):
    """Converts file-system-unfriendly characters in a backplane filename.
    """

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
    return(filename.lower().rstrip('_'))



#===============================================================================
def _construct_filename(array, title, dir):
    """Constructs a backplane filename.
    """

    # Construct base filename from title
    filename = _convert_filename('backplane-' + title)

    # Ensure unique filename by using the backplane key, if it exists
    # NOTE: if no backplane key exists, a non-unique filename could result
    try:
        key = array.key
        id = key[0]
        for item in key[1:]:
            if item != ():
                id = id + '_' + str(item)
        id = _convert_filename(id)
        filename = filename + '[' + id + ']' + '.png'
    except:
        filename = filename + '.png'

    # Add path
    filename = os.path.join(dir, filename)
    return(filename)



#===============================================================================
def _print(*x, printing=True):
    """Prints contignent upon verbosity.
    """
    if not printing:
        return

    print(*x)


#===============================================================================
def show_info(title, array, printing=True, saving=False, dir='./', refdir=None):
    """Internal method to print summary information and display images as
    desired.
    """

    import numbers

    if not printing and not saving and refdir==None:
        return

    _print(title, printing=printing)


    # Scalar summary
    if isinstance(array, numbers.Number):
        _print('  ', array, printing=printing)

    # Mask summary
    elif type(array.vals) == bool or \
            (isinstance(array.vals, np.ndarray) and \
             array.vals.dtype == np.dtype('bool')):
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

        filename = _construct_filename(array, title, dir)
        _save_image(image, filename)


    # Compare with reference array if refdir is known
    if refdir is not None:
        assert os.path.exists(refdir), \
            "No reference directory.  Use --no_compare, --no_exercises, or --reference."

        filename = _construct_filename(array, title, refdir)
        reference = _read_image(filename)
        if reference is not None:
            _compare_backplanes(image, reference)



#===============================================================================
def _diff_logs(old_log, new_log, verbose=False):
    """
    For each numeric entry in the new log file that differs from that in the
    old log file by more than 0.1%, this program prints out the header and the
    discrepant records, with the percentage changes in all numeric values
    appended.

    If the verbose option is specified, the program prints out the percentage
    change in every numeric value; in this case, a string of asterisks is
    appended to the ends of rows where any change exceeds 0.1%.
    """
    REGEX = re.compile('r[-+]?\d+\.?\d*[eE]?[-+]?\d*')

    if '--verbose' in sys.argv[1:]:
        verbose = True
        sys.argv.remove('--verbose')
    else:
        verbose = False

    with open(sys.argv[1]) as f:
        oldrecs = f.readlines()

    with open(sys.argv[2]) as f:
        newrecs = f.readlines()

    header = ''
    prev_header = ''
    first_test_results_found = False
    for k in range(min(len(oldrecs), len(newrecs))):
        oldrec = oldrecs[k]
        newrec = newrecs[k]

        oldvals = REGEX.findall(oldrec)
        newvals = REGEX.findall(newrec)

        if oldrec.startswith('Ran 1 test in'):
            if not first_test_results_found:
                print('File structure has changed; no tests performed')
            sys.exit()

        if oldrec.startswith('**'):
            first_test_results_found = True
            continue

        if not first_test_results_found:
            continue

        if oldrec[0].isupper():
            header = oldrec

            if oldrec != newrec:
                print('File structure has changed')
                sys.exit()

            continue

        if len(oldvals) != len(newvals):
            print()
            print(header[:-1])
            print(oldrec[:-1])
            print(newrec[:-1])
            print('Mismatch in number of numeric values')
            prev_header = header
            prev_oldrec = oldrec

        percentages = []
        discrepancy = False
        for j in range(min(len(oldvals), len(newvals))):
            x = float(oldvals[j])
            y = float(newvals[j])

            if x == 0. and y == 0.:
                percent = 0.
            else:
                percent = 200. * abs((x-y)/(x+y))

            percentages.append(percent)
            if percent > 0.1:
                discrepancy = True

        if not percentages:
            continue

        if verbose or discrepancy:
            if prev_header != header:
                print()
                print(header[:-1])
                prev_header = header

            suffixes = []
            for percent in percentages:
                suffixes.append('%.4f' % percent)

            if discrepancy and verbose:
                stars = ' ********'
            else:
                stars = ''

            print(oldrec[:-1])
            print(newrec[:-1], '  (' + ', '.join(suffixes) + ')' + stars)



#*******************************************************************************
class Backplane_Settings(object):
    ARGS = []
    DIFF = []
    EXERCISES_ONLY = False
    NO_EXERCISES = False
    NO_COMPARE = False
    SAVING = True
    LOGGING = False
    PRINTING = True
    UNDERSAMPLE = 16
#    PLANET_KEY = None
#    MOON_KEY = None
#    RING_KEY = None
    OUTPUT = './output/'
    REFERENCE = os.path.join(OUTPUT, 'reference_' + str(UNDERSAMPLE) + '/')


#===============================================================================
def backplane_unittester_args():
    """
    Parse command-line arguments for backplane unit tests.  Results are
    stored as Test_Backplane_Exercises attributes.
    """

    # Generic arguments
    if '--args' in sys.argv:
        k = sys.argv.index('--args')
        Backplane_Settings.ARGS = sys.argv[k+1:]
        del sys.argv[k:]


    # Basic controls
    if '--help' in sys.argv:
        print("Usage:")
        print("  python unittester.py [--output dir] [--no_output] [--exercises_only]")
        print("                       [--no_exercises] [--reference] [--undersample #]")
        print("                       [--test_level #] [--no_compare] [--silent] [--log]")
        print("                       [--diff old new] [--help] [--args arg1 arg2 ...]")
        exit()

    if '--silent' in sys.argv:
        Backplane_Settings.PRINTING = False
        sys.argv.remove('--silent')

    if '--diff' in sys.argv:
        k = sys.argv.index('--diff')
        logs = sys.argv[k+1:k+3]
        del sys.argv[k:k+3]
        _diff_logs(logs[0], logs[1], verbose=Backplane_Settings.PRINTING)
        exit()

    if '--exercises_only' in sys.argv:
        Backplane_Settings.EXERCISES_ONLY = True
        sys.argv.remove('--exercises_only')

    if '--no_exercises' in sys.argv:
        Backplane_Settings.NO_EXERCISES = True
        sys.argv.remove('--no_exercises')

    if '--no_compare' in sys.argv:
        Backplane_Settings.NO_COMPARE = True
        sys.argv.remove('--no_compare')

    if '--output' in sys.argv:
        k = sys.argv.index('--output')
        Backplane_Settings.OUTPUT = sys.argv[k+1]
        del sys.argv[k:k+2]

    if '--no_output' in sys.argv:
        Backplane_Settings.SAVING = False
        sys.argv.remove('--no_output')

    if '--log' in sys.argv:
        Backplane_Settings.LOGGING = True
        sys.argv.remove('--log')

    if '--undersample' in sys.argv:
        k = sys.argv.index('--undersample')
        Backplane_Settings.UNDERSAMPLE = int(sys.argv[k+1])
        del sys.argv[k:k+2]

    if '--reference' in sys.argv:
        Backplane_Settings.EXERCISES_ONLY = True
        Backplane_Settings.SAVING = True
        Backplane_Settings.OUTPUT = Backplane_Settings.REFERENCE
        sys.argv.remove('--reference')


    # Pre-defined test configurations
    test_level = 0
    if '--test_level' in sys.argv:
        k = sys.argv.index('--test_level')
        test_level = int(sys.argv[k+1])
        del sys.argv[k:k+2]

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


    # Implement NO_COMPARE
    if Backplane_Settings.NO_COMPARE:
        Backplane_Settings.REFERENCE = None

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

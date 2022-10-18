################################################################################
# oops/backplane/unittester_support.py
################################################################################

import os
import sys
import numpy as np


#===============================================================================
def save_image(image, filename, lo=None, hi=None):
    """Save an image file of a 2-D array.

    Input:
        image       a 2-D array.
        filename    the name of the output file, which should end with the
                    type, e.g., '.png' or '.jpg'
        lo          the array value to map to black; if None, then the
                    minimum value in the array is used.
        hi          the array value to map to white; if None, then the
                    maximum value in the array is used.
    """

    from PIL import Image

    image = np.asfarray(image)

    if lo is None:
        lo = image.min()

    if hi is None:
        hi = image.max()

    if hi == lo:
        bytes = np.zeros(image.shape, dtype='uint8')
    else:
        scaled = (image[::-1] - lo) / float(hi - lo)
        bytes = (256.*scaled).clip(0,255).astype('uint8')

    im = Image.frombytes('L', (bytes.shape[1], bytes.shape[0]), bytes)
    im.save(filename)



#===========================================================================
def show_info(title, array, printing=True, saving=False, dir='./'):
    """Internal method to print summary information and display images as
    desired.
    """

    import numbers

    if not printing and not saving:
        return

    if printing:
        print(title)

    # Scalar summary
    if isinstance(array, numbers.Number):
        print('  ', array)

    # Mask summary
    elif type(array.vals) == bool or \
            (isinstance(array.vals, np.ndarray) and \
             array.vals.dtype == np.dtype('bool')):
        count = np.sum(array.vals)
        total = np.size(array.vals)
        percent = int(count / float(total) * 100. + 0.5)
        print('  ', (count, total-count),
                    (percent, 100-percent), '(True, False pixels)')
        minval = 0.
        maxval = 1.

    # Unmasked backplane summary
    elif array.mask is False:
        minval = np.min(array.vals)
        maxval = np.max(array.vals)
        if minval == maxval:
            print('  ', minval)
        else:
            print('  ', (minval, maxval), '(min, max)')

    # Masked backplane summary
    else:
        print('  ', (array.min().as_builtin(),
                     array.max().as_builtin()), '(masked min, max)')
        total = np.size(array.mask)
        masked = np.sum(array.mask)
        percent = int(masked / float(total) * 100. + 0.5)
        print('  ', (masked, total-masked),
                    (percent, 100-percent), '(masked, unmasked pixels)')

        if total == masked:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
        else:
            minval = array.min().as_builtin()
            maxval = array.max().as_builtin()

    if saving and array.shape != ():
        os.makedirs(dir, exist_ok=True)

        if minval == maxval:
            maxval += 1.

        image = array.vals.copy()
        image[array.mask] = minval - 0.05 * (maxval - minval)

        filename = 'backplane-' + title + '.png'
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
        filename = os.path.join(dir, filename)
        save_image(image, filename)



#===========================================================================
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
    DIFF = []
    EXERCISES_ONLY = False
    NO_EXERCISES = False
    SAVING = False
    OUTPUT = './output/'
    LOGGING = False
    PRINTING = True
    PLANET_KEY = None
    MOON_KEY = None
    RING_KEY = None
    UNDERSAMPLE = 1


#===============================================================================
def backplane_unittester_args():
    """
    Parse command-line arguments for backplane unit tests.  Results are
    stored as Test_Backplane_Exercises attributes.
    
                  *** NOTE will not work with ipython ***
                  
    """

    # Basic controls
    if '--silent' in sys.argv:
        Backplane_Settings.PRINTING = False
        sys.argv.remove('--silent')

    if '--diff' in sys.argv:
        k = sys.argv.index('--diff')
        logs = sys.argv[k+1:k+3]
        del sys.argv[k:k+3]
        _diff_logs(logs[0], logs[1], verbose=Backplane_Settings.PRINTING)
        exit

    if '--exercises_only' in sys.argv:
        Backplane_Settings.EXERCISES_ONLY = True
        sys.argv.remove('--exercises_only')

    if '--no_exercises' in sys.argv:
        Backplane_Settings.NO_EXERCISES = True
        sys.argv.remove('--no_exercises')

    if '--png' in sys.argv:
        Backplane_Settings.SAVING = True
        sys.argv.remove('--png')

    if '--out' in sys.argv:
        Backplane_Settings.EXERCISES_ONLY = True
        Backplane_Settings.SAVING = True
        k = sys.argv.index('--out')
        Backplane_Settings.OUTPUT = sys.argv[k+1]
        del sys.argv[k:k+2]

    if '--log' in sys.argv:
        Backplane_Settings.LOGGING = True
        sys.argv.remove('--log')

    if '--reference' in sys.argv:
        Backplane_Settings.EXERCISES_ONLY = True
        Backplane_Settings.SAVING = True
        Backplane_Settings.OUTPUT = \
                       os.path.join(Backplane_Settings.OUTPUT, 'reference')
        Backplane_Settings.UNDERSAMPLE = 16
        sys.argv.remove('--reference')

    if '--undersample' in sys.argv:
        k = sys.argv.index('--undersample')
        Backplane_Settings.UNDERSAMPLE = int(sys.argv[k+1])
        del sys.argv[k:k+2]


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




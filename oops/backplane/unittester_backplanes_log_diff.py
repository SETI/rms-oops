#!/usr/bin/env python3
################################################################################
# Usage:
#   python unittester_backplanes_log_diff.py old_log new_log [--verbose]
# where:
#   old_log and new_log are logs from runs of unittester_backplanes.py, e.g.,
#       backplane-unittest-results-2019-07-21/unittest-log-2019-07-21.txt
#
# For each numeric entry in the new log file that differs from that in the old
# log file by more than 0.1%, this program prints out the header and the
# discrepant records, with the percentage changes in all numeric values
# appended.
#
# If the --verbose option is specified, the program prints out the percentage
# change in every numeric value; in this case, a string of asterisks is appended
# to the ends of rows where any change exceeds 0.1%.
################################################################################

import sys
import re

REGEX = re.compile(r'[-+]?\d+\.?\d*[eE]?[-+]?\d*')

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

################################################################################

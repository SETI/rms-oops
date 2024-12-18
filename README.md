[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-oops)](https://github.com/SETI/rms-oops/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-oops)](https://github.com/SETI/rms-oops/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-oops/run-tests.yml?branch=main)](https://github.com/SETI/rms-oops/actions)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-oops/main?logo=codecov)](https://codecov.io/gh/SETI/rms-oops)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-oops)](https://pypi.org/project/rms-oops)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-oops)](https://pypi.org/project/rms-oops)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-oops)](https://pypi.org/project/rms-oops)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-oops)](https://pypi.org/project/rms-oops)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-oops/latest)](https://github.com/SETI/rms-oops/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-oops)](https://github.com/SETI/rms-oops/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-oops)](https://github.com/SETI/rms-oops/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-oops)](https://github.com/SETI/rms-oops/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-oops)](https://github.com/SETI/rms-oops/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-oops)](https://github.com/SETI/rms-oops/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-oops)](https://github.com/SETI/rms-oops/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-oops)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-oops)](https://github.com/SETI/rms-oops/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-oops)

# rms-oops

This package is under development. Use with extreme caution.

# Environment Variables

- `OOPS_RESOURCES`: The top-level directory containing all files needed by OOPS. Unless
  overriden as described below, this environment variable is the only one that needs to be
  set. It is expected that the specified directory will contain the subdirectories:
  - `SPICE`: SPICE kernels and associated database.
  - `HST`: Reference and calibration files required for HST.
  - `JWST`: Reference and calibration files required for JWST.
  - `gold_master`: Gold master files for host tests.
  - `test_data`: Test input files.
- `SPICE_PATH`: The location of the SPICE kernel files; defaults to
  `${OOPS_RESOURCES}/SPICE`.
- `SPICE_SQLITE_DB_NAME`: The full path and filename of the SPICE SQlite database;
  defaults to `${SPICE_PATH}/SPICE.db`.
- `OOPS_TEST_DATA_PATH`: The location of the oops test files; defaults to
  `${OOPS_RESOURCES}/test_data`.
- `OOPS_GOLD_MASTER_PATH`: The location of the oops gold master test files; defaults to
  `${OOPS_RESOURCES}/gold_master`.
- `OOPS_BACKPLANE_OUTPUT_PATH`: The output path to use when writing backplanes
  for gold master tests; defaults to the current directory.
- `HST_IDC_PATH`: The location of HST IDC files; defaults to
  `${OOPS_RESOURCES}/HST/IDC`.
- `HST_SYN_PATH`: The location of HST SYN files; defaults to
  `${OOPS_RESOURCES}/HST/SYN`.

# Running Tests

- To run the main oops unit tests:

```sh
python -m unittest tests/unittester.py
```

- To run the host tests including golden master tests:

```sh
python -m unittest tests/hosts/unittester.py
```

- To run the main oops unit tests and the host tests:

```sh
python -m unittest tests/unittester_with_hosts.py
```

- To run the gold master tests for one instrument with the ability to specify command
  line options:

```sh
export PYTHONPATH=.
python tests/hosts/cassini/iss/gold_master.py --help
python tests/hosts/galileo/ssi/gold_master.py --help
```

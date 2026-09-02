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

# Repository Layout

- `src/oops`: The `oops` library.
- `src/spicedb`: The `spicedb` library.
- `programs/gold_master`: The gold master backplane test framework, imported as
  `programs.gold_master`. It is a runnable tool rather than part of the `oops` API, so
  it lives outside `src`.
- `tests`: The unit tests, mirroring `src/oops`, plus the host tests under
  `tests/hosts` and the `spicedb` tests under `tests/spicedb`.
- `scripts`: `setup-venv.sh`, `run-all-checks.sh`, and the automated test script CI
  runs.

# Development Setup

The library packages live under `src`, so they are importable only after an editable
install (or with `src` on `PYTHONPATH`). To create the virtual environment the check
script expects and install the package with its development extras:

```sh
./scripts/setup-venv.sh
source venv/bin/activate
```

To run the checks:

```sh
./scripts/run-all-checks.sh
```

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

The tests use pytest.

- To run the main oops unit tests:

```sh
pytest tests --ignore=tests/hosts --ignore=tests/spicedb
```

- To run the host tests, which are the gold master tests:

```sh
pytest tests/hosts
```

- To run the spicedb tests:

```sh
pytest tests/spicedb
```

- To run everything:

```sh
pytest tests
```

- To run the full set of quality gates (ruff, flake8, mypy, pyroma, bandit,
  vulture, the three test suites, and the documentation build):

```sh
./scripts/run-all-checks.sh
```

- To build the documentation on its own:

```sh
./scripts/run-all-checks.sh --sphinx
```

- To run the gold master tests for one instrument with the ability to specify command
  line options:

```sh
export PYTHONPATH=.
python tests/hosts/cassini/iss/gold_master.py --help
python tests/hosts/galileo/ssi/gold_master.py --help
```

- To compare against a set of gold master files somewhere other than the default, use
  `--gold-master` on either the instrument command or pytest. It overrides
  `$OOPS_GOLD_MASTER_PATH` and `$OOPS_RESOURCES` for that run only:

```sh
pytest tests/hosts --gold-master=/path/to/masters
PYTHONPATH=. python tests/hosts/cassini/iss/gold_master.py --gold-master=/path/to/masters
```

  The directory must have the standard layout, in which the files for one observation
  are found in `<path>/<module>/<basename>`, such as
  `<path>/oops.hosts.cassini.iss/W1573721822_1`. As with the environment variables, the
  path may name a cloud resource such as `gs://rms-oops-resources/gold_master`.

Installation and setup
======================

Requirements
------------

``oops`` runs on Python 3.11 or later, on macOS, Linux and Windows. It depends on
``rms-polymath`` for its array types, ``cspyce`` for the SPICE toolkit, ``rms-spicedb``
for kernel selection, ``rms-filecache`` for file access, and NumPy, SciPy and PIL;
``pip`` installs all of them.

Installing
----------

Into a virtual environment::

    python -m venv venv
    source venv/bin/activate
    pip install rms-oops

To work from a checkout instead, which is what the gold master tool and the host tests
need, see :doc:`the Developer's Guide </dev_guide/dev_guide_environment>`; in short::

    git clone https://github.com/SETI/rms-oops.git
    cd rms-oops
    ./scripts/setup-venv.sh
    source venv/bin/activate

The resource tree
-----------------

Nothing runs without external data. The library reads SPICE kernels and, for some
instruments, calibration files, and the test tooling reads test images and reference
arrays. They live in one directory tree with this layout:

::

    OOPS-Resources/
    ├── SPICE/                 The SPICE kernels, and SPICE.db, the SQLite database
    │                          that spicedb uses to select them by body and time
    ├── HST/
    │   ├── IDC/               Hubble distortion coefficient tables
    │   └── SYN/               Hubble throughput tables
    ├── JWST/                  Webb reference files
    ├── test_data/             The standard observations of each instrument,
    │                          e.g. cassini/ISS/W1573721822_1.IMG
    └── gold_master/           The reference backplanes of each standard observation,
                               e.g. cassini.iss/W1573721822_1/arrays/*.pickle

Point ``OOPS_RESOURCES`` at it, and everything else defaults to a subdirectory. The
Ring-Moon Systems Node maintains the tree; a copy is published at
``gs://rms-oops-resources``, and any of the variables below may name a cloud path such
as ``gs://rms-oops-resources/gold_master`` directly, because every file is read through
``filecache``.

Environment variables
---------------------

=============================== =================================== ======================
Variable                        Default                             Used for
=============================== =================================== ======================
``OOPS_RESOURCES``              (required)                          the root of the tree
``SPICE_PATH``                  ``$OOPS_RESOURCES/SPICE``           the kernels
``SPICE_SQLITE_DB_NAME``        ``$SPICE_PATH/SPICE.db``            the kernel database
``OOPS_TEST_DATA_PATH``         ``$OOPS_RESOURCES/test_data``       standard observations
``OOPS_GOLD_MASTER_PATH``       ``$OOPS_RESOURCES/gold_master``     reference backplanes
``OOPS_BACKPLANE_OUTPUT_PATH``  the current directory               gold master output
``HST_IDC_PATH``                ``$OOPS_RESOURCES/HST/IDC``         Hubble distortion
``HST_SYN_PATH``                ``$OOPS_RESOURCES/HST/SYN``         Hubble throughput
=============================== =================================== ======================

For example, in a shell profile::

    export OOPS_RESOURCES=/Users/Shared/OOPS-Resources

Checking the setup
------------------

This defines the Saturn system for one month and prints the shape model of the planet:

.. code-block:: python

    import oops
    from oops.body import Body

    kernels = Body.define_solar_system('2007-11-01', '2007-11-30', planets=6)
    print(kernels)
    print(Body.lookup('SATURN').surface)

It should print a list of a dozen kernel names, beginning with the leap-second and
planetary-constant kernels and ending with the Saturn ephemerides, and then a
:class:`~oops.surface.Spheroid`. An ``OSError`` from ``cspyce`` mentioning ``SPICE.db``,
or a ``KeyError`` for ``OOPS_RESOURCES``, means the variables above are not set in the
process that ran Python.

Always ``import oops`` before importing any of its subpackages or modules. The package
wires its classes together at import, and a module imported on its own is left
incomplete and fails in obscure ways.

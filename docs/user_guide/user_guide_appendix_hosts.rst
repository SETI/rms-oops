Appendix: the host modules
==========================

A host module reads the files of one instrument and returns an
:class:`~oops.Observation`. Each is imported by its full name, ``from oops.hosts.cassini
import iss``, and exposes ``from_file``; several also expose ``from_index`` to read a
whole PDS index table into a list of observations, and an ``initialize`` function to
furnish the mission's kernels with options before the first file is read. The host
modules are still being reworked, so their keywords vary, and the signatures below are
the current ones.

The common keywords are ``planets`` (restrict the kernels to one planet by number),
``asof`` (use only kernels available by that date), ``return_all_planets`` (load the
kernels of every planet rather than the mission's target), and for cameras
``fast_distortion`` (``True`` for a pre-inverted distortion polynomial, ``False`` for one
solved dynamically, ``None`` for a distortion-free field). ``**parameters`` and
``**kwargs`` absorb keywords the module does not use.

Cassini
-------

``oops.hosts.cassini.iss``
    ``from_file(filespec, *, fast_distortion=True, return_all_planets=False, frame=None,
    navigation=False, **kwargs)`` returns a :class:`~oops.observation.Snapshot` for an ISS
    NAC or WAC image or its label. ``frame`` substitutes a pointing frame;
    ``navigation=True`` wraps the frame so that its pointing can be fitted.
    ``from_index(filespec, ...)`` reads an index table.
    ``initialize(ck='reconstructed', planets=None, asof=None, spk='reconstructed',
    gapfill=True, mst_pck=True, irregulars=True)`` selects reconstructed or predicted
    C-kernels and SPKs (or ``'none'`` to leave kernel loading to the caller), gap-filling
    C-kernels, the small-moon constants and the irregular satellites.
``oops.hosts.cassini.vims``
    ``from_file(filespec, data=True, method='strict')`` returns a pair of observations,
    the visible and infrared channels of a VIMS cube.
``oops.hosts.cassini.uvis``
    ``from_file(filespec, data=True, enclose=False, method='strict', **parameters)``
    returns one or more observations for a UVIS spectrograph file.

Galileo
-------

``oops.hosts.galileo.ssi``
    ``from_file(filespec, return_all_planets=False, full_fov=False, method='strict',
    **parameters)`` returns a :class:`~oops.observation.Snapshot`; ``full_fov`` returns the
    full 800-line field for a summation-mode image. ``from_index(filespec,
    supplemental_filespec=None, full_fov=False, **parameters)`` and
    ``initialize(planets=None, asof=None, mst_pck=True, irregulars=True)``.

Juno
----

``oops.hosts.juno.junocam``
    ``from_file(filespec, fast_distortion=True, return_all_planets=False, snap=False,
    method='strict', **parameters)`` returns a list of observations, one per framelet;
    ``snap=True`` returns snapshots at the framelet midtimes. The wide, distorted field
    makes the inventory unreliable; the gold master tests turn it off.
``oops.hosts.juno.jiram``
    ``from_file(filespec, return_all_planets=False, method='strict', **parameters)``
    returns one or more observations for a JIRAM image (``.IMG``) or spectrum (``.DAT``).
``oops.hosts.juno.sru``
    ``from_file(filespec, return_all_planets=False, method='strict', **parameters)``
    returns a :class:`~oops.observation.Snapshot` for a star reference unit image.

New Horizons
------------

``oops.hosts.newhorizons.lorri``
    ``from_file(filespec, geom='spice', pointing='spice', fov_type='fast', asof=None,
    meta=None, **parameters)`` returns a :class:`~oops.observation.Snapshot`; ``geom`` and
    ``pointing`` may be ``'fits'`` to take the trajectory and pointing from the FITS
    header instead of SPICE, and ``fov_type`` is ``'fast'``, ``'slow'`` or ``'flat'``.
    ``from_index(filespec, fov_type='fast', asof=None, meta=None, **parameters)``.

Voyager
-------

``oops.hosts.voyager.iss``
    ``from_file(filespec, astrometry=False, action='error', method='strict',
    parameters=None)`` returns a :class:`~oops.observation.Snapshot` for a Voyager ISS
    image or label. ``from_index(filespec, geomed=False, action='ignore', omit=True,
    parameters={})``.

Hubble, Webb and Keck
---------------------

``oops.hosts.hst``
    ``from_file(filespec, **parameters)`` dispatches on the FITS header to ACS (HRC, SBC,
    WFC), NICMOS (NIC1, NIC2, NIC3), WFC3 (IR, UVIS) or WFPC2, each in its own subpackage.
    The distortion and throughput tables come from ``$HST_IDC_PATH`` and
    ``$HST_SYN_PATH``.
``oops.hosts.jwst``
    ``from_file(filespec, **options)`` dispatches to NIRCam, returning a
    :class:`~oops.observation.TimedImage` for calibrated and uncalibrated products.
``oops.hosts.keck``
    ``from_file(filespec, **parameters)`` reads a Keck II NIRC2 FITS file; distortion
    models and other instruments are not supported.

These three define the solar system over their mission's whole date range when they
are imported, so importing one loads kernels as a side effect.

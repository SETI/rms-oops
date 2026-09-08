Introduction
============

Who this guide is for
---------------------

This guide assumes a competent Python developer who is new to this codebase. It is
organized around the architecture: what each subsystem is responsible for, what contract
its abstract base class imposes, and what has to change, in which files, when a new class
or a new backplane is added. It does not repeat what the docstrings already say; the API
reference generated from them is linked from every chapter, and the
:doc:`private copy <dev_guide_api>` of that reference shows the private members that a
change to the library has to work with.

The :doc:`User's Guide </user_guide/user_guide>` is the other half of the documentation.
It shows how to load an observation, describe the solar system around it, and compute
geometry. Read its :doc:`introduction </user_guide/user_guide_introduction>` first if the
vocabulary of events, paths, frames and backplanes is new to you, because this guide uses
that vocabulary without pausing to define it.

Two areas are deliberately left out of this guide. The host modules under
``oops.hosts``, which read the files of particular instruments, are still being
reworked, so the process of adding a new instrument is not documented yet. The
:mod:`spicedb` package, which selects SPICE kernels from a database, is documented only
by its API reference.

What the package does
---------------------

``oops`` is an observation-geometry library for planetary science. Given an observation
from a spacecraft or telescope, it answers the question "what is at each pixel?": which
body or ring the line of sight intercepts, where on that surface, at what distance, under
what lighting, and at what time. The answers are *backplanes*: arrays with the shape of
the observation, one per geometric quantity.

Computing them requires four kinds of model, each with its own abstract class and
subpackage:

* the motion of a point through space (:class:`~oops.Path`) and the orientation of a
  coordinate frame (:class:`~oops.Frame`), which together describe spacecraft, planets and
  the instruments they carry;
* the shape of a target (:class:`~oops.Surface`) and the gravity field that shapes ring
  orbits (:class:`~oops.Gravity`), tied to a named target by :class:`~oops.Body`;
* the geometry of an instrument's field of view (:class:`~oops.FOV`), the timing of its
  exposure (:class:`~oops.Cadence`) and the meaning of its data numbers
  (:class:`~oops.Calibration`), assembled into an :class:`~oops.Observation`;
* the link between them: an :class:`~oops.Event` is a photon arriving at or departing from
  a point in spacetime, and the photon solvers of :class:`~oops.Path` and
  :class:`~oops.Surface` connect one event to another along a light path.

The :class:`~oops.Backplane` class drives all of this for every pixel at once and caches
each intermediate result, so that the hundred or so named backplanes share a handful of
expensive photon solutions.

Dependencies
------------

The library runs on Python 3.11 or later (``requires-python`` in ``pyproject.toml``) and
builds on four packages:

``polymath``
    The array types. Every geometric quantity in ``oops`` is a :class:`~oops.Scalar`,
    :class:`~oops.Vector3`, :class:`~oops.Matrix3` or another :class:`~oops.Qube`
    subclass, never a bare NumPy array. A ``Qube`` carries a mask and, optionally,
    derivatives, and both propagate through arithmetic. Reading ``.vals`` under the mask
    returns garbage; combine with ``.antimask`` first.

``cspyce``
    The SPICE toolkit. Ephemerides, orientations, body constants and time conversions all
    come from SPICE kernels, which must be furnished before any of them is asked for.
    ``oops`` calls ``cspyce.use_errors()`` and ``cspyce.use_aliases()`` at import, so SPICE
    errors arrive as Python exceptions and body names may be given by alias.

``spicedb``
    Selects the kernels for a body and a time range from a SQLite database of kernel
    metadata, and furnishes them. :meth:`~oops.Body.define_solar_system` and the host
    modules use it; nothing in the geometry core does.

``filecache``
    Every file path in the test tooling is an ``FCPath``, so that the resource tree may
    live on a local disk or in cloud storage without the code caring which.

NumPy and SciPy are used directly in a few places, chiefly the interpolators of the
``QuickPath`` and ``QuickFrame`` classes and the filters of the gold master comparison.
PIL writes the browse images of the gold master tool.

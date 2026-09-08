Events, paths and frames
========================

Overview
--------

This subsystem is the kinematic core: where things are, how frames are oriented, and
how a photon connects one point in spacetime to another. It is spread over
``oops/event.py``, ``oops/transform.py``, the ``oops.path`` subpackage and the
``oops.frame`` subpackage, and everything else in the library is built on it. The API
reference for the base classes is on the :doc:`oops package page </oops>` and the
concrete classes are on the :doc:`oops.path </oops_path>` and :doc:`oops.frame
</oops_frame>` pages.

Event
-----

An :class:`~oops.Event` is a time, a position and a velocity, relative to a path (its
``origin``) and expressed in a frame. The velocity travels as the ``t`` derivative of the
position, so ``state`` is a :class:`~oops.Vector3` with a ``d_dt`` and ``pos`` and ``vel``
are views of it. The constructor accepts ``(pos, vel)`` as a two-tuple and inserts a zero
velocity when none is given. The time is seconds TDB, the position km, and every
property need not have the same shape, only shapes that broadcast together.

The optional photon properties are what make an event more than a state vector:

``arr``, ``arr_ap``
    The direction of an arriving photon, actual and apparent (aberrated), in the event
    frame; ``arr_j2000`` and ``arr_ap_j2000`` are the same in J2000, and ``neg_arr`` and
    its variants are the reversed directions. Lengths are arbitrary.
``dep``, ``dep_ap``, and their J2000 forms
    The direction of a departing photon.
``arr_lt``, ``dep_lt``
    The light travel time, negative for an arrival and positive for a departure.
``perp``
    The surface normal at the event, set by the surface photon solver.
``vflat``
    The velocity within the surface, such as winds or orbital motion.

Each may be set exactly once after construction, and only one of the actual and apparent
forms may be set; the other is derived on demand. Setting a second time raises
``ValueError``. Every value entering an event is made read-only at the boundary, in the
constructor and in every setter, which is what allows events to be shared through the
backplane caches without copying.

``wrt(path, frame)`` re-expresses an event relative to another path and frame, using the
registries to find the linking path and frame; ``wrt_ssb`` reduces it to SSB coordinates
in J2000 and caches the result on the event, and ``from_ssb`` goes the other way. The
lighting methods ``incidence_angle``, ``emission_angle``, ``phase_angle`` and
``ra_and_dec`` compute from the photon directions and ``perp``; they default to
``apparent=False``, whereas the corresponding backplane methods default to
``apparent=True``.

Subfields are arbitrary extra attributes attached with ``insert_subfield``. They ride
along through ``copy``, ``mask_where``, ``shrink`` and ``unshrink``, which re-apply their
operation to every property and every subfield, and a subfield inserted into an event
that has an SSB twin is inserted into the twin as well, rotated into J2000 when it is a
vector. The backplane engine relies on this to tag each surface event with its ``body``,
``surface`` and ``event_key``.

Invariants for code that produces events:

* Give ``origin`` as a waypoint and ``frame`` as a wayframe; the constructor converts a
  path, a frame or an ID through ``as_waypoint`` and ``as_wayframe``.
* Never mutate an event. ``copy(omit=...)``, ``replace(...)``, ``mask_where``,
  ``without_derivs`` and the ``wrt`` family return new objects.
* A new setter must call ``empty_cache``, which discards the lazily computed SSB twin,
  J2000 transform, mask and shape.
* ``collapse_time`` replaces a time array whose spread is below
  :class:`~oops.config.EVENT_CONFIG` ``.collapse_threshold`` with a single value, which
  lets a later ``QuickPath`` window stay small. :class:`~oops.Backplane` applies the same
  idea to the observation times.

Transform
---------

A :class:`~oops.Transform` is a rotation matrix, the angular velocity of the rotating
frame, and the ``frame`` and ``reference`` it relates, with the ``origin`` about which it
rotates. ``rotate`` and ``unrotate`` apply it to positions, with the velocity correction
that a rotating frame requires, and two transforms compose. ``Transform.IDENTITY`` is
assigned in ``oops/__init__.py``.

Path
----

A :class:`~oops.Path` is a point moving through space. Its one abstract method is
``event_at_time(time, *, quick=None)``, which returns an event with at least a time, a
position and a velocity relative to the path's ``origin`` in its ``frame``. The path and
the time need not share a shape; broadcasting applies.

A subclass constructor must set ``_origin`` (a waypoint), ``_frame`` (a wayframe) and
``_shape``, then call ``self._register(path_id)`` and ``self.refresh()``. Registration
gives the path its ``path_id``, files it in ``Path._PATH_REGISTRY`` and its *waypoint*,
the canonical registered instance that events and surfaces refer to. A class that wants
instances with the same parameters to share one waypoint declares a class-level
``_WAYPOINTS = {}`` dictionary and implements ``_waypoint_key``; a class without one
makes every instance its own waypoint. A duplicate ID gets a numeric suffix. The static
methods ``as_path``, ``as_primary_path`` and ``as_waypoint`` accept a path or an ID and
raise ``KeyError`` for an ID that is not registered.

``Path._PATH_CACHE`` holds the linked paths that ``wrt`` derives, so that repeated
requests for the same origin and frame pair share one object. ``Path._reset_caches()``
clears the registry and the cache and re-seeds them with ``Path.SSB``. Subclasses append
themselves to ``Path._PATH_SUBCLASSES`` so that the reset can clear each class's own
waypoint dictionary. A class whose events vary with time and are expensive sets
``_USE_QUICKPATHS = True`` so that ``quick_path`` can interpolate it.

The concrete paths in ``oops.path`` are :class:`~oops.path.SpicePath` (an SPK body,
always registered under its SPICE name), :class:`~oops.path.CirclePath`,
:class:`~oops.path.KeplerPath` (a Fittable orbit), :class:`~oops.path.LinearPath`,
:class:`~oops.path.FixedPath`, :class:`~oops.path.CoordPath` and
:class:`~oops.path.LinearCoordPath` (a point at fixed surface coordinates),
:class:`~oops.path.MultiPath` (several paths as one shaped path),
:class:`~oops.path.PathShift` (a Fittable offset) and :class:`~oops.path.QuickPath`. The
base module also defines :class:`~oops.path.NullPath`, :class:`~oops.path.SSBPath`, and
the :class:`~oops.path.LinkedPath`, :class:`~oops.path.RelativePath`,
:class:`~oops.path.ReversedPath` and :class:`~oops.path.RotatedPath` classes that
``wrt`` assembles.

Frame
-----

A :class:`~oops.Frame` is the orientation of a coordinate frame. Its abstract method is
``transform_at_time(time, *, quick=None)``, which returns the :class:`~oops.Transform`
that rotates coordinates from the ``reference`` frame into this one. A rotating frame has
an ``origin``, the path about which it rotates, and coordinates handed to its transform
must be relative to that origin; an inertial frame has ``origin`` ``None``.
``transform_at_time_if_possible`` is an optional override that tolerates times where the
underlying data is missing, returning the times that worked; only
:class:`~oops.frame.SpiceFrame` overrides it, so that short gaps in a C-kernel can be
interpolated across. ``node_at_time`` is optional as well.

The registration machinery mirrors the path's: the constructor sets ``_reference``,
``_origin`` and ``_shape``, then calls ``_register(frame_id)`` and ``refresh()``; a class
sharing *wayframes* declares ``_WAYFRAMES = {}`` and ``_wayframe_key``; the class appends
itself to ``Frame._FRAME_SUBCLASSES``; ``Frame._reset_caches()`` re-seeds
``Frame.J2000``. A frame whose own transform varies with time sets
``_USE_QUICKFRAMES = True``.

The concrete frames are :class:`~oops.frame.SpiceFrame` (a C-kernel or PCK frame, always
registered under its SPICE name, with ``omega_type`` selecting how the angular velocity
is obtained), :class:`~oops.frame.Cmatrix`, :class:`~oops.frame.SpinFrame`,
:class:`~oops.frame.Rotation`, :class:`~oops.frame.PoleFrame`,
:class:`~oops.frame.RingFrame`, :class:`~oops.frame.InclinedFrame`,
:class:`~oops.frame.LaplaceFrame`, :class:`~oops.frame.SynchronousFrame`,
:class:`~oops.frame.TwoVectorFrame` (a frame defined by two direction vectors, which is
how a camera pointed at a target is built), :class:`~oops.frame.TrackerFrame`,
:class:`~oops.frame.PosTargFrame`, :class:`~oops.frame.SpiceType1Frame`,
:class:`~oops.frame.Navigation` and :class:`~oops.frame.FrameShift` (Fittable pointing
corrections) and :class:`~oops.frame.QuickFrame`.

The photon solvers
------------------

``Path.photon_to_event(arrival, ...)`` finds the event on the path from which a photon
departed so as to arrive at ``arrival``; ``photon_from_event(departure, ...)`` finds where
a photon departing from ``departure`` meets the path. Each returns a pair of events: the
event on the path, with ``dep`` and ``dep_lt`` filled in, and a copy of the given event
with ``arr`` and ``arr_lt`` (or the reverse). The solver iterates on the light travel
time, expressing both events in SSB coordinates, and is governed by
:class:`~oops.config.PATH_PHOTONS`: ``max_iterations`` (4), ``dlt_precision`` (the change
in light time below which iteration stops), ``dlt_limit`` (the largest allowed departure
from the initial estimate, which stops a divergent solution), and the precision goals.
The ``converge`` argument overrides any of those for one call; ``guess`` seeds the light
time; ``antimask`` restricts the work to part of a shaped event; ``quick`` is passed
through to the path and frame lookups.

The surface solver, ``Surface.photon_to_event`` and its siblings, does the same for a
photon arriving from a direction rather than from a path, iterating on the intercept of
the line of sight with the moving surface and filling in ``perp`` and ``vflat``.
:class:`~oops.config.SURFACE_PHOTONS` governs it, with ``max_iterations`` 6 and a
``collapse_threshold`` for reducing a spread of intercept times to one value.
:doc:`dev_guide_surfaces_bodies` continues there.

Both solvers are functions in a private ``_photon_solver`` module, assigned as methods of
the class at the end of ``path_.py`` and ``surface_.py``. They are private modules
because nothing should call them except through the class, but their contents are in the
:doc:`private API reference <dev_guide_api>`.

QuickPath and QuickFrame
------------------------

Evaluating a SPICE path or frame is expensive, and a backplane asks for the same path at
a million nearly equal times. :class:`~oops.path.QuickPath` and
:class:`~oops.frame.QuickFrame` interpolate the object over a window that covers the
requested times, extended by ``path_time_extension`` seconds at each end, sampled every
``path_time_step`` seconds (the keys of :class:`~oops.config.QUICK` ``.dictionary``).
``Path.quick_path(time, quick=...)`` returns one, reusing a cached window that covers the
times or extending a window that partly does, and keeping at most
``quickpath_cache_size`` windows per path. The ``quick`` argument threaded through the
library is ``None`` for the configured defaults, a dictionary of overrides, or ``False``
to bypass the interpolation entirely. Anything else silently returns the original
object, which is why ``quick=True`` disables the optimization rather than enabling it.
Set ``QUICK.flag = False`` to disable it globally.

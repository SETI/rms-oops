Architecture
============

The class hierarchy
-------------------

Every object in the library descends from :class:`~oops.oops.Oops`, an empty marker
class. Below it, the abstract classes each define a contract that a subpackage of
concrete classes implements, and a few mix-ins give some of them a parameter-fitting
protocol. The diagram shows the principal classes, the methods that define each
abstract contract, and how instances refer to one another.

.. mermaid::

    classDiagram
        direction TB

        class Oops {
            <<marker>>
        }
        class Fittable {
            <<mix-in>>
            params
            nparams
            set_params(params)
            _set_params(params)*
            freeze()
            refresh()
        }
        class Mutable {
            <<mix-in>>
            refresh()
            freeze()
            _refresh()
        }

        class Path {
            <<abstract>>
            path_id
            origin : Path
            frame : Frame
            shape
            event_at_time(time, quick)*
            photon_to_event(arrival)
            photon_from_event(departure)
            as_path(path)$
            as_waypoint(path)$
            _register(path_id)
        }
        class Frame {
            <<abstract>>
            frame_id
            reference : Frame
            origin : Path
            shape
            transform_at_time(time, quick)*
            transform_at_time_if_possible(time, quick)
            as_frame(frame)$
            as_wayframe(frame)$
            _register(frame_id)
        }
        class Transform {
            matrix : Matrix3
            omega : Vector3
            frame : Frame
            reference : Frame
            rotate(pos)
            unrotate(pos)
        }
        class Event {
            time : Scalar
            state : Vector3
            origin : Path
            frame : Frame
            arr, dep, arr_lt, dep_lt, perp, vflat
            wrt(path, frame)
            wrt_ssb()
            incidence_angle()
            emission_angle()
            phase_angle()
        }
        class Surface {
            <<abstract>>
            origin : Path
            frame : Frame
            COORDINATE_TYPE
            IS_VIRTUAL
            coords_from_vector3(pos)*
            vector3_from_coords(coords)*
            intercept(obs, los)*
            normal(pos)*
            photon_to_event(arrival)
            photon_from_event(departure)
        }
        class Gravity {
            <<abstract>>
            omega(a)*
            kappa(a)*
            nu(a)*
            potential(a)*
            solve_a(omega, kappa, nu)*
        }
        class Body {
            name
            path : Path
            frame : Frame
            surface : Surface
            gravity : Gravity
            ring_frame : Frame
            ring_body : Body
            parent : Body
            lookup(name)$
            define_solar_system()$
        }
        class LightSource {
            name
            photon_to_event(event)
        }

        class FOV {
            <<abstract>>
            uv_shape : Pair
            uv_los : Pair
            uv_scale : Pair
            xy_from_uvt(uv, time)*
            uv_from_xyt(xy, time)*
            los_from_uvt(uv, time)
            uv_from_los_t(los, time)
        }
        class Cadence {
            <<abstract>>
            time
            midtime
            shape
            time_at_tstep(tstep)*
            tstep_at_time(time)*
            time_range_at_tstep(tstep)*
            tstep_range_at_time(time)*
            time_shift(secs)*
            as_continuous()*
        }
        class Calibration {
            <<abstract>>
            extended_from_dn(dn, uv)*
            dn_from_extended(value, uv)*
            point_from_dn(dn, uv)*
            dn_from_point(value, uv)*
            prescale(factor, baseline)*
        }
        class Observation {
            <<abstract>>
            fov : FOV
            cadence : Cadence
            path : Path
            frame : Frame
            data
            uvt(indices)*
            uvt_range(indices)*
            time_range_at_uv(uv)*
            uv_range_at_time(time)*
            meshgrid()
            event_at_grid(meshgrid)
            inventory(bodies)
        }
        class Meshgrid {
            fov : FOV
            uv : Pair
            los(time)
            dlos_duv(time)
        }
        class Backplane {
            obs : Observation
            meshgrid : Meshgrid
            get_surface_event(event_key)
            get_gridless_event(event_key)
            register_backplane(key, array)
            evaluate(backplane_key)
            ring_radius(), incidence_angle(), ... 86 methods
        }

        Oops <|-- Fittable
        Oops <|-- Mutable
        Oops <|-- Event
        Oops <|-- Transform
        Oops <|-- Body
        Oops <|-- Gravity
        Oops <|-- Calibration
        Oops <|-- LightSource
        Oops <|-- Meshgrid
        Mutable <|-- Path
        Mutable <|-- Frame
        Mutable <|-- Surface
        Mutable <|-- FOV
        Mutable <|-- Cadence
        Mutable <|-- Observation
        Mutable <|-- Backplane

        Path --> Path : origin
        Path --> Frame : frame
        Frame --> Frame : reference
        Frame --> Path : origin
        Frame ..> Transform : produces
        Event --> Path : origin
        Event --> Frame : frame
        Path ..> Event : produces
        Surface --> Path : origin
        Surface --> Frame : frame
        Surface ..> Event : consumes, produces
        Body --> Path
        Body --> Frame
        Body --> Surface
        Body --> Gravity
        Observation --> FOV
        Observation --> Cadence
        Observation --> Path
        Observation --> Frame
        Meshgrid --> FOV
        Backplane --> Observation
        Backplane --> Meshgrid
        Backplane ..> Body : looks up
        Backplane ..> Event : caches

In the diagram a trailing ``*`` marks a method that every subclass must implement, a
trailing ``$`` marks a static method, and the ``<<abstract>>`` classes are abstract by
convention only: there is no ``abc`` anywhere in the library. Each abstract method is an
ordinary method whose body raises ``NotImplementedError``, under a banner comment that
reads "Each subclass must override..." or "Methods to be defined for each ... subclass".

The kinematic core
~~~~~~~~~~~~~~~~~~

:class:`~oops.Path` and :class:`~oops.Frame` are the foundation. A path answers "where is
this point at time *t*?" with an :class:`~oops.Event`, relative to another path (its
``origin``) in a coordinate frame (its ``frame``). A frame answers "how is this frame
oriented at time *t*?" with a :class:`~oops.Transform`, relative to another frame (its
``reference``); a rotating frame also has an ``origin``, the path about which it rotates.
Both chains end at fixed roots that the modules assign after the class statement:
``Path.SSB``, the solar system barycenter, and ``Frame.J2000``. Each class keeps a
registry of its instances by ID and a cache of the derived paths and frames that link any
two registered ones, so that ``event.wrt(path, frame)`` can re-express an event relative
to any pair, and every event can be reduced to SSB coordinates in J2000 through
``wrt_ssb``. :doc:`dev_guide_events_paths_frames` covers the registries, the photon
solvers, and the read-only discipline of events.

The physical models
~~~~~~~~~~~~~~~~~~~

:class:`~oops.Surface` is a 2-D surface anchored to a path and a frame: a spheroid, a ring
plane, a limb, an ansa. Its contract is intercepting a line of sight and converting
between positions and surface coordinates; the photon solver bound onto it turns an
arriving event into the event at the surface. :class:`~oops.Gravity` supplies the orbital
frequencies that ring surfaces and orbit paths need. :class:`~oops.Body` binds a name to
a path, a frame, a surface, a gravity field and the ring frame and ring body of a planet,
and :meth:`~oops.Body.define_solar_system` populates its registry from SPICE.
:class:`~oops.lightsource.LightSource` lives in the same registry, so that the Sun and a
star can serve as the source of illumination in a backplane key.
:doc:`dev_guide_surfaces_bodies` is their chapter.

The instrument models
~~~~~~~~~~~~~~~~~~~~~

An :class:`~oops.Observation` is a data array with an :class:`~oops.FOV` that maps pixel
coordinates *(u,v)* to lines of sight, a :class:`~oops.Cadence` that maps array indices
to times, a :class:`~oops.Calibration` that maps data numbers to physical units, and the
path and frame of the instrument. :class:`~oops.Meshgrid` samples the FOV at a chosen
set of pixel coordinates and caches the lines of sight. :doc:`dev_guide_observations`
covers the four.

The backplane engine
~~~~~~~~~~~~~~~~~~~~

:class:`~oops.Backplane` ties it together: for an observation and a meshgrid it builds
the arrival event of every pixel, solves the photon path back to each requested surface,
and caches the resulting events and every array derived from them under a key. The 86
backplane methods live in separate modules and are attached to the class at import.
:doc:`dev_guide_backplanes` describes the keys, the caches and the module pattern.

Import order and the cross-class attributes
-------------------------------------------

The subsystems refer to one another in a cycle: a path produces events, an event holds a
path, a frame's origin is a path, a surface has both. Python cannot import that cycle
directly, so each class body leaves a placeholder for the classes it cannot import
(``Path._Body = None``, ``Frame._Path = None``, ``Event.SSB = None``,
``Transform._Frame = None`` and so on), and the bottom of ``oops/__init__.py`` fills
every placeholder once every module is loaded. The order the modules are imported in is
fixed by that file and commented there.

Two consequences follow. First, ``import oops`` must come before importing any leaf
module, or those placeholders stay ``None`` and the failure surfaces far from its cause.
Second, a stub cannot describe a placeholder, because the source shows ``None`` while the
run-time value is a class; ``stubtest-allowlist.txt`` lists every such name, and a new
placeholder needs a line there.

The same file binds the photon solvers: ``path/path_.py`` and ``surface/surface_.py``
each end by importing their private ``_photon_solver`` module and assigning its
functions as methods of the class, so ``photon_to_event`` and ``photon_from_event`` are
methods that the class statement never mentions. They have to be written out by hand in
the stubs for the same reason.

Mutable and Fittable
--------------------

A :class:`~oops.Fittable` object has parameters that a fitting procedure can change:
``params``, ``nparams``, and a ``_set_params`` hook, plus ``freeze`` to declare the values
final. :class:`~oops.mutable.Mutable` is the companion protocol for any object that might
*contain* a Fittable, directly or through its sub-objects: ``refresh`` asks the object to
notice that something below it has changed and drop its cached derived values, and
``freeze`` propagates downward. Every abstract base whose instances can hold another
object, which is to say :class:`~oops.Path`, :class:`~oops.Frame`,
:class:`~oops.Surface`, :class:`~oops.FOV`, :class:`~oops.Cadence`,
:class:`~oops.Observation` and :class:`~oops.Backplane`, inherits Mutable. A class that
caches anything derived from a sub-object implements ``_refresh`` to discard it, and a
class with its own ``_refresh`` that overrides a parent's must call ``super()._refresh()``.
Every backplane method begins with ``self.refresh()`` for this reason.

An unfrozen Fittable is not registered under a shared key, because its parameters can
still change; ``Frame._register`` and ``Path._register`` note this, and ``_reregister``
files the object under its permanent key once it is frozen.

Caching and shared state
------------------------

Several layers cache, and all of them share objects rather than copying:

* The Path and Frame registries and link caches are process-wide class attributes.
  ``Path._reset_caches()`` and ``Frame._reset_caches()`` clear them, and
  :meth:`~oops.Body.reset_registry` clears the bodies as well. An event cached anywhere
  holds direct references to the waypoints and wayframes that were current when it was
  built; after a reset, a lookup by ID returns their replacements, so a cache built before
  the reset is stale. The backplane tests keep one solar system for the whole package for
  this reason.
* ``QuickPath`` and ``QuickFrame`` interpolate a path or frame over a time window and are
  cached on the path or frame itself (up to ``quickpath_cache_size`` of them, from
  :class:`~oops.config.QUICK`). They are used whenever ``quick`` is ``None`` or a
  dictionary; ``quick=False`` bypasses them. Passing ``True`` is the trap: it is not a
  dictionary, so the unoptimized object is silently returned.
* A bounded ``_Cache`` (``oops/_cache.py``) holds the events of ``KeplerPath`` and the
  transforms of several frames, keyed by the normalized value of the time argument; it
  returns ``None`` for a missing key rather than raising.
* An :class:`~oops.Event` caches its SSB twin, its transform to J2000, its mask and its
  shape lazily. Every value placed in an event is made read-only first, and an event's
  optional properties (``arr``, ``dep``, ``arr_lt`` and the rest) may be set exactly once.
* :class:`~oops.Backplane` caches observation events, surface events, intercepts,
  antimasks and every registered array, all read-only and shared with every later caller
  of the same key.

The rule that follows is the one in ``CLAUDE.md``: never mutate an event or a registered
array in place. Use ``copy``, ``replace``, ``mask_where`` or arithmetic, each of which
returns a new object.

Units and conventions
---------------------

Distances are km, times are seconds TDB measured from the J2000 epoch, velocities are
km/s, and angles are radians, everywhere and without exception; the ``polymath.Units``
machinery is effectively unused, so a value is not self-describing. The
``oops.constants`` module holds the conversions (``DPR``, ``RPD``, ``C``, ``AU``,
``SPD``). Pixel coordinates are *(u,v)* with *u* horizontal and *v* vertical, which is the
reverse of NumPy's index order; an observation records which array axis is which and
whether they are swapped. The observation frame has *z* along the line of sight, *x* to
the right and *y* downward. Light travel time is signed: ``arr_lt`` is negative and
``dep_lt`` positive.

Observations, fields of view, cadences and calibrations
=======================================================

Overview
--------

An :class:`~oops.Observation` is the instrument side of the geometry: a data array, an
:class:`~oops.FOV` that maps pixel coordinates to lines of sight, a
:class:`~oops.Cadence` that maps array indices to times, a :class:`~oops.Calibration`
that maps data numbers to physical units, and the path and frame of the instrument. A
:class:`~oops.Meshgrid` samples the FOV at chosen pixel coordinates and caches the lines
of sight for the backplane engine. The API reference pages are :doc:`oops.observation
</oops_observation>`, :doc:`oops.fov </oops_fov>`, :doc:`oops.cadence </oops_cadence>`
and :doc:`oops.calibration </oops_calibration>`, with the base classes and
:class:`~oops.Meshgrid` on the :doc:`oops package page </oops>`.

Pixel and frame conventions
---------------------------

Pixel coordinates are *(u,v)*: *u* horizontal, increasing to the right, and *v*
vertical. Integer values fall on pixel boundaries, so the center of the first pixel is
(0.5, 0.5). A data array indexes in NumPy order, which puts *v* first for an image read
from a FITS or VICAR file; the observation records ``u_axis``, ``v_axis``, ``t_axis`` and
``swap_uv`` so that ``uvt`` can translate array indices to *(u,v)* and time. The
observation frame has *z* along the boresight, *x* to the right and *y* downward, and an
FOV describes lines of sight in that frame with a gnomonic *(x,y)* pair whose *z*
component is 1, so that *x* and *y* are radians near the center.

FOV
---

The abstract :class:`~oops.FOV` has two methods a subclass must implement:

``xy_from_uvt(uv_pair, time=None, *, derivs=False, remask=False, **kwargs)``
    Pixel coordinates to gnomonic *(x,y)*, at a time for a time-dependent FOV.
``uv_from_xyt(xy_pair, time=None, *, derivs=False, remask=False, **kwargs)``
    The inverse.

Everything else derives from them: ``xy_from_uv`` and ``uv_from_xy`` drop the time,
``los_from_xy`` and ``xy_from_los`` convert to and from unit vectors, ``los_from_uvt``,
``uv_from_los_t`` and their time-free forms chain the two, ``area_factor`` is the solid
angle of a pixel relative to the nominal one, ``uv_is_outside`` and its siblings test the
bounds, ``nearest_uv`` clips, ``sphere_falls_inside`` supports the inventory, and the
``center_*`` and ``corner*_xy`` methods describe the extent. A subclass sets four
attributes in its constructor: ``uv_shape`` (a :class:`~oops.Pair`, possibly
non-integral), ``uv_los`` (the *(u,v)* of the nominal line of sight), ``uv_scale`` (the
approximate *dx/du* and *dy/dv*, with the sign of the second selecting whether *v*
increases downward; the first is always positive) and ``uv_area``. A time-dependent
subclass sets ``IS_TIME_INDEPENDENT = False``. The base class caches the center and
corner values under the names in ``_CACHED_NAMES`` and drops them in ``_refresh``; a
subclass with its own ``_refresh`` must call ``super()._refresh()``.

The concrete classes are :class:`~oops.fov.FlatFOV` (no distortion),
:class:`~oops.fov.PolynomialFOV` and :class:`~oops.fov.BarrelFOV` (polynomial and radial
distortion), :class:`~oops.fov.WCSFOV` (FITS SIP parameters), :class:`~oops.fov.GapFOV`,
:class:`~oops.fov.NullFOV` (an in-situ instrument), :class:`~oops.fov.TDIFOV` (a
time-delay-integration camera), and the wrappers :class:`~oops.fov.OffsetFOV`,
:class:`~oops.fov.Platescale` (both Fittable), :class:`~oops.fov.SliceFOV`,
:class:`~oops.fov.Subarray` and :class:`~oops.fov.SubsampledFOV`, which re-map another
FOV.

Cadence
-------

A :class:`~oops.Cadence` maps a time step, which may be fractional and may have one or
two dimensions, to a time. A subclass implements six methods: ``time_at_tstep``,
``time_range_at_tstep``, ``tstep_at_time``, ``tstep_range_at_time``, ``time_shift`` and
``as_continuous``. It also sets, in its constructor under a ``# Required attributes``
comment, the eight attributes the base class documents: ``time`` (the start and end),
``midtime``, ``lasttime`` (the start of the last step), ``shape``, ``is_continuous``,
``is_unique``, ``min_tstride`` and ``max_tstride``. ``time_is_inside``,
``time_is_outside`` and ``tstride_at_tstep`` derive from those.

The concrete cadences are :class:`~oops.cadence.SnapCadence` (one step, what a
:class:`~oops.observation.Snapshot` builds from a start time and an exposure),
:class:`~oops.cadence.Metronome` (uniform steps), :class:`~oops.cadence.Sequence`
(explicit times), :class:`~oops.cadence.Instant`, :class:`~oops.cadence.DualCadence` (a
two-dimensional cadence from a fast and a slow one), :class:`~oops.cadence.TDICadence`,
and the wrappers :class:`~oops.cadence.ReshapedCadence`,
:class:`~oops.cadence.ReversedCadence` and :class:`~oops.cadence.TimeShift` (Fittable).

Calibration
-----------

A :class:`~oops.Calibration` converts data numbers to a physical quantity and back, with
separate conversions for an extended source (per unit solid angle, so the pixel's area
factor matters) and a point source: ``extended_from_dn``, ``dn_from_extended``,
``point_from_dn``, ``dn_from_point``, plus ``prescale``, which folds an extra factor and
baseline into a new calibration. A subclass sets ``name``, ``factor``, ``baseline``,
``has_baseline``, ``shape`` and ``fov`` (``None`` when the calibration is independent of
position). Unlike the other abstract classes, it inherits :class:`~oops.oops.Oops`
directly rather than :class:`~oops.mutable.Mutable`. The concrete classes are
:class:`~oops.calibration.FlatCalib`, :class:`~oops.calibration.Radiance` (which applies
the FOV's area factor), :class:`~oops.calibration.RawCounts` and
:class:`~oops.calibration.NullCalib`.

Observation
-----------

:class:`~oops.Observation` is the most demanding contract. A subclass defines its own
``__init__`` and implements ``uvt``, ``uvt_range``, ``time_range_at_uv``,
``uv_range_at_time``, ``uv_range_at_tstep`` and ``time_shift``; ``uv_from_coords`` and
``inventory`` are implemented by the subclasses that can support them, which set
``_INVENTORY_IMPLEMENTED = True``. The base class supplies dimension-specific helpers
(``_time_range_at_uv_2d`` and the like) that a subclass delegates to according to how
its time axis couples to its spatial axes. The constructor sets ``cadence``, ``fov``,
``uv_shape``, ``u_axis``, ``v_axis``, ``swap_uv``, ``t_axis``, ``shape``, ``path``,
``frame`` and ``subfields``; ``time`` and ``midtime`` are properties reading the cadence,
and every extra keyword becomes a subfield and an attribute, which is how ``data``, the
instrument name and the file path travel with the observation.

The methods the rest of the library calls are ``meshgrid`` (a :class:`~oops.Meshgrid`
whose *(u,v)* axes sit where the observation's do), ``timegrid`` (the time of every
sample of that meshgrid), ``event_at_grid`` (an arrival event at the instrument with
``neg_arr_ap`` set to each line of sight), ``gridless_event`` (the same without
directions), ``uv_from_path`` and ``uv_from_ra_and_dec`` (where a target falls in the
field), ``inventory`` (which bodies are in view, treating each as a sphere, with
``return_type='full'`` giving the bounding box, range and resolution of each) and the
``parallel_*`` helpers for a co-mounted instrument.

The concrete observations are :class:`~oops.observation.Snapshot` (a 2-D image at one
time), :class:`~oops.observation.TimedImage` (a 2-D image whose pixels have distinct
times: pushbroom, raster and slit instruments of every kind, distinguished by the axis
labels), :class:`~oops.observation.Slit1D`, :class:`~oops.observation.RasterSlit1D`,
:class:`~oops.observation.Pixel` and :class:`~oops.observation.InSitu` (timing and a
path but no pointing).

Meshgrid
--------

A :class:`~oops.Meshgrid` is an array of *(u,v)* pairs within an FOV, with the derivative
*d(u,v)/d(u,v)* attached so that the derivatives of every downstream quantity with
respect to pixel position can be chained. ``Meshgrid.for_fov`` samples the pixel centers
of a whole FOV, with ``undersample``, ``oversample``, ``origin``, ``limit`` and ``swap``
options; ``for_shape`` places the *(u,v)* axes at given positions in a larger shape,
which is what ``Observation.meshgrid`` uses; ``for_fov_center`` is a single line of
sight. The line-of-sight methods take a time and cache their results by it, because a
time-dependent FOV gives a different answer at each.

Introduction
============

What oops is for
----------------

``oops`` computes the geometry of observations in planetary science. Given an image or
other data product from a spacecraft or telescope, it answers, for every pixel, what the
line of sight intercepts and under what conditions: which body or ring, at what
coordinates on its surface, at what distance and resolution, under what lighting
angles, at what time, and in which part of the sky. The result is a set of
*backplanes*, arrays with the same shape as the data, one per quantity, that can be
saved alongside the data, used to select pixels, or fed to a photometric or dynamical
analysis.

It does this from first principles rather than from metadata in the file. The
trajectory of the spacecraft, the rotation of the planet, the orbit of every moon and
the constants of every body come from SPICE kernels; the camera's distortion, timing and
pointing come from an instrument model; and the library solves the light-time and
aberration problems that connect them. The same machinery answers the inverse questions
too: where in the field of view a given body or point on its surface falls, and when a
given pixel was exposed.

.. note::

   This package is under development. Use with caution, and check results that matter
   against an independent source.

The workflow
------------

A typical session has four stages.

.. mermaid::

    flowchart LR
        A[Set up resources] --> B[Load the observation]
        B --> C[Define the solar system]
        C --> D[Compute backplanes]
        D --> E[Use the arrays]

#. **Set up the resources.** The library needs the SPICE kernels and a database that
   indexes them, reached through one environment variable. This is done once
   (:doc:`user_guide_installation`).
#. **Load the observation.** A *host* module for each supported instrument reads a data
   file and returns an :class:`~oops.Observation`: the data array plus a model of the
   field of view, the exposure timing, and the path and pointing of the instrument
   (:doc:`user_guide_observations`).
#. **Define the solar system.** :meth:`~oops.Body.define_solar_system` furnishes the
   kernels for a time range and registers a :class:`~oops.Body` for every planet, moon
   and ring, each with its path, orientation, shape and gravity. The host modules do this
   for you as part of loading an observation.
#. **Compute backplanes.** A :class:`~oops.Backplane` object wraps the observation. Each
   of its methods, ``ring_radius``, ``incidence_angle``, ``where_intercepted`` and about
   eighty more, returns an array for a named target, and the object caches the expensive
   light-path solutions so that related backplanes share the work
   (:doc:`user_guide_backplanes`).

The :doc:`gold master tool <user_guide_gold_master>` runs stages two through four from
the command line for any supported observation and writes every backplane out as an
array and as a browse image, which is the quickest way to see the geometric contents of
an image without writing code. For the geometry underneath the backplanes, paths,
frames, events and surfaces can be used directly (:doc:`user_guide_geometry`).

Vocabulary
----------

These terms recur throughout the library and this guide.

Backplane
    An array with the shape of the observation whose value at each pixel is one
    geometric quantity: the radius in the ring plane, the emission angle, whether the
    pixel is on a body. The :class:`~oops.Backplane` class computes them.
Body
    A named target: a planet, moon, ring or barycenter, with its path, frame, surface
    and gravity. Bodies are registered by name, in upper case, and referred to by that
    name everywhere: ``'SATURN'``, ``'EPIMETHEUS'``, ``'SATURN_MAIN_RINGS'``.
Path
    The motion of a point through space, such as a spacecraft, a planet's center or a
    point on a surface, relative to another path. Every path has an *origin* path and
    is expressed in a frame; the chain ends at the solar system barycenter, ``'SSB'``.
Frame
    The orientation of a coordinate frame as a function of time, relative to a
    *reference* frame; the chain ends at ``'J2000'``. A planet's rotating body frame, a
    spacecraft's pointing and a ring plane's frame are all frames.
Event
    A photon at a point in spacetime: a time, a position and a velocity relative to a
    path and in a frame, plus the directions and light travel times of the arriving
    and departing photons. Everything the library computes passes through events.
Surface
    A 2-D shape attached to a path and a frame: a spheroid or ellipsoid for a body, a
    ring plane, or the *virtual* surfaces of a limb and an ansa, which exist only from
    the observer's point of view.
Observation
    A data array with its field of view, timing, path and frame. A *snapshot* is an
    image exposed at one time; a *timed image* is one whose pixels have distinct times.
FOV
    The field of view: the mapping between pixel coordinates *(u,v)* and lines of sight,
    including distortion.
Meshgrid
    The set of pixel coordinates at which a backplane is evaluated. By default it is the
    center of every pixel; it can be undersampled or restricted.
Event key
    The address of a photon event in a backplane: which surface, illuminated by which
    source. ``'SATURN'`` names the surface of Saturn lit by the Sun; ``'SATURN:RING'`` its
    ring plane; ``'SATURN:LIMB'`` its limb.

Units and conventions
---------------------

Every quantity in the library is in one of three units and is never labeled with it:
distances are **km**, times are **seconds TDB** measured from the J2000 epoch (noon TDB
on 2000 January 1), and angles are **radians**. Velocities are km/s. A backplane named
``incidence_angle`` holds radians; the gold master tool converts to degrees when it
writes its files, and the titles of those files say so.

Values are ``polymath`` array types, not NumPy arrays. A :class:`~oops.Scalar` is an
array of numbers, a :class:`~oops.Vector3` an array of 3-vectors, a
:class:`~oops.Boolean` an array of booleans, and all of them carry a *mask* that marks
the elements with no valid value, such as pixels whose line of sight misses the body.
The values are in ``.vals``, the mask in ``.mask``, and the values where the mask is
false are garbage: index with ``.antimask`` before using them.

.. code-block:: python

    radius = bp.ring_radius('SATURN:RING')
    good = radius.vals[radius.antimask]          # the valid radii, as a NumPy array

Pixel coordinates are *(u,v)*, with *u* horizontal and increasing to the right and *v*
vertical. Integer coordinates fall on pixel boundaries, so the center of the first pixel
is (0.5, 0.5). The data array is indexed in NumPy order, which for an image read from a
FITS or VICAR file puts *v* first; the observation records which axis is which. The
observation's frame has *z* along the line of sight, *x* to the right and *y* downward.

Right ascension and declination, and the *(x,y)* of a field of view, are radians too;
``oops.DPR`` (degrees per radian) and ``oops.RPD`` convert.

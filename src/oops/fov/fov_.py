##########################################################################################
# oops/fov/fov_.py
##########################################################################################

import numpy as np

from polymath     import Boolean, Scalar, Pair, Vector3, Qube
from oops.config  import AREA_FACTOR
from oops.mutable import Mutable


class FOV(Mutable):
    """The FOV (Field of View) abstract class provides a description of the geometry of a
    field of view.

    The properties of an FOV are defined within a fixed coordinate frame, with the
    positive Z axis oriented near the center of the line of sight. The x and y axes are
    effectively in the plane of the FOV, with the x-axis oriented horizontally and the
    y-axis pointed downward. The values for (x,y) are implemented using a "pinhole camera"
    or "gnomonic" model, in which the z-component has unit length. Therefore, near the
    center of the field of view, the units of x and y are radians. However, the scale
    shifts at greater x and y, because the magnitude of the vector is
    sqrt(1 + x**2 + y**2).

    The FOV converts between the actual line of sight vector (x,y,z) and an internal
    coordinate system (ICS) that typically defines a pixel grid. It also accommodates any
    spatial distortion of the field of view. The ICS coordinates (u,v) are linear with the
    grid of pixels and, typically, a unit step in (u,v) shifts the position by one pixel.
    The u-axis points rightward in the default display orientation of a data array, and
    the v-axis points either upward or downward.

    Although ICS coordinates (u,v) are defined here in units of pixels, the FOV concept
    can be used to describe an arbitrary field of view, consisting of detectors of
    arbitrary shape. In this case, (u,v) are simply a convenient set of coordinates to use
    in a frame that describes the layout of detectors on the focal plane of the
    instrument.

    The class also allows for the possibility that the field of view has additional
    dependencies on wavelength, etc. Additional arguments and keyword values can be passed
    through these methods and into the subclass methods.

    Properties:
        uv_los (Pair): The `(u,v)` coordinates of the nominal line of sight.
        uv_scale (Pair): The approximate ratios `dx/du` and `dy/dv`. For example, if
            `(u,v)` are in units of arcseconds, then `uv_scale` is::

                Pair((pi/180/3600., pi/180/3600.)).

            Use the sign of the second element to define the direction of increasing `v`:
            negative for up, positive for down. Note that, by its definition, uv_scale[0]
            must _always_ be positive.
        uv_shape (Pair): The size of the field of view in pixels. This number can be
            non-integral if the detector is not composed of a rectangular array of pixels.
        uv_area (float): The nominal area of a region defined by unit steps in `(u,v)`,
            e.g., the size of a pixel in steradians.
    """

    # Override this class attribute to False for FOV subclasses that have time-dependence
    IS_TIME_INDEPENDENT = True

    # Values derived from the FOV geometry and saved on first use. Any change to the FOV,
    # such as a Fittable subclass receiving new parameters, invalidates all of them.
    _CACHED_NAMES = ('_center_xy_filled', '_center_los_filled', '_center_dlos_duv_filled',
                     '_outer_radius_filled', '_inner_radius_filled', '_corner00_filled',
                     '_corner01_filled', '_corner10_filled', '_corner11_filled')

    def _refresh(self):
        """Discard every cached value, because a change to this FOV invalidates them.

        A subclass that defines its own `_refresh` must call this one as well.
        """

        for name in FOV._CACHED_NAMES:
            self.__dict__.pop(name, None)

    ######################################################################################
    # Methods to be defined for each FOV subclass
    ######################################################################################

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the FOV's frame.
        """

        raise NotImplementedError(type(self).__name__ + '.xy_from_uvt is not implemented')

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` coordinates in the FOV.
        """

        raise NotImplementedError(type(self).__name__ + '.uv_from_xyt is not implemented')

    ######################################################################################
    # Derived methods, to override only if necessary
    ######################################################################################

    def xy_from_uv(self, uv_pair, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)`,
        assuming the FOV is time-independent.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the FOV's frame.
        """

        if not self.IS_TIME_INDEPENDENT:
            raise NotImplementedError(type(self).__name__ + '.xy_from_uv '
                                      'is not implemented; FOV is time-dependent')

        return self.xy_from_uvt(uv_pair, derivs=derivs, remask=remask, **kwargs)

    def uv_from_xy(self, xy_pair, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates,
        assuming the FOV is time-independent.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` Pair.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` coordinates in the FOV.
        """

        if not self.IS_TIME_INDEPENDENT:
            raise NotImplementedError(type(self).__name__ + '.uv_from_xy '
                                      'is not implemented; FOV is time-dependent')

        return self.uv_from_xyt(xy_pair, derivs=derivs, remask=remask, **kwargs)

    def area_factor(self, uv_pair, time=None, *, remask=False, **kwargs):
        """The relative area of a pixel or other sensor at `(u,v)` at the specified time
        (although any dependence on time should be very small).

        The returned value is proportional to the solid angle subtended by a unit step in
        `(u,v)`, divided by the nominal pixel area `uv_area`. It is therefore unitless,
        and it is one wherever a pixel subtends the nominal solid angle.

        Parameters:
            uv_pair (Pair, ndarray, or tuple): `(u,v)` coordinates in the FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Scalar: Relative area of the pixel at FOV coordinates `(u,v)`.
        """

        # Prepare for the partial derivatives
        uv_pair = Pair.as_pair(uv_pair).wod
        uv_pair = uv_pair.with_deriv('uv', Pair.IDENTITY, method='insert')
        xy_pair = self.xy_from_uvt(uv_pair, time=time, derivs=True, remask=remask,
                                   **kwargs)

        # These are the values returned prior to January 2023. They measure the area in
        # the plane z = 1 rather than the solid angle, omitting the factor of 1/los^3
        # below. This option is preserved for backward compatibility and because it
        # simplifies some of the Calibration unit tests.
        if AREA_FACTOR.old:
            dx_du = xy_pair.d_duv.vals[...,0,0]
            dx_dv = xy_pair.d_duv.vals[...,0,1]
            dy_du = xy_pair.d_duv.vals[...,1,0]
            dy_dv = xy_pair.d_duv.vals[...,1,1]
            cross_product = dx_du * dy_dv - dx_dv * dy_du
            return Scalar(np.abs(cross_product) / self.uv_area, xy_pair.mask)

        # (x,y) are defined on the assumption that z = 1, so equal areas in the (x,y)
        # plane do not subtend equal solid angles; the area must be converted to
        # steradians.
        #
        # The unit line of sight is
        #   p(x,y) = (x, y, 1) / los,
        # where los(x,y) = sqrt(1 + x^2 + y^2)
        #
        # Differentiating,
        #   dp/dx = (e_x - x * p / los) / los
        #   dp/dy = (e_y - y * p / los) / los
        #
        # where e_x and e_y are the unit vectors along the x and y axes. Their cross
        # product simplifies to
        #   dp/dx X dp/dy = (x, y, 1) / los^4
        #
        # and |(x,y,1)| = los, so the solid angle subtended by an area element of the
        # plane z = 1 is
        #   dOmega = dx dy / los^3
        #
        # The area factor is therefore the area of the (x,y) parallelogram spanned by unit
        # steps in u and v, divided by los^3 and by the nominal pixel area.

        (x,y) = xy_pair.to_scalars(recursive=False)
        los = (1 + x**2 + y**2).sqrt()

        # Construct the cross product
        dx_du = xy_pair.d_duv.vals[...,0,0]
        dx_dv = xy_pair.d_duv.vals[...,0,1]
        dy_du = xy_pair.d_duv.vals[...,1,0]
        dy_dv = xy_pair.d_duv.vals[...,1,1]
        cross_product = dx_du * dy_dv - dx_dv * dy_du

        return Scalar(np.abs(cross_product) / (los.vals**3 * self.uv_area),
                      xy_pair.mask)

    def los_from_xy(self, xy_pair, *, derivs=False):
        """The unit line-of-sight vector for camera coordinates `(x,y)`, assuming a
        pinhole camera model.

        Note that this vector points in the direction _opposite_ to the path of arriving
        photons.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            derivs (bool, optional): True to propagate any derivatives of `(x,y)` forward
                into the returned line-of-sight vector.

        Returns:
            Vector3: Unit vector in the direction of the line of sight, in the FOV's
            frame.
        """

        # Convert to Pair if necessary
        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)

        # In the pinhole camera model, the z-component is always 1
        (x,y) = Pair.to_scalars(xy_pair)
        return Vector3.from_scalars(x,y,1.).unit(recursive=derivs)

    def xy_from_los(self, los, *, derivs=False):
        """Camera frame coordinates `(x,y)` given a line of sight.

        Lines of sight point outward from the camera, near the Z-axis, and are therefore
        opposite to the direction in which a photon is moving. The length of the vector is
        ignored.

        Parameters:
            los (Vector3): Line of sight in this FOV's coordinate frame.
            derivs (bool, optional): True to propagate any derivatives of `los` into the
                returned coordinates.

        Returns:
            Pair: `(x,y)` coordinates in the FOV's frame.
        """

        # Scale to z=1 and then convert to Pair
        los = Vector3.as_vector3(los, recursive=derivs)
        z = los.to_scalar(2)
        los = los / z

        return los.to_pair((0,1))

    def los_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The unit line of sight vector in the camera's frame, given FOV coordinates
        `(u,v)` at the specified time.

        Note that the line of sight points in the direction _opposite_ to that of the
        arriving photons.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): True to propagate any derivatives of `(u,v)` into the
                returned line of sight.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Vector3: Direction of the line of sight in the FOV's frame.
        """

        xy_pair = self.xy_from_uvt(uv_pair, time=time, derivs=derivs, remask=remask,
                                   **kwargs)
        return self.los_from_xy(xy_pair, derivs=derivs)

    def los_from_uv(self, uv_pair, *, derivs=False, remask=False, **kwargs):
        """The unit line of sight vector given FOV coordinates `(u,v)`, assuming this FOV
        is time-independent.

        Note that the line of sight points in the direction _opposite_ to that of the
        arriving photons.

        Parameters:
            uv_pair (Pair, ndarray, or tuple): `(u,v)` coordinates in the FOV.
            derivs (bool, optional): True to propagate any derivatives of `(u,v)` into the
                returned line of sight.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Vector3: Direction of the line of sight in the FOV's frame.
        """

        xy_pair = self.xy_from_uv(uv_pair, derivs=derivs, remask=remask, **kwargs)
        return self.los_from_xy(xy_pair, derivs=derivs)

    def uv_from_los_t(self, los, time=None, *, derivs=False, remask=False, **kwargs):
        """The FOV coordinates `(u,v)` given a line of sight vector in the FOV's frame at
        the specified time.

        Note that `los` points in the direction _opposite_ to that of the arriving photon.

        Parameters:
            los (Vector3): The line of sight in the FOV's frame.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): True to propagate any derivatives of `los` into the
                returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: `(u,v)` coordinates in the FOV.
        """

        xy_pair = self.xy_from_los(los, derivs=derivs)
        return self.uv_from_xyt(xy_pair, time=time, derivs=derivs,  remask=remask,
                                **kwargs)

    def uv_from_los(self, los, *, derivs=False, remask=False, **kwargs):
        """The FOV coordinates `(u,v)` given a line of sight vector, assuming the FOV is
        time-independent.

        Note that `los` points in the direction _opposite_ to that of the arriving photon.

        Parameters:
            los (Vector3): Direction of the line of sight in the FOV's frame.
            derivs (bool, optional): True to propagate any derivatives of `los` into the
                returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: `(u,v)` coordinates in the FOV.
        """

        if not self.IS_TIME_INDEPENDENT:
            raise NotImplementedError(type(self).__name__ + '.uv_from_los '
                                      'is not implemented; FOV is time-dependent')

        return self.uv_from_los_t(los, derivs=derivs, remask=remask, **kwargs)

    def offset_angles_from_duv(self, duv, *, time=None, origin=None):
        """The rotation angles defined by a `(u,v)` pixel offset in an FOV.

        Parameters:
            duv (Pair): `(u,v)` pixel offsets. These define the coordinates of a feature
                in a Navigation frame as offsets from that same feature in the reference
                frame.
            time (Scalar, optional): Absolute time in seconds TDB.
            origin (Pair, optional): The `(u,v)` coordinates of the reference location at
                which this offset is determined. By default, this is the center of the
                FOV.

        Returns:
            tuple[Scalar, Scalar]: The two rotation angles in radians.
        """

        # Locate the origin LOS vector
        if origin is None:
            center_uv = self.uv_shape/2.
        else:
            center_uv = Pair.as_pair(origin)

        los0 = self.los_from_uvt(center_uv, time=time)

        # Determine the LOS vector associated with this pointing offset
        los1 = self.los_from_uvt(center_uv - duv, time=time)

        # Return the rotation angles
        return los0.offset_angles(los1)

    def offset_duv_from_angles(self, angles, *, time=None, origin=None):
        """The `(u,v)` pixel offset in the FOV associated with the given pair of rotation
        angles.

        Parameters:
            angles (tuple[Scalar, Scalar]): Two offset angles in radians. The first
                rotation is about the Y axis of this FOV's frame and the second is about
                the X axis.
            time (Scalar, optional): Absolute time in seconds TDB.
            origin (Pair, optional): The `(u,v)` coordinates of the reference location at
                which this offset is determined. By default, this is the center of the
                FOV.

        Returns:
            Pair: The `(u,v)` pixel offset.
        """

        # Locate the origin and the reference LOS vector
        if origin is None:
            center_uv = self.uv_shape/2.
        else:
            center_uv = Pair.as_pair(origin)

        los0 = self.los_from_uvt(center_uv, time=time)

        # Determine the offset LOS
        los1 = los0.spin(Vector3.YAXIS, angles[0])
        los1 = los1.spin(Vector3.XAXIS, angles[1])

        # Return the pixel offset
        return center_uv - self.uv_from_los_t(los1, time=time)

    ######################################################################################
    # Boundary tests
    ######################################################################################

    def uv_is_outside(self, uv_pair, time=None, *, uv_min=None, uv_max=None,
                      inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.

        Returns:
            Boolean: True where the point is outside the FOV.
        """

        # Interpret the (u,v) coordinates
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        (u,v) = uv_pair.to_scalars()

        # Fill in the corners
        if uv_min is None:
            uv_min = Pair.INT00

        if uv_max is None:
            uv_max = self.uv_shape

        (umin, vmin) = uv_min.vals
        (umax, vmax) = uv_max.vals

        # Create the mask
        result = (Qube.is_outside(u, umin, umax, inclusive=inclusive) |
                  Qube.is_outside(v, vmin, vmax, inclusive=inclusive))
        return Boolean(result, uv_pair.mask)

    def u_or_v_is_outside(self, uv_pair, uv_index, *, uv_min=None, uv_max=None,
                          inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV along one axis.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            uv_index (int): 0 to test u-coordinates; 1 to test v-coordinates.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV, of which only the element selected by `uv_index` is used; None for
                the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV, of which only the element selected by `uv_index` is used; None for
                the full FOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.

        Returns:
            Boolean: True where the point is outside the FOV along the specified axis.
        """

        # Interpret the (u,v) coordinates
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        uv_coord = uv_pair.to_scalar(uv_index, recursive=False)

        # Fill in the corners
        if uv_min is None:
            uv_min = Pair.INT00

        if uv_max is None:
            uv_max = self.uv_shape

        # Create the mask
        result = Qube.is_outside(uv_coord.vals, uv_min.vals[uv_index],
                                 uv_max.vals[uv_index], inclusive=inclusive)
        return Boolean(result, uv_pair.mask)

    def xy_is_outside(self, xy_pair, time=None, *, inclusive=True,
                      uv_min=None, uv_max=None, **kwargs):
        """A Boolean mask identifying coordinates outside the FOV.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Boolean: True where `xy_pair` is outside the FOV.
        """

        uv = self.uv_from_xyt(xy_pair, time=time, derivs=False, **kwargs)
        return self.uv_is_outside(uv, time=time, uv_min=uv_min, uv_max=uv_max,
                                  inclusive=inclusive)

    def los_is_outside(self, los, time=None, *, inclusive=True, uv_min=None, uv_max=None,
                       **kwargs):
        """A Boolean mask identifying lines of sight outside the FOV.

        Parameters:
            los (Vector3): An outward line-of-sight vector.
            time (Scalar, optional): Absolute time in seconds TDB.
            inclusive (bool, optional): True to interpret coordinates at the upper end of
                each range as inside the FOV; False to interpret them as outside.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Boolean: True where `los` is outside the FOV.
        """

        xy = self.xy_from_los(los, derivs=False)
        return self.xy_is_outside(xy, time=time, uv_min=uv_min, uv_max=uv_max,
                                  inclusive=inclusive, **kwargs)

    def nearest_uv(self, uv_pair, *, remask=False):
        """The closest `(u,v)` coordinates inside the FOV.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            remask (bool, optional): True to mask the points outside the FOV's boundary.

        Returns:
            Pair: Nearest `(u,v)` coordinates to `uv_pair`.
        """

        clipped = Pair.as_pair(uv_pair, recursive=False)

        clipped = uv_pair.copy(readonly=False)
        clipped.vals[...,0] = clipped.vals[...,0].clip(0, self.uv_shape.vals[0])
        clipped.vals[...,1] = clipped.vals[...,1].clip(0, self.uv_shape.vals[1])

        if remask:
            return Pair(clipped, Qube.or_(uv_pair.mask, (clipped != uv_pair)))
        else:
            return clipped

    ######################################################################################
    # Self-check support
    ######################################################################################

    def max_inversion_error(self, steps=30):
        """Sample the FOV and return the largest error in pixels resulting from
        `(u,v) -> (x,y) -> (u,v)`.

        Parameters:
            steps (int, optional): The number of samples per axis.

        Returns:
            float: The largest error in `(u,v) -> (x,y) -> (u,v)` at the sampled points.
        """

        # Sample the FOV uniformly along each axis
        du = self.uv_shape.vals[0] / (steps - 1.)
        dv = self.uv_shape.vals[1] / (steps - 1.)
        u = np.arange(0., self.uv_shape.vals[0] + du/2., du)
        v = np.arange(0., self.uv_shape.vals[1] + dv/2., dv)

        uv = Pair.combos(u,v)
        xy = self.xy_from_uvt(uv, derivs=False)
        uv_test = self.uv_from_xyt(xy, derivs=False)

        return (uv_test - uv).norm().max(builtins=True)

    ######################################################################################
    # Properties and methods to support body inventories
    #
    # These might need to be overridden for FOV subclasses that are not rectangular.
    ######################################################################################

    def center_xy(self, time=None):
        """Location of the center of the FOV at the specified time.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Pair: `(x,y)` coordinates of the FOV center.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """

        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.center_xy '
                                          'requires a time; FOV is time-dependent')
            return self.xy_from_uvt(self.uv_shape/2., time=time)

        if not hasattr(self, '_center_xy_filled'):
            self._center_xy_filled = self.xy_from_uvt(self.uv_shape/2.)

        return self._center_xy_filled

    def center_los(self, time=None):
        """The unit line of sight defining the center of the FOV at the specified time.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Vector3: The unit line of sight.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """


        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.center_los '
                                          'requires a time; FOV is time-dependent')
            return self.los_from_xy(self.center_xy(time=time)).unit()

        if not hasattr(self, '_center_los_filled'):
            self._center_los_filled = self.los_from_xy(self.center_xy()).unit()

        return self._center_los_filled

    @property
    def center_dlos_duv(self):
        """The line of sight derivative matrix `dlos/d(u,v)` at the FOV center.

        Returns:
            Vector3: `dlos/d(u,v)`, a Vector3 with a `(u,v)` denominator.

        Raises:
            NotImplementedError: If this FOV is time-dependent, because this property
                takes no time at which to evaluate it.
        """

        if not hasattr(self, '_center_dlos_duv_filled'):
            center_uv = self.uv_shape/2.
            center_uv.insert_deriv('uv', Pair.IDENTITY)
            los = self.los_from_uvt(center_uv, derivs=True)
            self._center_dlos_duv_filled = los.d_duv

        return self._center_dlos_duv_filled

    @property
    def outer_radius(self):
        """The radius of a circle circumscribing the entire FOV.

        Returns:
            float: Radius value in radians.

        Raises:
            NotImplementedError: If this FOV is time-dependent, because this property
                takes no time at which to evaluate it.
        """

        if not hasattr(self, '_outer_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]
            uv_corners = Pair([(0.,0.), (0.,vmax), (umax,0.), (umax,vmax)])

            seps = self.center_los().sep(self.los_from_uvt(uv_corners))
            self._outer_radius_filled = seps.max(builtins=True)

        return self._outer_radius_filled

    @property
    def inner_radius(self):
        """The radius of a circle entirely enclosed within the FOV.

        Returns:
            float: Radius value in radians.

        Raises:
            NotImplementedError: If this FOV is time-dependent, because this property
                takes no time at which to evaluate it.
        """

        if not hasattr(self, '_inner_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]
            umid = umax/2.
            vmid = vmax/2.

            uv_edges = Pair([(0.,vmid), (umax,vmid), (umid,0.), (umid,vmax)])

            seps = self.center_los().sep(self.los_from_uvt(uv_edges))
            self._inner_radius_filled = seps.min(builtins=True)

        return self._inner_radius_filled

    def corner00_xy(self, time=None):
        """The `(x,y)` coordinates where `(u,v) = (0,0)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Pair: The `(x,y)` coordinates.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """

        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.corner00_xy '
                                          'requires a time; FOV is time-dependent')
            return self.xy_from_uvt(Pair.ZEROS, time=time)

        if not hasattr(self, '_corner00_filled'):
            self._corner00_filled = self.xy_from_uvt(Pair.ZEROS)

        return self._corner00_filled

    def corner01_xy(self, time=None):
        """The `(x,y)` coordinates where `(u,v) = (0,v_max)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Pair: The `(x,y)` coordinates.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """

        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.corner01_xy '
                                          'requires a time; FOV is time-dependent')
            return self.xy_from_uvt(Pair((0, self.uv_shape.vals[1])), time=time)

        if not hasattr(self, '_corner01_filled'):
            self._corner01_filled = self.xy_from_uvt(Pair((0, self.uv_shape.vals[1])))

        return self._corner01_filled

    def corner10_xy(self, time=None):
        """The `(x,y)` coordinates where `(u,v) = (u_max,0)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Pair: The `(x,y)` coordinates.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """

        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.corner10_xy '
                                          'requires a time; FOV is time-dependent')
            return self.xy_from_uvt(Pair((self.uv_shape.vals[0], 0)), time=time)

        if not hasattr(self, '_corner10_filled'):
            self._corner10_filled = self.xy_from_uvt(Pair((self.uv_shape.vals[0], 0)))

        return self._corner10_filled

    def corner11_xy(self, time=None):
        """The `(x,y)` coordinates where `(u,v) = (u_max,v_max)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB.

        Returns:
            Pair: The `(x,y)` coordinates.

        Raises:
            NotImplementedError: If this FOV is time-dependent and no `time` is given.
        """

        if not self.IS_TIME_INDEPENDENT:
            if time is None:
                raise NotImplementedError(type(self).__name__ + '.corner11_xy '
                                          'requires a time; FOV is time-dependent')
            return self.xy_from_uvt(self.uv_shape, time=time)

        if not hasattr(self, '_corner11_filled'):
            self._corner11_filled = self.xy_from_uvt(self.uv_shape)

        return self._corner11_filled

    def sphere_falls_inside(self, center, radius, *, time=None, border=0.):
        """True if any piece of a sphere falls inside this FOV.

        Parameters:
            center (Vector3): The apparent vector to the center of a sphere in the frame
                of the FOV, at the given time (km).
            radius (Scalar): The radius of the sphere (km).
            time (Scalar, optional): Absolute time in seconds TDB.
            border (float, optional): Angular extension of the FOV to allow for pointing
                uncertainties (radians).

        Returns:
            Boolean: True if any piece of a sphere falls inside.
        """

        # Perform quick tests based on the separation angles
        sphere_center_los = Vector3.as_vector3(center, recursive=False)

        radius_angle = (radius / sphere_center_los.norm()).arcsin()
        center_los = self.center_los(time=time)
        center_sep = center_los.sep(sphere_center_los)

        if center_sep > self.outer_radius + border + radius_angle:
            return False
        if center_sep <= self.inner_radius + border + radius_angle:
            return True

        # Find the point on the image that falls closest to the center of the sphere
        sphere_center_uv = self.uv_from_los_t(sphere_center_los, time=time)
        nearest_fov_uv  = self.nearest_uv(sphere_center_uv)
        nearest_fov_los = self.los_from_uvt(nearest_fov_uv, time=time)

        # Allow for the border region when returning True or False
        return nearest_fov_los.sep(sphere_center_los) <= radius_angle + border

##########################################################################################

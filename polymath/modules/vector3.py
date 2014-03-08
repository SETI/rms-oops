################################################################################
# polymath/modules/vector3.py: Vector3 subclass of PolyMath Vector
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube   import Qube
from scalar import Scalar
from vector import Vector
from units  import Units

class Vector3(Vector):
    """A vector with a fixed length of three."""

    NRANK = 1           # the number of numerator axes.
    NUMER = (3,)        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    @staticmethod
    def as_vector3(arg):
        if type(arg) == Vector3: return arg

        if isinstance(arg, Qube):

            # Collapse a 1x3 or 3x1 Matrix down to a Vector
            if arg.numer == (1,3) or arg.numer == (3,1):
                return arg.reshape_items((3,), (Vector3, Qube.MATRIX_CLASS))

        return Vector3(arg)

    @staticmethod
    def from_scalars(x,y,z):
        """A Vector3 constructed by combining given x, y and z components.

        Derivates are ignored. Denominator items are disallowed.
        """

        x = Scalar.as_scalar(x)
        y = Scalar.as_scalar(y) #.confirm_units(x.units)
        z = Scalar.as_scalar(z) #.confirm_units(x.units)

        if x.denom or y.denom or z.denom:
            raise NotImplementedError('denominator axes are disallowed')

        (xx, yy, zz) = np.broadcast_arrays(x.values, y.values, z.values)

        new_values = np.empty(xx.shape + (3,), dtype=xx.dtype)
        new_values[...,0] = xx
        new_values[...,1] = yy
        new_values[...,2] = zz
        return Vector3(new_values, x.mask | y.mask | z.mask, x.units)

    ### Most operations are inherited from Vector. These include:
    #     def extract_scalar(self, axis, recursive=True)
    #     def as_scalars(self, recursive=True)
    #     def as_column(self, recursive=True)
    #     def as_row(self, recursive=True)
    #     def as_diagonal(self, recursive=True)
    #     def dot(self, arg, recursive=True)
    #     def norm(self, recursive=True)
    #     def unit(self, recursive=True)
    #     def cross(self, arg, recursive=True)
    #     def ucross(self, arg, recursive=True)
    #     def outer(self, arg, recursive=True)
    #     def perp(self, arg, recursive=True)
    #     def proj(self, arg, recursive=True)
    #     def sep(self, arg, recursive=True)
    #     def cross_product_as_matrix(self, recursive=True)
    #     def element_mul(self, arg, recursive=True):
    #     def element_div(self, arg, recursive=True):
    #     def __abs__(self)

    def spin(self, pole, angle=None, recursive=True):
        """Returns the result of rotating this Vector3 around the given pole
        vector by the given angle. If angle is None, then the rotation angle
        is pole.norm().arcsin().
        """

        pole = Vector3.as_vector3(pole)
        if not recursive:
            pole = pole.without_derivs()
            self = self.without_derivs()

        if angle is None:
            norm = pole.norm()
            angle = norm.arcsin()
            zaxis = pole / norm
        else:
            angle = Scalar.as_scalar(angle)
            if not recursive: angle = angle.without_derivs()

            mask = (angle == 0.)
            if np.any(mask):
                pole = pole.mask_where_eq(Vector3.ZEROS, Vector3.ZAXIS,
                                          remask=False)
            zaxis = pole.unit()

        z = self.dot(zaxis)
        perp = self - z * zaxis
        r = perp.norm()
        perp = perp.mask_where_eq(Vector3.ZEROS, Vector3.XAXIS, remask=False)
        xaxis = perp.unit()
        yaxis = zaxis.cross(xaxis)
        return r * (angle.cos() * xaxis + angle.sin() * yaxis) + z * zaxis

# A set of useful class constants
Vector3.ZERO   = Vector3((0.,0.,0.)).as_readonly()
Vector3.XAXIS  = Vector3((1.,0.,0.)).as_readonly()
Vector3.YAXIS  = Vector3((0.,1.,0.)).as_readonly()
Vector3.ZAXIS  = Vector3((0.,0.,1.)).as_readonly()
Vector3.MASKED = Vector3((1,1,1), True).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.VECTOR3_CLASS = Vector3

################################################################################


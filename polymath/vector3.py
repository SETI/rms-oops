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

TWOPI = 2. * np.pi

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
    def as_vector3(arg, recursive=True):
        if type(arg) == Vector3:
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Qube):

            # Collapse a 1x3 or 3x1 Matrix down to a Vector
            if arg.numer in ((1,3), (3,1)):
                return arg.flatten_numer(Vector3, recursive)

            # For any suitable Qube, move numerator items to the denominator
            if arg.rank > 1 and arg.numer[0] == 3:
                arg = arg.split_items(1, Vector3)

            arg = Vector3(arg, example=arg)
            if recursive: return arg
            return arg.without_derivs()

        return Vector3(arg)

    @staticmethod
    def from_scalars(x, y, z, recursive=True):
        """A Vector3 constructed by combining three scalars.

        Inputs:
            x, y, z     Three Scalars defining the vector's x, y and z
                        components. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found among x, y and z. Default is True.
        """

        return Vector.from_scalars(x, y, z, recursive=recursive,
                                            classes=[Vector3])

    @staticmethod
    def from_ra_dec_length(ra, dec, length=1., recursive=True):
        """Vector3 from right ascension, declination and optional length.

        Inputs:
            ra, dec     Scalars of right ascension and declination, in radians.
                        They need not have the same shape, but it must be
                        possible to cast them to the same shape.

            length      A Scalar of lengths; default 1.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives in ra, dec and length. Default is True.
        """

        ra  = Scalar.as_scalar(ra, recursive)
        dec = Scalar.as_scalar(dec, recursive)

        cos_dec = dec.cos()
        x = cos_dec * ra.cos()
        y = cos_dec * ra.sin()
        z = dec.sin()

        result = Vector3.from_scalars(x, y, z, recursive)

        if length is 1.:
            return result
        else:
            return Scalar.as_scalar(length, recursive) * result

    def to_ra_dec_length(self, recursive=True):
        """A tuple (ra, dec, length) from this Vector3.

        Inputs:
            recursive   True to include the derivatives. Default is True.
        """

        (x,y,z) = self.to_scalars(recursive)
        length = self.norm(recursive)

        ra = y.arctan2(x) % TWOPI
        dec = (z/length).arcsin()

        return (ra, dec, length)

    @staticmethod
    def from_cylindrical(radius, longitude, z=0., recursive=True):
        """Vector3 from cylindrical coordinates.

        Inputs:
            radius      Scalar radius, distance from the cylindrical axis.
            longitude   Scalar longitude in radians. Zero is along the x-axis.
            z           Distance above/below the equatorial plane, default 0.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives in radius, longitude and z. Default is True.
        """

        radius  = Scalar.as_scalar(radius, recursive)
        longitude = Scalar.as_scalar(longitude, recursive)
        z = Scalar.as_scalar(z, recursive)

        x = radius * longitude.cos(recursive)
        y = radius * longitude.sin(recursive)

        return Vector3.from_scalars(x, y, z, recursive)

    def to_cylindrical(self, recursive=True):
        """A tuple (radius, longitude, z) from this Vector3.

        Inputs:
            recursive   True to include the derivatives. Default is True.
        """

        (x,y,z) = self.to_scalars(recursive)
        radius = (x**2 + y**2).sqrt(recursive)

        longitude = y.arctan2(x,recursive) % TWOPI

        return (radius, longitude, z)

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
        perp = perp.mask_where_eq(Vector3.ZERO, Vector3.XAXIS, remask=False)
        xaxis = perp.unit()
        yaxis = zaxis.cross(xaxis)
        return r * (angle.cos() * xaxis + angle.sin() * yaxis) + z * zaxis

# A set of useful class constants
Vector3.ZERO   = Vector3((0.,0.,0.)).as_readonly()
Vector3.ONES   = Vector3((1.,1.,1.)).as_readonly()
Vector3.XAXIS  = Vector3((1.,0.,0.)).as_readonly()
Vector3.YAXIS  = Vector3((0.,1.,0.)).as_readonly()
Vector3.ZAXIS  = Vector3((0.,0.,1.)).as_readonly()
Vector3.MASKED = Vector3((1,1,1), True).as_readonly()

Vector3.ZERO_POS_VEL = Vector3((0.,0.,0.)).as_readonly()
Vector3.ZERO_POS_VEL.insert_deriv('t', Vector3.ZERO).as_readonly()

Vector3.IDENTITY = Vector3([(1,0,0),(0,1,0),(0,0,1)], drank=1).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.VECTOR3_CLASS = Vector3

################################################################################


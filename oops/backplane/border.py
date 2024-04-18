################################################################################
# oops/backplanes/border.py: Boolean backplanes
################################################################################

import numpy as np

from polymath       import Boolean
from oops.backplane import Backplane


def border_above(self, backplane_key, value):
    """The locus of points surrounding the region >= a specified value."""

    return self._border_above_or_below(+1, backplane_key, value)


def border_below(self, backplane_key, value):
    """The locus of points surrounding the region <= a specified value."""

    return self._border_above_or_below(-1, backplane_key, value)


def border_atop(self, backplane_key, value):
    """The locus of points straddling the points closest to a border.

    This backplane is True for the pixels that fall closest to the transition
    from below to above.
    """

    backplane_key = self.standardize_backplane_key(backplane_key)
    key = ('border_atop', backplane_key, value)
    if key in self.backplanes:
        return self.get_backplane(key)

    absval = self.evaluate(backplane_key) - value
    sign = absval.sign()
    absval = absval * sign

    border = (absval == 0.)

    axes = len(absval.shape)
    for axis in range(axes):
        xabs = absval.vals.swapaxes(0, axis)
        xsign = sign.vals.swapaxes(0, axis)
        xborder = border.vals.swapaxes(0, axis)

        xborder[:-1] |= ((xsign[:-1] == -xsign[1:]) &
                         (xabs[:-1] <= xabs[1:]))
        xborder[1:]  |= ((xsign[1:] == -xsign[:-1]) &
                         (xabs[1:] <= xabs[:-1]))

    return self.register_backplane(key, border)


def _border_above_or_below(self, sign, backplane_key, value):
    """The locus of points <= or >= a specified value."""

    backplane_key = self.standardize_backplane_key(backplane_key)

    if sign > 0:
        key = ('border_above', backplane_key, value)
    else:
        key = ('border_below', backplane_key, value)

    if key in self.backplanes:
        return self.get_backplane(key)

    backplane = sign * (self.evaluate(backplane_key) - value)
    border = np.zeros(self.meshgrid.shape, dtype='bool')

    axes = len(backplane.shape)
    for axis in range(axes):
        xbackplane = backplane.vals.swapaxes(0, axis)
        xborder = border.swapaxes(0, axis)

        xborder[:-1] |= ((xbackplane[:-1] >= 0) &
                         (xbackplane[1:]  <  0))
        xborder[1:]  |= ((xbackplane[1:]  >= 0) &
                         (xbackplane[:-1] < 0))

    return self.register_backplane(key, Boolean(border & backplane.antimask))

#===============================================================================
def border_inside(self, backplane_key):
    """Defines the locus of True pixels adjacent to a region of False pixels."""

    return self._border_outside_or_inside(backplane_key, is_inside=True)


def border_outside(self, backplane_key):
    """Defines the locus of False pixels adjacent to a region of True pixels."""

    return self._border_outside_or_inside(backplane_key, is_inside=False)


def _border_outside_or_inside(self, backplane_key, is_inside=True):
    """The locus of points that fall on the outer edge of a mask.

    "Outside" identifies the first False pixels outside each area of True
    pixels; "Inside" identifies the last True pixels adjacent to an area of
    False pixels.
    """

    backplane_key = self.standardize_backplane_key(backplane_key)

    if is_inside:
        key = ('border_inside', backplane_key)
    else:
        key = ('border_outside', backplane_key)

    if key in self.backplanes:
        return self.get_backplane(key)

    backplane = self.evaluate(backplane_key)
    if backplane.dtype() != 'bool':
        raise ValueError('border operation requires boolean mask, not '
                         + backplane.dtype())

    # Reverse the backplane if is_inside is False
    if not is_inside:
        backplane = backplane.logical_not()

    border = np.zeros(backplane.shape, dtype='bool')

    axes = len(backplane.shape)
    for axis in range(axes):
        xbackplane = backplane.vals.swapaxes(0, axis)
        xborder = border.swapaxes(0, axis)

        xborder[:-1] |= ((xbackplane[:-1] ^ xbackplane[1:]) &
                          xbackplane[:-1])
        xborder[1:]  |= ((xbackplane[1:] ^ xbackplane[:-1]) &
                          xbackplane[1:])

    return self.register_backplane(key, Boolean(border))

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################

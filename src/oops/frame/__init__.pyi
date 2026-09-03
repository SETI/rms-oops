##########################################################################################
# oops/frame/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.frame`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from oops.frame.cmatrix import Cmatrix as Cmatrix
from oops.frame.frame_ import (Frame as Frame, J2000Frame as J2000Frame,
                               LinkedFrame as LinkedFrame, NullFrame as NullFrame,
                               ReversedFrame as ReversedFrame)
from oops.frame.frameshift import FrameShift as FrameShift
from oops.frame.inclinedframe import InclinedFrame as InclinedFrame
from oops.frame.laplaceframe import LaplaceFrame as LaplaceFrame
from oops.frame.navigation import Navigation as Navigation
from oops.frame.poleframe import PoleFrame as PoleFrame
from oops.frame.postargframe import PosTargFrame as PosTargFrame
from oops.frame.quickframe import QuickFrame as QuickFrame
from oops.frame.ringframe import RingFrame as RingFrame
from oops.frame.rotation import Rotation as Rotation
from oops.frame.spiceframe import SpiceFrame as SpiceFrame
from oops.frame.spicetype1frame import SpiceType1Frame as SpiceType1Frame
from oops.frame.spinframe import SpinFrame as SpinFrame
from oops.frame.synchronousframe import SynchronousFrame as SynchronousFrame
from oops.frame.trackerframe import TrackerFrame as TrackerFrame
from oops.frame.twovectorframe import TwoVectorFrame as TwoVectorFrame

__all__ = ['Frame', 'NullFrame', 'J2000Frame', 'LinkedFrame', 'ReversedFrame', 'Cmatrix',
           'FrameShift', 'InclinedFrame', 'LaplaceFrame', 'Navigation', 'PoleFrame',
           'PosTargFrame', 'QuickFrame', 'RingFrame', 'Rotation', 'SpiceFrame',
           'SpiceType1Frame', 'SpinFrame', 'SynchronousFrame', 'TrackerFrame',
           'TwoVectorFrame']

##########################################################################################

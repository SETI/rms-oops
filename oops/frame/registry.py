################################################################################
# oops/frame/registry.py: Global Frame Registry
################################################################################

# The frames registry is keyed in two ways. If the key is a string, then this
# constitutes the primary definition of the frame. The reference_id of the frame
# must already be in the registry, and the new frame's ID cannot be in the
# registry.

# We also create secondary definitions of a frame, where it is defined relative
# to a different reference frame. These are entered into the registry keyed by a
# tuple: (frame_id, reference_id). This saves the effort of re-creating a frame
# used repeatedly.

# NOTE: This file has the property that it does NOT need to import the Frame
# class. This is necessary so that modules that must load earlier (such as
# Event and Transform) will not have circular dependencies.

J2000 = None
REGISTRY = {}
FRAME_CLASS = None

def lookup(key): return REGISTRY[key]

def as_frame(frame):
    """Returns a Frame object given the registered name or the object itself."""

    global REGISTRY

    if frame is None: return None

    try:
        test = frame.frame_id
        return frame
    except AttributeError:
        return REGISTRY[frame]

def as_id(frame):
    """Returns a Frame ID given the object or a registered ID."""

    if frame is None: return None

    try:
        return frame.frame_id
    except AttributeError: 
        return frame

def as_primary(frame):
    """Returns the primary definition of a Frame object, based on a registered
    name or a Frame object."""

    global REGISTRY

    try:
        return REGISTRY[frame.frame_id]
    except AttributeError:
        return REGISTRY[frame]

def is_id(item):
    """Returns True if the item is a valid frame ID."""

    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

def is_frame(item):
    """Returns true if the item is a Frame object."""

    global FRAME_CLASS
    return isinstance(item, FRAME_CLASS)

def initialize_registry():
    """Initializes the registry. It is not generally necessary to call this
    function, but it can be used to reset the registry for purposes of
    debugging."""

    global FRAME_CLASS
    FRAME_CLASS.initialize_registry()

################################################################################
# Frame Generator
################################################################################

def connect(target, reference):
    """Returns a Frame object that transforms from the given reference Frame
    to the given target Frame.

    Input:
        target      a Frame object or the registered ID of the destination
                    frame.
        reference   a Frame object or the registered ID of the starting frame.

    The shape of the connected frame will will be the result of broadcasting
    the shapes of the target and reference.
    """

    global FRAME_CLASS

    return FRAME_CLASS.connect(target, reference)

################################################################################

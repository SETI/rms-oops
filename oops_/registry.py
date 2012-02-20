################################################################################
# oops_/registry.py: Global Frame and Path Registry
#
# 2/20/12 MRS - Adapted from frame/registry.py and path/registry.py. This is a
#   cleaner way to accomplish the same goal.
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
FRAME_REGISTRY = {}
FRAME_CLASS = None

def frame_lookup(key): return FRAME_REGISTRY[key]

def as_frame(frame):
    """Returns a Frame object given the registered name or the object itself."""

    global FRAME_REGISTRY

    if frame is None: return None

    try:
        test = frame.frame_id
        return frame
    except AttributeError:
        return FRAME_REGISTRY[frame]

def as_frame_id(frame):
    """Returns a Frame ID given the object or a registered ID."""

    if frame is None: return None

    try:
        return frame.frame_id
    except AttributeError: 
        return frame

def as_primary_frame(frame):
    """Returns the primary definition of a Frame object, based on a registered
    name or a Frame object."""

    global FRAME_REGISTRY

    try:
        return FRAME_REGISTRY[frame.frame_id]
    except AttributeError:
        return FRAME_REGISTRY[frame]

def is_id(item):
    """Returns True if the item is a valid frame ID."""

    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

def is_frame(item):
    """Returns true if the item is a Frame object."""

    global FRAME_CLASS
    return isinstance(item, FRAME_CLASS)

def initialize_frame_registry():
    """Initializes the registry. It is not generally necessary to call this
    function, but it can be used to reset the registry for purposes of
    debugging."""

    global FRAME_CLASS
    FRAME_CLASS.initialize_registry()

def connect_frames(target, reference):
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

# The global registry is keyed in two ways. If the key is a string, then
# this constitutes the primary definition of the path. The origin_id
# of the path must already be in the registry, and the new path's ID
# cannot be in the registry.

# We also create secondary definitions of a path, where it is defined
# relative to a different reference frame and/or with respect to a different
# origin. These are entered into the registry keyed twice, by a tuple:
#   (path_id, origin_id)
# and a triple:
#   (path_id, origin_id, frame_id).
# This saves the effort of re-creating paths used repeatedly.

# NOTE: This file has the property that it does NOT need to import the Path
# class. This is necessary so that modules that must load earlier (such as
# Event and Transform) will not have circular dependencies.

SSB = None
PATH_REGISTRY = {}
PATH_CLASS = None

def path_lookup(key): return PATH_REGISTRY[key]

def as_path(path):
    """Returns a Path object given the registered name or the object
    itself."""

    if path is None: return None

    try:
        test = path.path_id
        return path
    except AttributeError:
        return PATH_REGISTRY[path]

def as_path_id(path):
    """Returns a path ID given the object or a registered ID."""

    if path is None: return None

    try:
        return path.path_id
    except AttributeError:
        return path

def as_primary_path(path):
    """Returns the primary definition of a Path object, based on a
    registered name or a Path object."""

    try:
        return PATH_REGISTRY[path.path_id]
    except AttributeError:
        return PATH_REGISTRY[path]

def is_path(item):
    """Returns true if the item is a Path object."""

    global PATH_CLASS
    return isinstance(item, PATH_CLASS)

def initialize_path_registry():
    """Initializes the registry. It is not generally necessary to call this
    function, but it can be used to reset the registry for purposes of
    debugging."""

    PATH_CLASS.initialize_registry()

def connect_paths(target, origin, frame="J2000"):
    """Returns a path that creates event objects in which vectors point
    from any origin path to any target path, using any coordinate frame.

    Input:
        target      the Path object or ID of the target path.
        origin      the Path object or ID of the origin path.
        frame       the Frame object of ID of the coordinate frame to use;
                    use None for the default frame of the origin.
    """

    global PATH_CLASS
    return PATH_CLASS.connect(target, origin, frame)

################################################################################

def is_id(item):
    """Returns True if the item is a valid path ID."""

    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

################################################################################

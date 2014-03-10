################################################################################
# oops/registry.py: Global Frame and Path Registry
################################################################################

################################################################################
# Frame Registry
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
TEMPORARY_FRAME_ID = 10000

def is_frame(item):
    """Return True if the item is a Frame object."""

    return isinstance(item, FRAME_CLASS)

def frame_lookup(key):
    """Return a frame from the registry given one of its keys."""

    return FRAME_REGISTRY[key]

def as_frame(frame):
    """Return a Frame object given the registered name or the object itself."""

    if frame is None: return None
    if is_frame(frame): return frame
    return frame_lookup(frame)

def as_frame_id(frame):
    """Return a Frame ID given the object or a registered ID."""

    if frame is None: return None
    if is_frame(frame): return frame.frame_id
    return frame

def as_primary_frame(frame):
    """Return the primary definition of a Frame object.
    
    The result is based on a registered name or a Frame object."""

    return frame_lookup(as_frame_id(frame))

def temporary_frame_id():
    """Return a temporary frame ID.
    
    This is assigned once and never re-used."""

    global TEMPORARY_FRAME_ID

    while True:
        frame_id = "TEMPORARY_" + str(TEMPORARY_FRAME_ID)
        if not FRAME_REGISTRY.has_key(frame_id):
            return frame_id
        TEMPORARY_FRAME_ID += 1

def initialize_frame_registry():
    """Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging."""

    FRAME_CLASS.initialize_registry()

def connect_frames(target, reference):
    """Return a Frame object that transforms between two frames.

    Input:
        target      a Frame object or the registered ID of the destination
                    frame.
        reference   a Frame object or the registered ID of the starting frame.

    The shape of the connected frame will will be the result of broadcasting
    the shapes of the target and reference.
    """

    return FRAME_CLASS.connect(target, reference)

################################################################################
# Path Registry
################################################################################

# The global path registry is keyed in two ways. If the key is a string, then
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
TEMPORARY_PATH_ID = 10000

def is_path(item):
    """Return True if the item is a Path object."""

    return isinstance(item, PATH_CLASS)

def path_lookup(key):
    """Return a path from the registry given one of its keys."""

    return PATH_REGISTRY[key]

def is_valid_path_id(item):
    """Return True if the item is a validly-formatted path ID."""

    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

def as_path(path):
    """Return a Path object given the registered name or the object itself."""

    if path is None: return None
    if is_path(path): return path
    return path_lookup(path)

def as_path_id(path):
    """Return a path ID given the object or a registered ID."""

    if path is None: return None
    if is_path(path): return path.path_id
    return path

def as_primary_path(path):
    """Return the primary definition of a Path object.
    
    The result is based on a registered name or a Path object.
    """

    return path_lookup(as_path_id(path))

def temporary_path_id():
    """Return a temporary path ID. This is assigned once and never re-used."""

    global TEMPORARY_PATH_ID

    while True:
        path_id = "TEMPORARY_" + str(TEMPORARY_PATH_ID)
        if not PATH_REGISTRY.has_key(path_id):
            return path_id
        TEMPORARY_PATH_ID += 1

def initialize_path_registry():
    """Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging."""

    PATH_CLASS.initialize_registry()

def connect_paths(target, origin, frame="J2000"):
    """Return a path that connects two paths.
    
    The returned path creates event objects in which vectors point from any
    origin path to any target path, using any coordinate frame.

    Input:
        target      the Path object or ID of the target path.
        origin      the Path object or ID of the origin path.
        frame       the Frame object of ID of the coordinate frame to use;
                    use None for the default frame of the origin.
    """

    return PATH_CLASS.connect(target, origin, frame)

################################################################################
# Body Registry
################################################################################

# A dictionary associating the names of solar system bodies with their Body
# objects.

BODY_REGISTRY = {}
BODY_CLASS = None

def is_body(item):
    """Return True if the item is a Body object."""

    return isinstance(item, BODY_CLASS)

def body_lookup(key):
    """Return a body from the registry given its name."""

    global BODY_REGSITRY
    return BODY_REGISTRY[key]

def body_exists(key):
    """Return True if the body's name exists in the registry."""

    global BODY_REGSITRY
    return BODY_REGISTRY.has_key(key)

def as_body(body):
    """Return a body object given the registered name or the object itself."""

    if is_body(body): return body
    return body_lookup(body)

def as_body_name(body):
    """Return a body name given the registered name or the object itself."""

    if is_body(body): return body.name
    return body

def initialize_body_registry():
    """Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging.
    """

    global BODY_REGISTRY
    BODY_REGISTRY = {}

################################################################################
# General Functions
################################################################################

def initialize():
    initialize_path_registry()
    initialize_frame_registry()
    initialize_body_registry()

################################################################################

# There are no unit tests for registry because it is amply tested in all other
# modules.

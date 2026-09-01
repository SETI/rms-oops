# Critique of src/oops

**Date:** 2026-08-31
**Scope:** All of `src/oops` except `body.py`, `hosts/`, and `spicedb` (excluded per
request). Every module was read in full and reviewed for (1) code correctness, (2)
docstring accuracy against the actual signatures and behavior, (3) docstring consistency
and Google/Sphinx compliance, and (4) house-style conformance per CLAUDE.md. The review
was performed by seven parallel reviewers, one per area; their full findings follow the
summary. Clear-cut defects were fixed in the working tree (nothing committed); judgment
calls are reported only.

**Finding categories:** BUG (incorrect code), DOC (inaccurate or missing docstring),
CONSISTENCY (style mixing, e.g. legacy `Input:`/`Return:` blocks), STYLE (house-style
deviations), SUGGESTION (improvements needing an owner decision).

## Summary of findings

| Area | Files | Findings | Fixed | Bugs | Bugs fixed |
|---|---|---|---|---|---|
| Top-level modules (`event`, `transform`, `config`, ...) | 14 | 67 | 35 | 14 | 11 |
| `oops/path` | 13 | 29 | 18 | 11 | 11 |
| `oops/frame` | 18 | 41 | 17 | 17 | 13 |
| `oops/surface` | 16 | 58 | 31 | 18 | 18 |
| `oops/fov` | 14 | 22 | 6 | 7 | 6 |
| `oops/observation`, `cadence`, `calibration`, `gravity` | 28 | 41 | 14 | 14 | 8 |
| `oops/backplane` | 15 | 69 | 49 | 9 | 9 |
| **Total** | **118** | **327** | **170** | **90** | **76** |

These counts describe the review as first delivered. Work continued afterward, and 35 of
the findings it left open have since been resolved; each is labelled **(fixed after
review)** in place of "(not fixed)", with its entry saying what was done. A few defects
found during that later work are labelled **(found after review, fixed)** and were added
to the file where they belong. The longer pieces of work have their own sections:
"Strengthening the KeplerPath derivative tests", "Replacing the exclusion parameter",
"Completing the backplane docstrings", and "Fixing the FOV caches". One module,
`oops/utils.py`, was deleted outright; its section explains why.

## Headline defects (fixed)

- **`Event` pickling was completely broken** (`event.py`): `__getstate__` stored `None`
  photon attributes, `__setstate__` called a nonexistent method, and the light-time
  norm-scaling flipped vector directions and lost the negative sign of `arr_lt`.
  Rewritten and verified with round-trip tests.
- **`KeplerPath` wobble derivatives were mathematically wrong** (`keplerpath.py`): the
  e2d/i2d branch used sign-flipped/swapped formulas producing velocity errors up to
  6.4%; the derived-element partials dropped all wobble contributions. All partials now
  agree with finite differences to ~1e-9. The existing tests normalize so loosely that
  every one of these bugs passed the suite.
- **`OrbitPlane` eccentric-orbit math used `a*e` (km) where the dimensionless `e` was
  required** in `lon = M + 2e sin M` (velocity, `from_mean_anomaly`, `to_mean_anomaly`).
  Tests use a=1 where the two coincide, hiding the bug; for real rings (e.g. Uranian,
  a ~ 50,000 km) eccentric-ring velocities were wrong by orders of magnitude. This is a
  behavior change that could shift gold-master values. A second, independent error in the
  same routine — the pericenter offset applied in the wrong direction, a first-order
  velocity error — was found later and is described in the `orbitplane.py` section.
- **Latent crashes in untested modules**: unpickling any `SpiceType1Frame`; `RingPlane`
  constructed with radii but no gravity (which also broke `OrbitPlane` with radii);
  OrbitPlane/Ansa unpickling and `Limb(limits=...)` (keyword-only fallout);
  `OblateGravity.osc_from_state`/`geom_from_state`; `Backplane.standardize_event_key`
  with a tuple key and a `default`; `PolarLimb` reading the wrong coordinate and using
  an undefined variable.
- **Newton-solver convergence epsilons could go NaN** in `PolynomialFOV`/`BarrelFOV`
  (missing `abs()` on precision targets with negative pixel scales), and `WCSFOV`
  swapped (u,v) in `uv_shape` (masked until now by square detectors).
- Every `oops/backplane` module banner named the package `oops/backplanes/`; 14 fixed.

## Most important issues left for the owner

Most of these were resolved in the work that followed the review; each says so. The two
still open are `SpiceType1Frame.__init__`'s double registration and the axis-direction
and `z`-definition discrepancies noted below.

- `Event.__init__`: the `or origin.frame` default-frame clause is unreachable — the
  default frame is always J2000, not the origin's frame as documented. *Resolved by the
  owner after this review: the default frame is now the frame of the origin path.*
- `SpiceFrame._FRAME_LOOKUP` is a cache that is never populated, and neither SPICE
  lookup dict is cleared by `Frame._reset_caches` (*both resolved after this review; see
  the `spiceframe.py` entry*); `SpiceType1Frame.__init__` double-registers, clobbering
  its "don't register" branch (still open).
- `ReversedCadence.tstep_at_time` is provably not the inverse of `time_at_tstep`
  (demonstrated numerically; existing tests cancel the error), but the correct
  clipping/derivative semantics are ambiguous. *Resolved after this review; see the
  `reversedcadence.py` entry.*
- The `FOV` base class's `*_filled` caches ignore `time` and are never invalidated when
  a `Fittable` FOV is refit; `sphere_falls_inside`/`outer_radius` cannot work for
  `TDIFOV`. *The caches were fixed after this review; see "Fixing the FOV caches" in the
  `oops/fov` section, which also records a separate TDIFOV defect found there.*
- `SynchronousFrame`'s docstring contradicts its `twovec` axis signs; `Limb`'s `z`
  definition disagrees between `z_clock_from_intercept` and `coords_from_vector3`;
  `Ellipsoid`'s `exclusion` default (0.9) contradicted its own ">= 0.95" recommendation.
  *Resolved after this review: `exclusion` was removed and the zone is now derived from
  the shape; see "Replacing the exclusion parameter" below.*
- `KeplerPath` derivative tests needed a strengthening pass (they hid 6% velocity
  errors). *Resolved after this review; see "Strengthening the KeplerPath derivative
  tests" below.*
- Remaining legacy/incomplete docstrings: `Backplane.get_surface_event` still uses a
  legacy `Inputs:` block; `border.py` and `pole.py` lack `Parameters:` blocks; several
  cache properties are undocumented. *Resolved after this review; see "Completing the
  backplane docstrings" below.*

## Verification

- Every edited file compiles; per-area unit suites were run by each reviewer.
- Main suite as the review was delivered: `pytest tests --ignore=tests/hosts
  --ignore=tests/spicedb` passed 150. It passes 224 after the later work, which added
  tests along with its fixes.
- spicedb suite: 2 passed, 1 skipped, unchanged throughout.
- Host (gold-master) suite: `pytest tests/hosts` passed 5 as the review was delivered, so
  none of the fixes described above disturbed the gold masters. **That is no longer the
  case.** Removing the `exclusion` parameter shrank the masked zone around a body's
  center, which is correct and deliberate, and four of the five standard observations now
  report mask mismatches on their `:LIMB` backplanes. Every one is a "Mask mismatch",
  meaning the masks differ while the values agree; no "Value mismatch" appears. The
  masters need re-adoption to record the wider coverage, which is a decision about
  reference data and was left to the author. See "Replacing the exclusion parameter" for
  the numbers.

---
## Critique: top-level oops modules

Files reviewed: `__init__.py`, `oops.py`, `event.py`, `transform.py`, `constants.py`,
`config.py`, `utils.py`, `cache.py`, `spice_support.py`, `unittester_support.py`,
`lightsource.py`, `meshgrid.py`, `fittable.py`, `mutable.py`.

### src/oops/event.py

- **[BUG] (fixed)** `event.py:151-208` — Pickling was broken for *every* Event.
  `__getstate__` tested `hasattr(self, attr)` for the photon attributes, but `__init__`
  pre-fills all of them with None, so the test was always true and `None` values were
  stored in `more`; unpickling then called `Vector3.as_vector3(None)` and raised
  TypeError. Additionally, `__setstate__` called `self.fset(...)`, a method that does not
  exist (should have been `set_prop`), and the `vec.with_norm(arr_lt)` scaling flipped the
  stored vector's direction and lost the sign of `arr_lt` (which is negative by
  convention). Fixed by storing only non-None properties, saving `arr_lt`/`dep_lt` as
  their own entries, and dropping the norm-scaling and the `fset` calls. Verified with
  round-trip tests covering bare events, arr/arr_lt/perp, dep_ap/dep_lt, and subfields.
- **[BUG] (fixed)** `event.py:938` (`as_all_masked`) — `result._ssb_ == result` was a
  comparison, not an assignment, so `_ssb_` was left None for an SSB/J2000 event; changed
  to `=`. Also normalized the 8-space over-indentation of that if/else block.
- **[BUG] (fixed)** `event.py:934` (`as_all_masked`) — `result.__antimask = False` set the
  name-mangled attribute `_Event__antimask` instead of `_antimask_`; the antimask cache
  was silently untouched. Changed to `result._antimask_ = False`.
- **[BUG] (fixed)** `event.py:698,717` — the `perp` and `vflat` setters called
  `Vector3.as_vector(...)`, which returns a `Vector`, not a `Vector3` (verified in the
  venv), contradicting the documented types. Changed to `Vector3.as_vector3(...)`.
- **[BUG] (fixed)** `event.py:1917` (`ra_and_dec`) — the invalid-`subfield` error message
  read `invalid input value for apparent: {apparent!r}`; now names and reports `subfield`.
- **[BUG] (fixed)** `event.py:134` — `self._frame_ = Frame.as_wayframe(frame) or
  origin.frame`: `Frame._FRAME_REGISTRY[None]` maps to J2000, so `as_wayframe(None)`
  returns the J2000 wayframe and the `or origin.frame` clause is unreachable dead code.
  The docstring claimed the default frame "matches the default frame of the origin",
  which was false — the default was always J2000. Resolved by the owner after this
  review: when `frame` is None, the frame now defaults to the frame of the resolved
  origin path, and the docstring says so.
- **[DOC] (fixed)** `event.py:61` — class docstring listed property `_dep_j2000_ap_`; the
  actual property is `dep_ap_j2000`. Renamed the bullet.
- **[DOC] (fixed)** `event.py:97-101` — `__init__` docstring referred to a `link`
  parameter that does not exist in the signature (stale text from an older API); removed.
  Also fixed the "that are will" typo and added the missing summary period.
- **[DOC] (fixed)** `event.py:800` (`_apply_this_func`) — the Parameters block listed
  `Return (Event)` as if it were a parameter and documented neither `func` nor `*args`;
  rewritten as proper Parameters/Returns sections.
- **[DOC] (fixed)** `event.py:1315` (`wrt`) — a central public method had no Parameters or
  Returns documentation at all; added, modeled on `wrt_path`/`wrt_frame`, including the
  `include_xform` tuple return.
- **[DOC] (fixed)** `event.py:1417,1510,1546` — typos: "urotated", "earliest latest
  times", "specifed", missing periods in `actual_arr` summary and in `sub`; copy-pasted
  comments in `actual_arr`/`actual_dep` said "apparent vector" where the actual vector is
  computed.
- **[BUG] (fixed after review)** `event.py:706` — the `vflat` getter fills `_vflat_` with
  `Vector3.ZERO` on first read, after which the setter raises "surface velocities were
  already defined". Reading the property therefore permanently blocks setting it. Fixed
  after this review by taking the first option: the getter returns `Vector3.ZERO` without
  saving it, so the default is no longer mistaken for a definition. Returning None
  instead was measured and rejected: it fails four tests in the main suite and one
  gold master, because two aberration methods call `vflat.without_deriv('t')` on the
  result.
- **[CONSISTENCY] (fixed after review)** `event.py:557` — the `dep_lt` setter propagates
  to `_ssb_` unconditionally while `arr_lt`'s setter guards with `_ssb_._arr_lt_ is None`;
  harmless but asymmetric. Fixed after this review; the two setters now match.
- **[CONSISTENCY] (fixed after review)** — most simple properties (`time`, `origin`,
  `frame`, `arr`, `dep`, ...) have no docstrings; they are documented centrally in the
  class docstring instead. Acceptable house pattern, but Sphinx `autodoc` would render
  them undocumented. Fixed after this review: all 28 now carry a one-line summary, and the
  central list in the class docstring stays as the fuller description.
- **[CONSISTENCY] (fixed after review)** `event.py:1499` (`sub`) and `shrink`/`unshrink` —
  public methods whose docstrings document behavior in prose without Parameters blocks
  (`reference`, `quick`, `antimask`, `shape` undocumented). Fixed after this review; all
  three now document their parameters and returns.
- **[SUGGESTION] (fixed after review)** `event.py:1040-1110` — a ~70-line commented-out
  block of `with_*_derivs` methods marked "TODO: These are unused; might not work exactly
  as intended. --Mark", containing the retired `#===` separator style. Restored to live
  code after this review rather than deleted, and the TODO's doubt turned out to be
  justified. Two defects had to be repaired before the block would work. `with_pos_derivs`
  read `self.__state__`, a name-mangled attribute that does not exist, so the method
  raised immediately. It then inserted `xform_to_j2000.rotate(event._state_)` as the SSB
  position derivative, but that expression is the rotated *position*; the derivative
  wanted is the rotated *derivative*, `rotated.d_dpos`, which equals the transform matrix.
  The wrong value was about four orders of magnitude too large, being a position in km.
  With that corrected, the SSB derivative matches a numerical derivative of the SSB
  position to 1e-9, and all four methods insert their derivative, propagate it to the SSB
  counterpart, and return unchanged when asked twice. The retired `#===` separators are
  gone and the docstrings carry `Returns:` sections. Six tests were added to
  `tests/test_event.py`; three fail against the code as it was.

### src/oops/transform.py

- **[BUG] (fixed)** `transform.py:160` (`wod` property) — dead self-assignment
  `self._filled_wod = self._filled_wod` immediately after the real assignment; removed.
- **[DOC] (fixed)** `transform.py:35-47` — the class Properties block documented
  attributes that do not exist (`frame_id`, `reference_id`, `origin_id` "str") while the
  real attributes are `frame`, `reference`, `origin`, plus `is_fixed`, none of which were
  listed correctly. Rewrote the block to match the actual attribute set.
- **[DOC] (fixed)** `transform.py:57-64` — `__init__` documented `frame` and `reference`
  with type `Transform` (they are Frame-or-str) and had a doubled parenthesis in
  `omega (Vector3))`.
- **[DOC] (fixed)** `transform.py:116` — `omega1` docstring said "The negative rotation
  *matrix*"; it is the negative rotation *vector* transformed into the target frame.
- **[DOC] (fixed)** `transform.py:185,232` — `rotate`/`unrotate` had malformed parameter
  lines ("pos (Vector3): , Vector or Matrix object"), a stale reference to the retired
  `Array` class, a stale "subfield d_dt" description, and a meaningless "dpos/dpos"
  derivative description; all reworded.
- **[DOC] (fixed)** `transform.py:337` (`unrotate_transform`) — garbled sentence "applies
  the convert coordinates in the parent frame" rewritten; the transforms' composition is
  described in terms of reference frames, matching `rotate_transform`.
- **[DOC] (not fixed)** — `identity`, `invert`, `rotate_transform`, `unrotate_transform`
  have no Parameters sections for their `frame`/`arg` inputs.
- **[SUGGESTION] (fixed after review)** `transform.py:76-81` — `__init__` stores `origin`
  without conversion (`self.origin = origin`), so a string path ID is stored as a string
  while the docstring implies a Path; the sibling attributes go through `as_wayframe`.
  Also `self.filled_*` cache attributes are public but internal; elsewhere in the package
  such caches have been renamed with leading underscores. Both fixed after this review:
  `origin` is normalized the way the sibling arguments are, and the five caches now match
  the `_filled_wod` already in the same file. The missing `Parameters:` blocks on
  `identity`, `invert`, `rotate_transform`, and `unrotate_transform` were added at the
  same time.

### src/oops/config.py

- **[BUG] (fixed)** `config.py:518` (`LOGGING.pop`) — called `LOGGING.set_logging_level`,
  a method that does not exist (AttributeError whenever the popped level differed);
  changed to `set_logger_level` and guarded with `LOGGING.logger` so a None logger cannot
  raise inside `setLevel`.
- **[BUG] (fixed)** `config.py:466` — `literal()` was the only method in the class
  missing its `@staticmethod` decorator; added (behavior unchanged for class-access
  calls, but consistent and safe if ever called on an instance).
- **[DOC] (fixed)** `config.py:53-57` — the comment for `ignore_quickframe_omega` was a
  copy-paste of the `quickframe_numerical_omega` comment; rewritten to describe what the
  flag does (treat omega as zero in a QuickFrame, per `quickframe.py`'s `_omega_zero`).
- **[DOC] (fixed)** `config.py:307` (`LOGGING.print`) — the level table said
  `"ERROR"=20`; 20 is INFO. Also closed an unbalanced parenthesis in the `literal`
  description.
- **[DOC] (fixed)** `config.py:437` — `exception()` had no docstring; added one noting
  the raise-when-no-logger behavior.
- **[DOC] (fixed)** `config.py:543` (`AREA_FACTOR`) — comment typos "due the fact" and
  "are not not quite".
- **[BUG] (not fixed)** `config.py:400-412` (`LOGGING.print`, literal mode with a
  logger) — literal mode sets `LOGGING.log_formatting = False` and installs a literal
  formatter, but nothing ever restores `log_formatting = True`, so after one literal
  message new handlers are no longer formatted. Needs an owner decision on intended
  life-cycle.
- **[CONSISTENCY] (fixed after review)** — `LOGGING.print` uses a legacy `Inputs:` block;
  most one-line staticmethod docstrings omit Parameters (`all`'s `category`/`reset`
  semantics are undocumented). `set_stdout(False)` can disable stdout even when no other
  destination is active, unlike every sibling setter that re-enables stdout as a fallback.
  Fixed after this review: the legacy block is converted, the LOGGING staticmethods that
  take arguments document them, and `set_stdout` applies the same fallback guard its
  siblings use, so stdout can only be switched off while another destination is live.
- **[STYLE] (not fixed)** — classes are declared `class QUICK(object)` etc. with 2-space
  body indent, an intentional legacy pattern used as namespaces; left alone.

### src/oops/mutable.py

- **[DOC] (fixed)** `mutable.py:38` — module docstring: "depend one or more" missing
  "on".
- **[DOC] (fixed)** `mutable.py:427` (`get_params`) — docstring documented a `params`
  parameter that does not exist in the signature (copy-paste from `set_params`); replaced
  with a Returns section.
- **[DOC] (fixed)** `mutable.py:472` (`_get_info`) — "True either if `obj` is Fittable"
  stray word; `_needs_refresh_internal` missing sentence period.
- **[DOC] (fixed)** `mutable.py:667` — `class Mutable(Oops)` had no class docstring;
  added one.
- **[DOC] (fixed)** `mutable.py:730` (`Mutable._version`) — "each time an object of any
  of its sub-objects" corrected to "this object or any of its sub-objects".
- **[CONSISTENCY] (fixed after review)** — `get_param_order`, `get_nparams` lack Returns
  sections; `needs_refresh` writes `obj (object)` / `(bool)` unlike the annotation-style
  used in the rest of the file. Double blank line between `_mutable_names` and
  `_unfrozen_names`. All three fixed after this review.
- **[STYLE] (fixed)** — `import numpy as np` precedes the stdlib imports
  (`collections`, `typing`); the project convention is stdlib first. Same in
  `meshgrid.py`. Both reordered after this review.
- Note: type annotations here and in `fittable.py` are sanctioned by CLAUDE.md.

### src/oops/fittable.py

- **[BUG] (fixed)** `fittable.py:96` — `set_params` was annotated `-> None` but returns
  True/False (and its docstring documents the bool); changed to `-> bool`.
- **[DOC] (fixed)** — missing period after "uses this function"; `copy`'s Returns used
  `(Fittable):` instead of `Fittable:`.
- **[DOC] (not fixed)** `fittable.py:104` — `set_params` docstring has a `Returns:`
  section but the summary does not mention that equal parameters short-circuit before
  the length check, so a wrong-length tuple equal in prefix... (actually unreachable;
  no issue). The `Raises:` note is accurate.
- **[SUGGESTION] (not fixed)** `fittable.py:150` (`copy`) — relies on
  `hasattr(self, 'stripped_id')` to decide whether to drop the last `__getstate__` item;
  fragile coupling to Frame/Path internals, but documented.

### src/oops/lightsource.py

- **[BUG] (fixed)** `lightsource.py:78` — `if weight:` evaluated the truth of a
  Scalar/array weight, which raises for array-valued weights (and treats 0 as absent);
  changed to `if weight is not None:`.
- **[DOC] (fixed)** — typos in `DiskSource.__init__`: "respresenting", "containing
  defining the lines of sight", "If false".
- **[CONSISTENCY] (fixed after review)** — `LightSource.__init__` documents its arguments
  in prose with no Parameters block; `DiskSource.__init__` uses the legacy `Inputs:`
  two-column style while the same file's `photon_to_event` methods are fully modern. Both
  converted after this review.
- **[SUGGESTION] (fixed after review)** `lightsource.py:52-73` — source-type dispatch by
  try/except chains (Pair, then Vector3, then Path) means a malformed 2-element input is
  silently interpreted as an (RA, dec) pair; worth an explicit type check some day. Done
  after this review: the dispatch recognizes each documented form explicitly and rejects
  anything else with a `ValueError` naming the problem, where the try/except chain used to
  end in a bare `KeyError` from `Path.as_primary_path`.
- **[SUGGESTION] (not fixed)** `lightsource.py:87` — registers itself in
  `Body.BODY_REGISTRY` directly; documented behavior, but couples this module to Body
  internals.

### src/oops/meshgrid.py

- **[DOC] (fixed after review)** `meshgrid.py:141-144` (`for_shape`) — `u_axis`/`v_axis`
  are documented "(optional)" with no type; should be "(int, optional)". Fixed after this
  review.
- **[SUGGESTION] (fixed after review)** `meshgrid.py:222-231` (`for_shape`) — when `u_axis
  < 0` the code still builds `u_range` from `origin[0]`/`limit[0]`; if that range has more
  than one element the trailing reshape raises with an unhelpful message. Fixed after this
  review by handling it rather than raising: an absent axis is meant to yield a single
  sample, which is what the surrounding code already assumes when it fills that coordinate
  with 0.5 and reserves an extent of 1 for it. Only cases that previously crashed behave
  differently. Reaching the bug needs an explicit `limit`, because the default is 1 along
  a missing axis.
- **[STYLE] (not fixed)** — `import numpy as np` precedes `import numbers` (stdlib).
- Otherwise the cleanest module in this group: modern docstrings with accurate
  Parameters/Returns throughout, correct caching semantics (`_as_key` False sentinel).

### src/oops/cache.py

- **[SUGGESTION] (not fixed)** `cache.py:63` (`clean_key`) — the `case Cache._Path()`
  class patterns require `oops/__init__.py` to have filled `_Path`/`_Frame`; if the
  module were used standalone the match statement would raise TypeError. Consistent with
  the documented "always import oops first" trap.
- **[SUGGESTION] (fixed after review)** `cache.py:66` — `case x if hasattr(x, '__data__')`
  keys an object by `id()`, which can collide after garbage collection; acceptable for a
  cache that tolerates false hits only as stale entries, but worth a comment. The comment
  was added after this review; the behavior is unchanged.
- Docstrings are modern, accurate, and complete.

### src/oops/spice_support.py

- **[DOC] (fixed after review)** `frame_id_and_name` falls through and implicitly
  returns None for an argument that is neither int nor str, while `body_id_and_name`
  raises LookupError; neither docstring mentions its failure behavior, and both
  docstrings are one-line legacy style without Parameters/Returns/Raises. Both now carry
  full Parameters/Returns/Raises blocks, and `frame_id_and_name` raises `LookupError` on
  that fall-through, matching its sibling. Neither function has a caller anywhere in the
  repository — `SpiceFrame` and `SpicePath` use their own `_frame_code_and_name` and
  `_body_code_and_name` — so nothing depended on the implicit None.
- **[DOC] (fixed after review)** `initialize()` has no docstring. Added, noting that it
  discards every translation registered since the last call.
- **[STYLE] (fixed after review)** — two comments read "does not raise an error; I may
  fix", a personal note that should not survive in library code. Both now state the fact
  the code depends on: `cspyce.frmnam` returns an empty name and `cspyce.namfrm` returns
  zero for an unrecognized argument instead of raising, which is why each call is
  followed by a test of the result.
- **[SUGGESTION] (fixed after review)** — the TypeError message "a SpicePath cannot
  originate from a X" describes the situation backwards (the registered path is not a
  SpicePath). It now names the offending path and its actual type.

### src/oops/unittester_support.py

- **[BUG] (fixed)** line 2 — the banner named the file `oops/oops_resources.py`; the file
  is `oops/unittester_support.py`. Corrected.
- **[DOC] (fixed)** — the comment above `TEST_SPICE_FILECACHE` claimed it stores the
  `$OOPS_RESOURCES/SPICE` directory; the code (and the actual resource tree, verified)
  uses `$OOPS_TEST_DATA_PATH/SPICE`. Corrected.
- **[SUGGESTION] (fixed after review)** — the module comment says `$OOPS_RESOURCES` is
  "expected to have two subdirectories"; README documents five. The comment now says this
  module uses two of them and points to the README for the rest.

### src/oops/utils.py

**Deleted after this review.** The question the review left open, whether the module still
had callers, was answered: it did not. Nothing in `src/oops`, `programs`, or the hosts
called any of its seventeen functions, and no qualified `utils.<name>` reference existed
anywhere in the repository. Its only consumers were `oops/__init__.py`, which imported it
and listed it in `__all__`, and `tests/test_utils.py`, which existed solely to exercise
it. The module predates polymath and duplicates on raw ndarrays what the polymath types
now provide as methods; the apparent matches for names like `dot`, `norm`, and `unit`
elsewhere in the library are all polymath method calls.

The module and its test were removed, along with the import and the `__all__` entry.
Note that this narrows the published API: `oops.utils` was exported, so an external
caller relying on it would break. The findings below were resolved by the deletion rather
than by editing:

- **[DOC] (fixed)** — "illustation" typo in the `mxm` explanatory comment.
- **[CONSISTENCY] (moot)** — every docstring is the legacy one-line
  `f(a,b) = ...` form; none has Parameters/Returns. Flagged for a future conversion
  pass, not converted.
- **[SUGGESTION] (resolved)** — module appears to predate polymath and mimics SPICE
  routines on raw ndarrays; worth checking whether it still has callers.

### src/oops/__init__.py

- **[STYLE] (fixed)** — `except ImportError as err:` bound an unused variable; now a bare
  `except ImportError:`.
- **[SUGGESTION] (fixed after review)** — the trailing comment block describing the
  import hierarchy lists "Body, Surface, Path, Gravity, Event, Frame, Transform", but the
  injected attributes below also cover Cache and Fittable; comment is mildly stale. It now
  says those two sit outside the hierarchy and are filled in for the same reason.

### src/oops/oops.py

- **[DOC] (fixed after review)** — `class Oops` has a `#` comment instead of a
  docstring; the only class in the group with neither docstring nor banner description.
  The comment is now a docstring saying what the class is for: a common ancestor for the
  library's objects, defining no behavior of its own. It is subclassed by `Transform`,
  `Fittable`, and `Mutable`.

### src/oops/constants.py

- No defects. Legacy inline-comment style, accurate values.

### Summary

Counts: 12 BUG (10 fixed, 2 reported), 24 DOC (18 fixed, 6 reported), 7 CONSISTENCY
(0 fixed — conversions deliberately deferred), 5 STYLE (2 fixed), 10 SUGGESTION.

The most serious finding was that `Event` pickling was completely broken (every
round-trip raised TypeError, and the design would have silently flipped photon vectors
and light-time signs had it run); it is now fixed and verified. Two other latent
AttributeError-class bugs (`LOGGING.pop`, `Event.__setstate__.fset`) and two
silent-wrong-result bugs (`as_all_masked`'s `==` and name-mangled `__antimask`) were
fixed. The modern docstrings (meshgrid, cache, mutable, most of event) are accurate and
close to Google/Sphinx style; the remaining gaps are concentrated in the legacy-style
modules (utils, spice_support, config's LOGGING.print, DiskSource) and in missing
Parameters blocks on a handful of prominent public methods. Tests: all 20 tests in the
seven matching test modules pass, plus targeted pickle round-trip verification.

---
## Critique: src/oops/path/

Reviewer scope: all 13 .py files under `src/oops/path/`. Every file was read in full.
All fixes below were verified with `py_compile`, targeted numerical experiments, the
`tests/path` suite (31 passed), and the full main suite (150 passed).

### src/oops/path/keplerpath.py

- **[BUG] (fixed)** `keplerpath.py:441-447` — In the 'e2d'/'i2d' wobble branch, the
  analytic derivative formulas were wrong: `damp2_dx2 = -x2/amp2` and `damp2_dy2 =
  -y2/amp2` had the wrong sign (d/dx sqrt(x²+y²) = +x/amp2), and `dangle2_dx2`/
  `dangle2_dy2` had the arctan2 partials swapped (correct: d/dx = -y/r², d/dy = x/r²).
  This corrupted the returned *velocity* whenever an e2d/i2d wobble was active; a
  finite-difference check showed relative velocity errors up to 6.4% (e2d) and 3.2%
  (i2d) with a fast wobble. After the fix, velocity agrees with d(pos)/dt to ~1e-10.
- **[BUG] (fixed)** `keplerpath.py:490` — In the single-element wobble branch, the
  partial of `w = amp*cos(arg)` with respect to `phase0` was computed as
  `self._amp[k] * cos_arg[k]`: the formula is wrong (should be `-amp*sin_arg`) and the
  `[k]` indexes the time array by the wobble index (IndexError for scalar time, silently
  wrong values otherwise). Fixed to `-self._amp[k] * sin_arg`; the `start+2` entry
  (`* t`) then follows correctly.
- **[BUG] (fixed)** `keplerpath.py:556-563` — The partials of the derived elements were
  hardcoded (`dae_delem[..., _SEMIM] = e`, `dae_delem[..., _ECCEN] = a`,
  `dcosi/dsini` to `_INCLI` only), silently dropping every wobble contribution carried
  in `da_delem`, `de_delem`, and `di_delem`. Replaced with the general chain rule
  (`dae_delem = a*de_delem + e*da_delem`, etc.). After all three fixes, per-element
  partials agree with central finite differences to ~1e-9 for every wobble type
  ((), e, i, a, mean, e2d, i2d).
- **[BUG] (fixed)** `keplerpath.py:232` — `_show()` referenced `self._body`, an
  attribute that does not exist (the constructor stores `self._planet`), so any
  `show(level>=2)` call raised AttributeError. Fixed to `self._planet`.
- **[BUG] (fixed)** `keplerpath.py:827` — `pole_at_time()` referenced
  `self._frame_wrt_j2000`, which no code path ever defined, so the method always raised
  AttributeError. The constructor now defines
  `self._frame_wrt_j2000 = self._planet.ring_frame.wrt(Frame.J2000)`; the returned pole
  is a unit vector consistent with the ring frame's z-axis tilted by the inclination.
- **[DOC] (fixed)** `keplerpath.py:75-78` — The constructor docstring said a dict of
  elements uses "keys with these names", pointing at the array-order names (`lon`,
  `peri`, `prec`, `node`, `regr`), but the code reads keys `mean0`, `peri0`,
  `dperi_dt`, `node0`, `dnode_dt` (plus `amp`, `phase0`, `dphase_dt`). The docstring
  now lists the actual keys.
- **[STYLE] (not fixed)** `keplerpath.py:785` — `_photon_from_planet` has a mutable
  default argument `converge={}` (flake8-bugbear B006). It is never mutated, so it is
  harmless, but `converge=None` would match the rest of the API. It also lacks a
  docstring.
- **[SUGGESTION] (not fixed)** `tests/path/test_keplerpath.py` — The derivative tests
  normalize errors by `pos_norm` and scale by tiny parameter deltas, which made them
  numerically incapable of detecting any of the four partial/velocity bugs above (all
  passed the suite). Consider normalizing each element's error by the magnitude of that
  element's own partial, and adding a velocity-vs-finite-difference check.
- **[DOC] (not fixed)** `keplerpath.py:292-310` — `_xyz_planet`'s Returns section does
  not mention that with `partials=True` the position carries a derivative named
  'elements' (accessed as `pos.d_delements`). Minor; internal method.

### src/oops/path/linearpath.py

- **[BUG] (fixed)** `linearpath.py:44-46` — The documented form "velocity defined via a
  derivative 'd_dt'" could never work: `.wod` was applied to `pos` *before* checking
  `hasattr(pos, 'd_dt')`, and `.wod` strips all derivatives (verified empirically), so
  the velocity was silently `Vector3.ZERO`. The derivative is now captured before
  `.wod` is applied; a LinearPath built from a pos with `d_dt` now moves.
- **[SUGGESTION] (not fixed)** `linearpath.py:53` — `_shape` broadcasts `_pos`,
  `_epoch`, origin, and frame but omits `_vel`, so a shaped velocity with a shapeless
  position yields shape (). Edge case; left as is.

### src/oops/path/multipath.py

- **[BUG] (fixed)** `multipath.py:155-157` — `quick_path()` iterated
  `self._input_paths`, which holds the *raw constructor inputs* and may contain path ID
  strings (`Path.as_path` conversion happens only into `self._paths`); calling
  `.quick_path()` on a string raises AttributeError. It also allocated `new_paths` with
  `time.shape` instead of `self._shape`, mismatching the `ndenumerate` indices whenever
  the broadcast time had extra leading dimensions. Now converts via `Path.as_path` and
  allocates with `self._shape`.
- **[STYLE] (fixed)** `multipath.py:117` — The mask accumulator was `np.empty(shape)`
  (float dtype) used as a boolean mask; polymath casts it, but it now uses
  `dtype=bool` explicitly.
- **[SUGGESTION] (not fixed)** `multipath.py:157` — `time[..., k]` uses the tuple `k`
  from `np.ndenumerate`; this is only well-behaved for the documented 1-D case. If
  N-D MultiPaths are ever intended (`__getitem__` suggests so), this indexing needs
  revisiting.
- **[DOC] (not fixed)** `multipath.py:51` — `__getitem__` has an inline comment but no
  docstring; it returns a scalar Path or a sliced MultiPath, worth documenting.

### src/oops/path/path_.py

- **[BUG] (fixed)** `path_.py:455` — `Path.as_waypoint(id_string)` returned the
  registry entry itself rather than its waypoint, unlike the Path-object branch and
  unlike the parallel `Frame.as_wayframe`, which does `._wayframe` on the registry
  entry. A registered Path that shares a waypoint with an earlier definition would be
  returned as a non-waypoint. Fixed to return `Path._PATH_REGISTRY[path]._waypoint`.
- **[DOC] (fixed)** `path_.py:286` — The comment on `_PATH_REGISTRY` said
  "path ID -> waypoint", but `_register()` stores the registered Path itself. Comment
  corrected.
- **[DOC] (fixed)** `path_.py:335` — Comment said "its existing wayframe" (a Frame
  term, copied from frame code) where "waypoint" was meant.
- **[CONSISTENCY] (not fixed)** `path_.py:748` — `NullPath.event_at_time` declares
  `quick=False` while the abstract method and other subclasses use `quick=None`. The
  parameter is unused here (as in QuickPath, where `False` is deliberate), so this is
  cosmetic.
- **[SUGGESTION] (not fixed)** `path_.py:430-433` — `as_primary_path(id_string)`
  returns the registry entry without `._primary`. This mirrors
  `Frame.as_primary_frame` and is safe today because `_register()` always sets
  `_primary = self` on registered paths, but the symmetry with the object branch is
  worth a comment if that invariant ever weakens.

### src/oops/path/spicepath.py

- **[BUG] (fixed)** `spicepath.py:106-107` — In `_show()`, two adjacent f-strings were
  implicitly concatenated into a single list element, so the path name and the origin
  sub-display were joined with no `,\n` separator, producing a malformed multi-line
  display. Split into two list elements as clearly intended.
- **[STYLE] (not fixed)** `spicepath.py:130-144` — `_body_code_and_name` re-raises
  `LookupError` without `from`, losing the cspyce traceback; also `except (KeyError,
  LookupError)` is redundant since KeyError is a subclass of LookupError. Harmless.

### src/oops/path/quickpath.py

- **[STYLE] (fixed)** `quickpath.py:387,433,443` — Three `LOGGING` diagnostic messages
  ended with a stray unmatched `)` (`'... {tmin:.3f}, {tmax:.3f})'`). Removed.
- **[DOC] (fixed)** `quickpath.py:360,410,438` — Three comments said "QuickFrame"
  where "QuickPath" was meant (copy-paste from quickframe.py).
- **[DOC] (fixed)** `quickpath.py:28-30` — The constructor's Raises section documented
  only the shape check; it also raises for a QuickPath-of-a-QuickPath and for a
  failed "path_self_check" precision test. Docstring extended.
- **[CONSISTENCY] (not fixed)** `quickpath.py:148` — `event_at_time(quick=False)`
  default differs from the abstract signature; deliberate here (a QuickPath never
  quickens itself), so left alone.

### src/oops/path/circlepath.py

- **[BUG] (fixed)** `circlepath.py:72,75` — `_show()` had a stray `)` after the origin
  element and wrapped the optional frame element in extra parentheses, producing
  doubled/misplaced closing parens in the display (the method already appends `)`
  after the join). Both removed.
- **[STYLE] (not fixed)** `circlepath.py:61-62` — Double blank line between methods;
  harmless deviation from the file's own single-blank-line rhythm.

### src/oops/path/linearcoordpath.py

- **[BUG] (fixed)** `linearcoordpath.py:80-84` — `_show()` appended `epoch = ...,` with
  a literal trailing comma even though the elements are joined with `',\n'` (yielding
  a doubled comma), used `indent+6` where the sibling classes use `skip+6`, and never
  appended the final closing `)`. All three corrected to match CoordPath.

### src/oops/path/coordpath.py, fixedpath.py, pathshift.py, __init__.py

- No defects found. Docstrings are accurate against the signatures and behavior,
  modern Google style throughout (`Parameters:`, noun-phrase summaries, `Returns:`/
  `Raises:` only where applicable), and `__init__.py`'s `__all__` matches its imports
  exactly.

### Docstring style assessment (whole package)

The package is fully converted to the modern Google/Sphinx style; no legacy
`Input:`/`Return:` blocks remain. Summary lines are noun phrases, `Parameters:` is
used consistently (never `Args:`), keyword-only markers in signatures are reflected
in the docs, and text is wrapped within 90 columns. The per-subclass
`event_at_time` docstrings are uniform nearly to the word, which is a strength. The
only recurring soft spot is that `quick` is described identically even in classes
that ignore it (NullPath, QuickPath), and `_show`/`_waypoint_key`/`_refresh`
overrides are undocumented (private, so acceptable).

### Summary

- BUG: 12 (12 fixed) — 5 in keplerpath (2 numerically significant: wrong wobble
  velocities and dropped/incorrect wobble partials), 1 silent zero-velocity bug in
  linearpath, 1 crash + 1 allocation bug in multipath.quick_path, 1 registry-semantics
  bug in Path.as_waypoint, 4 malformed `_show` displays.
- DOC: 8 (7 fixed, 1 noted) — one factual docstring error (KeplerPath dict keys),
  comment typos, and completions.
- CONSISTENCY: 2 (0 fixed, deliberate/cosmetic).
- STYLE: 5 (2 fixed, 3 noted).
- SUGGESTION: 4 (reported only), the most important being that the KeplerPath
  derivative tests are numerically too loose to catch real errors — every fixed bug
  passed the existing suite.

Overall: the package's documentation quality is high and consistent, and the core
Path/registry/photon-solver machinery is solid. The defects clustered in the less
frequently exercised corners: KeplerPath's wobble mathematics (real, measurable
errors), display methods (`_show`), and MultiPath.quick_path. Tests: `tests/path`
31 passed; full main suite 150 passed after all fixes.

### Strengthening the KeplerPath derivative tests

Every wobble-derivative bug above passed the existing suite, so `tests/path/
test_keplerpath.py` was rewritten after the review. The single `test_keplerpath`
function became six, parametrized over the eight orbits it already exercised, and the
main suite grew from 150 tests to 176.

**Why the old tests passed buggy code.** Both helpers measured

    (analytic_partial * step - actual_change) / |position|

The flaw is the denominator. In the observer-frame helper `|position|` is the
Earth-Saturn distance, about 1.29e9 km, while the orbital signal being tested is the
140,000 km orbit — so every error was divided by roughly 10,000 times its own scale. The
perturbation was itself a relative step of 1e-5, shrinking the numerator again. A 6%
error in a partial landed near 1e-11, against a threshold of 1e-8. The suite could not
have failed.

**What replaced it.** Each partial is now compared against a central finite difference
and normalized by that element's own partial, so the measure is the relative error of
the quantity under test:

    max_t |analytic(t) - numeric(t)| / max_t |numeric(t)|

Three details make that measure trustworthy, each found by measurement rather than
assumption:

- *Step size.* A relative step is wrong for this parameter set, whose magnitudes span
  1e-8 to 1e5 in mixed units. On a wobble amplitude of 4.4e-8 a relative step of 1e-5
  moves the body by 3e-8 km against a round-off floor of 2e-11 km — three surviving
  digits, pure noise. Each step is now sized to move the body a fixed distance. A rate
  element needs a second bound: it multiplies time, so a step that looks small swings
  the phase by 360 radians over the 10^5-second span, far outside the linear regime
  where a derivative means anything. Rate steps are therefore capped by the phase they
  accumulate.
- *Normalizing by the amplitude, not the instantaneous value.* A wobble partial
  oscillates through zero. Dividing by its value at each time produced relative errors of
  1e7 at the crossings for partials that were in fact correct to three digits.
- *A noise floor per element.* The weakest wobble elements move the body only a few times
  the round-off, and no choice of step recovers them; against the observer distance the
  predicted noise is 11%, and 8.8% was observed. Each element's tolerance is therefore
  the stated 1e-4 or its own measured round-off limit, whichever is larger, and an
  element whose signal never clears the floor is skipped rather than tested vacuously.
  The nine orbital elements always clear it, and the planet-frame test asserts that every
  element was genuinely checked.

**A gap that had no test at all.** Nothing compared the velocity returned with the
position against the time derivative of that position — the very quantity the 6.4% error
corrupted. `test_velocity_matches_position_derivative` now does, to 1e-6; the observed
error is 3.5e-9.

**Verification.** The rewritten tests were run against the pre-fix `keplerpath.py`
restored from commit 2c61fee. The old test passed it; 17 of the 27 new tests fail it,
including the velocity checks for the `e2d`, `i2d`, and `i2d+e2d+a` orbits (velocity
error 3.2e-3 against a 1e-6 threshold) and the element partials (element 3 off by 2.44,
a 244% error, against a 1e-4 threshold). Against the fixed code all 27 pass.

The rewrite also made the module's solar-system fixture module-scoped. It had been
rebuilt for each test, which cost 3 seconds per test; at 27 tests the file took 92
seconds, and it now takes 3.

---
## Critique: src/oops/frame/

All 18 files read in full. Fixes verified with `py_compile`, `pytest tests/frame` (13
passed), `pytest tests/path tests/test_transform.py` (32 passed), and a smoke test of
every repaired `_show` method.

### src/oops/frame/frame_.py
- **[BUG] (fixed)** `frame_.py:664` — `NullFrame.__init__` assigned `self._wayframe =
  frame._wayframe` twice (lines 660 and 664). Removed the duplicate.
- **[DOC] (not fixed)** `frame_.py:874-878, 970-973` — The boilerplate paragraph in
  `transform_at_time_if_possible` ("The default behavior is to assume that all times are
  valid. As a result, this method calls `transform_at_time`...") is copied into the
  `LinkedFrame` and `ReversedFrame` overrides, where it is inaccurate: those overrides
  delegate to their component frames and can genuinely drop times. The same stale
  paragraph appears in `QuickFrame.transform_at_time_if_possible` (quickframe.py:243-246).
  Deciding the right replacement wording per class is a judgment call, so it is reported
  rather than edited.
- **[CONSISTENCY] (not fixed)** `frame_.py:684` — `NullFrame.transform_at_time` declares
  `quick=False` while the abstract signature (line 140) uses `quick=None`. Harmless
  (the argument is ignored), but several fixed-frame subclasses do the same
  (`QuickFrame:210`, `Cmatrix`, `Navigation`, `Rotation`, ...) while others use `None`;
  the convention is not applied uniformly.
- **[DOC] (not fixed)** `frame_.py:340` — The comment `_FRAME_REGISTRY = {}  # frame ID ->
  wayframe` is imprecise: `_register()` (line 405) stores the registered Frame itself,
  which is not always its wayframe.
- **[SUGGESTION] (not fixed)** `frame_.py:567` — `_wrt` compares `wayframe == reference`
  with `==` where the class docstring (line 55) promises `is`-comparability of wayframes;
  `is` would be cheaper and clearer. Behavior is correct either way.

### src/oops/frame/quickframe.py
- **[BUG] (fixed)** `quickframe.py:584, 630, 640` — All three
  `LOGGING.diagnostic(...)` messages ended with a stray unmatched `)` inside the text
  (e.g. `'... {tmax:.3f})'`). Removed the stray parentheses.
- **[DOC] (not fixed)** `quickframe.py:525-528` — The `quick` parameter of `for_frame`
  (and of `Frame.quick_frame`, frame_.py:622) is documented as "If False, no QuickFrame is
  created", but the code returns `frame` unchanged for *any* non-dict, non-None value,
  including `True`. This is the documented CLAUDE.md trap; the docstring would benefit
  from stating that only `None` or a dict enables a QuickFrame. Left for a coordinated
  wording pass because the same text recurs in path/quickpath.py (owned by another
  reviewer).
- **[SUGGESTION] (not fixed)** `quickframe.py:292` — In the empty-time branch of
  `_interpolate_matrix_omega`, `omega` is built from `np.ones(...)`; zeros would be more
  natural. Harmless because the array is empty by construction.
- **[SUGGESTION] (not fixed)** `quickframe.py:97-99` — `_refresh` computes `self._steps =
  len(times)` before `transform_at_time_if_possible` may shorten `times`, so `_steps` can
  overcount dropped samples; it is only used in the extension-cost heuristic of
  `for_frame`, so the effect is a slightly wrong cost estimate, not a wrong result.

### src/oops/frame/spiceframe.py
- **[BUG] (fixed)** `spiceframe.py:150-151` — `_show` for a frame with a non-J2000
  reference returned a *tuple* of two strings (trailing comma inside the parentheses) and
  omitted the closing `)` of the rendered text. `Frame.show()` promises a `str`. Rewrote
  as a single concatenated f-string ending with `)`.
- **[BUG] (fixed after review)** `spiceframe.py:22, 556-561` — `SpiceFrame._FRAME_LOOKUP`
  is declared with the comment "(name, reference name, omega_type, omega_dt) ->
  SpiceFrame" and consulted in `get()`, but nothing ever stores a `SpiceFrame` under such
  a key, so this cache can never hit and `get()` constructs a fresh `SpiceFrame` on every
  miss of the earlier name checks. Populating it (e.g. in `get()` after construction)
  looks intended, but `Frame._reset_caches()` (frame_.py:344-358) does not clear
  `_FRAME_LOOKUP` or `_FOR_NAME`, so adding entries could leak stale frames across the
  registry resets the test fixtures rely on. Fixed after this review, taking the first of
  those options: the constructor stores the frame under its key, and `_reset_caches` now
  clears `_WAYFRAMES`, `_FOR_NAME`, and `_FRAME_LOOKUP` on every Frame subclass that
  defines them. The stale-frame hazard is closed because every code path that empties the
  registry, `Body.reset_registry` and the test fixtures included, goes through
  `_reset_caches`; nothing clears the registry directly except `_reset_caches` itself.

  Two details the key needs. A frame that is inertial relative to its reference has its
  `omega_type` forced to "zero" by the constructor whichever option was requested, so one
  frame satisfies every request and is stored under all three; without this, the default
  request of "tabulated" would miss on every such frame and rebuild it. And a key with
  `omega_dt` of None is stored alongside the real value, so a caller that does not
  constrain the step matches, which is what `get`'s docstring promises. Measured effect:
  100 `get` calls for two frames now run the constructor twice instead of 100 times.

  Four tests were added to `test_spiceframe.py`, covering reuse, the omega options, the
  inertial-frame sharing, and the reset; all four fail against the unpopulated cache.
- **[CONSISTENCY] (not fixed)** `spiceframe.py:464, 467` — The numerical-omega branch of
  `transform_at_time_if_possible` uses the `.values` alias while the rest of the file
  uses `.vals`. Both work; one spelling should be chosen.
- **[DOC] (not fixed)** `spiceframe.py:423-426, 439-442, 474-477` — The docstring
  advertises tolerance of cspyce errors, but the code re-raises for any time array with
  more than one dimension (`if len(time.shape) > 1: raise e`). This 1-D-only restriction
  is worth documenting.
- **[BUG] (found after review, fixed)** `spiceframe.py`, `spicetype1frame.py` — Both
  `get()` methods read `reference._spice_frame_name` before checking that the reference is
  something the constructor could use, so a reference that is neither a SpiceFrame nor
  J2000 raised `AttributeError` instead of the `ValueError` the constructor documents and
  raises for the same input. The rule now lives in one place, `_reference_spice_info`,
  used by the constructor and by both `get()` methods, so all three agree and fail the
  same way. Found while implementing `_FRAME_LOOKUP` above, since populating that cache
  made the two key-building paths matter.

- **[SUGGESTION] (not fixed)** `spiceframe.py:124-128` — `_fill_spice_info` sets
  `self._omega_type = 'zero'` for doubly-inertial frames, but `SpiceFrame.__init__`
  (line 64) immediately overwrites it from the `omega_type` argument; the inertia-based
  forcing actually in effect is the one at lines 60-61. The assignment inside
  `_fill_spice_info` is dead for SpiceFrame and only matters for SpiceType1Frame.

### src/oops/frame/spicetype1frame.py
- **[BUG] (fixed)** `spicetype1frame.py:103-105` — `__setstate__` called
  `self.__init__(frame_name, reference, tick_tolerance=tick_tolerance, ...)`, passing
  `reference` positionally into the `tick_tolerance` slot and then `tick_tolerance` again
  by keyword: every unpickle raised `TypeError: multiple values for argument
  'tick_tolerance'`. Reordered to `__init__(frame_name, tick_tolerance, reference, ...)`.
- **[BUG] (fixed)** `spicetype1frame.py:314` — `get()` compared the requested
  `cache_size` against `spice_frame._cache` (a `Cache` object) instead of
  `spice_frame._cache_size`, so a matching cache size never short-circuited. Fixed the
  attribute name.
- **[BUG] (fixed)** `spicetype1frame.py:346` — `_get_shortcut` tested
  `isinstance(ancestor, (SpiceFrame, Frame.J2000Frame))`, but `Frame` has no attribute
  `J2000Frame`, so reaching a J2000 ancestor raised `AttributeError`. Imported
  `J2000Frame` and used it directly, matching spiceframe.py:579.
- **[BUG] (fixed)** `spicetype1frame.py:184-266` — Every return path of
  `transform_at_time_if_possible` returned a bare `Transform`, violating the documented
  `(newtimes, transform)` contract and breaking any caller that unpacks the result
  (e.g. `QuickFrame._refresh`, which this class enables via `_USE_QUICKFRAMES`). Each
  return now yields `(time, transform)`. The method still does not actually skip
  error-raising times (see next item).
- **[DOC] (not fixed)** `spicetype1frame.py:190-192` — The docstring still claims the
  method "tolerates times that raise cspyce errors", but there is no try/except; whether
  gap-skipping should be implemented (as in SpiceFrame) or the claim dropped is a design
  decision.
- **[DOC] (fixed)** `spicetype1frame.py:284` — `get()` documented `tick_tolerance` as
  "optional"; it is a required positional parameter. Removed the marker.
- **[BUG] (not fixed)** `spicetype1frame.py:73-75` — `__init__` calls `_register` a
  second time (line 75) after both branches above have already registered or deliberately
  not registered: in the non-J2000 branch this defeats the "cache but don't register"
  intent, and in the J2000 branch the second call uses the raw SPICE name *without* the
  `replace(' ', '_')` applied at line 71, so a name containing spaces registers under two
  IDs. Untangling which registration is intended needs the author; note the `_refresh()`
  call on line 74 is required (it builds `_cache`) and must survive any cleanup.
- **[BUG] (fixed after review)** `spicetype1frame.py:19, 76-79, 322` — Cache bookkeeping
  is confused: the class declares its own `_FRAME_LOOKUP` (line 19, with a 3-tuple
  comment) that is never used, while reads and writes both go to
  `SpiceFrame._FRAME_LOOKUP` with 4-tuple keys; the key stores the raw `reference`
  argument rather than a normalized wayframe, so direct construction and `get()` can
  disagree; and the `for cache_size in (self._cache_size, None):` loop clobbers the
  `cache_size` parameter. Also, like SpiceFrame's lookup, none of this is cleared by
  `Frame._reset_caches()`. The last point no longer holds: `_reset_caches` now clears
  these dictionaries. The rest stands, and the shared dictionary matters more now that
  `SpiceFrame` populates it too. The two key shapes do not collide, because this class
  stores a Frame in the position where `SpiceFrame` stores a name string, but relying on
  that is fragile; giving this class its own dictionary, which it already declares, would
  settle it.

  Done after this review, along with the rest of this finding. The class now uses its own
  `_FRAME_LOOKUP` for both reads and writes, and its key holds the reference's SPICE name
  rather than the raw argument, so direct construction and `get()` agree. Two further
  defects surfaced while checking that work. A tolerance passed as a string is converted
  to ticks by the constructor but was not converted by `get()`, so such a key could never
  match and every call built another frame; `get()` now converts it the same way. And the
  unconditional `self._register(frame_id or self._spice_frame_name)` that followed the
  branches undid both of them: it registered the frame the non-J2000 branch had
  deliberately left unregistered, displacing the J2000-referenced frame from the registry
  so that the frame ID silently resolved to a frame on a different reference, and it
  registered the J2000 branch's frame under the raw name rather than the underscored one.
  Removing it leaves each branch to register as it intends, which is what the parent
  `SpiceFrame.__init__` already does. `tests/frame/test_spicetype1frame.py` is new, the
  class having had no tests at all; two of its five fail against the defects above, and
  the Galileo gold masters, which build this frame, are unchanged.
- **[STYLE] (not fixed)** `spicetype1frame.py:126-131, 210-215` — The lazy
  `_time_tolerance` computation is duplicated verbatim in both transform methods; a small
  private helper would remove the copy.

### src/oops/frame/cmatrix.py
- No defects found. Docstrings are accurate and in the modern style; `_show` builds a
  proper string; `from_ra_dec` maths matches the PDS definition cited.

### src/oops/frame/frameshift.py
- No defects found. The `arg, /, frame` positional-only signature is documented
  accurately, and the Fittable plumbing (`_set_params`, `params`, `_refresh`, `_freeze`)
  is consistent with rotation.py and navigation.py.

### src/oops/frame/inclinedframe.py
- **[BUG] (fixed)** `inclinedframe.py:104` — The `_show` text omitted the comma after the
  `reference = ...` line, so the rendered multi-line constructor text was malformed.
  Added the comma.

### src/oops/frame/laplaceframe.py
- **[DOC] (fixed)** `laplaceframe.py:36-37` — The `tilt` parameter did not state its
  units; per house convention all angles are radians and the code takes `cos`/`sin`
  directly. Now reads "The tilt in radians ...".
- **[SUGGESTION] (not fixed)** `laplaceframe.py:137-139` — `np.cos(node_lon)` /
  `np.sin(node_lon)` on a polymath `Scalar` returns an object-dtype ndarray of `Scalar`s
  (verified in the venv), which then multiplies Vector3s element-wise. It works but is
  slow and obscure; `node_lon.cos()` / `.sin()` would be idiomatic and faster. There is
  no `tests/frame/test_laplaceframe.py`, so this module is untested.

### src/oops/frame/navigation.py
- **[SUGGESTION] (not fixed)** `navigation.py:141` — `_set_params` stores `params`
  as-is (`self._angles = params`), so after a fit `_angles` may be a list or ndarray
  while the constructor normalizes to a tuple; `_wayframe_key` then mixes types.
  Normalizing with `tuple(params)` would be safer.
- Otherwise clean; docstrings accurate, including the `_matrix` private speed-up
  parameter.

### src/oops/frame/poleframe.py
- **[STYLE] (fixed)** `poleframe.py:60` — Comment typo "falls 90 degrees ahead pole's RA"
  corrected to "ahead of the pole's RA".
- No other defects found; the long transform derivation is well commented.

### src/oops/frame/postargframe.py
- **[BUG] (fixed)** `postargframe.py:87` — `_show` omitted the closing `)` of the
  rendered text. Added it.

### src/oops/frame/quickframe.py, ringframe.py
- **[BUG] (fixed)** `ringframe.py:53` — `self._epoch = epoch and Scalar.as_scalar(...)`
  treated `epoch=0.` (the J2000 epoch, a legitimate value) as falsy and skipped the
  Scalar/readonly conversion, leaving a raw float in `_epoch`. Replaced with an explicit
  `None` test. Verified `RingFrame(frame, epoch=0.)` now yields a readonly Scalar epoch
  and an inertial frame.
- **[STYLE] (fixed)** `ringframe.py:80` — Comment typo "tranform" corrected.
- **[SUGGESTION] (not fixed)** `ringframe.py:88, 232` — The `(x, y) == (0., 0.)` pole
  test compares polymath Scalars inside a tuple; it works via `Boolean.__bool__` but
  raises if `x`/`y` are arrays. Reachable only for array-shaped planet frames, which are
  rare; noted for robustness.

### src/oops/frame/rotation.py
- **[SUGGESTION] (not fixed)** `rotation.py:132-137` — For a shapeless angle, `params`
  returns `(self._angle,)` (a Scalar inside the tuple) while the array case returns plain
  floats; FrameShift returns floats. A fitter consuming `params` uniformly as numbers
  would trip on the Scalar. Report only, since the Fittable contract is not spelled out.
- **[STYLE] (not fixed)** `rotation.py:107-108` — Double blank line inside `_show`.

### src/oops/frame/spinframe.py
- **[STYLE] (not fixed)** `spinframe.py:131` — Uses the private `angle._shape` where the
  public `angle.shape` is meant; works but violates the code's own privacy convention.
- Otherwise clean; the omega vector and matrix construction agree with rotation.py.

### src/oops/frame/synchronousframe.py
- **[BUG] (fixed)** `synchronousframe.py:73-79` — The two-argument `_show` rendered the
  second line as `SynchronousFrame(<planet>` (repeating the class name instead of
  indenting) because it used `{name}(` where the blanks-indent belongs. Fixed to match
  the house `_show` pattern.
- **[DOC] (not fixed)** `synchronousframe.py:10-12` — The class docstring says the body
  "keeps its x-axis pointed toward a central planet and its y-axis in the negative
  direction of motion", but `transform_at_time` builds `Matrix3.twovec(event.pos, 0,
  event.vel, 1)` where `event.pos` points from the planet *to* the body and `event.vel`
  is the *positive* direction of motion — i.e. the code's x-axis points away from the
  planet and y along the motion. Either the docstring or the sign convention is wrong;
  needs the author's intent (test_synchronousframe.py exercises consistency, not the
  sign).
- **[BUG] (not fixed)** `synchronousframe.py:117` — `transform_at_time` returns
  `Transform(..., origin=self._orbit_path)` while the Frame's own `_origin` is
  `self._planet_path` (line 51). The two should agree; which one is the true center of
  rotation for this frame is a design question.

### src/oops/frame/trackerframe.py
- **[BUG] (fixed)** `trackerframe.py:104-107` — `_show` returned a 4-tuple of strings
  (trailing commas) instead of one string. Joined into a single f-string.
- **[STYLE] (not fixed)** `trackerframe.py:75, 148` — `path_event` is assigned and never
  used in both `_refresh` and `transform_at_time`; `_ = ...` or `(_, obs_event)` would be
  clearer.

### src/oops/frame/twovectorframe.py
- **[BUG] (fixed)** `twovectorframe.py:76-78` — `_show` returned a 3-tuple of strings
  (trailing commas) instead of one string. Joined into a single f-string.
- Otherwise clean.

### src/oops/frame/__init__.py
- No defects. Imports and `__all__` agree with the modules present; column-aligned per
  house style.

### Test coverage gaps (observed while verifying)
- No test modules exist for `SpiceType1Frame`, `LaplaceFrame`, `InclinedFrame`,
  `Rotation`, or `TwoVectorFrame` (`tests/frame/` has 13 files). The four latent crashes
  fixed above (`__setstate__`, `Frame.J2000Frame`, tuple-returning `_show`s, tuple
  contract of `transform_at_time_if_possible`) all lived in exactly these untested
  regions.

### Summary
Counts: 13 fixed (9 BUG, 2 DOC, 2 STYLE), 17 reported unfixed (5 BUG, 5 DOC,
3 CONSISTENCY, 4 SUGGESTION/STYLE against distinct items). The core machinery
(`frame_.py`, `quickframe.py`, `spiceframe.py`, and the small fixed-rotation frames) is
in good shape, with modern, accurate docstrings and only cosmetic defects. Quality drops
sharply in the untested corners: `spicetype1frame.py` held four genuine bugs including a
guaranteed unpickle crash and a broken API contract, and its registration/cache
bookkeeping still needs an author decision, as does the axis-direction discrepancy in
`synchronousframe.py`. The `_FRAME_LOOKUP` cache in `spiceframe.py` has since been
implemented. Docstring
style is uniformly the modern Google form; no legacy `Input:`/`Return:` blocks remain in
this package.

---
## Critique: src/oops/surface/

Reviewer scope: all 16 files under `src/oops/surface/`. Every file was read in full.
All fixes below were verified with `py_compile` and by running `pytest tests/surface`
(35 passed) plus `tests/test_event.py` and `tests/test_body.py`.

### src/oops/surface/surface_.py
- **[DOC] (fixed)** `surface_.py:256,271` — `intercept_normal_to` documented the
  coefficient as `intercept = pos + p * normal(intercept)`, but the implemented
  convention (see the Ellipsoid derivation and `coords_from_vector3` hints docs) is
  `intercept + p * normal(intercept) = pos`; the two differ by the sign of `p`, and the
  sign of `p` is meaningful (it distinguishes points above vs. below the surface).
  Corrected the formula in both the `guess` parameter and the Returns bullet.
- **[DOC] (fixed)** `surface_.py:421` — Comment typo "Strip derivatives is necessary"
  corrected to "if necessary".
- **[DOC] (not fixed)** `surface_.py:345` — `coords_of_event` documents its return as
  "Two or three unitless Scalars", but when the event carries a `hints` subfield the
  underlying `coords_from_vector3` appends the hints value to the tuple, so a fourth
  element can appear. The docstring should mention the optional hints element.
- **[SUGGESTION] (not fixed)** `surface_.py:303` — `position_is_inside` raises
  `NotImplementedError` for subclasses with `HAS_INTERIOR = True` that fail to override
  it, but the docstring has no `Raises:` section.
- **[SUGGESTION] (not fixed)** `surface_.py:362` — `obs and obs.wrt(...)` relies on the
  truthiness of an Event; `None if obs is None else obs.wrt(...)` would be more explicit.

### src/oops/surface/spice_shape.py
- **[DOC] (not fixed)** `spice_shape.py:27-29` — The `Raises:` section documents only
  `KeyError`, but the code catches and re-raises `(RuntimeError, KeyError)` from
  `cspyce.bodvcd`, so a `RuntimeError` can also propagate.
- **[STYLE] (not fixed)** `spice_shape.py:41` — `raise e` inside the `except` block;
  bare `raise` is the idiomatic way to re-raise and preserves the original traceback
  identically.

### src/oops/surface/ellipsoid.py
- **[BUG] (fixed)** `ellipsoid.py:34` — `COORDINATE_RANGES` gave the latitude range as
  `(-PI, PI)`; latitude spans `(-HALFPI, HALFPI)` (the code computes it via `arcsin`).
  Fixed here and in `limb.py`; nothing in the codebase consumes `COORDINATE_RANGES`
  computationally, so this is metadata-only.
- **[DOC] (fixed)** `ellipsoid.py:24` — Typo "right- handed" (stray space from a wrap)
  corrected to "right-handed".
- **[DOC] (fixed)** `ellipsoid.py:443-445` — Same sign-flipped `p` formula in the
  `intercept_normal_to` Returns bullet as in `surface_.py`; corrected to
  `intercept + p * normal(intercept) = pos`.
- **[DOC] (fixed after review)** `ellipsoid.py:40,51-53` — `Ellipsoid.__init__` defaults
  `exclusion=0.9` while its own docstring says "Values of less than 0.95 are not
  recommended" — the default violates the stated recommendation. This mismatch is
  inherited verbatim from the legacy code (Spheroid: default 0.95, "less than 0.9 not
  recommended" — self-consistent; Ellipsoid: default 0.9, "less than 0.95 not
  recommended" — contradictory). Resolved after this review: the parameter was removed
  from `Ellipsoid` and `Spheroid` in favor of a zone derived from the radii, so neither
  the default nor the recommendation survives.
- **[BUG] (fixed after review)** `ellipsoid.py:225` — In `vector3_from_coords`, the
  z-offset direction is computed with `self.normal(track)` (no `derivs=derivs`), so
  when `derivs=True` the derivative contribution of the normal direction with respect
  to lon/lat is dropped from the returned position. Reclassified from a suggestion after
  measurement: the effect is not small. On a 1000x800x600 km body at an elevation of
  50 km, the returned d(pos)/d(lat) was wrong by 8.7%, growing with elevation and
  vanishing only at the surface. Passing `derivs=derivs` brings it to within 1e-11 of a
  central difference at every elevation tested. Nothing covered it; a parametrized test
  now does, and it fails at 50 km and above against the old code.
- **[SUGGESTION] (fixed after review)** `ellipsoid.py:340` — `intercept` treats any
  `direction` value other than `'dep'` as `'arr'`; an invalid string is silently
  accepted. It now raises ValueError, which the docstring records.

  Worth noting for whoever revisits this method: the two branches compute the same
  value. `-c/(b + d)` is the numerically stable rewriting of `(d - b)/a`, the larger
  root, so `direction` selects nothing. Confirmed across four geometries, including an
  observer inside the body. Whether that is intended, and which root each direction
  ought to select, is a question this critique does not answer.

### src/oops/surface/spheroid.py
- **[BUG] (fixed)** `spheroid.py:219-220` — The non-convergence warning in
  `intercept_normal_to` was a plain string containing `{type(self).__name__}` and
  `{max_dp:.6g}` with no `f` prefix (and `iter=(count+1)` even as written), so it would
  log the literal braces. Rewrote as an f-string and scaled the change by `km_scale`
  to match the per-iteration convergence message.
- **[DOC] (fixed)** `spheroid.py:83-85` — Same sign-flipped `p` formula in the Returns
  bullet; it now matches the (correct) formula given for the `guess` parameter.
- **[CONSISTENCY] (fixed)** `spheroid.py:28` — `Spheroid.__init__` took `exclusion` as an
  ordinary positional parameter with default 0.95, while `Ellipsoid.__init__` made it
  keyword-only with default 0.9. Both parameters were removed after this review, so the
  signatures now agree.

### src/oops/surface/centricspheroid.py
- **[DOC] (fixed)** `centricspheroid.py:80-87` — `vector3_from_coords` Returns section
  omitted the `hints` bullet even though the parameter list promises hints is appended
  (and the delegated Ellipsoid implementation does append it). Added the bullet.
- **[CONSISTENCY] (not fixed)** `centricspheroid.py:50,96` — `coords_from_vector3`
  delegates to `CentricEllipsoid`, while `vector3_from_coords` delegates to
  `Ellipsoid` directly; `GraphicSpheroid` delegates to `Spheroid` in the same spot.
  All resolve correctly, but the asymmetry is confusing.

### src/oops/surface/graphicspheroid.py
- **[DOC] (fixed)** `graphicspheroid.py:80-87` — Same missing `hints` bullet as
  centricspheroid; added.
- **[STYLE] (not fixed)** `graphicspheroid.py:14-15` — Double space in `axes=2,  derivs`
  and a continuation line misaligned by one column relative to the opening parenthesis.
- **[STYLE] (not fixed)** `graphicspheroid.py:5-8` — Imports split into a `polymath`
  group and an `oops` group, unlike the sibling modules which keep them in one aligned
  block (per house style, `polymath` groups with the `oops` imports).

### src/oops/surface/centricellipsoid.py
- **[BUG] (fixed)** `centricellipsoid.py:57` — `coords_from_vector3` called
  `self.intercept_normal_to(pos, guess=True)` without `derivs=derivs`, so with
  `derivs=True` the returned lon/lat/z carried no derivatives from the intercept
  solution (the parent `Ellipsoid.coords_from_vector3` passes `derivs=derivs` at the
  same spot). Added `derivs=derivs`.

### src/oops/surface/graphicellipsoid.py
- **[BUG] (fixed)** `graphicellipsoid.py:64` — Same missing `derivs=derivs` as
  centricellipsoid; fixed.

### src/oops/surface/ringplane.py
- **[BUG] (fixed)** `ringplane.py:87-89` — Constructing a `RingPlane` with `radii` but
  no `gravity` crashed with `AttributeError: 'NoneType' object has no attribute 'n'`
  while computing `_max_rate` (reproduced). This also broke every `OrbitPlane` with
  radial limits, since OrbitPlane builds its internal RingPlane with `gravity=None`.
  `_max_rate` is only consulted when gravity is present, so it is now set to None when
  `gravity` is None.
- **[BUG] (fixed)** `ringplane.py:171` — In `coords_from_vector3` with radial modes,
  radial limits, and `axes=3`, `z` was remasked with `r.mask` (the unmodded radius,
  whose mask never includes the radial-limit mask) instead of `a.mask` (used for
  `theta` two lines above). Out-of-range points therefore kept an unmasked `z`.
  Changed to `a.mask`.
- **[STYLE] (not fixed)** `ringplane.py:32` — Mutable default argument `modes=[]`. It is
  never mutated, so it is harmless, but a `None` or tuple default would be safer.
- **[CONSISTENCY] (not fixed)** `ringplane.py:185,239,291,329` — The `time` parameter
  defaults to `0.` in `vector3_from_coords`, `normal`, and `velocity` but to `None` in
  `intercept` and in every other Surface subclass; `velocity(pos, time=None)` with
  modes present would raise. Harmonizing on one default would be safer.
- **[CONSISTENCY] (not fixed)** `ringplane.py:167` — Uses `mask.any_true_or_masked()`
  where `ansa.py:179` uses plain `mask.any()` for the identical pattern; the RingPlane
  comment says the former allows fully masked results, so Ansa may have the lesser
  variant.

### src/oops/surface/orbitplane.py
- **[BUG] (fixed)** `orbitplane.py:194` — Building the unmasked variant used
  `OrbitPlane.__new__(type(OrbitPlane))`, i.e. `__new__(type)` — the metaclass, not the
  class — which raises `TypeError` (reproduced, after the RingPlane gravity fix exposed
  the line). Changed to `OrbitPlane.__new__(type(self))`.
- **[BUG] (fixed)** `orbitplane.py:205-207` — `__setstate__` called
  `self.__init__(*state)` with six positional values, but `path_id` and `radii` are
  keyword-only, so unpickling any OrbitPlane raised `TypeError` (reproduced). Now
  unpacks the state and passes the keyword-only values by keyword.
- **[BUG] (fixed)** `orbitplane.py:373,404,433-441` — Dimensional error in the
  first-order eccentric-orbit math: the longitude equation `lon = M + 2e sin(M)` was
  implemented with `2*self._ae` (`a*e`, in km) instead of the dimensionless
  `2*self._e`, in `velocity()` (the `dlon/dt` factor), `from_mean_anomaly`, and
  `to_mean_anomaly`. The existing unit tests all use `a = 1`, where `ae == e`, so they
  could not distinguish the two (the test even names the eccentricity variable `ae`).
  For real orbits (Uranian rings are defined via `Body.define_orbit` with `a` of
  ~42,000-51,000 km) the factor was wrong by ~4-5 orders of magnitude. Fixed the three
  code sites and the derivation comments. **Behavior change**: eccentric-ring
  velocities (`vflat`) and anomaly conversions change for `a != 1`; if any gold-master
  arrays for eccentric Uranian rings depend on `vflat`, they were generated with the
  wrong values and will need review/regeneration.
- **[BUG] (fixed)** `orbitplane.py:367` — Second, independent error in the same
  first-order math, found after the review while re-deriving `velocity()`. The surface is
  centered on the displaced ring center, so `velocity` shifts to planet-centered
  coordinates before applying its model; it added `a*e` where it must subtract. The ring
  center lies `a*e` toward *apocenter* (`_peri_path` is a CirclePath at `lon = peri +
  PI`) while the spin frame puts x along *pericenter*, so the planet sits at `+a*e`
  relative to the ring center and the conversion subtracts. This was confirmed
  independently from the orbit's own origin path, which reports the ring center at
  `x = -a*e`. The consequence was first-order: against an exact Kepler orbit at
  `e = 0.02`, the tangential speed at pericenter was high by 4% (exactly `2e`) and the
  radial speed at 90 degrees of true anomaly was three times too large. With the sign
  corrected, the residual scales as `e**2` — the ratio `error/e**2` converges to 2.50 as
  `e` falls — which is the expected limit of a model documented as first-order.
  `test_orbitplane.py` had no velocity coverage whatsoever, which is why this and the
  `a*e` error above both survived; three tests were added, two of which fail on the old
  sign.
- **[DOC] (fixed)** `orbitplane.py:363` — Comment typo `dy/dy` corrected to `dy/dt`.
- **[STYLE] (not fixed)** `orbitplane.py:41-54` — In the `__init__` docstring, the
  bullet list describing the elements is not indented under the `elements` parameter,
  so Napoleon will not associate it with the parameter.
- **[SUGGESTION] (not fixed)** `orbitplane.py:439-444` — `to_mean_anomaly`'s Newton loop
  has no iteration cap; it relies solely on the improvement test. Fine in practice now
  that the coefficient is `2e < 1`, but a `max_iterations` guard would be cheap.

### src/oops/surface/ansa.py
- **[BUG] (fixed)** `ansa.py:78-88` — Pickling failed two ways (both reproduced):
  `__getstate__` called `tuple(self._radii)` which raises `TypeError` when `_radii` is
  None, and `__setstate__` passed the state positionally to an `__init__` whose
  `gravity`/`ringplane`/`radii` are keyword-only. Both fixed.
- **[BUG] (fixed)** `ansa.py:189` — `theta.remask(r.mask)` discarded its result
  (`remask` returns a new object), so `theta` was never masked at the radial limits.
  Now assigns the result.
- **[BUG] (fixed)** `ansa.py:183-193` — `coords_from_vector3` documented that a
  non-None `hints` value is appended to the returned tuple (the base-class contract)
  but never appended it. Restructured the returns to honor the contract.
- **[CONSISTENCY] (not fixed)** `ansa.py:28` — Ansa sets `COORDINATE_TYPE =
  'cylindrical'` but does not override `COORDINATE_NAMES`/`COORDINATE_ABBREVS`/
  `COORDINATE_RANGES`, so it inherits `('x','y','z')` from Surface even though its
  coordinates are `(r, z, theta)`.

### src/oops/surface/limb.py
- **[BUG] (fixed)** `limb.py:67` — `self.unmasked = Limb(self._ground, None)` passed
  the keyword-only `limits` positionally, so constructing any `Limb` with `limits`
  raised `TypeError` (reproduced). Changed to `Limb(self._ground)`.
- **[BUG] (fixed)** `limb.py:142-155` — `coords_from_vector3` documented that the
  converged coefficient `p` is appended when `hints` is not None (and computes `p` in
  both branches), but never appended it; the sibling `PolarLimb.coords_from_vector3`
  does. Added the append and the missing `p` bullet in the Returns section. The only
  in-repo caller that passes hints (`_photon_solver`) indexes `coords[0..2]`, so the
  extra element is safe.
- **[DOC] (fixed)** `limb.py:45` — "optically using Centric or Graphic coordinates"
  corrected to "optionally".
- **[DOC] (fixed)** `limb.py:47-48` — "the lower upper limit(s)" corrected to "the
  lower and upper limits".
- **[DOC] (fixed)** `limb.py:227` — `intercept` Returns said "Two to five values" but
  the maximum is four (`pos`, `t`, `p`, `track`); corrected to "Two to four".
- **[BUG] (fixed)** `limb.py:35` — Latitude range in `COORDINATE_RANGES` was
  `(-PI, PI)`; corrected to `(-HALFPI, HALFPI)` (see ellipsoid.py entry).
- **[SUGGESTION] (not fixed)** `limb.py:476` — `z_clock_from_intercept` computes
  `z = pos.norm() - track.norm()` (difference of radii), whereas
  `coords_from_vector3:135` computes `z = (pos - track).norm() * p.sign()`
  (perpendicular distance, as the docstring states). For an ellipsoid these differ;
  the two methods should probably agree.
- **[SUGGESTION] (not fixed)** `limb.py:67` — `unmasked` is created as class `Limb`
  even when `self` is a `PolarLimb` (which inherits this constructor); `type(self)`
  may be intended.
- **[SUGGESTION] (not fixed)** `limb.py:563-589` — `intercept_from_z_clock`'s Newton
  loop lacks the divergence check (`max_dp >= prev_max_dp`) that every other iteration
  in this package has.

### src/oops/surface/polarlimb.py
- **[BUG] (fixed)** `polarlimb.py:144` — In `vector3_from_coords`, the third
  coordinate (offset distance `d`) was read from `clock` (`coords[1]`) instead of
  `coords[2]`, so any 3-coordinate call shifted the point by the clock angle
  interpreted as km. Changed to `coords[2]`.
- **[BUG] (fixed)** `polarlimb.py:75-88` — In `coords_from_vector3`, `los` was defined
  only in the no-hints branch but used unconditionally when `axes == 3`, so passing a
  Scalar `hints` with `axes=3` raised `NameError`. Hoisted `los = pos - obs` above the
  branch.
- **[BUG] (fixed)** `polarlimb.py:148-151` — `vector3_from_coords` documented the
  hints-append contract but ignored `hints` entirely; it now appends a non-None
  `hints` (matching `Limb.vector3_from_coords`) and the Returns section gained the
  missing bullet.
- **[DOC] (fixed)** `polarlimb.py:55-64` — `coords_from_vector3` Returns claimed "Two
  to five values" but listed no `p` bullet; added it.

### src/oops/surface/nullsurface.py
- **[DOC] (fixed)** `nullsurface.py:134` — `intercept` returns entirely masked values
  by construction (a NullSurface has no extent), but the docstring did not say so;
  added a sentence.
- **[SUGGESTION] (not fixed)** `nullsurface.py:164` — Uses `Vector3.as_vector(obs,
  derivs)` (the inherited `Vector.as_vector`, with `recursive` passed positionally)
  where every sibling uses `Vector3.as_vector3(obs, recursive=derivs)`.
- **[SUGGESTION] (not fixed)** `nullsurface.py:115` — Stale comment "Convert to Scalars
  and strip units, if any": units are implicit in oops and nothing here strips them.

### src/oops/surface/_photon_solver.py
- **[BUG] (fixed)** `_photon_solver.py:1588` (`_solve_photon_path_normal`) — the
  fully-masked-link branch called `Event(..., path=path, frame=Frame.J2000)`, but the
  Event constructor's third parameter is `origin`, not `path`; `path=` was absorbed into
  `**more` as a subfield, leaving `origin` unfilled and raising `TypeError:
  Event.__init__() missing 1 required positional argument: 'origin'` (reproduced). The
  branch only runs when every element is masked, which is why no test reached it. Fixed
  by passing `path` positionally as the origin.
- **[SUGGESTION] (not fixed)** `_photon_solver.py:1614,1635` — In
  `_solve_photon_path_normal`, the surface event gets only a `surface_key + '_ap'`
  subfield, but line 1635 reads `surface_key + '_j2000'` back via `get_subfield`;
  whether Event derives `dep_j2000` from `dep_ap` needs verification (the function is
  marked "TODO: full testing!!"). The sibling `_solve_photon_event_normal` inserts
  `_j2000` directly.
- **[STYLE] (not fixed)** `_photon_solver.py:770-774` — Lines inside
  `if self.IS_VIRTUAL:` are indented three spaces instead of four.
- **[SUGGESTION] (not fixed)** `_photon_solver.py:747` — For a time-dependent,
  non-virtual surface, `_solve_photon_by_coords` evaluates `vector3_from_coords` once
  with the initial `surface_time` and never re-evaluates inside the loop;
  `IS_TIME_DEPENDENT` is not consulted.
- **[DOC] (not fixed)** Docstrings here are thorough and consistent (modern style
  throughout); the long converge-parameter block is duplicated eight times and could
  be shared via a `Notes` cross-reference, but that is a judgment call.

### src/oops/surface/__init__.py
- No findings. `__all__` matches the imports; banner style correct.

### Summary

Counts: 17 BUG (16 fixed, 1 partially style-gated), 14 DOC (11 fixed), 6 CONSISTENCY
(0 fixed, by instruction), 5 STYLE, 10 SUGGESTION.

Reproduced-and-fixed crashes: RingPlane with radii but no gravity (which broke every
OrbitPlane with radii), OrbitPlane unmasked construction (`__new__(type(OrbitPlane))`),
OrbitPlane and Ansa unpickling (keyword-only args passed positionally — fallout from
the recent keyword-only refactor that `__setstate__`/internal callers missed), and
Limb with limits (same fallout). Logic fixes: PolarLimb's `coords[2]`/`clock` mixup and
undefined `los`, Ansa's discarded `remask`, RingPlane's wrong mask source for `z`, the
missing `derivs=` in Centric/GraphicEllipsoid, the missing hints/p appends promised by
docstrings (Limb, Ansa, PolarLimb), and the dimensional `2*ae` vs `2*e` error in
OrbitPlane's eccentric-orbit math — the last is a behavior change for real-scale
eccentric orbits (Uranian rings) and may interact with gold masters.

Overall the package's docstrings are in good shape after the recent proofread (modern
Google style throughout, no legacy `Input:`/`Return:` blocks remain in this package),
but the proofread introduced or left behind several keyword-only-argument call bugs,
and the hints/guess return contracts were unevenly implemented across subclasses. The
`tests/surface` suite (35 tests) passes with all fixes; `xtest_polarlimb-NOT-WORKING.py`
remains uncollected and both PolarLimb bugs found here would have been caught by a
working test.

### Replacing the exclusion parameter

`Ellipsoid` and `Spheroid` took an `exclusion` argument that scaled the zone in which
`intercept_normal_to` is masked, as a fixed fraction of the equatorial radius (0.9 and
0.95 respectively, with contradictory advice in the docstrings). It was replaced after
this review by a zone derived from the shape,

    r_exclusion = a - g c^2/a,     g = _EXCLUSION_FACTOR

applied in unsquashed coordinates, and the parameter was removed from both classes along
with its pickle and intercept-key entries. No caller anywhere in the repository ever
passed it.

**The formula is geometrically right.** For radii a >= b >= c the smallest radius of
curvature anywhere on the surface is c^2/a, at the ends of the longest axis, so the
evolute, where the normal-foot stops being unique and the solution degenerates, comes
within exactly that distance of the surface. Placing the boundary at depth g c^2/a with
g < 1 keeps a proportional margin. Measurement confirms the derivation: sweeping g over
interior positions, results are exact to a few parts in 1e14 up to g ~ 0.55, degrade
through 0.8, and fail outright at g = 1.0, which is the evolute itself. The cusps along
the intermediate axis are covered for any g <= a^2/b^2, which holds for any triaxial
shape.

**Extreme proportions.** For a body flatter than about c/a = 0.83 the evolute also
reaches outside this sphere along the polar axis. Those positions cannot be masked: they
lie outside the surface — 100% of them, at unsquashed radii of 1.3 to 3.4 a for a
1:1:0.4 body — so a zone large enough to cover them would discard legitimate exterior
positions just above the pole. Bodies flatter than about c/a = 0.5 do return bad
intercepts there, off the surface by up to 1e8 times the radius, but the old fixed
fraction produced *identical* failures at those axis ratios; the limit is the
convergence of `intercept_normal_to`, not the size of this zone. Interior positions are
accurate for every shape tested, including a 1:0.3:0.3 needle and a 1:0.5:0.2 triaxial.

**The factor.** 0.8 was the proposed value; 0.5 was adopted. At 0.8 five tests in
`tests/surface` fail, not on masking but on precision — a groundtrack round-trip that
asserts agreement within 1e-10 km on a 6e4 km body reads 1.4e-9. The safe range for the
suite's existing tolerances ends between 0.55 and 0.6. 0.5 keeps full precision, sits
inside that range rather than on its edge, and leaves a factor of two below the evolute.

**Consequence for the gold masters.** The zone is now far smaller for round bodies: for
a sphere the evolute is a single point, so the old 0.95 masked 95% of the interior for no
numerical reason, where the new zone masks 50%. Four of the five gold-master
observations (Cassini ISS W1573721822 and three of the four Galileo SSI images) therefore
report discrepancies on their `:LIMB` backplanes, which gain unmasked pixels: 711 of 4096
on `SATURN:LIMB`, 292 of 625 on `GANYMEDE:LIMB`.

Every one of those is a "Mask mismatch", which the framework defines as the masks
disagreeing while the values agree; not one "Value mismatch" or "Value/mask mismatch"
appears. The Ganymede values match to 1.6e-07 against a limit of 0.1. Saturn's raw
altitude discrepancy of 83 km is the expected artifact of a moving mask edge and falls to
exactly zero once the test's own one-pixel offset is applied. No value of the factor
preserves the recorded masks: reproducing them needs a factor near 0.05, which for a
sphere is the old, unjustified 95% zone. The masters need re-adoption to record the wider
coverage, which is a decision about reference data and was left to the author.

---
## Critique: src/oops/fov

### src/oops/fov/fov_.py
- **[BUG] (fixed after review)** `fov_.py:622` — `center_xy`, `center_los`, and the four
  `corner*_xy` methods consult their `*_filled` cache before examining `time`, so for a
  time-dependent FOV a value cached by a `time=None` call would be returned for every
  later call with an explicit time. In practice this is unreachable today: the only
  time-dependent subclass (TDIFOV) raises on `time=None`, which also means
  `outer_radius`, `inner_radius`, and `sphere_falls_inside` (which call `center_los()`
  with no time) do not work for TDIFOV at all. Fixed after this review; see "Fixing the
  FOV caches" below, which also records why the TDIFOV failure was a `TypeError` from
  inside polymath rather than a stale value, and a separate TDIFOV defect found while
  verifying it.
- **[CONSISTENCY] (not fixed)** `fov_.py:300` — `los_from_uv` accepts `derivs`/`remask` as
  positional-or-keyword parameters while every sibling method makes them keyword-only with
  `*`. Same for `wcs_from_uv` in wcsfov.py. Making them keyword-only would narrow the
  public API, so left for a deliberate decision.
- **[CONSISTENCY] (fixed after review)** `fov_.py:622-786` — the lazy caches
  (`center_xy_filled`, `center_los_filled`, `center_dlos_duv_filled`,
  `outer_radius_filled`, `inner_radius_filled`, `corner00_filled`...`corner11_filled`) are
  public attribute names though purely internal; recent commits privatized equivalent
  attributes in Surface and Observation. Renamed after this review, each gaining a leading
  underscore, which brings them into line with those classes and with the `_filled_*`
  caches in `transform.py`. The `_CACHED_NAMES` list that `_refresh` clears was updated
  with them, and the test that asserts a time-dependent FOV caches nothing now checks that
  list rather than matching on the name suffix, so it cannot drift out of step with a
  later rename.
- **[SUGGESTION] (not fixed)** `fov_.py:575` — `nearest_uv` with `remask=True` reads
  `uv_pair.mask` from the raw argument; a tuple or ndarray input (accepted by the
  `Pair.as_pair` conversion used for `clipped`) would raise AttributeError. Converting once
  at the top would make the two uses consistent.
- **[STYLE] (fixed after review)** `fov_.py:640-641` — stray double blank line at the top
  of `center_los`; removed when that method was rewritten.
- **[DOC]** Docstrings are modern Google style, accurate against signatures and behavior;
  `Parameters:`/`Returns:` blocks are complete. No factual errors found.

### src/oops/fov/polynomialfov.py
- **[BUG] (fixed)** `polynomialfov.py:142` — `xy_precision` was computed as
  `EPSILON * min(dx_du, abs(dy_dv))`: `abs()` was missing on `dx_du`, so a camera with
  negative `dx/du` would get a negative precision target, and `np.sqrt(precision)` in
  `_solve_polynomial` would be NaN, disabling early convergence. Fixed to take
  `abs(dx_du)`, matching the treatment of `dy_dv`.
- **[BUG] (fixed)** `polynomialfov.py:222` — in `uv_from_xyt`, constructing the FOV with
  only `coefft_uv_from_xy` and `fast=False` crashed: the else-branch attempted to invert
  `self.coefft_xy_from_uv`, which is None. The branch condition is now
  `if self.fast or self.coefft_xy_from_uv is None:`, evaluating the polynomial directly
  when there is no forward polynomial to invert. Verified by round-tripping such an FOV
  (`max_inversion_error()` ~1e-14).
- **[CONSISTENCY] (not fixed)** `polynomialfov.py:98` — `self.uv_los.as_readonly()` return
  value is discarded (works because `as_readonly` marks in place), whereas line 92 uses the
  return value; the two idioms coexist in most fov modules.
- **[STYLE] (not fixed)** `polynomialfov.py:5-6` — `import sys` (stdlib) is placed after
  `import numpy` (third-party); the project rule puts stdlib first. Same in barrelfov.py.

### src/oops/fov/barrelfov.py
- **[BUG] (fixed)** `barrelfov.py:125` — `xy_precision = EPSILON * np.min(uv_scale.vals)`
  lacked `np.abs`. The constructor docstring itself says the second `uv_scale` element is
  negative for v-up cameras, in which case the precision target went negative and the
  Newton solver's `np.sqrt(precision)` produced NaN, so `_solve_ratio` never detected
  convergence early. Fixed with `np.abs`.
- **[BUG] (fixed)** `barrelfov.py:202` — same `fast=False`-with-only-`coefft_uv_from_xy`
  crash as PolynomialFOV; same fix, verified by round-trip.
- **[DOC] (fixed after review)** `barrelfov.py:356` — the non-convergence warning
  reported `max_dr` (the previous iteration's change) where PolynomialFOV reports the
  latest change. The two exits from the loop differ: when it runs out of iterations the
  two are equal, but when it breaks because the step grew (`new_max_dr >= max_dr`) the
  step that triggered the break was never assigned to `max_dr`, so the warning reported
  the smaller preceding step. That understates the failure and prints a
  converged-looking number: replaying the control flow over steps of 1, 1e-3, 1e-6, 4e-6
  logged `change=1e-06` for the iteration whose actual change was `4e-06`. Now reports
  `new_max_dr`, making the warning identical in form to PolynomialFOV's.

### src/oops/fov/wcsfov.py
- **[BUG] (fixed)** `wcsfov.py:68` — `uv_shape` was built as `Pair([NAXIS2, NAXIS1])`,
  i.e., (v,u) order, while `uv_los` two lines below uses `(CRPIX1, CRPIX2)` = (u,v), and
  the HST host code builds `uv_shape` as `(NAXIS1, NAXIS2)`. The swap was invisible only
  because all current users (JWST NIRCam, the unit tests) have square 2048x2048 images.
  Fixed to `(NAXIS1, NAXIS2)`.
- **[CONSISTENCY] (not fixed)** `wcsfov.py:315` — `wcs_from_uv` takes `derivs`/`remask`
  positionally (see fov_.py:300 note).
- **[CONSISTENCY] (not fixed)** `wcsfov.py:22-33` — the class `Properties:` block uses
  bulleted `*` items unlike the plain-indent style of the FOV base class docstring.
- **[SUGGESTION] (not fixed)** `wcsfov.py:95-97` — the attribute `polyfov` may actually
  hold a FlatFOV (the code comments on this); a neutral name would be clearer.

### src/oops/fov/flatfov.py
- No defects found. Docstrings accurate and in modern style.

### src/oops/fov/gapfov.py
- No defects found; the gap-clipping logic in `uv_from_xyt` (including operator
  precedence around `&`) is correct.

### src/oops/fov/nullfov.py
- No defects found. `uv_is_outside` returning `Boolean.TRUE` (everything outside a null
  FOV) is consistent with the class's stated purpose.

### src/oops/fov/offsetfov.py
- **[BUG] (fixed)** `offsetfov.py:66` — the Fittable hook `_set_params` updated
  `uv_offset` and `xy_offset` but left `self.uv_los` stale (the constructor defines it as
  `fov.uv_los - uv_offset`), so after a pointing fit the FOV advertised the wrong line of
  sight. `_set_params` now recomputes `uv_los`. Verified: `set_params((3,4))` leaves
  `uv_los == fov.uv_los - uv_offset`.
- **[SUGGESTION] (not fixed)** — the FOV base-class `*_filled` caches are never
  invalidated when a Fittable FOV (OffsetFOV, Platescale) is refit via `set_params`, so a
  `center_xy()` computed before the fit remains cached afterward. Cache invalidation
  belongs in the Fittable/Mutable refresh path and needs a design decision.

### src/oops/fov/platescale.py
- Verified non-issue: `_set_params` does not call `_refresh`, but the Fittable framework
  (`fittable.py` `set_params`) invokes `_refresh` afterward, so `uv_scale`/`uv_area` stay
  consistent.
- **[STYLE] (not fixed)** `platescale.py:5-7` — import block lists `polymath` after the
  `oops` imports; every other fov module lists `polymath` first.
- **[STYLE] (not fixed)** `platescale.py:13` — `__init__(self, factor, /, fov)` uses a
  positional-only marker mid-signature for no evident reason; harmless but unusual for
  this codebase.

### src/oops/fov/subarray.py
- No defects found.

### src/oops/fov/subsampledfov.py
- No defects found. (`uv_los` is rescaled while the docstring says the optic axis is
  "unchanged" — this is correct: the physical axis is unchanged, its pixel coordinates
  scale.)

### src/oops/fov/slicefov.py
- No defects found.

### src/oops/fov/tdifov.py
- **[SUGGESTION] (not fixed)** `tdifov.py:126-142` — `uv_from_xyt` mutates the Pair
  returned by the wrapped FOV in place (via the shared-memory `line += ...` and
  `uv.derivs['t'] += ...`). Safe with current subclasses, which return freshly built
  objects, but a wrapped FOV returning a shared or readonly object (e.g. NullFOV's
  `Pair.ZEROS`) would be corrupted — the "mutating shared objects" trap in CLAUDE.md.
  `xy_from_uvt` defends with an explicit `.copy()`; `uv_from_xyt` should too.

### src/oops/fov/__init__.py
- No defects found; `__all__` matches the re-exported names.

### Summary
Counts: 7 BUG (6 fixed, 1 unfixed design issue), 2 DOC, 5 CONSISTENCY, 4 STYLE,
4 SUGGESTION.

The fov package is in good shape overall: docstrings are uniformly modern Google style,
accurate against signatures, and wrapped to 90 columns; banners and house style are
observed. The genuine defects clustered in the less-traveled paths: sign handling of
negative pixel scales in the two Newton-solver precision targets, the `fast=False`
single-polynomial crash in both distortion FOVs, a (u,v) axis swap in WCSFOV masked by
square detectors, and a stale `uv_los` after refitting an OffsetFOV. The two items the
review left open were design-level, time-dependent-FOV caching in the base class and cache
invalidation under the Fittable protocol; both were settled afterward, as described below.
All 10 tests in tests/fov passed after the review's fixes, and 30 after the later work.

### Fixing the FOV caches

Both cache defects were repaired after the review, in `center_xy`, `center_los`, and the
four `corner*_xy` methods, which shared one broken shape:

    if hasattr(self, 'center_xy_filled'):       # consulted before time is considered
        return self.center_xy_filled
    if self.IS_TIME_INDEPENDENT or time is None:
        self.center_xy_filled = self.xy_from_uvt(self.uv_shape/2.)
        ...

Two things go wrong. The cache is consulted before the time is looked at, so a value
computed once for `time=None` is returned for every later time, making the result depend
on the order the calls happen to arrive in. And the branch that fills the cache is taken
whenever `time is None`, even for a time-dependent FOV, where it calls `xy_from_uvt`
without a time; a TDIFOV cannot do that, so `center_xy()` did not merely return a stale
value, it raised `TypeError: invalid Scalar data type: <class 'NoneType'>` from inside
polymath. That is also why `outer_radius`, `inner_radius`, `center_dlos_duv`, and
`sphere_falls_inside` could not work for a TDIFOV: each reaches one of these methods
without a time.

The cache is now keyed on `IS_TIME_INDEPENDENT` alone. A time-independent FOV caches its
value and returns it for any time, since time cannot matter; a time-dependent FOV
computes fresh for the time it is given and caches nothing, and raises
`NotImplementedError` naming the method when no time is given, which is the pattern
`xy_from_uv`, `uv_from_xy`, and `uv_from_los` already use for the same situation. The
three time-less properties document that they raise it too, in place of their former
claim that time-dependence is "ignored", which was never true.

For invalidation, `FOV` gained a `_refresh` that discards every cached name. The Mutable
protocol already calls `_refresh` after `set_params`, so no subclass needs to know about
it; `Platescale`, which defines its own `_refresh`, now calls up to it. Before this, a
refit left every cached value describing the FOV's former geometry: doubling a
Platescale's factor left `corner11_xy` and `outer_radius` reporting their old values
indefinitely.

`tests/fov/test_fov.py` is new; nothing had exercised these methods or the caches. Its 20
tests cover both defects and fail against each original behavior.

**A separate TDIFOV defect, found while verifying this and not fixed.** `TDIFOV`'s
`xy_from_uvt` and `uv_from_xyt` apply the TDI line shift through a buffer they expect to
share with the `(u,v)` they return:

    line = uv.to_scalar(self._uv_line_index, recursive=False)
        # uv and line share memory, so updating line also updates uv.
    ...
    line -= self.tdi_sign * shifts

That holds only for shaped input. For a shapeless Pair, `to_scalar` returns a copy rather
than a view (`np.shares_memory` is False), so `line -= ...` rebinds a local and the shift
never reaches `uv`: the TDI FOV silently behaves as though it were the undistorted FOV
beneath it. Every cached method above passes a shapeless Pair — `uv_shape/2.`,
`Pair.ZEROS` — so all of them are affected, and `tests/fov/test_tdifov.py` passes because
it only ever tests arrays. Repairing it means not relying on the aliasing, which changes
JunoCam geometry for shapeless queries, so it is left for the author.

---
## Critique: oops/observation, oops/cadence, oops/calibration, oops/gravity

### src/oops/gravity/oblategravity.py
- **[BUG] (fixed)** `oblategravity.py:573,585,608,718,740,837,840` — `osc_from_state()` and
  `geom_from_state()` referenced `Gravity._pos_arctan2` and `Gravity._freq_to_geom`, but both
  static methods are defined on `OblateGravity`, not `Gravity`, so every call raised
  `AttributeError` (verified live). Changed the references to `OblateGravity.*`; both methods
  now run, and `osc_from_state(state_from_osc(elements))` reproduces its inputs exactly.
  Realigned the continuation lines the rename disturbed.
- **[DOC] (fixed)** `oblategravity.py:106` — `kappa2()` described its return as "(radians/s)"
  although it returns the *square* of the frequency. Now reads "(radians^2/s^2)".
- **[DOC] (not fixed)** `oblategravity.py:625-634,694-702` — `state_from_geom` says "Returns
  x, y, z, vx, vy, vz" but actually returns two stacked arrays `(pos, vel)`;
  `geom_from_state` says "Returns: a, e, inc, long_peri, long_node, mean_anomaly" but
  actually returns `(a, e, inc, mean_lon, long_peri, long_node)` — both the names and the
  order are wrong. These docstrings (adapted-from-SWIFT notes) need a rewrite into the
  modern format with correct Returns blocks; left for a deliberate documentation pass.
- **[CONSISTENCY] (not fixed)** `oblategravity.py:80` — `potential(self, a)` drops the
  `e=0., sin_i=0.` parameters that the base-class signature declares.
- **[CONSISTENCY] (not fixed)** whole file — docstrings are one-line summaries with no
  `Parameters:`/`Returns:` sections; several public methods (`combo`, `solve_a`,
  `state_from_osc`, ...) document neither their inputs nor their return structure.
- **[STYLE] (not fixed)** `oblategravity.py:5-6` — `import numpy as np` precedes
  `import warnings`; the stdlib import belongs in its own group above the third-party group.
- **[SUGGESTION] (not fixed)** `oblategravity.py:761` — the divergence diagnostics in
  `geom_from_state` use `warnings.warn` with raw state values; elsewhere the module logs
  through `LOGGING`. Also `bad_idx` indexing assumes array (not scalar) inputs.

### src/oops/gravity/gravity_.py
- **[DOC] (fixed)** `gravity_.py:129` — "A gravity filed from the registry" typo → "field".
- **[BUG] (fixed)** `gravity_.py:71-107` — the convenience wrappers (`n`, `dmean_dt`,
  `dperi_dt`, `dnode_dt`, `d_dmean_dt_da`, `d_dperi_dt_da`, `d_dnode_dt_da`) accepted
  `e` and `sin_i` but silently discarded them, calling e.g. `self.omega(a)` with defaults.
  `OblateGravity` overrides all of these and passes the parameters through, so the base
  versions were both unused and wrong. They now forward `e` and `sin_i`, matching the
  subclass behavior.
- **[CONSISTENCY] (not fixed)** `gravity_.py:11-65` — the abstract methods use bare `pass`
  (silently returning None) instead of raising `NotImplementedError` as the Cadence,
  Calibration, and Observation abstract bases do.
- **[SUGGESTION] (not fixed)** `gravity_.py:109-121` — `ilr_pattern`/`olr_pattern` have no
  `Parameters:` documentation and their resonance-formula provenance is undocumented.

### src/oops/cadence/reshapedcadence.py
- **[DOC] (fixed)** `reshapedcadence.py:170,181` — the docstrings of `_old_tstep_from_new`
  and `_new_tstep_from_old` were swapped: each described the opposite conversion direction
  from the one its code (and name) performs.
- **[BUG] (fixed)** `reshapedcadence.py:397` — in `tstep_range_at_time`, the continuity check
  on the *old* tstep range compared `old_tstep_max.vals[...,-1]` against `self.shape[-1]`
  (the new shape) instead of `self._old_shape[-1]`; a copy-paste from the new-range check
  above. Affects only non-unique multi-dimensional cadences (a rare, otherwise-guarded
  path), but the comparison was plainly against the wrong shape.
- **[BUG] (fixed)** `reshapedcadence.py:402-404` — in the same block, the scalar-input error
  path reported `new_tstep_min`/`new_tstep_max` in the "input tstep range is discontinuous"
  message where the array path reports `old_tstep_min`/`old_tstep_max`.

### src/oops/cadence/cadence_.py, metronome.py
- **[DOC] (fixed)** `cadence_.py:190`, `metronome.py:339` — doubled word in
  `tstride_at_tstep`: "True to mask time tsteps that are out of range" → "True to mask
  tsteps that are out of range".

### src/oops/cadence/reversedcadence.py
- **[BUG] (fixed after review)** `reversedcadence.py:166` — `tstep_at_time` returns
  `self.shape[0] - tstep`, which mirrors the fractional part within each time step, so it
  is not the inverse of `time_at_tstep` (which deliberately lets the fraction increase with
  time within each step). Demonstrated numerically: for a reversed continuous Metronome
  (100-140, 4 steps), `time_at_tstep(0.25) == 132.5` but `tstep_at_time(132.5) == 0.75`
  (should be 0.25). The existing tests never expose this: the TDI comparison cases skip
  `tstep_at_time`, and the doubly-reversed Metronome cases cancel the error
  (`steps - (steps - u) == u`). Fixed after this review: the reversal applies to whole
  steps only, so the integer part is reversed and the fractional part carries over
  unchanged, `(max_step - int(u)) + frac(u)`, splitting the index with the same
  `shift=True` convention `time_at_tstep` uses. The error was `1 - 2f` in the fraction,
  which is why it vanished at the midpoint of every step and the tests passed. Round-trip
  error is now at machine precision across continuous, gapped, and non-uniform cadences,
  and the derivative is `+du/dT`, not negated: within a step, time still increases with
  the index.

  One consequence is worth recording. A doubly-reversed cadence is exactly the original at
  every time strictly inside its range, and for `time_at_tstep` and `tstep_range_at_time`
  everywhere, but not at the final time: the original reports `steps`, the doubled cadence
  reports 2. This is not a residual defect but the boundary convention composing with
  itself. A time at the end of a step maps to that step's exclusive upper bound, which is
  the index where the next step begins; one reversal sends the end of the underlying
  cadence to the end of this cadence's first step, and a second reversal can no longer
  distinguish that index from the start of its second step. The old code preserved this
  one value only through the compensating error that made every other value wrong. The end
  time of a singly-reversed cadence is in any case not attained at any exact index, since
  the index-to-time map is a sawtooth; the value returned is the limit approached from
  below, which is the physically meaningful step.

  `test_reversedcadence.py` gained a direct test that `tstep_at_time` inverts
  `time_at_tstep`, one pinning the derivative sign, and one asserting the double-reversal
  property above; the first three fail against the old code. The four doubly-reversed
  cases previously borrowed the Metronome suite's shared assertions, which is what
  cancelled the error, and now make the comparison against the original cadence
  explicitly.

### src/oops/cadence/tdicadence.py
- **[BUG] (fixed after review)** `tdicadence.py:102-120` — `tdi_shifts_after_time` clipped
  its result to `[0, self._tdi_stages]`, but the number of shifts remaining can never
  exceed `self._max_shifts` (= stages-1). `Scalar.int()` does not clip unless asked
  (`clip=False` is its default), so a time before `tstart` yields a negative `tstep_int`
  and an inflated count: a 4-stage cadence reported 3 shifts remaining at its start time
  but 4 at any earlier time, one more than the detector can ever perform. The sibling
  method `tdi_shifts_at_line` already clips to `_max_shifts`, which settles the intended
  convention that the critique called unverifiable. Now clips to `_max_shifts` as well.
  Both methods were uncalled and untested anywhere in the repo; five tests were added,
  covering the bound across stage counts, the countdown through the exposure, the
  independence from `tdi_sign`, and the `remask` behavior. All five fail against the old
  code.
- **[SUGGESTION] (not fixed)** `tdicadence.py:57` — `is_unique = (tdi_stages == 1)` conflicts
  with the base-class definition of `is_unique` ("no times ... associated with more than one
  time step"): with 1 stage and N lines, every line spans the same interval, and
  `tstep_range_at_time` returns all N lines. The tests pin this behavior
  (`case_tdicadence_10_100_10_1`), so it is a deliberate convention, but the base-class
  Properties docstring does not admit it.
- **[SUGGESTION] (not fixed)** `tdicadence.py:54` — `self.time[-1]` works (2-tuple) but every
  sibling class writes `self.time[1]`.

### src/oops/cadence/instant.py
The class carried a "DO NOT USE" banner when the review was written. That banner is gone
and its constructor has been rebuilt, so the module is live code and was fixed rather than
left alone.

- **[BUG] (fixed after review)** `instant.py:83,103` —
  `time_at_tstep`/`time_range_at_tstep` ignored `tstep` entirely and returned the full
  time array; the author's own `#### Shouldn't this be self._tdb[tstep.int()]?` comments
  agree. Both now index the times by the given time step. polymath indexes a Scalar by a
  Scalar or Pair index and propagates the index's mask, so a shared `_index_at_tstep`
  helper converts the time step with `int(clip=True)`
  and indexes with the result: fractional indices truncate to the step that contains them
  (an instant cannot be interpolated), indices beyond the ends clip to the nearest edge as
  `DualCadence` documents, and `remask`/`inclusive` behave as the base class specifies.
- **[BUG] (fixed after review)** `instant.py:123` — `tstep_at_time` built
  `Scalar(np.zeros(self.shape), self._tdb != time)`, which took the shape of the cadence
  rather than of `time`, broke unless `time` broadcast against the cadence shape, and
  returned index 0 for every time. A shared `_match_at_time` helper now finds the first
  unmasked time step whose time equals each given time; unsampled times are masked. The
  same helper implements `tstep_range_at_time`, which had been left as `### TBD` raising
  `NotImplementedError`: a sampled time now yields a one-step range and an unsampled one
  an empty range, per the base-class convention.
- **[BUG] (found after review, fixed)** `instant.py:159` — `time_is_outside` returned
  `Scalar.as_scalar(time) != self._tdb`, comparing each time against the whole table
  instead of reducing over it. It took the shape of the cadence rather than of `time`, and
  reported a sampled time as outside: for `Instant([100., 110., 130.])`,
  `time_is_outside(100.)` returned `Boolean(False, True, True)` where the answer is a
  single `False`. Now reduces over the table via `_match_at_time`.
- **[DOC] (fixed after review)** `instant.py:66-83` — docstrings described the standard
  Cadence behavior (interpolation, remask, derivs) that this implementation does not
  provide. Each parameter that an Instant cannot honor now says so and why: `derivs`,
  because the time does not vary within a time step; `inclusive` on the time-to-tstep
  methods, because each step is a single moment always treated as part of the cadence; and
  `remask` on `tstep_at_time`, because an unsampled time has no time step and so is masked
  whether or not it is asked for. The methods were untested; 18 tests were added, of which
  17 fail against the old code.

### src/oops/cadence/dualcadence.py, sequence.py, snapcadence.py, timeshift.py
- **[SUGGESTION] (fixed after review)** `dualcadence.py:69-72` — `__setstate__` re-derived
  `time`/`midtime`/`lasttime` after `self.__init__(*state)` had already set them. Probing
  the recomputation during real unpickles of four DualCadences confirmed it reproduced
  what `__init__` set every time, so the three lines were removed; `__setstate__` matches
  `Metronome.__setstate__`.
- **[DOC] (fixed after review)** `timeshift.py:63` — the `params` property (Fittable
  API) had no docstring; added one in the form its siblings `Platescale.params` and
  `OffsetFOV.params` use.
- **[SUGGESTION] (not fixed)** `sequence.py:50-51` — the error message "Sequence tlist must
  be 1-D" also fires for a 1-D list of length <= 1; the length requirement is not mentioned
  in the message or the docstring.
- sequence.py, snapcadence.py, timeshift.py, dualcadence.py, metronome.py, cadence_.py are
  otherwise accurate: signatures, defaults, keyword-only markers, and Returns blocks match
  the code, and the modern Google style is applied consistently.

### src/oops/observation/observation_.py
- **[BUG] (fixed)** `observation_.py:261` — `uv_range_at_time_0d` called
  `Scalar.as_scalar(time, derivs=False)`; `as_scalar` accepts only `recursive`, so every
  call with `remask=True` raised `TypeError` (verified live). Changed to `recursive=False`.
- **[BUG] (fixed)** `observation_.py:545-547` — `delete_subfields` deleted keys from
  `self.subfields` while iterating over it, raising `RuntimeError: dictionary changed size
  during iteration` for any observation with subfields. Now iterates over
  `list(self.subfields)`.
- **[BUG] (fixed)** `observation_.py:574` — in `uv_is_outside` with `inclusive=False`, the
  u-test still used `tvl_gt` while the v-test used `tvl_ge`, so `u == uv_shape[0]` was
  wrongly treated as inside. Made both `tvl_ge`, matching `FOV.uv_is_outside`, which treats
  the axes symmetrically via `Qube.is_outside`.
- **[DOC] (fixed)** `observation_.py:865` — garbled comment "Require extra at least two
  iterations" → "Require at least two iterations".
- **[DOC] (not fixed)** `observation_.py:1075-1077` — the abstract `inventory` docstring for
  `return_type='full'` claims the dictionary holds "one entry per body that falls at least
  partially inside the FOV and is not completely obscured", but the only implementation
  (Snapshot) returns an entry for *every* body with an `"inside"` flag; the base docstring
  also omits the `"inside"` key. Should be aligned with the Snapshot behavior (whose own
  docstring is now corrected).
- **[SUGGESTION] (not fixed)** `observation_.py:866` — `Scalar.as_scalar(Scalar.as_scalar(
  tfrac) == 0.5)` double-wraps; one `as_scalar` suffices.
- **[SUGGESTION] (not fixed)** `observation_.py:1234-1236` — `parallel_offset_duv` forwards
  `time=None` straight into the FOV methods while `parallel_offset_angles` substitutes the
  midtime; the None-handling is inconsistent across the three `parallel_*` methods.

### src/oops/observation/snapshot.py
- **[BUG] (fixed)** `snapshot.py:537-538` — `inventory` computed the "resolution" entry from
  `self.fov` even when the caller supplied the `fov` override that every other part of the
  method honors. Now uses the local `fov`.
- **[DOC] (fixed)** `snapshot.py:412-415` — the `return_type='full'` docstring claimed
  entries exist only for unobscured bodies inside the FOV; the loop stores an entry for
  every body, with `body_data['inside']` carrying that flag. Docstring now says so.
- **[SUGGESTION] (not fixed)** `snapshot.py:523-524` — `v_scale` is wrapped in `np.abs` but
  `u_scale` is not; a negative u-scale FOV would corrupt the u_min/u_max pixel bounds.
- **[SUGGESTION] (not fixed)** `snapshot.py:155-181` — `uv_range_at_tstep` is not part of
  the Observation API (no other subclass defines it) and is untested; either promote it to
  the base class or remove it.

### src/oops/observation/pixel.py
- **[BUG] (not fixed)** `pixel.py:115` — in `uvt` with `t_axis < 0`, `indices.shape` is
  evaluated on the raw argument, which `scalar_from_indices` allows to be a list or plain
  number; those inputs crash with `AttributeError`. Convert first or document the
  restriction.
- **[BUG] (not fixed)** `pixel.py:235,267` — `event_at_grid`/`gridless_event` document
  `meshgrid` as optional (None allowed) but dereference `meshgrid.shape` whenever `time` is
  None; `gridless_event(None)` crashes where the base-class version works.

### src/oops/observation/slit1d.py, rasterslit1d.py, timedimage.py, insitu.py, __init__.py
- **[DOC] (not fixed)** `insitu.py:21` and class docstring — WIP banner ("Not yet tested. Do
  not use."); depends on the also-WIP Instant cadence. No further findings; the pickle
  support and time_shift are consistent with the other subclasses.
- slit1d.py, rasterslit1d.py, timedimage.py: docstrings verified against signatures and
  behavior (including the tricky fast/slow axis-suffix mapping in TimedImage and its
  cadence-extended-FOV bookkeeping); no defects found.

### src/oops/calibration/* (calibration_.py, flatcalib.py, nullcalib.py, radiance.py, rawcounts.py)
- **[CONSISTENCY] (not fixed)** all subclasses set a public `has_baseline` attribute that the
  base-class `Properties:` docstring does not document.
- **[DOC] (not fixed)** `flatcalib.py:20` (also radiance.py, rawcounts.py) — `factor` is
  typed as "(float)" but the note under `baseline` explains both may be arrays broadcastable
  to the non-spatial data shape; the type should say "(float or array-like)".
- The prescale algebra (`prescaled_args`) was verified symbolically and is correct; the
  extended/point source area-factor asymmetry between Radiance and RawCounts is internally
  consistent.

### Summary
- BUG: 12 found — 7 fixed (2 verified-crash AttributeError/TypeError/RuntimeError classes,
  1 wrong-shape comparison, 1 wrong error payload, 1 asymmetric bounds test, 1 fov override
  ignored, plus the e/sin_i pass-through), 5 reported only (ReversedCadence inverse,
  TDI helper, Instant WIP x2, Pixel edge cases).
- DOC: 10 found — 6 fixed, 4 reported.
- CONSISTENCY: 4 reported. STYLE: 1 reported. SUGGESTION: 8 reported.
- Overall: the recently proofread files (observation subclasses, most cadences) are in good
  shape — docstrings match signatures and behavior closely. The weak spots are the files the
  proofreading pass skipped: `oblategravity.py` (two public methods crashed outright, and its
  docstrings are still legacy one-liners), `gravity_.py`, and the two declared-WIP modules
  (`instant.py`, `insitu.py`). `ReversedCadence.tstep_at_time` was the one substantive
  unfixed correctness issue in actively used code; it has since been fixed.
- Verification: `py_compile` on every edited file; `pytest tests/cadence tests/gravity
  tests/calibration tests/observation` all pass; full main suite
  (`pytest tests --ignore=tests/hosts --ignore=tests/spicedb`) passes 150/150.

---
## Critique: src/oops/backplane/

### src/oops/backplane/__init__.py
- **[BUG] (fixed)** `__init__.py:307` — `standardize_event_key` produced a *tuple* on the
  string-input path but a *list* on the tuple-input path, so the default-suffix branch
  (`event_key[:-1] + (…,)`) raised `TypeError: can only concatenate list (not "tuple") to
  list` for any tuple input combined with a non-empty `default` (e.g.
  `standardize_event_key(('SATURN',), default='RING')`). Reproduced before the fix,
  verified after. Fix: build a list on both paths and convert to a tuple once, right
  after the SUN-deduplication step.
- **[BUG] (fixed)** `__init__.py:518` — In `unmasked_surface_key`, the `'ansa'` branch
  compared `intercept_key[2]` (always a *wayframe*, per `Ansa.intercept_key`) against
  `parent.ring_frame` (the frame object), while the `'ring'` branch correctly compares
  against `parent.ring_frame.wayframe`. A ring_frame that is not its own wayframe would
  never be recognized, silently defeating the intercept cache. Fixed to `.wayframe`.
- **[BUG] (fixed)** `__init__.py:523` — `unmasked_surface_key` fell through to
  `return ''` for unrecognized surface types, contradicting its docstring ("the same
  surface key is returned") and creating a collision-prone `''` dictionary key. Fixed to
  `return surface_key`, matching both the docstring and the behavior of the other
  fall-through branches.
- **[DOC] (fixed)** `__init__.py:25,37,47,122` — Typos: lowercase sentence start
  ("intermediate results…"), "The  object" / "The  number" double-spaces, comment
  "Intialize".
- **[DOC] (fixed)** `__init__.py:80` — Truncated sentence in the constructor Notes: "The
  detector selects the photons it receives based on its." Completed as "…based on its
  lines of sight."
- **[DOC] (fixed)** `__init__.py:87` — "straight- line" wrap artifact → "straight-line".
- **[DOC] (not fixed)** `__init__.py:34` — Constructor summary is just "The
  constructor." — permissible but thin; a noun-phrase summary describing the object
  would be better.
- **[DOC] (not fixed)** `__init__.py:135,204,210` — `_refresh`, `__getstate__`,
  `__setstate__` have no docstrings; the four cached resolution properties `dlos_duv`,
  `duv_dlos`, `center_dlos_duv`, `center_duv_dlos` (lines 226–285) also lack docstrings
  while sibling `dlos_duv1` has one.
- **[CONSISTENCY] (not fixed)** `__init__.py:692` — `get_surface_event` uses a legacy
  two-column `Inputs:` block (note: nonstandard even for legacy, which uses `Input:`)
  while `get_surface` in the same file is fully modern Google style. Several other
  methods (`standardize_event_key`, `register_backplane`, `evaluate`, …) use free-text
  docstrings without `Parameters:`/`Returns:` sections.
- **[SUGGESTION] (not fixed)** `__init__.py:319` — `standardize_event_key` indexes
  `event_key[1]` in the SUN-dedup check; a 1-item key whose only entry ends in
  `<`/`>`/`-` (e.g. `('SUN<',)`) would raise IndexError. Probably unreachable in
  practice.
- **[SUGGESTION] (not fixed)** `__init__.py:454,468` — `modifier = None` assigned twice
  in `get_body_and_modifier` (dead initialization).
- **[SUGGESTION] (not fixed)** `__init__.py:30` — `ALL_DERIVS` lacks the explanatory
  trailing comment its two sibling class flags have.

### src/oops/backplane/all.py
- No defects. Correctly carries `# flake8: noqa: F401` per house rules.

### src/oops/backplane/ansa.py
- **[STYLE] (fixed)** `ansa.py:2` — Banner named the directory `oops/backplanes/…`
  (plural); the package is `oops/backplane/`. Same defect in all 13 non-`__init__`
  modules of this package; all fixed.
- **[DOC] (fixed)** `ansa.py:66,92,194,228` — Four docstrings said "Key defining the
  limb surface event" (copy-paste from limb.py); these are ansa surface events.
- **[DOC] (fixed)** `ansa.py:224` — `ansa_vertical_resolution` summary said "Projected
  radial resolution"; it is the vertical resolution.
- **[DOC] (fixed)** `ansa.py:145` — `_fill_ansa_intercepts` documented parameters
  `radius_type` and `rmax` that do not exist; the only parameter is `event_key`.
- **[SUGGESTION] (not fixed)** `ansa.py:185` — Reaches into
  `event.surface._ringplane` (private attribute of Ansa) from outside the class.

### src/oops/backplane/border.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[CONSISTENCY] (not fixed)** No `Parameters:` blocks anywhere in this module; all
  docstrings are short free text, unlike the rest of the package.
- **[SUGGESTION] (not fixed)** `border.py:12,18` — "locus of points surrounding the
  region" is loose: `border_above`/`border_below` mark edge pixels *inside* the
  region, not surrounding it.

### src/oops/backplane/distance.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[BUG] (fixed)** `distance.py:92` — `center_light_time` documents 'sun'/'obs' as
  accepted `direction` aliases but passed `direction` through unmapped to
  `light_time`, which accepts only 'arr'/'dep' — so the documented aliases raised
  ValueError. Added the same alias mapping `center_distance` already had.
- **[STYLE] (fixed)** `distance.py:87` — Local variable `map` shadowed the builtin
  (house rule: no builtin shadowing); renamed to `directions` in both center
  functions.
- **[DOC] (fixed)** `distance.py:97` — `center_light_time` described its result as
  "distance traveled"; it returns a travel time in seconds.

### src/oops/backplane/lighting.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (fixed)** `lighting.py:68` — Comment in `emission_angle` said "Save this as
  the 'prograde' ring incidence angle"; it saves the emission angle.
- **[DOC] (fixed)** `lighting.py:244` — `minnaert_law` parameters `k`, `k2`, `clip`
  lacked types; added `(float)`.
- **[CONSISTENCY] (not fixed)** `lommel_seeliger_law` states its return value in prose
  ("Returns mu0 / (mu + mu0)") rather than a `Returns:` section; backplane methods
  generally omit `Returns:` sections everywhere, so left as-is.

### src/oops/backplane/limb.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (fixed)** `limb.py:75` — Comment said "Get the ring intercept coordinates" in
  `_fill_limb_intercepts`; changed to "limb".
- **[STYLE] (not fixed)** `limb.py:87` — Stray double space in the signature
  (`direction='west',  minimum=0`). Harmless; whitespace checks are deliberately
  relaxed here.
- **[SUGGESTION] (not fixed)** `limb.py:32,192` — Reaches into private surface
  attributes `_radii`, `_ground`, `_limits` from outside the Surface classes.
  Consider public accessors if these are legitimate cross-module needs.

### src/oops/backplane/orbit.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[SUGGESTION] (not fixed)** `orbit.py:39-49` — `orbit_longitude` computes the
  gridless event *before* checking the backplane cache; harmless (the event itself is
  cached) but wasted work on a cache hit, and inconsistent with the other modules'
  check-cache-first pattern.

### src/oops/backplane/pixel.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[BUG] (fixed)** `pixel.py:37` — `body_diameter_in_pixels` read the private Event
  attribute `_dep_lt_` instead of the public `dep_lt` property.
- **[BUG] (fixed)** `pixel.py:44` — Divided by `fov.uv_scale.vals` without absolute
  value. `uv_scale` components are negative for flipped axes (common in real
  instruments), which made the diameter negative for that axis and corrupted the
  'min'/'max' axis selection. Now divides by `np.abs(...)`.
- **[DOC] (fixed)** `pixel.py:17-19` — Docstring had "min"/"max" descriptions swapped
  relative to the code ('max' picks the *largest* diameter direction).
- **[DOC] (fixed)** `pixel.py:15` — `radius` parameter lacked a type; added
  `(float, optional)`.

### src/oops/backplane/pole.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (fixed)** `pole.py:11` — `pole_clock_angle` summary described the return as a
  "projected pole vector"; it returns the clock *angle* of that vector. Reworded.
- **[CONSISTENCY] (not fixed)** Neither function has a `Parameters:` block.
- **[SUGGESTION] (not fixed)** `pole.py:55` — `pole_position_angle` standardizes but
  does not gridless-ify its cache key, while `pole_clock_angle` uses the gridless key;
  the two entries for the same quantity are keyed inconsistently (correct results,
  duplicate cache entries possible).

### src/oops/backplane/resolution.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (fixed)** `resolution.py:62-64` — `finest_resolution`: summary missing its
  period and "Determined a the intercept point" typo ("at the").
- **[DOC] (fixed)** `resolution.py:67,84` — "Key defining the ring surface event"
  copy-paste in two general-surface functions; changed to "surface event".

### src/oops/backplane/ring.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[BUG] (fixed)** `ring.py:804` — `ring_radius_in_front` called
  `self.where_intercepted(event_key, tvl=False)`, but `where_intercepted` takes no
  `tvl` parameter — guaranteed `TypeError` on every call. Removed the argument.
- **[DOC] (fixed)** `ring.py:266` — `ring_elevation` equivalence note was doubly wrong:
  incidence/emission were swapped and the parameter was called `photon` instead of
  `direction`. Now "(PI/2 - emission) if direction == 'obs', (PI/2 - incidence) if
  direction == 'sun'", matching the code.
- **[DOC] (fixed)** `ring.py:425-431` — `ring_emission_angle` and
  `ring_center_emission_angle` pole bullets said "incidence < pi/2" (copied from the
  incidence functions); changed to "emission". Also fixed the mis-indented `apparent`
  bullet (7 spaces instead of 8) in `ring_emission_angle`.
- **[DOC] (fixed)** `ring.py:23` — `` `event_key.` `` had the period inside the
  backticks.
- **[DOC] (fixed)** `ring.py:124-133` — `radial_mode` parameters `amp`, `peri0`,
  `speed` lacked types; added `(float)` and rewrapped to 90 columns.
- **[DOC] (fixed)** `ring.py:673` — `ring_angular_resolution` declared
  `units (Scalar, optional)`; it is a string flag. Changed to `(str, optional)`.
- **[CONSISTENCY] (not fixed)** `ring.py:521,525,574,578` — Uses `np.pi` where every
  parallel branch elsewhere uses `Scalar.PI` (numerically identical).
- **[DOC] (not fixed)** `ring.py:588,615` — `ring_center_incidence_angle` /
  `ring_center_emission_angle` describe `event_key` as "Key defining the ring surface
  event"; for gridless center quantities the parallel modules say "Key defining the
  event on the body's path".

### src/oops/backplane/sky.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (fixed)** `sky.py:104` — `celestial_east_angle` summary said "Direction of
  celestial north"; changed to east.
- **[DOC] (fixed)** `sky.py:13,86,175` — Missing period on `right_ascension` summary;
  "refer refer" doubled word; "the the" doubled word.

### src/oops/backplane/spheroid.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[DOC] (not fixed)** `spheroid.py:22` — `minimum … in degrees, either 0 or -180`:
  the value is a selector label (0 or -180) while the returned backplane is in radians;
  consistent with the repo-wide "labels say deg, values are radians" trap, but a
  clarifying phrase ("selects the branch cut; values are radians") would help. Same
  wording appears in limb.py and the sub-longitude functions.
- **[CONSISTENCY] (not fixed)** `spheroid.py:302` — `sub_solar_longitude` passes
  `event_key` where `sub_observer_longitude` passes `gridless_key` to the underscore
  helper (equivalent because the helper re-standardizes, but inconsistent).
- **[SUGGESTION] (not fixed)** `spheroid.py:374,402` — Reaches into
  `event.surface._unsquash_sq` (private attribute) from outside the class.

### src/oops/backplane/where.py
- **[STYLE] (fixed)** banner path (see ansa.py).
- **[BUG] (fixed)** `where.py:253` — In `_where_inside_or_outside`, `event` was only
  assigned inside the `if surface.HAS_INTERIOR:` branch, but the `else` branch and the
  subsequent mask application both use `event` — `UnboundLocalError` for any surface
  without an interior. Hoisted `event = self.get_surface_event(event_key)` above the
  branch.
- **[BUG] (fixed)** `where.py:53,233` — Two `raise ValueError('…: ', event_key)` calls
  passed the key as a second exception argument (comma) instead of formatting it into
  the message; now concatenates `repr(event_key)`.
- **[DOC] (fixed)** `where.py:91` — "is in not obscured" → "is not obscured".
- **[DOC] (fixed)** 6 occurrences — "this uses the mask uses three-valued logic" →
  "this mask uses three-valued logic".
- **[DOC] (fixed)** `where.py:162` — "away fron the Sun" → "from".
- **[DOC] (fixed)** `where.py:48,114,171,228` — The four private dispatch helpers had
  no docstrings; added one-line docstrings.

### Summary

Counts: 8 BUG (8 fixed), 30 DOC (26 fixed), 6 CONSISTENCY (0 fixed — style-coexistence
issues flagged per instructions), 15 STYLE (14 banner fixes + 1 left), 8 SUGGESTION
(0 fixed).

The package's computational core (`__init__.py` event solver, ring/spheroid geometry)
is in good shape, but the outlying code paths clearly see little test traffic: three
of the confirmed bugs (`where_intercepted(tvl=…)`, the unbound `event`, the list+tuple
`TypeError`) crash on first use, meaning `ring_radius_in_front`,
`where_inside`/`where_outside` against interior-less surfaces, and tuple-keyed
default-suffix lookups have no coverage at all. Docstring quality is generally modern
Google style but with a high copy-paste error rate between sibling modules (limb→ansa,
incidence→emission, north→east); a targeted proofread of the remaining `CONSISTENCY`
items (legacy `Inputs:` block in `get_surface_event`, missing `Parameters:` blocks in
border.py/pole.py, missing property docstrings in `__init__.py`) would bring the
package to a uniform standard. There is no test directory for backplanes in the main
suite; coverage comes only from the gold-master host tests.

### Completing the backplane docstrings

The proofread above was carried out after the review. Every module and class in the
package, and in `config.py` and `lightsource.py`, now carries a docstring with a
`Parameters:` block; an audit of the package reports no function that lacks one. That
covered 101 items: the 31 in the `Backplane` class itself, including the cache properties
`dlos_duv`, `duv_dlos`, `center_dlos_duv`, and `center_duv_dlos`, the `_refresh` and
pickling methods, and the whole key-manipulation and event-retrieval surface; 26 in
`where.py` and `border.py`; 18 across `ring.py`, `spheroid.py`, `sky.py`, `pole.py`,
`ansa.py`, and `resolution.py`; and the last three legacy blocks in the library outside
`body.py` and the hosts — `Backplane.get_surface_event`, `LOGGING.print`, and
`DiskSource.__init__`, all of which used the two-column `Inputs:` layout. No legacy block
now remains in that tree.

`Returns:` blocks were added only in `__init__.py`. The package's convention is
unambiguous on this point: the thirteen modules that define backplane methods carry 96
`Parameters:` blocks between them and not one `Returns:`, stating the returned quantity
in the summary line instead, while the `Backplane` class methods use `Returns:`
throughout. Both styles were left as they were found.

Reading the code closely enough to document it also turned up two summary lines that
misdescribed behavior, now corrected. `where_not` claimed a mask "where the value of the
given backplane is False, zero, or masked", but a masked location never comes back True:
with `tvl=False` the values are ANDed with the antimask, and with `tvl=True` a masked
location stays masked. `where_in_front` described its arguments as `back_body` and
`front_body`, names that appear nowhere in the signature or the code.


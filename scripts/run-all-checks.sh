#!/usr/bin/env bash
#
# rms-oops - Run All Checks Script
#
# This script runs linting, the pytest suites, and Markdown lint as separate
# checks. In parallel mode all requested checks run concurrently.
#
# Usage:
#   ./scripts/run-all-checks.sh [options]
#
# Options:
#   -p, --parallel         Run all requested checks in parallel (default)
#   -s, --sequential       Run all requested checks sequentially
#   -c, --code             Run all code checks (sets each RUN_* code flag true)
#   -m, --markdown         Run only PyMarkdown (RUN_PYMARKDOWN)
#   --flake8               Run flake8 only, i.e. the continuation-line checks
#                          (may combine with other --* flags)
#   --ruff-check           Run ruff check only
#   --mypy                 Run mypy only (the tests; src has no annotations)
#   --stubtest             Run stubtest only (the .pyi stubs vs the runtime API)
#   --pyroma               Run pyroma only
#   --bandit               Run bandit only
#   --vulture              Run vulture only
#   --sphinx               Build the documentation only
#   --pytest               Run the main oops test suite only
#   --pytest-hosts         Run the host (gold master) test suite only
#   --pytest-spicedb       Run the spicedb tests only
#   --pip-audit            Run pip-audit only
#   --pymarkdown           Run PyMarkdown scan only
#   -h, --help             Show this help message
#
# Requires the virtualenv created by ./scripts/setup-venv.sh.
#
# Environment:
#   VENV or VENV_PATH        Path to virtualenv (default: $PROJECT_ROOT/venv)
#   CLEANUP_GRACE_PERIOD     Seconds to wait for graceful shutdown (default: 5)
#
#   The test suites read the resource tree named by OOPS_RESOURCES (see README.md).
#   Without it they fail on missing SPICE kernels and gold masters, which is an
#   environment problem rather than a code defect.
#
#   RUN_* (set by this script from CLI or full-run defaults): RUN_FLAKE8,
#   RUN_RUFF_CHECK, RUN_MYPY, RUN_PYROMA, RUN_BANDIT, RUN_VULTURE, RUN_PYTEST,
#   RUN_PYTEST_HOSTS, RUN_PYTEST_SPICEDB, RUN_PIP_AUDIT, RUN_SPHINX,
#   RUN_PYMARKDOWN
#
#   Per-check toggles (true/false). Each check runs only if both RUN_* and
#   ENABLE_* are true (RUN_* from CLI or defaults below; ENABLE_* from env):
#     ENABLE_FLAKE8            continuation-line checks only (default: true)
#     ENABLE_RUFF_CHECK        the linter of record (default: true)
#     ENABLE_MYPY              the tests only (default: true)
#     ENABLE_STUBTEST          .pyi stubs match the runtime API (default: true)
#     ENABLE_PYROMA            packaging metadata, --min=10 (default: true)
#     ENABLE_BANDIT            security scan of src (default: true)
#     ENABLE_VULTURE           dead code in src (default: true)
#     ENABLE_PYTEST            main suite, tests/ minus hosts and spicedb
#                              (default: true)
#     ENABLE_PYTEST_HOSTS      host suite, tests/hosts (default: true)
#     ENABLE_PYTEST_SPICEDB    spicedb tests, tests/spicedb (default: true)
#     ENABLE_PIP_AUDIT         (default: false)
#     ENABLE_SPHINX            documentation build, warnings as errors
#                              (default: true)
#     ENABLE_PYMARKDOWN        PyMarkdown scan (default: false)
#
#   pip-audit and PyMarkdown remain off by default: pip-audit reports findings
#   against pinned upstream dependencies this repository does not control, and
#   PyMarkdown reports pre-existing findings in the Markdown. Turning either on
#   is a cleanup project rather than a gate.
#
#   `ruff format` is deliberately absent. The house style aligns assignments,
#   imports, and trailing comments in columns, which the formatter would undo;
#   [tool.ruff.format] records the quote style for anyone running it by hand.
#
# Checks (each run separately):
#   Code:     the three pytest suites, and optionally ruff check (the linter of
#             record), flake8 (the continuation-line checks alone), and
#             pip-audit.
#
#             The three suites are separate pytest invocations rather than one
#             `pytest tests` run so that each keeps its own process, as it had
#             under unittest. They pass either way; running them apart keeps a
#             failure attributable to one suite.
#   Markdown: pymarkdown scan .claude/ README.md CONTRIBUTING.md
#
# The CI workflow runs scripts/automated_tests/oops_main_test.sh, which wraps the
# same test suites with the self-hosted runner's reinstall step and coverage
# reporting. Keep the two in step if either changes.
#
# Exit codes:
#   0 - All requested checks passed
#   1 - One or more checks failed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Default options
PARALLEL=true
RUN_FLAKE8=false
RUN_RUFF_CHECK=false
RUN_MYPY=false
RUN_STUBTEST=false
RUN_PYROMA=false
RUN_BANDIT=false
RUN_VULTURE=false
RUN_PYTEST=false
RUN_PYTEST_HOSTS=false
RUN_PYTEST_SPICEDB=false
RUN_PIP_AUDIT=false
RUN_SPHINX=false
RUN_PYMARKDOWN=false
SCOPE_SPECIFIED=false

# Per-check defaults (override by exporting before invoking this script, or
# permanently change here)
: "${ENABLE_FLAKE8:=true}"
: "${ENABLE_RUFF_CHECK:=true}"
: "${ENABLE_MYPY:=true}"
: "${ENABLE_STUBTEST:=true}"
: "${ENABLE_PYROMA:=true}"
: "${ENABLE_BANDIT:=true}"
: "${ENABLE_VULTURE:=true}"
: "${ENABLE_PYTEST:=true}"
: "${ENABLE_PYTEST_HOSTS:=true}"
: "${ENABLE_PYTEST_SPICEDB:=true}"
: "${ENABLE_PIP_AUDIT:=false}"
: "${ENABLE_SPHINX:=true}"
: "${ENABLE_PYMARKDOWN:=false}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${VENV:-${VENV_PATH:-$PROJECT_ROOT/venv}}"

# Track failures and final exit code
FAILED_CHECKS=()
EXIT_CODE=0

# Temp directory for parallel output and status files
TEMP_DIR=$(mktemp -d)

# Grace period (seconds) before SIGKILL after SIGTERM
CLEANUP_GRACE_PERIOD=${CLEANUP_GRACE_PERIOD:-5}
if ! echo "$CLEANUP_GRACE_PERIOD" | grep -qE '^[0-9]+$'; then
    echo "Error: CLEANUP_GRACE_PERIOD must be a non-negative integer (got: $CLEANUP_GRACE_PERIOD)" >&2
    exit 1
fi

_wait_or_kill() {
    local pid=$1
    [ -z "$pid" ] && return 0
    kill -TERM "$pid" 2>/dev/null || true
    local waited=0
    while [ "$waited" -lt "$CLEANUP_GRACE_PERIOD" ]; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    return 0
}

_cleanup() {
    rm -rf "$TEMP_DIR"
}

# On INT/TERM: kill all background check jobs with grace period, then exit
_cleanup_and_exit() {
    local sig_code=$1
    local pids
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            _wait_or_kill "$pid"
        done
    fi
    _cleanup
    exit "$sig_code"
}
trap '_cleanup_and_exit 130' SIGINT
trap '_cleanup_and_exit 143' SIGTERM
trap _cleanup EXIT

print_header() {
    echo -e "\n${BOLD}${BLUE}===================================================${RESET}"
    echo -e "${BOLD}${BLUE}  $1${RESET}"
    echo -e "${BOLD}${BLUE}===================================================${RESET}\n"
}

print_section() {
    echo -e "\n${BOLD}${YELLOW}>>> $1${RESET}\n"
}

print_success() {
    echo -e "${GREEN}✓${RESET} $1"
}

print_error() {
    echo -e "${RED}✗${RESET} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${RESET} $1"
}

show_usage() {
    sed -n '/^# Usage:/,/^# Exit codes:/p' "$0" | sed 's/^# //g' | sed 's/^#//g'
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -s|--sequential)
            PARALLEL=false
            shift
            ;;
        -c|--code)
            RUN_FLAKE8=true
            RUN_RUFF_CHECK=true
            RUN_MYPY=true
            RUN_STUBTEST=true
            RUN_PYROMA=true
            RUN_BANDIT=true
            RUN_VULTURE=true
            RUN_PYTEST=true
            RUN_PYTEST_HOSTS=true
            RUN_PYTEST_SPICEDB=true
            RUN_PIP_AUDIT=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        -m|--markdown)
            RUN_PYMARKDOWN=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --flake8)
            RUN_FLAKE8=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --mypy)
            RUN_MYPY=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --stubtest)
            RUN_STUBTEST=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pyroma)
            RUN_PYROMA=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --bandit)
            RUN_BANDIT=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --vulture)
            RUN_VULTURE=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --sphinx)
            RUN_SPHINX=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --ruff-check)
            RUN_RUFF_CHECK=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pytest)
            RUN_PYTEST=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pytest-hosts)
            RUN_PYTEST_HOSTS=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pytest-spicedb)
            RUN_PYTEST_SPICEDB=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pip-audit)
            RUN_PIP_AUDIT=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --pymarkdown)
            RUN_PYMARKDOWN=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${RESET}" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Default: run all checks (each RUN_* true; ENABLE_* still filters per repo)
if [ "$SCOPE_SPECIFIED" = false ]; then
    RUN_FLAKE8=true
    RUN_RUFF_CHECK=true
    RUN_MYPY=true
    RUN_STUBTEST=true
    RUN_PYROMA=true
    RUN_BANDIT=true
    RUN_VULTURE=true
    RUN_PYTEST=true
    RUN_PYTEST_HOSTS=true
    RUN_PYTEST_SPICEDB=true
    RUN_PIP_AUDIT=true
    RUN_SPHINX=true
    RUN_PYMARKDOWN=true
fi

START_TIME=$(date +%s)

print_header "rms-oops - Running All Checks"

if [ "$PARALLEL" = true ]; then
    print_info "Running checks in PARALLEL mode"
else
    print_info "Running checks in SEQUENTIAL mode"
fi

# True if at least one code check is both selected (RUN_*) and enabled (ENABLE_*).
_code_checks_any_scheduled() {
    [ "$RUN_FLAKE8" = true ] && [ "$ENABLE_FLAKE8" = true ] && return 0
    [ "$RUN_RUFF_CHECK" = true ] && [ "$ENABLE_RUFF_CHECK" = true ] && return 0
    [ "$RUN_PYTEST" = true ] && [ "$ENABLE_PYTEST" = true ] && return 0
    [ "$RUN_PYTEST_HOSTS" = true ] && [ "$ENABLE_PYTEST_HOSTS" = true ] && return 0
    [ "$RUN_PYTEST_SPICEDB" = true ] && [ "$ENABLE_PYTEST_SPICEDB" = true ] && return 0
    [ "$RUN_PIP_AUDIT" = true ] && [ "$ENABLE_PIP_AUDIT" = true ] && return 0
    [ "$RUN_MYPY" = true ] && [ "$ENABLE_MYPY" = true ] && return 0
    [ "$RUN_STUBTEST" = true ] && [ "$ENABLE_STUBTEST" = true ] && return 0
    [ "$RUN_PYROMA" = true ] && [ "$ENABLE_PYROMA" = true ] && return 0
    [ "$RUN_BANDIT" = true ] && [ "$ENABLE_BANDIT" = true ] && return 0
    [ "$RUN_VULTURE" = true ] && [ "$ENABLE_VULTURE" = true ] && return 0
    return 1
}

# ---- Code checks (flake8, ruff, the pytest suites, pip-audit) ----
run_code_checks() {
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "Code Checks"

    cd "$PROJECT_ROOT" || exit 1

    if ! _code_checks_any_scheduled; then
        print_info "No code checks scheduled (RUN_* and ENABLE_*); skipping code checks"
        return 0
    fi

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV; run ./scripts/setup-venv.sh"
        [ -n "$status_file" ] && echo "Code - Virtual environment not found" >> "$status_file"
        return 1
    fi

    # shellcheck source=/dev/null
    source "$VENV/bin/activate"

    local failed=false
    local failed_checks=""

    if [ "$RUN_FLAKE8" = true ] && [ "$ENABLE_FLAKE8" = true ]; then
        # Ruff is the linter of record. It implements no rule in the E121-E133 range, so
        # the continuation-line indent checks come from flake8 and nothing else. `.flake8`
        # selects that range on its own, so the flag below only restates it; every other
        # check is ruff's. This is the same split rms-polymath uses.
        print_info "Running flake8 (continuation-line checks only)..."
        if python -m flake8 --select=E12,E13 src programs tests; then
            print_success "Flake8 passed"
        else
            print_error "Flake8 failed"
            failed=true
            failed_checks="${failed_checks}Code - Flake8"$'\n'
        fi
    fi

    if [ "$RUN_RUFF_CHECK" = true ] && [ "$ENABLE_RUFF_CHECK" = true ]; then
        print_info "Running ruff check..."
        if python -m ruff check .; then
            print_success "Ruff check passed"
        else
            print_error "Ruff check failed"
            failed=true
            failed_checks="${failed_checks}Code - Ruff check"$'\n'
        fi
    fi

    if [ "$RUN_PYTEST_SPICEDB" = true ] && [ "$ENABLE_PYTEST_SPICEDB" = true ]; then
        print_info "Running the spicedb tests..."
        if python -m pytest tests/spicedb; then
            print_success "Spicedb tests passed"
        else
            print_error "Spicedb tests failed"
            failed=true
            failed_checks="${failed_checks}Code - Spicedb tests"$'\n'
        fi
    fi

    if [ "$RUN_PYTEST" = true ] && [ "$ENABLE_PYTEST" = true ]; then
        print_info "Running the main oops test suite..."
        if python -m pytest tests --ignore=tests/hosts --ignore=tests/spicedb; then
            print_success "Main test suite passed"
        else
            print_error "Main test suite failed"
            failed=true
            failed_checks="${failed_checks}Code - Main test suite"$'\n'
        fi
    fi

    if [ "$RUN_PYTEST_HOSTS" = true ] && [ "$ENABLE_PYTEST_HOSTS" = true ]; then
        print_info "Running the host test suite (gold masters)..."
        if python -m pytest tests/hosts; then
            print_success "Host test suite passed"
        else
            print_error "Host test suite failed"
            failed=true
            failed_checks="${failed_checks}Code - Host test suite"$'\n'
        fi
    fi

    if [ "$RUN_MYPY" = true ] && [ "$ENABLE_MYPY" = true ]; then
        # The tests only; src carries no annotations by house rule. The scope is set by
        # `files` in [tool.mypy].
        print_info "Running mypy on the tests..."
        if python -m mypy; then
            print_success "Mypy passed"
        else
            print_error "Mypy failed"
            failed=true
            failed_checks="${failed_checks}Code - Mypy"$'\n'
        fi
    fi

    if [ "$RUN_STUBTEST" = true ] && [ "$ENABLE_STUBTEST" = true ]; then
        # A stub replaces its module outright for a type checker, so whatever a stub
        # omits is invisible downstream. stubtest compares each one against the imported
        # module, which is what keeps them complete. MYPYPATH names src/ because the two
        # library packages live there and the editable install's import hook is invisible
        # to mypy. The allowlist holds the handful of leaked loop variables that exist at
        # run time and are deliberately not published.
        print_info "Running stubtest (.pyi stubs vs the runtime API)..."
        if MYPYPATH=src python -m mypy.stubtest oops spicedb programs \
                --mypy-config-file pyproject.toml \
                --allowlist stubtest-allowlist.txt; then
            print_success "Stubtest passed"
        else
            print_error "Stubtest failed"
            failed=true
            failed_checks="${failed_checks}Code - Stubtest"$'\n'
        fi
    fi

    if [ "$RUN_PYROMA" = true ] && [ "$ENABLE_PYROMA" = true ]; then
        print_info "Running pyroma..."
        if python -m pyroma --min=10 .; then
            print_success "Pyroma passed"
        else
            print_error "Pyroma failed"
            failed=true
            failed_checks="${failed_checks}Code - Pyroma"$'\n'
        fi
    fi

    if [ "$RUN_BANDIT" = true ] && [ "$ENABLE_BANDIT" = true ]; then
        print_info "Running bandit..."
        if python -m bandit -q -c pyproject.toml -r src; then
            print_success "Bandit passed"
        else
            print_error "Bandit failed"
            failed=true
            failed_checks="${failed_checks}Code - Bandit"$'\n'
        fi
    fi

    if [ "$RUN_VULTURE" = true ] && [ "$ENABLE_VULTURE" = true ]; then
        print_info "Running vulture..."
        if python -m vulture; then
            print_success "Vulture passed"
        else
            print_error "Vulture failed"
            failed=true
            failed_checks="${failed_checks}Code - Vulture"$'\n'
        fi
    fi

    if [ "$RUN_PIP_AUDIT" = true ] && [ "$ENABLE_PIP_AUDIT" = true ]; then
        print_info "Running pip-audit..."
        if python -m pip_audit; then
            print_success "Pip-audit passed"
        else
            print_error "Pip-audit failed"
            failed=true
            failed_checks="${failed_checks}Code - Pip-audit"$'\n'
        fi
    fi

    deactivate 2>/dev/null || true

    if [ "$failed" = true ]; then
        [ -n "$status_file" ] && printf '%s' "$failed_checks" >> "$status_file"
        return 1
    fi
    return 0
}

# ---- Documentation build (Sphinx) ----
run_docs_checks() {
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "Documentation (Sphinx)"

    cd "$PROJECT_ROOT" || exit 1

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV; run ./scripts/setup-venv.sh"
        [ -n "$status_file" ] && echo "Docs - Virtual environment not found" >> "$status_file"
        return 1
    fi

    # shellcheck source=/dev/null
    source "$VENV/bin/activate"

    # -W turns every warning into an error, so a broken reference or an unparseable
    # docstring fails the build rather than scrolling past. -n adds nitpicky mode, which
    # reports every cross-reference and parameter type that resolves to nothing; the
    # handful with no target to resolve to are listed in `nitpick_ignore` in docs/conf.py.
    # -E rebuilds from scratch, so a stale cache cannot hide a warning that a previous run
    # already reported.
    print_info "Building the documentation..."
    if python -m sphinx -W -n -E -b html docs docs/_build/html; then
        print_success "Documentation build passed"
        deactivate 2>/dev/null || true
        return 0
    else
        print_error "Documentation build failed"
        [ -n "$status_file" ] && echo "Docs - Sphinx build" >> "$status_file"
        deactivate 2>/dev/null || true
        return 1
    fi
}

# ---- Markdown lint only (PyMarkdown) ----
run_markdown_checks() {
    local output_file="${1:-}"
    local status_file="${2:-}"

    if [ -n "$output_file" ]; then
        exec > "$output_file" 2>&1
    fi

    print_section "Markdown Lint (PyMarkdown)"

    cd "$PROJECT_ROOT" || exit 1

    if [ ! -f "$VENV/bin/activate" ]; then
        print_error "Virtual environment not found at $VENV; run ./scripts/setup-venv.sh"
        [ -n "$status_file" ] && echo "Markdown - Virtual environment not found" >> "$status_file"
        return 1
    fi

    # shellcheck source=/dev/null
    source "$VENV/bin/activate"

    print_info "Running PyMarkdown scan (.claude/, root *.md)..."
    local scan_paths=()
    [ -d ".claude/" ] && scan_paths+=(".claude/")
    [ -f "README.md" ] && scan_paths+=("README.md")
    [ -f "CONTRIBUTING.md" ] && scan_paths+=("CONTRIBUTING.md")
    if [ ${#scan_paths[@]} -eq 0 ]; then
        print_info "No Markdown files/directories found to scan"
        deactivate 2>/dev/null || true
        return 0
    fi
    if python -m pymarkdown scan "${scan_paths[@]}"; then
        print_success "PyMarkdown scan passed"
        deactivate 2>/dev/null || true
        return 0
    else
        print_error "PyMarkdown scan failed"
        [ -n "$status_file" ] && echo "Markdown - PyMarkdown scan" >> "$status_file"
        deactivate 2>/dev/null || true
        return 1
    fi
}

# ---- Collect status from a status file into FAILED_CHECKS ----
_collect_status() {
    local status_file=$1
    if [ -f "$status_file" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && FAILED_CHECKS+=("$line")
        done < "$status_file"
    fi
}

# ---- Run requested checks ----
if [ "$PARALLEL" = true ]; then
    print_info "Running requested checks in parallel, please wait..."

    pids=()
    temp_files=()
    status_files=()

    if _code_checks_any_scheduled; then
        code_output="$TEMP_DIR/code.log"
        code_status="$TEMP_DIR/code.status"
        temp_files+=("$code_output")
        status_files+=("$code_status")
        run_code_checks "$code_output" "$code_status" &
        pids+=($!)
    fi

    if [ "$RUN_SPHINX" = true ] && [ "$ENABLE_SPHINX" = true ]; then
        docs_output="$TEMP_DIR/docs.log"
        docs_status="$TEMP_DIR/docs.status"
        temp_files+=("$docs_output")
        status_files+=("$docs_status")
        run_docs_checks "$docs_output" "$docs_status" &
        pids+=($!)
    fi

    if [ "$RUN_PYMARKDOWN" = true ] && [ "$ENABLE_PYMARKDOWN" = true ]; then
        markdown_output="$TEMP_DIR/markdown.log"
        markdown_status="$TEMP_DIR/markdown.status"
        temp_files+=("$markdown_output")
        status_files+=("$markdown_status")
        run_markdown_checks "$markdown_output" "$markdown_status" &
        pids+=($!)
    fi

    # Wait for all jobs; any non-zero exit sets EXIT_CODE=1
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            EXIT_CODE=1
        fi
    done

    # Collect named failures from status files
    for status_file in "${status_files[@]}"; do
        _collect_status "$status_file"
    done

    # Safety net: if any status file had content, ensure EXIT_CODE reflects it
    [ ${#FAILED_CHECKS[@]} -gt 0 ] && EXIT_CODE=1

    # Print all outputs in a fixed order
    echo ""
    for log_file in "${temp_files[@]}"; do
        [ -f "$log_file" ] && cat "$log_file"
    done
else
    # Sequential — pass a status file so FAILED_CHECKS is populated
    if _code_checks_any_scheduled; then
        code_status="$TEMP_DIR/code.status"
        if ! run_code_checks "" "$code_status"; then
            EXIT_CODE=1
        fi
        _collect_status "$code_status"
    fi

    if [ "$RUN_SPHINX" = true ] && [ "$ENABLE_SPHINX" = true ]; then
        docs_status="$TEMP_DIR/docs.status"
        if ! run_docs_checks "" "$docs_status"; then
            EXIT_CODE=1
        fi
        _collect_status "$docs_status"
    fi

    if [ "$RUN_PYMARKDOWN" = true ] && [ "$ENABLE_PYMARKDOWN" = true ]; then
        markdown_status="$TEMP_DIR/markdown.status"
        if ! run_markdown_checks "" "$markdown_status"; then
            EXIT_CODE=1
        fi
        _collect_status "$markdown_status"
    fi
fi

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
ELAPSED_SECONDS=$((ELAPSED % 60))

print_header "Summary"

if [ "$EXIT_CODE" -eq 0 ]; then
    print_success "All checks passed!"
    echo -e "${GREEN}${BOLD}✓ SUCCESS${RESET} - All checks completed successfully"
else
    print_error "Some checks failed:"
    if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
        echo -e "  ${RED}✗${RESET} One or more checks failed (see output above)"
    else
        for check in "${FAILED_CHECKS[@]}"; do
            echo -e "  ${RED}✗${RESET} $check"
        done
        echo -e "${RED}${BOLD}✗ FAILURE${RESET} - ${#FAILED_CHECKS[@]} check(s) failed"
    fi
fi

echo ""
print_info "Total time: ${MINUTES}m ${ELAPSED_SECONDS}s"
echo ""

exit "$EXIT_CODE"

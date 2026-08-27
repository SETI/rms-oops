#!/usr/bin/env bash
#
# rms-oops - Run All Checks Script
#
# This script runs linting, the unit test suites, and Markdown lint as separate
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
#   --flake8               Run flake8 only (may combine with other --* flags)
#   --ruff-check           Run ruff check only
#   --unittest             Run the main oops unit test suite only
#   --unittest-hosts       Run the host (gold master) unit test suite only
#   --unittest-spicedb     Run the spicedb unit tests only
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
#   RUN_RUFF_CHECK, RUN_UNITTEST, RUN_UNITTEST_HOSTS, RUN_UNITTEST_SPICEDB,
#   RUN_PIP_AUDIT, RUN_PYMARKDOWN
#
#   Per-check toggles (true/false). Each check runs only if both RUN_* and
#   ENABLE_* are true (RUN_* from CLI or defaults below; ENABLE_* from env):
#     ENABLE_FLAKE8            (default: false; see the note below)
#     ENABLE_RUFF_CHECK        (default: false; see the note below)
#     ENABLE_UNITTEST          main suite, tests/unittester.py (default: true)
#     ENABLE_UNITTEST_HOSTS    host suite, tests/hosts/unittester.py (default: true)
#     ENABLE_UNITTEST_SPICEDB  spicedb tests (default: true)
#     ENABLE_PIP_AUDIT         (default: false)
#     ENABLE_PYMARKDOWN        PyMarkdown scan (default: false)
#
#   flake8, ruff check, pip-audit, and PyMarkdown are configured but off by
#   default: each currently reports pre-existing findings against the legacy
#   modules and Markdown, so turning one on is a cleanup project rather than a
#   gate. .github/workflows/run-lint.yml is dispatch-only for the same reason.
#
# Checks (each run separately):
#   Code:     the three unittest suites, and optionally flake8 (the linter of
#             record; .flake8 is authoritative), ruff check, and pip-audit.
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
RUN_UNITTEST=false
RUN_UNITTEST_HOSTS=false
RUN_UNITTEST_SPICEDB=false
RUN_PIP_AUDIT=false
RUN_PYMARKDOWN=false
SCOPE_SPECIFIED=false

# Per-check defaults (override by exporting before invoking this script, or
# permanently change here)
: "${ENABLE_FLAKE8:=false}"
: "${ENABLE_RUFF_CHECK:=false}"
: "${ENABLE_UNITTEST:=true}"
: "${ENABLE_UNITTEST_HOSTS:=true}"
: "${ENABLE_UNITTEST_SPICEDB:=true}"
: "${ENABLE_PIP_AUDIT:=false}"
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
            RUN_UNITTEST=true
            RUN_UNITTEST_HOSTS=true
            RUN_UNITTEST_SPICEDB=true
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
        --ruff-check)
            RUN_RUFF_CHECK=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --unittest)
            RUN_UNITTEST=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --unittest-hosts)
            RUN_UNITTEST_HOSTS=true
            SCOPE_SPECIFIED=true
            shift
            ;;
        --unittest-spicedb)
            RUN_UNITTEST_SPICEDB=true
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
    RUN_UNITTEST=true
    RUN_UNITTEST_HOSTS=true
    RUN_UNITTEST_SPICEDB=true
    RUN_PIP_AUDIT=true
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
    [ "$RUN_UNITTEST" = true ] && [ "$ENABLE_UNITTEST" = true ] && return 0
    [ "$RUN_UNITTEST_HOSTS" = true ] && [ "$ENABLE_UNITTEST_HOSTS" = true ] && return 0
    [ "$RUN_UNITTEST_SPICEDB" = true ] && [ "$ENABLE_UNITTEST_SPICEDB" = true ] && return 0
    [ "$RUN_PIP_AUDIT" = true ] && [ "$ENABLE_PIP_AUDIT" = true ] && return 0
    return 1
}

# ---- Code checks (flake8, ruff, the unittest suites, pip-audit) ----
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
        print_info "Running flake8..."
        if python -m flake8 src programs; then
            print_success "Flake8 passed"
        else
            print_error "Flake8 failed"
            failed=true
            failed_checks="${failed_checks}Code - Flake8"$'\n'
        fi
    fi

    if [ "$RUN_RUFF_CHECK" = true ] && [ "$ENABLE_RUFF_CHECK" = true ]; then
        print_info "Running ruff check..."
        if python -m ruff check src programs tests; then
            print_success "Ruff check passed"
        else
            print_error "Ruff check failed"
            failed=true
            failed_checks="${failed_checks}Code - Ruff check"$'\n'
        fi
    fi

    if [ "$RUN_UNITTEST_SPICEDB" = true ] && [ "$ENABLE_UNITTEST_SPICEDB" = true ]; then
        print_info "Running the spicedb unit tests..."
        if python -m unittest spicedb; then
            print_success "Spicedb unit tests passed"
        else
            print_error "Spicedb unit tests failed"
            failed=true
            failed_checks="${failed_checks}Code - Spicedb unit tests"$'\n'
        fi
    fi

    if [ "$RUN_UNITTEST" = true ] && [ "$ENABLE_UNITTEST" = true ]; then
        print_info "Running the main oops unit test suite..."
        if python -m unittest tests/unittester.py; then
            print_success "Main unit test suite passed"
        else
            print_error "Main unit test suite failed"
            failed=true
            failed_checks="${failed_checks}Code - Main unit test suite"$'\n'
        fi
    fi

    if [ "$RUN_UNITTEST_HOSTS" = true ] && [ "$ENABLE_UNITTEST_HOSTS" = true ]; then
        print_info "Running the host unit test suite (gold masters)..."
        if python -m unittest tests/hosts/unittester.py; then
            print_success "Host unit test suite passed"
        else
            print_error "Host unit test suite failed"
            failed=true
            failed_checks="${failed_checks}Code - Host unit test suite"$'\n'
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

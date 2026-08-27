#!/usr/bin/env bash
#
# rms-oops - Virtual Environment Bootstrap
#
# Creates the virtualenv that scripts/run-all-checks.sh expects and installs
# the project in editable mode with its development extras.
# Safe to re-run: an existing environment is reused and its packages upgraded.
#
# Usage:
#   ./scripts/setup-venv.sh [options]
#
# Options:
#   -r, --recreate         Delete an existing virtualenv and build a fresh one
#   -p, --python CMD       Interpreter used to create the venv (default: python3)
#   -h, --help             Show this help message
#
# Environment:
#   VENV or VENV_PATH      Path to virtualenv (default: $PROJECT_ROOT/venv)
#
# Exit codes:
#   0 - Environment ready
#   1 - Bootstrap failed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

print_info() { echo -e "${BLUE}==>${RESET} $1"; }
print_success() { echo -e "${GREEN}✓${RESET} $1"; }
print_error() { echo -e "${RED}✗${RESET} $1" >&2; }

# Minimum interpreter version, kept in sync with requires-python in pyproject.toml
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${VENV:-${VENV_PATH:-$PROJECT_ROOT/venv}}"

RECREATE=false
PYTHON_CMD=python3

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--recreate)
            RECREATE=true
            shift
            ;;
        -p|--python)
            if [ $# -lt 2 ]; then
                print_error "Option $1 requires an argument"
                exit 1
            fi
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT" || exit 1

if ! command -v "$PYTHON_CMD" > /dev/null 2>&1; then
    print_error "Interpreter not found: $PYTHON_CMD"
    echo "Use --python to name a different interpreter." >&2
    exit 1
fi

# Refuse an interpreter older than the project supports, rather than building an
# environment whose failures would surface later as confusing import errors.
if ! "$PYTHON_CMD" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_PYTHON_MAJOR, $MIN_PYTHON_MINOR) else 1)"; then
    print_error "$PYTHON_CMD is older than the required Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR"
    "$PYTHON_CMD" --version >&2
    exit 1
fi

if [ "$RECREATE" = true ] && [ -d "$VENV" ]; then
    print_info "Removing existing virtualenv at $VENV"
    rm -rf "$VENV"
fi

if [ -f "$VENV/bin/activate" ]; then
    print_info "Reusing existing virtualenv at $VENV"
else
    if [ -e "$VENV" ]; then
        print_error "$VENV exists but is not a virtualenv; move it aside or use --recreate"
        exit 1
    fi
    print_info "Creating virtualenv at $VENV"
    "$PYTHON_CMD" -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

print_info "Upgrading pip"
python -m pip install --upgrade pip

print_info "Installing rms-oops with its dev extras"
python -m pip install -e ".[dev]"

echo
print_success "Virtual environment ready at $VENV"
echo -e "  Activate it with: ${BOLD}source ${VENV#"$PROJECT_ROOT"/}/bin/activate${RESET}"
echo -e "  Run the checks with: ${BOLD}./scripts/run-all-checks.sh${RESET}"

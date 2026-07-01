#!/bin/bash
# Planet Bridging — build and publish the planetbridging PyPI package.
#
#   ./publish.sh              # build + interactive upload to PyPI
#   ./publish.sh --test       # upload to TestPyPI instead
#   ./publish.sh -y             # skip confirmation (CI / scripted release)
#
# Prerequisites:
#   - PyPI API token in ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD env vars
#   - go toolchain (bundles loom-stream for this platform into the wheel)
#
# The wheel ships the Python package, bedrock POC code, multi-platform loom-stream
# binaries, and examples. Fixture npz files are generated on first use into
# ~/.planetbridging/fixtures/ (keeps the wheel under PyPI's 100MB limit).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UPLOAD_TEST=0
AUTO_YES=0
for arg in "$@"; do
  case "$arg" in
    --test) UPLOAD_TEST=1 ;;
    -y|--yes) AUTO_YES=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)"
      exit 1
      ;;
  esac
done

PKG_NAME="$(grep -E '^name = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')"
PKG_VERSION="$(grep -E '^version = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')"

echo "=== Building and Publishing ${PKG_NAME} to PyPI ==="
echo ""

pick_python() {
  for cmd in python python3; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import build, twine" 2>/dev/null; then
      echo "$cmd"
      return 0
    fi
  done
  return 1
}

if PYTHON="$(pick_python)"; then
  echo "Using $PYTHON ($($PYTHON --version 2>&1))"
else
  VENV=".publish-venv"
  echo "Packaging tools not found — bootstrapping $VENV ..."
  if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV" 2>/dev/null || python -m venv "$VENV"
  fi
  "$VENV/bin/pip" install -q -U pip build twine
  PYTHON="$VENV/bin/python"
  echo "Using $PYTHON ($($PYTHON --version 2>&1))"
fi
echo ""

if [[ ! -f "LICENSE" ]]; then
  echo "❌ LICENSE missing in $SCRIPT_DIR"
  exit 1
fi

if [[ ! -f "src/planetbridging/__init__.py" ]]; then
  echo "❌ src/planetbridging package not found"
  exit 1
fi

echo "Package: ${PKG_NAME} ${PKG_VERSION}"
echo "Bundling bedrock data + loom-stream (linux/windows/mac arm64) …"
echo ""

if [[ ! -x "$SCRIPT_DIR/scripts/prepare_pypi_bundle.sh" ]]; then
  echo "❌ scripts/prepare_pypi_bundle.sh missing"
  exit 1
fi
"$SCRIPT_DIR/scripts/prepare_pypi_bundle.sh"
echo ""

echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

echo "Building sdist + wheel..."
"$PYTHON" -m build

echo ""
echo "✓ Build complete!"
echo ""
echo "Files:"
ls -lh dist/
echo ""

if ! "$PYTHON" -m twine check dist/*; then
  echo "❌ twine check failed"
  exit 1
fi
echo "✓ Package passes twine checks"
echo ""

REPO_URL="https://pypi.org/project/${PKG_NAME}/"
if [[ "$UPLOAD_TEST" -eq 1 ]]; then
  REPO_URL="https://test.pypi.org/project/${PKG_NAME}/"
  echo "Target: TestPyPI ($REPO_URL)"
else
  echo "Target: PyPI ($REPO_URL)"
fi
echo ""

if [[ "$AUTO_YES" -eq 1 ]]; then
  REPLY=y
else
  read -r -p "Upload ${PKG_NAME} ${PKG_VERSION}? (y/N): " -n 1 REPLY
  echo
fi

if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
  echo "Upload cancelled."
  echo ""
  echo "Upload manually:"
  if [[ "$UPLOAD_TEST" -eq 1 ]]; then
    echo "  $PYTHON -m twine upload --repository testpypi dist/*"
  else
    echo "  $PYTHON -m twine upload dist/*"
  fi
  exit 0
fi

echo "Uploading..."
if [[ "$UPLOAD_TEST" -eq 1 ]]; then
  "$PYTHON" -m twine upload --repository testpypi dist/*.whl
else
  # Wheel only — sdist duplicates the bundle and can exceed PyPI 100MB limit.
  "$PYTHON" -m twine upload dist/*.whl
fi

echo ""
echo "=== Published Successfully ==="
echo "View at: $REPO_URL"
echo ""
echo "Install:"
if [[ "$UPLOAD_TEST" -eq 1 ]]; then
  echo "  pip install -i https://test.pypi.org/simple/ ${PKG_NAME}==${PKG_VERSION}"
else
  echo "  pip install ${PKG_NAME}[pytorch,welvet]"
  echo "  # stream live weights → .entity, reload with welvet"
fi
echo ""
echo "No repo clone or go build required for end users on this platform."

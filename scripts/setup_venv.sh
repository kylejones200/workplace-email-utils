#!/bin/bash
# Setup script for creating and configuring the virtual environment

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo "Setting up virtual environment in ${REPO_ROOT}..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# shellcheck source=/dev/null
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing package in editable mode (workplace_email_utils)..."
pip install -e .

# Check for FAISS on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo ""
    echo "Note: Apple Silicon detected (M1/M2/M3)."
    echo "If FAISS installation fails, try:"
    echo "  pip install faiss-cpu --no-build-isolation"
    echo "  or"
    echo "  conda install -c conda-forge faiss-cpu"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"


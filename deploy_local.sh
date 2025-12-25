#!/bin/bash
set -e

echo "=== FaceFusion Local Deployment ==="

# Find a suitable Python version
PYTHON_CMD=""
if [ -f "/opt/homebrew/bin/python3.11" ]; then
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif [ -f "/usr/bin/python3" ]; then
    PYTHON_CMD="/usr/bin/python3"
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Create virtual environment if it doesn't exist or if we want to force recreation
if [ -d "venv" ]; then
    # Check if existing venv uses the same python version
    VENV_PYTHON_VERSION=$(./venv/bin/python3 --version 2>&1 | awk '{print $2}')
    TARGET_PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    
    # Simple check major.minor match
    if [[ "$VENV_PYTHON_VERSION" != "$TARGET_PYTHON_VERSION" ]]; then
        echo "Virtual environment Python version ($VENV_PYTHON_VERSION) differs from target ($TARGET_PYTHON_VERSION). Recreating..."
        rm -rf venv
    else
        echo "Virtual environment Python version matches: $VENV_PYTHON_VERSION"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Export environment variable to fix pydantic/pyo3 build issues on newer Python versions
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Install dependencies using the provided installer script
# We use --skip-conda because we are using venv
# We use --onnxruntime default because we are on macOS
echo "Installing dependencies..."
# First, ensure we don't have broken requirements
python install.py --onnxruntime default --skip-conda

echo "Deployment setup complete."

# Launch the application
echo "Launching FaceFusion on port 8081..."
exec python facefusion.py run --server-port 8081

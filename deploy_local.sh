#!/bin/bash
set -e

echo "=== FaceFusion Local Deployment ==="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies using the provided installer script
# We use --skip-conda because we are using venv
# We use --onnxruntime default because we are on macOS
echo "Installing dependencies..."
python install.py --onnxruntime default --skip-conda

echo "Deployment setup complete."

# Launch the application
echo "Launching FaceFusion on port 8081..."
# Using exec to replace the shell process with the python process
exec python facefusion.py run --server-port 8081

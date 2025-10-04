#!/bin/bash
# GameDash Setup Script

echo "Setting up GameDash - Gaming Dashboard for 320x1480 displays..."

# Create virtual environment
if [ ! -d "gamedash_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv gamedash_venv
fi

# Activate and install dependencies
echo "Installing Python dependencies..."
source gamedash_venv/bin/activate
pip install -r requirements.txt
deactivate

# Make script executable
chmod +x gamedash.py

# Make run script executable
chmod +x run.sh

# Create symlink in user bin (optional)
if [ ! -f ~/.local/bin/gamedash ]; then
    mkdir -p ~/.local/bin
    ln -s "$(pwd)/run.sh" ~/.local/bin/gamedash
    echo "Created symlink: ~/.local/bin/gamedash"
    echo "Make sure ~/.local/bin is in your PATH"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  ./run.sh             # Run from current directory"
echo "  gamedash             # Run from anywhere (if ~/.local/bin is in PATH)"
echo ""
echo "For 320x1480 display, run in a terminal sized to fit your monitor:"
echo "  alacritty --dimensions 40x100 -e ./run.sh"
echo "  or resize your terminal to match the display"

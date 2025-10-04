# GameDash

A terminal-based gaming dashboard optimized for horizontal small monitors (1480x320 landscape displays). Shows real-time GPU metrics, FPS, and game-related stats with beautiful ASCII graphs.

## Features

- **Real-time GPU Monitoring**: Supports both AMD and NVIDIA graphics cards
- **Dual-line Graph**: Visualizes GPU usage and memory usage over time
- **FPS Tracking**: Displays current, average, min, and max FPS with frame time
- **System Stats**: CPU usage, RAM usage, GPU temperature, clock speed, and power consumption
- **Process Detection**: Shows the top process using the GPU
- **Optimized Display**: Designed for 1480x320 landscape displays (works on any terminal size)

## Screenshots

The dashboard displays:
- Large dual-line graph showing GPU usage (cyan) and GPU memory (yellow) over time
- Real-time meters for GPU, memory, CPU, and temperature
- FPS statistics and sparkline graph
- GPU clock speed and power consumption

## Requirements

- Python 3.7+
- Linux OS (tested on Ubuntu/Debian-based systems)
- AMD or NVIDIA GPU (optional, will show N/A if not detected)
- Terminal emulator that supports Unicode and colors

## Installation

1. Clone or download this repository
2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all required dependencies
- Make scripts executable
- Create a launcher script
- Optionally create a symlink in `~/.local/bin` for easy access

## Usage

After installation, you can run GameDash in several ways:

### Run from the installation directory:
```bash
./run.sh
```

### Run from anywhere (if symlink was created):
```bash
gamedash
```

### For 320x1480 displays with specific terminal:
```bash
alacritty --dimensions 40x100 -e ./run.sh
```

Or just resize your terminal window to fit your horizontal monitor.

## GPU Support

### AMD GPUs
- Uses sysfs interface (no additional drivers needed)
- Reads from `/sys/class/drm/card*/device/`
- Supports all modern AMD GPUs with amdgpu driver

### NVIDIA GPUs
- Uses pynvml library (nvidia-ml-py)
- Requires NVIDIA drivers to be installed
- Supports GPU usage, memory, temperature, power, and clock monitoring

## Dependencies

All dependencies are listed in `requirements.txt`:
- textual - TUI framework
- psutil - System and process monitoring
- nvidia-ml-py - NVIDIA GPU monitoring (optional)
- rich - Terminal formatting and styling

## Customization

You can modify `gamedash.py` to:
- Adjust graph update frequency (default: 30Hz)
- Change graph history length (default: 120 points)
- Customize colors and styling in the CSS section
- Adjust meter thresholds and colors

## Troubleshooting

### GPU not detected
- For AMD: Check if `/sys/class/drm/card*/device/gpu_busy_percent` exists
- For NVIDIA: Ensure NVIDIA drivers are installed and `nvidia-smi` works

### Permission issues
- Some GPU metrics may require root access
- Try running with `sudo` if certain stats show as 0

### Display issues
- Ensure your terminal supports Unicode characters
- Try a different terminal emulator (alacritty, kitty, gnome-terminal)
- Increase terminal font size if text is too small

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## Author

Created for horizontal small monitor gaming setups.

## Acknowledgments

- Built with [Textual](https://github.com/Textualize/textual) TUI framework
- Uses [Rich](https://github.com/Textualize/rich) for terminal formatting

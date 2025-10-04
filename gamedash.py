#!/usr/bin/env python3
"""
GameDash - Terminal-based gaming dashboard optimized for 1480x320 landscape displays
Shows GPU metrics, FPS, and game-related stats in real-time
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Label
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.console import Console
from rich.table import Table
import psutil
import time
from collections import deque
from datetime import datetime
import subprocess
import os
import glob

# Try to import GPU libraries
try:
    import pynvml
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False


class GPUStats:
    """GPU statistics collector - supports both AMD and NVIDIA"""

    def __init__(self):
        self.has_gpu = False
        self.gpu_name = "N/A"
        self.gpu_type = None  # 'nvidia' or 'amd'
        self.top_process = "N/A"

        # Try AMD first (sysfs)
        self._init_amd()

        # If AMD not found, try NVIDIA
        if not self.has_gpu and HAS_NVIDIA:
            self._init_nvidia()

    def _init_amd(self):
        """Initialize AMD GPU monitoring via sysfs"""
        try:
            # Find AMD GPU card (look for card with gpu_busy_percent)
            for card_path in glob.glob('/sys/class/drm/card?'):
                busy_file = os.path.join(card_path, 'device', 'gpu_busy_percent')
                if os.path.exists(busy_file):
                    self.card_path = os.path.join(card_path, 'device')

                    # Get GPU name from lspci
                    try:
                        uevent_file = os.path.join(self.card_path, 'uevent')
                        with open(uevent_file, 'r') as f:
                            for line in f:
                                if 'PCI_SLOT_NAME' in line:
                                    pci_id = line.split('=')[1].strip()
                                    result = subprocess.run(['lspci', '-s', pci_id],
                                                          capture_output=True, text=True, timeout=1)
                                    if result.returncode == 0:
                                        # Parse GPU name from lspci output
                                        # Example: "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [RX 9070/9070 XT] (rev c0)"
                                        gpu_line = result.stdout.strip()
                                        # Find the part after "[AMD/ATI]"
                                        if '[AMD/ATI]' in gpu_line:
                                            after_amd = gpu_line.split('[AMD/ATI]')[1].strip()
                                            # Extract content in brackets if present
                                            if '[' in after_amd and ']' in after_amd:
                                                self.gpu_name = after_amd[after_amd.find('[')+1:after_amd.find(']')]
                                            else:
                                                self.gpu_name = after_amd.split('(')[0].strip()
                                        else:
                                            # Fallback: just take everything after the third colon
                                            parts = gpu_line.split(':')
                                            if len(parts) >= 3:
                                                self.gpu_name = ':'.join(parts[2:]).split('(')[0].strip()
                    except:
                        self.gpu_name = "AMD GPU"

                    # Find hwmon directory
                    hwmon_base = os.path.join(self.card_path, 'hwmon')
                    if os.path.exists(hwmon_base):
                        hwmon_dirs = glob.glob(os.path.join(hwmon_base, 'hwmon*'))
                        if hwmon_dirs:
                            self.hwmon_path = hwmon_dirs[0]

                    self.has_gpu = True
                    self.gpu_type = 'amd'
                    break
        except:
            pass

    def _init_nvidia(self):
        """Initialize NVIDIA GPU monitoring"""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            self.has_gpu = True
            self.gpu_type = 'nvidia'
        except:
            pass

    def _read_sysfs(self, path):
        """Read a sysfs file and return integer value"""
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def get_stats(self):
        """Get current GPU statistics"""
        if not self.has_gpu:
            return {
                'usage': 0,
                'memory_used': 0,
                'memory_total': 0,
                'temp': 0,
                'power': 0,
                'clock': 0
            }

        if self.gpu_type == 'amd':
            return self._get_amd_stats()
        elif self.gpu_type == 'nvidia':
            return self._get_nvidia_stats()

        return {'usage': 0, 'memory_used': 0, 'memory_total': 0, 'temp': 0, 'power': 0, 'clock': 0}

    def _get_amd_stats(self):
        """Get AMD GPU stats from sysfs"""
        try:
            usage = self._read_sysfs(os.path.join(self.card_path, 'gpu_busy_percent'))

            mem_used = self._read_sysfs(os.path.join(self.card_path, 'mem_info_vram_used'))
            mem_total = self._read_sysfs(os.path.join(self.card_path, 'mem_info_vram_total'))

            temp = 0
            if hasattr(self, 'hwmon_path'):
                temp = self._read_sysfs(os.path.join(self.hwmon_path, 'temp1_input')) / 1000

            power = 0
            if hasattr(self, 'hwmon_path'):
                power = self._read_sysfs(os.path.join(self.hwmon_path, 'power1_average')) / 1000000.0

            # Get current GPU clock
            clock = 0
            try:
                sclk_file = os.path.join(self.card_path, 'pp_dpm_sclk')
                with open(sclk_file, 'r') as f:
                    for line in f:
                        if '*' in line:  # Current clock marked with *
                            parts = line.split(':')
                            if len(parts) > 1:
                                clock_str = parts[1].strip().replace('Mhz', '').replace('*', '').strip()
                                clock = int(clock_str)
                            break
            except:
                pass

            return {
                'usage': usage,
                'memory_used': mem_used // (1024**2),  # Convert to MB
                'memory_total': mem_total // (1024**2),  # Convert to MB
                'temp': int(temp),
                'power': power,
                'clock': clock
            }
        except:
            return {'usage': 0, 'memory_used': 0, 'memory_total': 0, 'temp': 0, 'power': 0, 'clock': 0}

    def _get_nvidia_stats(self):
        """Get NVIDIA GPU stats"""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            except:
                power = 0

            try:
                clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            except:
                clock = 0

            return {
                'usage': util.gpu,
                'memory_used': mem_info.used // (1024**2),  # MB
                'memory_total': mem_info.total // (1024**2),  # MB
                'temp': temp,
                'power': power,
                'clock': clock
            }
        except:
            return {'usage': 0, 'memory_used': 0, 'memory_total': 0, 'temp': 0, 'power': 0, 'clock': 0}

    def get_top_process(self):
        """Get the top process using the GPU"""
        if not self.has_gpu:
            return "N/A"

        process_name = ""
        if self.gpu_type == 'nvidia':
            process_name = self._get_nvidia_top_process()
        elif self.gpu_type == 'amd':
            process_name = self._get_amd_top_process()
        else:
            return "N/A"

        # Remove underscores from process name
        return process_name.replace('_', '')

    def _get_nvidia_top_process(self):
        """Get top NVIDIA GPU process"""
        try:
            # Try graphics processes first (games, etc)
            try:
                procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
            except:
                # Fall back to compute processes
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)

            if procs:
                # Get the process with most memory usage
                top_proc = max(procs, key=lambda p: p.usedGpuMemory)
                pid = top_proc.pid
                # Get process name
                try:
                    with open(f'/proc/{pid}/comm', 'r') as f:
                        return f.read().strip()
                except:
                    return f"PID {pid}"
        except:
            pass
        return "N/A"

    def _get_amd_top_process(self):
        """Get top AMD GPU process (via top CPU usage)"""
        try:
            # Get all processes sorted by CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    # Filter out system processes and low usage
                    if proc.info['cpu_percent'] > 1.0:
                        processes.append((proc.info['name'], proc.info['cpu_percent']))
                except:
                    continue

            if processes:
                # Sort by CPU usage and return top process
                processes.sort(key=lambda x: x[1], reverse=True)
                return processes[0][0]
        except:
            pass
        return "N/A"


class FPSMonitor:
    """FPS monitoring (simulated for now, can be extended with MangoHud integration)"""

    def __init__(self):
        self.fps_history = deque(maxlen=60)
        self.frame_times = deque(maxlen=60)
        self.last_update = time.time()

    def get_fps(self):
        """Get current FPS (simulated based on GPU usage for demo)"""
        # In a real implementation, this would read from MangoHud or game APIs
        # For now, we'll simulate based on time
        current_time = time.time()
        delta = current_time - self.last_update
        self.last_update = current_time

        # Simulated FPS (replace with actual FPS reading)
        import random
        fps = random.randint(55, 144)  # Placeholder
        self.fps_history.append(fps)
        self.frame_times.append(1000.0 / fps if fps > 0 else 0)

        return {
            'current': fps,
            'avg': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
            'min': min(self.fps_history) if self.fps_history else 0,
            'max': max(self.fps_history) if self.fps_history else 0,
            'frametime': self.frame_times[-1] if self.frame_times else 0
        }


class MeterWidget(Static):
    """A meter widget showing percentage with bar"""

    value = reactive(0)
    label_text = reactive("")

    def __init__(self, label: str, max_val: int = 100, suffix: str = "%", **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.max_val = max_val
        self.suffix = suffix

    def render(self) -> str:
        # Create a horizontal bar - compact single line
        bar_width = 15
        filled = int((self.value / self.max_val) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on value
        if self.value > 80:
            color = "red"
        elif self.value > 60:
            color = "yellow"
        else:
            color = "green"

        return f"[bold]{self.label_text}[/bold] [{color}]{bar}[/{color}] {self.value:.0f}{self.suffix}"


class GraphWidget(Static):
    """A simple ASCII graph widget"""

    def __init__(self, label: str, max_points: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.data = deque(maxlen=max_points)
        self.max_points = max_points

    def add_point(self, value: float):
        self.data.append(value)
        self.refresh()

    def render(self) -> str:
        if not self.data:
            return f"[bold]{self.label}[/bold]\n" + "░" * 30

        # Create a simple sparkline - compact
        height = 4
        max_val = max(self.data) if self.data else 1
        min_val = min(self.data) if self.data else 0
        range_val = max_val - min_val if max_val != min_val else 1

        # Build graph
        lines = []
        for h in range(height, 0, -1):
            line = ""
            threshold = min_val + (range_val * h / height)
            for val in self.data:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)

        graph = "\n".join(lines)
        return f"{graph}\n\n[bold]{self.label}[/bold] Max:{max_val:.0f}"


class DualLineGraphWidget(Static):
    """A dual-line graph widget for displaying two metrics"""

    def __init__(self, label: str, line1_name: str, line2_name: str, max_points: int = 120, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.line1_name = line1_name
        self.line2_name = line2_name
        self.line1_data = deque(maxlen=max_points)
        self.line2_data = deque(maxlen=max_points)
        self.max_points = max_points
        self.process_name = "N/A"

    def add_points(self, value1: float, value2: float, process_name: str = "N/A"):
        self.line1_data.append(value1)
        self.line2_data.append(value2)
        self.process_name = process_name
        self.refresh()

    def render(self) -> str:
        if not self.line1_data and not self.line2_data:
            return f"[bold]{self.label}[/bold]\n" + "░" * 80

        # Graph dimensions - reduced for compact display
        height = 9
        width = len(self.line1_data) if self.line1_data else 1

        # Find max values for scaling (fixed at 100 for percentage)
        max_val = 100
        min_val = 0
        range_val = 100

        # Build graph with both lines - always show both
        lines = []
        for h in range(height, 0, -1):
            line = ""
            threshold = min_val + (range_val * h / height)
            threshold_next = min_val + (range_val * (h-1) / height) if h > 1 else min_val

            for i in range(max(len(self.line1_data), len(self.line2_data))):
                val1 = self.line1_data[i] if i < len(self.line1_data) else 0
                val2 = self.line2_data[i] if i < len(self.line2_data) else 0

                val1_above = val1 >= threshold
                val2_above = val2 >= threshold

                # Draw both lines with blocks
                if val1_above and val2_above:
                    line += "[white]█[/white]"  # Both above
                elif val1_above:
                    line += "[cyan]█[/cyan]"  # GPU Usage
                elif val2_above:
                    line += "[yellow]█[/yellow]"  # Memory
                else:
                    line += " "

            # Add vertical scale labels
            line_index = height - h
            if line_index == 0:  # Top line
                lines.append(f"100% {line}")
            elif line_index == height // 2:  # Middle line
                lines.append(f" 50% {line}")
            elif line_index == height - 1:  # Bottom line
                lines.append(f"  0% {line}")
            else:
                lines.append(f"     {line}")

        graph = "\n".join(lines)
        legend = f"[cyan]█[/cyan] {self.line1_name} [yellow]█[/yellow] {self.line2_name}"
        process_line = f"Process: {self.process_name}"

        return f"{graph}\n\n{legend}\n\n{process_line}"


class GameDashboard(App):
    """Main dashboard application"""

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        display: none;
    }

    Footer {
        display: none;
    }

    #main_container {
        width: 100%;
        height: 100%;
    }

    #graph_panel {
        width: 75%;
        height: 100%;
        padding: 0;
    }

    #stats_panel {
        width: 25%;
        height: 100%;
        padding: 0 0 0 1;
    }

    #dual_graph {
        height: 1fr;
        width: 100%;
    }

    #gpu_name {
        height: 1;
        width: 100%;
        padding: 0 0 0 1;
    }

    #gpu_meter, #mem_meter, #cpu_meter, #temp_meter {
        height: 1;
        margin: 0;
    }

    #fps_display {
        height: 1;
        margin: 0 0 1 0;
        padding: 0;
    }

    #fps_graph {
        height: 6;
        margin: 0;
    }

    .stat_line {
        height: 1;
        margin: 0;
    }
    """

    TITLE = "GameDash"
    SUB_TITLE = "Gaming Dashboard"

    def __init__(self):
        super().__init__()
        self.gpu_stats = GPUStats()
        self.fps_monitor = FPSMonitor()
        self.dual_graph = None
        self.fps_graph = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main_container"):
            # Left side: Large dual-line graph (GPU Usage + GPU Memory)
            with Container(id="graph_panel"):
                # GPU Name display
                gpu_name_text = f"[bold cyan]{self.gpu_stats.gpu_name}[/bold cyan]"
                yield Static(gpu_name_text, id="gpu_name")
                yield Static("")  # Blank line

                self.dual_graph = DualLineGraphWidget(
                    "GPU Metrics History",
                    "GPU Usage",
                    "GPU Memory",
                    max_points=120,
                    id="dual_graph"
                )
                yield self.dual_graph

            # Right side: Stats and meters
            with Vertical(id="stats_panel"):
                # FPS Display
                yield Static("", id="fps_display")

                # Meters
                yield MeterWidget("GPU ", id="gpu_meter")
                yield MeterWidget("MEM ", id="mem_meter")
                yield MeterWidget("CPU ", id="cpu_meter")
                yield MeterWidget("TEMP", max_val=100, suffix="°C", id="temp_meter")

                yield Static("")  # Blank line

                # FPS Graph
                self.fps_graph = GraphWidget("FPS", max_points=35)
                yield self.fps_graph

                yield Static("")  # Blank line

                # Additional stats
                yield Static("", id="stats_line1", classes="stat_line")
                yield Static("", id="stats_line2", classes="stat_line")

    def on_mount(self) -> None:
        self.set_interval(1/30, self.update_stats)  # 30Hz update

    def update_stats(self) -> None:
        # Get GPU stats
        gpu_data = self.gpu_stats.get_stats()
        fps_data = self.fps_monitor.get_fps()
        top_process = self.gpu_stats.get_top_process()

        # Update meters
        self.query_one("#gpu_meter", MeterWidget).value = gpu_data['usage']

        mem_percent = (gpu_data['memory_used'] / gpu_data['memory_total'] * 100) if gpu_data['memory_total'] > 0 else 0
        self.query_one("#mem_meter", MeterWidget).value = mem_percent

        cpu_percent = psutil.cpu_percent()
        self.query_one("#cpu_meter", MeterWidget).value = cpu_percent

        self.query_one("#temp_meter", MeterWidget).value = gpu_data['temp']

        # Update FPS display
        fps_text = f"[bold cyan]FPS:{fps_data['current']}[/bold cyan] Avg:{fps_data['avg']:.0f} Min:{fps_data['min']} Max:{fps_data['max']} ({fps_data['frametime']:.1f}ms)"
        self.query_one("#fps_display", Static).update(fps_text)

        # Update dual-line graph (GPU Usage + GPU Memory)
        if self.dual_graph:
            self.dual_graph.add_points(gpu_data['usage'], mem_percent, top_process)

        # Update FPS graph
        if self.fps_graph:
            self.fps_graph.add_point(fps_data['current'])

        # Update stats lines
        ram = psutil.virtual_memory()
        stats_text1 = f"GPU: {gpu_data['usage']:.0f}%  Mem: {gpu_data['memory_used']}MB/{gpu_data['memory_total']}MB  Temp: {gpu_data['temp']}°C"
        stats_text2 = f"RAM: {ram.percent:.1f}%  Clk: {gpu_data['clock']}MHz  Pwr: {gpu_data['power']:.1f}W"
        self.query_one("#stats_line1", Static).update(stats_text1)
        self.query_one("#stats_line2", Static).update(stats_text2)


def main():
    app = GameDashboard()
    app.run()


if __name__ == "__main__":
    main()

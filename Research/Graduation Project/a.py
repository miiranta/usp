import os
import platform
import psutil
import socket
import subprocess
import cpuinfo
import GPUtil
from pathlib import Path

def get_size(bytes, suffix="B"):
    """Scale bytes to KB, MB, GB, etc."""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

# =======================
#  SYSTEM INFORMATION
# =======================
def system_info():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Python Version: {platform.python_version()}")

# =======================
#  CPU INFORMATION
# =======================
def cpu_info():
    print("\n" + "="*40, "CPU Information", "="*40)
    info = cpuinfo.get_cpu_info()
    print(f"Brand: {info.get('brand_raw', 'Unknown')}")
    print(f"Arch: {info.get('arch', 'Unknown')}")
    print(f"Bits: {info.get('bits', 'Unknown')}")
    print(f"Cores: {psutil.cpu_count(logical=False)}")
    print(f"Threads: {psutil.cpu_count(logical=True)}")
    freq = psutil.cpu_freq()
    if freq:
        print(f"Max Frequency: {freq.max:.2f} MHz")
        print(f"Current Frequency: {freq.current:.2f} MHz")
    print(f"CPU Usage Per Core: {psutil.cpu_percent(percpu=True)}")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

# =======================
#  MEMORY INFORMATION
# =======================
def get_ram_type():
    """Try to detect RAM type without sudo (heuristic)."""
    # Try /sys/class/dmi/id/memory_* (may work without sudo)
    mem_dirs = list(Path("/sys/devices/system/memory/").glob("memory*"))
    if mem_dirs:
        for path in mem_dirs:
            device = Path(f"/sys/devices/system/memory/{path.name}/device")
            if device.exists():
                type_file = device / "type"
                if type_file.exists():
                    try:
                        ram_type = type_file.read_text().strip()
                        if ram_type:
                            return ram_type
                    except Exception:
                        pass

    # Try /proc/meminfo and guess
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        if "DDR5" in meminfo:
            return "DDR5"
        elif "DDR4" in meminfo:
            return "DDR4"
        elif "DDR3" in meminfo:
            return "DDR3"
    except:
        pass

    # Fallback heuristic based on CPU age
    brand = cpuinfo.get_cpu_info().get("brand_raw", "").lower()
    if "ryzen 7" in brand or "intel core i9" in brand or "13th" in brand:
        return "Likely DDR5"
    elif "ryzen 5" in brand or "intel core i7" in brand or "10th" in brand:
        return "Likely DDR4"
    else:
        return "Unknown (no root access)"

def memory_info():
    print("\n" + "="*40, "Memory Information", "="*40)
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

    swap = psutil.swap_memory()
    print("\nSwap Memory:")
    print(f"Total: {get_size(swap.total)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Percentage: {swap.percent}%")

    print("\nDetected RAM Type:", get_ram_type())

# =======================
#  DISK INFORMATION
# =======================
def disk_info():
    print("\n" + "="*40, "Disk Information", "="*40)
    partitions = psutil.disk_partitions()
    total_space = 0
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        print(f"  Total Size: {get_size(usage.total)}")
        print(f"  Used: {get_size(usage.used)}")
        print(f"  Free: {get_size(usage.free)}")
        print(f"  Percentage: {usage.percent}%")
        total_space += usage.total
    print(f"\nTotal Disk Space Across All Drives: {get_size(total_space)}")

# =======================
#  NETWORK INFORMATION
# =======================
def network_info():
    print("\n" + "="*40, "Network Information", "="*40)
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.gaierror:
        ip_address = "Unknown"
    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip_address}")
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

# =======================
#  GPU INFORMATION
# =======================
def gpu_info():
    print("\n" + "="*40, "GPU Information", "="*40)
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU detected.")
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
            print(f"  Load: {gpu.load * 100:.1f}%")
            print(f"  Free Memory: {get_size(gpu.memoryFree * 1024**2)}")
            print(f"  Used Memory: {get_size(gpu.memoryUsed * 1024**2)}")
            print(f"  Total Memory: {get_size(gpu.memoryTotal * 1024**2)}")
            print(f"  Temperature: {gpu.temperature} °C")
    except Exception as e:
        print("GPU info unavailable:", e)

# =======================
#  MAIN EXECUTION
# =======================
def main():
    system_info()
    cpu_info()
    memory_info()
    disk_info()
    network_info()
    gpu_info()

if __name__ == "__main__":
    main()

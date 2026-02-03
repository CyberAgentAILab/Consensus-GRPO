import os
import shutil
import time
import psutil
import subprocess
import datetime
import argparse

def get_memory_data():
    mem = psutil.virtual_memory()
    return {
        "used_gb": mem.used / (1024**3),
        "total_gb": mem.total / (1024**3),
        "percent": mem.percent
    }

def get_disk_data(path):
    if os.path.ismount(path):
        total, used, free = shutil.disk_usage(path)
        return {
            "path": path,
            "used_gb": used / (1024**3),
            "total_gb": total / (1024**3),
            "percent": (used / total) * 100
        }
    return {
        "path": path,
        "used_gb": 0,
        "total_gb": 0,
        "percent": 0,
        "error": "not found"
    }

def get_vram_data():
    """Get raw GPU VRAM usage data"""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(", "))
            return {
                "used_mb": used,
                "total_mb": total,
                "percent": (used / total) * 100
            }
    except Exception:
        pass
    
    return {
        "used_mb": 0,
        "total_mb": 0,
        "percent": 0,
        "error": "unavailable"
    }

def format_detailed(memory_data, disk_data_list, vram_data):
    log_entry = "-" * 50 + "\n"
    log_entry += f"RAM Usage: {memory_data['used_gb']:.2f} GB / {memory_data['total_gb']:.2f} GB ({memory_data['percent']}%)\n"
    
    for disk_data in disk_data_list:
        if "error" in disk_data:
            log_entry += f"Disk ({disk_data['path']}) not found.\n"
        else:
            log_entry += f"Disk ({disk_data['path']}) Usage: {disk_data['used_gb']:.2f} GB / {disk_data['total_gb']:.2f} GB ({disk_data['percent']:.2f}%)\n"
    
    if "error" in vram_data:
        log_entry += "GPU VRAM Usage: Unable to retrieve. Is an NVIDIA GPU available?\n"
    else:
        log_entry += f"GPU VRAM Usage: {vram_data['used_mb']} MB / {vram_data['total_mb']} MB ({vram_data['percent']:.2f}%)\n"
    
    log_entry += "-" * 50 + "\n"
    return log_entry

def format_summary(memory_data, disk_data_list, vram_data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"[{timestamp}] RAM: {memory_data['used_gb']:.2f}/{memory_data['total_gb']:.2f}GB ({memory_data['percent']}%) | "
    
    disk_parts = []
    for disk_data in disk_data_list:
        if "error" in disk_data:
            disk_parts.append(f"Disk({disk_data['path']}): N/A")
        else:
            disk_parts.append(f"Disk({disk_data['path']}): {disk_data['used_gb']:.2f}/{disk_data['total_gb']:.2f}GB ({disk_data['percent']:.2f}%)")
    
    summary += " | ".join(disk_parts)
    
    if "error" in vram_data:
        summary += " | GPU: N/A"
    else:
        summary += f" | GPU: {vram_data['used_mb']}/{vram_data['total_mb']}MB ({vram_data['percent']:.2f}%)"
    
    return summary + "\n"

def get_csv_header(disk_paths):
    header = ["timestamp", "ram_used_gb", "ram_total_gb", "ram_percent"]
    
    for path in disk_paths:
        path_safe = path.replace("/", "_").strip("_")
        if not path_safe:
            path_safe = "root"
        header.extend([
            f"disk_{path_safe}_used_gb",
            f"disk_{path_safe}_total_gb", 
            f"disk_{path_safe}_percent"
        ])
    
    header.extend(["gpu_vram_used_mb", "gpu_vram_total_mb", "gpu_vram_percent"])
    return ",".join(header)

def format_csv(memory_data, disk_data_list, vram_data, disk_paths):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        timestamp,
        f"{memory_data['used_gb']:.2f}",
        f"{memory_data['total_gb']:.2f}",
        f"{memory_data['percent']:.1f}"
    ]
    
    disk_data_dict = {d["path"]: d for d in disk_data_list}
    
    for path in disk_paths:
        if path in disk_data_dict and "error" not in disk_data_dict[path]:
            disk_data = disk_data_dict[path]
            row.extend([
                f"{disk_data['used_gb']:.2f}",
                f"{disk_data['total_gb']:.2f}",
                f"{disk_data['percent']:.1f}"
            ])
        else:
            row.extend(["N/A", "N/A", "N/A"])
    
    if "error" in vram_data:
        row.extend(["N/A", "N/A", "N/A"])
    else:
        row.extend([
            str(vram_data["used_mb"]),
            str(vram_data["total_mb"]),
            f"{vram_data['percent']:.1f}"
        ])
    
    return ",".join(row)

def log_resources(disk_paths=None, interval=5, log_file="resource_log.txt", format_type="detailed", quiet=False):

    if disk_paths is None:
        disk_paths = ['/', '/data']
    
    write_header = format_type == "csv" and not os.path.exists(log_file)
    
    with open(log_file, "a") as f:
        if write_header:
            header = get_csv_header(disk_paths)
            f.write(header + "\n")
            if not quiet:
                print(header)
        
        while True:
            memory_data = get_memory_data()
            disk_data_list = [get_disk_data(path) for path in disk_paths]
            vram_data = get_vram_data()
            
            log_entry = ""
            
            if format_type == "detailed":
                log_entry = format_detailed(memory_data, disk_data_list, vram_data)
            elif format_type == "summary":
                log_entry = format_summary(memory_data, disk_data_list, vram_data)
            elif format_type == "csv":
                log_entry = format_csv(memory_data, disk_data_list, vram_data, disk_paths) + "\n"
            else:
                log_entry = format_detailed(memory_data, disk_data_list, vram_data)
            
            if not quiet:
                print(log_entry, end="")
            
            f.write(log_entry)
            f.flush()
            
            time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor and log system resources.")
    parser.add_argument("--disk_paths", nargs="+", default=None, help="List of disk paths to monitor.")
    parser.add_argument("--interval", type=float, default=5, help="Time interval between logging in seconds.")
    parser.add_argument("--log_file", type=str, default="resource_log.txt", help="Path to log file.")
    parser.add_argument("--format", type=str, choices=["detailed", "summary", "csv"], default="detailed",
                        help="Output format: detailed (multi-line), summary (single-line), or csv.")
    parser.add_argument("--quiet", action="store_true", 
                        help="Suppress output to stdout (only write to log file).")
    args = parser.parse_args()

    log_resources(args.disk_paths, args.interval, args.log_file, args.format, args.quiet)

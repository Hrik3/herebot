"""
system_stats.py - Lightweight library for instant host disk, CPU, and RAM usage.

This module provides fast, cross-platform system statistics using `psutil`.
All functions are designed to be called repeatedly with minimal overhead.

Dependencies:
    psutil (install via: pip install psutil)

Example:
    from system_stats import get_cpu_usage, get_ram_usage, get_disk_usage

    print("CPU:", get_cpu_usage())
    print("RAM:", get_ram_usage())
    print("Disk:", get_disk_usage('/'))
"""

import os
import time
from functools import lru_cache

try:
    import psutil
except ImportError:
    raise ImportError(
        "psutil is required for system_stats. "
        "Install it with: pip install psutil"
    )


# ----------------------------------------------------------------------
#  CPU Usage
# ----------------------------------------------------------------------
def get_cpu_usage(interval: float = 0.1, per_cpu: bool = False):
    """
    Return CPU usage as a percentage.

    Args:
        interval: Sampling interval in seconds. 
                  Use 0.0 for instantaneous (may be less accurate on some OSes).
        per_cpu: If True, return list of per-core percentages.

    Returns:
        float or list[float]: Overall CPU usage percentage, or list per CPU.
    """
    # psutil.cpu_percent(interval) measures overall system CPU usage
    # over the given interval.
    if per_cpu:
        return psutil.cpu_percent(interval=interval, percpu=True)
    else:
        return psutil.cpu_percent(interval=interval)


def get_cpu_count(logical: bool = True) -> int:
    """Return number of CPU cores."""
    return psutil.cpu_count(logical=logical)


def get_cpu_freq():
    """
    Return current CPU frequency as a dict with keys:
        current, min, max (in MHz).
    May return empty dict if not available.
    """
    freq = psutil.cpu_freq()
    if freq:
        return {
            "current": freq.current,
            "min": freq.min,
            "max": freq.max,
        }
    return {}


def get_cpu_stats():
    """
    Return a dictionary with comprehensive CPU statistics.
    Includes usage, count, frequency, and load average (if available).
    """
    stats = {
        "percent": get_cpu_usage(interval=0.1),
        "per_cpu_percent": get_cpu_usage(interval=0.1, per_cpu=True),
        "logical_count": get_cpu_count(logical=True),
        "physical_count": get_cpu_count(logical=False),
        "frequency": get_cpu_freq(),
    }
    # Load average (Unix only, returns (1min, 5min, 15min))
    if hasattr(psutil, "getloadavg"):
        try:
            stats["load_avg"] = psutil.getloadavg()
        except OSError:
            stats["load_avg"] = None
    else:
        stats["load_avg"] = None
    return stats


# ----------------------------------------------------------------------
#  RAM / Memory Usage
# ----------------------------------------------------------------------
def get_ram_usage():
    """
    Return virtual memory usage as a dictionary with keys:
        total, available, percent, used, free, active, inactive, buffers, cached,
        shared, slab (platform-dependent).
    Units are in bytes.
    """
    mem = psutil.virtual_memory()
    # Convert to dict for easy access
    return mem._asdict()


def get_swap_usage():
    """Return swap memory usage as a dictionary (total, used, free, percent)."""
    swap = psutil.swap_memory()
    return swap._asdict()


def get_memory_stats():
    """Return both RAM and swap statistics."""
    return {
        "ram": get_ram_usage(),
        "swap": get_swap_usage(),
    }


# ----------------------------------------------------------------------
#  Disk Usage
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _disk_usage_cached(path):
    """Cached version of disk_usage to avoid repeated syscalls for same path."""
    return psutil.disk_usage(path)


def get_disk_usage(path: str = "/", use_cache: bool = True):
    """
    Return disk usage statistics for the given path.
    Args:
        path: Filesystem path (e.g., '/', 'C:\\')
        use_cache: If True, cache results for 2 seconds (fast repeated calls).

    Returns:
        Dictionary with keys: total, used, free, percent.
    """
    if use_cache:
        # Simple time‑based cache: clear after 2 seconds.
        # Since lru_cache has no TTL, we manually check.
        now = time.monotonic()
        if not hasattr(get_disk_usage, "_cache_timestamp") or \
           now - get_disk_usage._cache_timestamp > 2.0:
            get_disk_usage._cache_timestamp = now
            get_disk_usage._cache_path = path
            get_disk_usage._cache_value = psutil.disk_usage(path)._asdict()
        elif get_disk_usage._cache_path != path:
            # Different path, refresh
            get_disk_usage._cache_path = path
            get_disk_usage._cache_value = psutil.disk_usage(path)._asdict()
        return get_disk_usage._cache_value
    else:
        return psutil.disk_usage(path)._asdict()


def get_disk_partitions(all: bool = False):
    """
    Return list of mounted disk partitions.
    Args:
        all: If True, return all partitions (including loop, fake, etc.)
    Returns:
        List of dicts with keys: device, mountpoint, fstype, opts.
    """
    partitions = psutil.disk_partitions(all=all)
    return [p._asdict() for p in partitions]


def get_disk_io_counters(perdisk: bool = False):
    """
    Return disk I/O statistics.
    Args:
        perdisk: If True, return dict per disk, else aggregate.
    Returns:
        Dictionary with read_count, write_count, read_bytes, write_bytes, etc.
    """
    io = psutil.disk_io_counters(perdisk=perdisk)
    if io is None:
        return {} if not perdisk else {}
    if perdisk:
        return {disk: counters._asdict() for disk, counters in io.items()}
    else:
        return io._asdict()


def get_disk_stats(path: str = "/"):
    """Return combined disk usage and I/O for the given path."""
    return {
        "usage": get_disk_usage(path),
        "io": get_disk_io_counters(perdisk=False),
    }


# ----------------------------------------------------------------------
#  Combined / Convenience
# ----------------------------------------------------------------------
def get_all_stats():
    """
    Return a dictionary with CPU, memory, and disk stats for the root partition.
    Useful for a quick snapshot.
    """
    return {
        "timestamp": time.time(),
        "cpu": get_cpu_stats(),
        "memory": get_memory_stats(),
        "disk": get_disk_stats("/"),
    }


# ----------------------------------------------------------------------
#  Simple CLI test (if run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=== System Statistics (instant) ===")
    print("\n--- CPU ---")
    print(f"CPU Usage: {get_cpu_usage(interval=0.1):.1f}%")
    print(f"Per Core: {get_cpu_usage(interval=0.1, per_cpu=True)}")
    print(f"CPU Count (logical): {get_cpu_count(logical=True)}")
    print(f"CPU Count (physical): {get_cpu_count(logical=False)}")
    print(f"CPU Frequency: {get_cpu_freq()}")

    print("\n--- Memory ---")
    ram = get_ram_usage()
    print(f"RAM: {ram['used'] / 2**30:.2f} GB / {ram['total'] / 2**30:.2f} GB ({ram['percent']:.1f}%)")
    swap = get_swap_usage()
    print(f"Swap: {swap['used'] / 2**30:.2f} GB / {swap['total'] / 2**30:.2f} GB ({swap['percent']:.1f}%)")

    print("\n--- Disk (/) ---")
    disk = get_disk_usage('/')
    print(f"Disk: {disk['used'] / 2**30:.2f} GB / {disk['total'] / 2**30:.2f} GB ({disk['percent']:.1f}%)")

    # Full dump
    # print("\n--- Full Stats ---")
    # print(json.dumps(get_all_stats(), indent=2, default=str))

"""
Example:

import system_stats

# Get individual stats
cpu = system_stats.get_cpu_usage()
ram = system_stats.get_ram_usage()
disk = system_stats.get_disk_usage('/')

# Or get everything at once
stats = system_stats.get_all_stats()
print(f"CPU: {stats['cpu']['percent']}%")
print(f"RAM: {stats['memory']['ram']['percent']}%")
print(f"Disk: {stats['disk']['usage']['percent']}%")
"""
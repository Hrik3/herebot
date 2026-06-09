## Doc for __init__
---

This Python script provides a lightweight library named `system_stats.py` for retrieving various system statistics such as CPU usage, memory (RAM and swap) details, disk space utilization, and I/O counters using the `psutil` module. The key components include functions to measure overall and per-core CPU usage (`get_cpu_usage()`), count logical and physical CPUs (`get_cpu_count()`), retrieve current CPU frequency (`get_cpu_freq()`), gather comprehensive CPU statistics including load average (`get_cpu_stats()` if available on Unix systems), get RAM and swap memory details (`get_ram_usage()`, `get_swap_usage()`, `get_memory_stats()`), disk usage for a given path with caching support (`get_disk_usage()`), list mounted partitions (`get_disk_partitions()`), I/O counters per disk or aggregated (`get_disk_io_counters()`), and combined stats for CPU, memory, and disk (`get_all_stats()`). The script also includes a simple CLI test that prints out system statistics when run as the main module.

---


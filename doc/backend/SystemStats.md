## Doc for SystemStats
---

This Python script, named `system_stats.py`, provides a lightweight library for retrieving system statistics such as CPU usage, RAM usage, and disk space utilization using the `psutil` module. The key components include functions to measure overall and per-core CPU usage (`get_cpu_usage()`), count logical and physical CPUs (`get_cpu_count()`), retrieve current CPU frequency (`get_cpu_freq()`), and gather comprehensive CPU statistics including load average (`get_cpu_stats()`) if available on Unix systems. For memory, it offers methods to get RAM and swap space details (`get_ram_usage()`, `get_swap_usage()`, `get_memory_stats()`). Disk-related functions cover disk usage for a given path with caching support (`get_disk_usage()`), listing mounted partitions (`get_disk_partitions()`), and I/O counters per disk or aggregated (`get_disk_io_counters()`). Additionally, there is a convenience function to retrieve all stats in one call (`get_all_stats()`) which can be useful for quick snapshots. The script also includes a simple CLI test that prints out CPU, memory, and root partition disk statistics when run as the main module.

---

---

## Global Functions

### Functions

#### `get_cpu_usage`

The function `get_cpu_usage` is designed to retrieve the current CPU usage as a percentage. It leverages the `psutil` library for this purpose.

### Parameters:
- **interval (float = 0.1)**: Sampling interval in seconds over which CPU usage is measured.
    - If set to `0.0`, it measures instantaneous CPU usage, though this may be less accurate on some operating systems.
- **per_cpu (bool = False)**: Determines whether the function returns overall system-wide CPU usage or a list of percentages for each individual core.

### Return Value:
- Returns either a single float representing the overall CPU usage percentage if `per_cpu` is set to `False`.
- If `per_cpu` is set to `True`, it returns a list of floats, where each element represents the CPU usage percentage for an individual core.

### Important Behaviour/Side-effects:
The function relies on the underlying operating system's ability to provide accurate instantaneous or sampled CPU usage data. The accuracy and availability of per-core measurements can vary depending on the OS and hardware configuration.

---

#### `get_cpu_count`

The function `get_cpu_count` returns the number of CPU cores, either logical (threads) or physical.

### Parameters:
- **logical (bool = True)**: Determines whether to count logical CPUs (threads) or physical CPUs.
  - If set to `True`, it counts all logical processors (including hyper-threaded threads).
  - If set to `False`, it counts only the physical CPU cores.

### Return Value:
- Returns an integer representing the number of CPU cores based on the specified parameter (`logical`).

### Important Behaviour/Side-effects:
The function relies on the underlying operating system's ability to provide accurate information about logical and physical CPUs. The return value can vary depending on whether hyper-threading is enabled or not.

### Purpose:
This function provides a simple way to determine the number of CPU cores available in the system, either counting all threads (logical) or only the physical cores.

---

#### `get_cpu_freq`

The function `get_cpu_freq` retrieves the current CPU frequency and its minimum and maximum frequencies, returning them as a dictionary.

### Parameters:
- None

### Return Value:
- Returns a dictionary with keys `"current"`, `"min"`, and `"max"` representing the current, minimum, and maximum CPU frequencies in MHz.
- If no information is available, it returns an empty dictionary `{}`.

### Important Behaviour/Side-effects:
The function relies on `psutil.cpu_freq()` to fetch the CPU frequency details. It may return an empty dictionary if the system does not provide this information.

### Purpose:
This function provides a simple way to retrieve and access the current CPU frequency along with its minimum and maximum limits, encapsulated in a dictionary for easy use.

---

#### `get_cpu_stats`

The function `get_cpu_stats` returns a dictionary containing comprehensive CPU statistics.

### Parameters:
- None

### Return Value:
- Returns a dictionary with the following keys and values:
  - `"percent"`: Overall CPU usage percentage.
  - `"per_cpu_percent"`: List of per-core CPU usage percentages.
  - `"logical_count"`: Number of logical CPUs (threads).
  - `"physical_count"`: Number of physical CPUs.
  - `"frequency"`: Dictionary containing the current, minimum, and maximum CPU frequencies in MHz.
  - `"load_avg"`: Tuple representing the load average over 1 minute, 5 minutes, and 15 minutes. This key is only included if available on Unix systems.

### Important Behaviour/Side-effects:
- The function calls other functions such as `get_cpu_usage`, `get_cpu_count`, and `get_cpu_freq` to gather various CPU statistics.
- On non-Unix operating systems or when an OSError occurs, the `"load_avg"` key will be set to `None`.

### Purpose:
This function provides a convenient way to retrieve multiple CPU-related metrics in one call, encapsulated within a dictionary for easy access and use.

---

#### `get_ram_usage`

The function `get_ram_usage` retrieves the virtual memory usage statistics and returns them as a dictionary.

### Parameters:
- None

### Return Value:
- Returns a dictionary with keys representing various RAM metrics, including `"total"`, `"available"`, `"percent"`, `"used"`, `"free"`, `"active"` (if available), `"inactive"` (if available), `"buffers"`, `"cached"`, `"shared"`, and `"slab"` (platform-dependent). All values are in bytes.

### Important Behaviour/Side-effects:
- The function uses `psutil.virtual_memory()` to fetch the memory usage statistics.
- Some keys like `"active`", `"inactive"`, etc., may not be available on all platforms, depending on the underlying operating system's capabilities.

### Purpose:
This function provides a convenient way to retrieve detailed virtual memory (RAM) usage information in one call, encapsulated within a dictionary for easy access and use.

---

#### `get_swap_usage`

The function `get_swap_usage` retrieves the swap memory usage statistics and returns them as a dictionary.

### Parameters:
- None

### Return Value:
- Returns a dictionary with keys representing various swap metrics, including `"total"`, `"used"`, `"free"`, and `"percent"` (percentage of total used).

### Important Behaviour/Side-effects:
- The function uses `psutil.swap_memory()` to fetch the swap memory usage statistics.
- All values are in bytes.

### Purpose:
This function provides a convenient way to retrieve detailed swap memory usage information in one call, encapsulated within a dictionary for easy access and use.

---

#### `get_memory_stats`

The function `get_memory_stats` retrieves and returns a combined dictionary containing both RAM and swap memory usage statistics.

### Parameters:
- None

### Return Value:
- Returns a nested dictionary with two keys:
  - `"ram"`: A dictionary of virtual memory (RAM) usage metrics as returned by `get_ram_usage()`.
  - `"swap"`: A dictionary of swap memory usage metrics as returned by `get_swap_usage()`.

### Important Behaviour/Side-effects:
- The function calls `get_ram_usage` and `get_swap_usage` to gather the respective statistics.
- Both dictionaries contain various memory metrics in bytes, such as total, available, percent used, etc.

### Purpose:
This function provides a convenient way to retrieve both RAM and swap memory usage information in one call, encapsulated within a single dictionary for easy access and use.

---

#### `_disk_usage_cached`

The function `_disk_usage_cached` retrieves the disk usage statistics for a given filesystem path and caches the result to minimize repeated system calls.

### Parameters:
- **path (str)**: The filesystem path for which to retrieve disk usage statistics (e.g., `'/'`, `'C:\\'').

### Return Value:
- Returns a dictionary with keys representing various disk metrics, including `"total"`, `"used"`, `"free"`, and `"percent"` (percentage of total used).

### Important Behaviour/Side-effects:
- This function is intended to be cached using the `lru_cache` decorator from Python's functools module.
- The caching mechanism avoids repeated system calls for the same path, improving performance when calling this function multiple times with the same input.

### Purpose:
This internal helper function provides a way to efficiently retrieve disk usage statistics by leveraging caching. It is designed to be used in conjunction with an external cache decorator like `lru_cache`.

---

#### `get_disk_usage`

The function `get_disk_usage` retrieves disk usage statistics for a given filesystem path, with an option to use caching.

### Parameters:
- **path (str = "/")**: The filesystem path for which to retrieve disk usage statistics. Defaults to the root directory (`'/'`).
- **use_cache (bool = True)**: If set to `True`, enables caching of results for 2 seconds to avoid repeated system calls for the same path.

### Return Value:
- Returns a dictionary with keys representing various disk metrics, including `"total"`, `"used"`, `"free"`, and `"percent"` (percentage of total used).

### Important Behaviour/Side-effects:
- If `use_cache` is set to `True`, the function uses a simple time-based cache mechanism. The cached value will be refreshed after 2 seconds or when a different path is requested.
- Caching logic involves manually checking timestamps since Python's `lru_cache` does not support TTL (time-to-live) functionality.

### Purpose:
This function provides an efficient way to retrieve disk usage statistics, optionally leveraging caching to minimize repeated system calls for the same path. This can improve performance when calling this function multiple times with the same input within a short period.

---

#### `get_disk_partitions`

The function `get_disk_partitions` retrieves a list of mounted disk partitions.

### Parameters:
- **all (bool = False)**: If set to `True`, includes all partitions, including loopback and other non-standard filesystems. Defaults to `False`.

### Return Value:
- Returns a list of dictionaries, where each dictionary represents a partition with keys `"device"`, `"mountpoint"`, `"fstype"` (filesystem type), and `"opts"` (options).

### Important Behaviour/Side-effects:
- The function uses `psutil.disk_partitions(all=all)` to fetch the partitions.
- If `all` is set to `False`, only standard mounted filesystems are returned. Setting it to `True` includes additional partitions like loopback devices.

### Purpose:
This function provides a convenient way to retrieve information about all mounted disk partitions, optionally including non-standard ones, in one call and formats the output as a list of dictionaries for easy access and use.

---

#### `get_disk_io_counters`

The function `get_disk_io_counters` retrieves disk I/O statistics.

### Parameters:
- **perdisk (bool = False)**: If set to `True`, returns a dictionary with I/O statistics per disk. Defaults to `False`.

### Return Value:
- Returns a dictionary containing various I/O metrics such as read count, write count, read bytes, and write bytes.
  - If `perdisk` is `False`, the returned dictionary contains aggregated I/O counters for all disks.
  - If `perdisk` is `True`, returns a nested dictionary where each key represents a disk identifier and its value is another dictionary containing the I/O statistics.

### Important Behaviour/Side-effects:
- The function uses `psutil.disk_io_counters(perdisk=perdisk)` to fetch the I/O counters.
- If no disks are detected or an error occurs, it returns an empty dictionary `{}` for both aggregated (`False`) and per-disk (`True`) cases.

### Purpose:
This function provides a convenient way to retrieve disk I/O statistics either in aggregate form or per individual disk, depending on the specified parameter. The output is formatted as dictionaries for easy access and use.

---

#### `get_disk_stats`

The function `get_disk_stats` retrieves a combination of disk usage and I/O statistics for a specified filesystem path.

### Parameters:
- **path (str = "/")**: The filesystem path for which to retrieve the combined disk stats. Defaults to the root directory (`'/'`).

### Return Value:
- Returns a dictionary with two keys:
  - `"usage"`: A nested dictionary containing disk usage statistics as returned by `get_disk_usage(path)`.
  - `"io"`: A dictionary containing aggregated I/O counters for all disks as returned by `get_disk_io_counters(perdisk=False)`.

### Important Behaviour/Side-effects:
- The function calls `get_disk_usage` to retrieve the disk usage metrics.
- It also calls `get_disk_io_counters` with `perdisk=False` to get aggregated I/O statistics across all disks.

### Purpose:
This function provides a convenient way to retrieve both disk usage and I/O statistics in one call, encapsulated within a single dictionary for easy access and use.

---

#### `get_all_stats`

The function `get_all_stats` retrieves and returns a comprehensive dictionary containing CPU, memory, and disk statistics for the root partition.

### Parameters:
- None

### Return Value:
- Returns a nested dictionary with keys representing different categories of system stats:
  - `"timestamp"`: The current time in seconds since the epoch.
  - `"cpu"`: A dictionary containing various CPU-related metrics as returned by `get_cpu_stats()`.
  - `"memory"`: A dictionary containing both RAM and swap memory usage statistics as returned by `get_memory_stats()`.
  - `"disk"`: A nested dictionary containing disk usage and I/O stats for the root partition (`"/"`) as returned by `get_disk_stats("/")`.

### Important Behaviour/Side-effects:
- The function calls several other functions to gather detailed system metrics, including `get_cpu_stats`, `get_memory_stats`, and `get_disk_stats`.
- It provides a quick snapshot of overall system health in one call.

### Purpose:
This function offers an efficient way to retrieve comprehensive system statistics (CPU usage, memory usage, disk usage, and I/O) for the root partition in one call. The output is formatted as a dictionary for easy access and use.

---


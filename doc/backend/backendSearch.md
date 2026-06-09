## Doc for backendSearch
---

This Python script defines a class `AdaptiveSearchBackend` for live semantic search using an active IncrementalCSR memory system. The key components include methods to initialize the backend with configuration settings and database paths, load full configurations from JSON files, retrieve system statistics (CPU, RAM, disk usage) at regular intervals, get the current live matrix used for searching, perform searches by transforming input text into a vector and finding top matches in the live CSR memory, and clean up or close resources. The class also tracks performance metrics such as search count, hits from live data, and system stats calls. Additionally, it uses caching mechanisms to minimize redundant computations and improve efficiency.

---

### AdaptiveSearchBackend
The `AdaptiveSearchBackend` class provides a live semantic search backend using an active IncrementalCSR memory system. It initializes with configuration settings and database paths, loads full configurations from JSON files, and manages caching for both the live matrix used in searches and system statistics (CPU, RAM, disk usage). Key attributes include the live CSR memory (`live_csr`), a vectorizer for transforming input text into vectors, performance counters like search count and hits, and intervals for cache refreshes. The class is responsible for performing semantic searches by comparing query vectors with cached data and tracking system performance metrics.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `AdaptiveSearchBackend` class. It initializes various attributes and settings when an instance of the class is created.

### Parameters:
- **config_path (str = "search_config.json")**: Path to the configuration file.
- **csr_config (str = "config_listner.json")**: Configuration path for IncrementalCSR.
- **tier1_db (str = "tier1.duckdb")**: Database path for tier 1 data storage.
- **memory_bridge (object)**: Optional bridge object for memory management.

### Important Behaviour/Side-effects:
- The function sets up the configuration file path and loads full configurations from a JSON file using `_load_full_config()`.
- It initializes attributes related to experience memory, system stats cache, and performance counters.
  - `self.memory_bridge`: Stores an optional memory bridge object.
  - `self.live_csr`: Initializes live CSR (IncrementalCSR) either through the provided `memory_bridge` or by creating a new instance with `csr_config`.
  - `self.vectorizer`: Retrieves the vectorizer from the live CSR for transforming input text into vectors.
  - Caching mechanisms are set up for both live matrix and system stats, including intervals (`live_cache_interval`, `stats_cache_interval`) to control cache refresh times.
- Initializes performance counters such as search count, live hits, and stats calls.

### Purpose:
This constructor sets up the necessary attributes and configurations required for performing semantic searches using an active IncrementalCSR memory system. It ensures that all relevant components are initialized properly when creating a new instance of `AdaptiveSearchBackend`.

---

#### `_load_full_config`

The function `_load_full_config` loads and sets configuration parameters from a JSON file into the instance attributes of the `AdaptiveSearchBackend` class.

### Parameters:
- None

### Return Value:
- None (void function)

### Important Behaviour/Side-effects:
- Opens and reads the specified configuration file (`self.config_path`) using Python's built-in `json.load()` method.
- Iterates through each key-value pair in the loaded JSON object, setting corresponding attributes of the instance using `setattr()`.
- Sets default values for specific attributes if they are not present in the configuration:
  - `n_features`: Number of features (default: 10000)
  - `cpu_threshold`: CPU usage threshold percentage (default: 80)
  - `ram_threshold`: RAM usage threshold percentage (default: 85)
  - `degraded_search_ratio`: Degraded search ratio (default: 0.5)
  - `confidence_threshold`: Confidence threshold for matching scores (default: 0.4)

### Purpose:
This function initializes the instance attributes with configuration settings from a JSON file, ensuring that all necessary parameters are properly set when creating an instance of the `AdaptiveSearchBackend` class. If certain keys are missing in the JSON config, it sets default values to ensure consistent behavior.

---

#### `_get_system_stats`

The function `_get_system_stats` retrieves and caches the current system statistics.

### Parameters:
- None

### Return Value:
- Returns a dictionary containing the cached system stats, which includes CPU, memory (RAM), and disk usage details as provided by `system_stats.get_all_stats()`.

### Important Behaviour/Side-effects:
- Checks if there is no existing cache (`self._stats_cache`) or if the current time exceeds the specified interval since the last update (`now - self._stats_cache_time > self.stats_cache_interval`).
- If either condition is met, it updates the system stats by calling `system_stats.get_all_stats()`, sets the `_stats_cache_time` to the current time, and increments the number of times this function has been called (`self.stats_calls`).

### Purpose:
This function ensures that system statistics are retrieved only when necessary (i.e., if there is no existing cache or if enough time has passed since the last retrieval), thus minimizing redundant calls and improving efficiency. The cached stats can be accessed repeatedly without additional overhead until the interval expires.

---

#### `_get_live_matrix`

The function `_get_live_matrix` retrieves and caches the live matrix used for semantic searches.

### Parameters:
- None

### Return Value:
- Returns a cached version of the full CSR (Compressed Sparse Row) matrix loaded from `self.live_csr.load_full()`.

### Important Behaviour/Side-effects:
- Checks if there is no existing cache (`self._live_cache`) or if enough time has passed since the last update (`now - self._live_cache_time > self.live_cache_interval`).
- If either condition is met, it updates the live matrix by calling `self.live_csr.load_full()`, sets `_live_cache_time` to the current time.

### Purpose:
This function ensures that the full CSR matrix used for semantic searches is retrieved only when necessary (i.e., if there is no existing cache or if enough time has passed since the last retrieval), thus minimizing redundant calls and improving efficiency. The cached matrix can be accessed repeatedly without additional overhead until the interval expires.

---

#### `search`

The function `search` performs a semantic search by comparing the input text with entries in the live CSR matrix.

### Parameters:
- **text (str)**: The query text to be searched.
- **top_k (int = 5)**: Number of top matches to retrieve.

### Return Value:
- Returns a list of tuples containing the type ("experience"), index, and score for each match if the best score meets or exceeds the confidence threshold. Otherwise, returns an empty list `[]`.

### Important Behaviour/Side-effects:
- Increments the search count (`self.search_count`) every time this function is called.
- Transforms the input text into a vector using the instance's vectorizer and converts it to CSR format.
- Retrieves the live matrix from `_get_live_matrix()`.
- Computes dot products between the query vector and each row in the live matrix, resulting in scores for matches.
- Filters top `top_k` indices based on highest scores and sorts them if necessary.
- Increments the number of successful hits (`self.live_hits`) if the best score meets or exceeds the confidence threshold.

### Purpose:
This function performs a semantic search by comparing input text with entries in the live CSR matrix, returning up to `top_k` matches that meet the specified confidence threshold. It ensures efficient and consistent retrieval of relevant results while tracking performance metrics such as search count and hits.

---

#### `cleanup`

The function `cleanup` clears the cached data for both the live matrix and system stats.

### Parameters:
- **force (bool = False)**: If set to `True`, forces clearing of all caches regardless of their current state. Defaults to `False`.

### Return Value:
- None (void function)

### Important Behaviour/Side-effects:
- Sets `_live_cache` and `_stats_cache` to `None` if the `force` parameter is `True`.
- This clears any cached data, ensuring that subsequent calls will refresh these caches.

### Purpose:
This function provides a way to clear the cache for live matrix and system stats, which can be useful in scenarios where fresh data needs to be retrieved or when cleaning up resources. If called with `force=True`, it ensures immediate clearing of all relevant caches regardless of their current state.

---

#### `close`

The function `close` clears the cached data for both the live matrix and system stats.

### Parameters:
- None

### Return Value:
- None (void function)

### Important Behaviour/Side-effects:
- Sets `_live_cache` and `_stats_cache` to `None`, effectively clearing any stored cache values.
- This ensures that subsequent calls will refresh these caches, ensuring fresh data retrieval.

### Purpose:
This function is used to clean up resources by explicitly clearing the cached live matrix and system stats. It helps in releasing memory and ensuring that new queries or requests retrieve updated information rather than relying on potentially outdated cached data.

---

#### `__del__`

The function `__del__` is a destructor that calls the `close` method when an instance of the class is about to be destroyed.

### Parameters:
- None

### Return Value:
- None (void function)

### Important Behaviour/Side-effects:
- Calls the `self.close()` method, which clears the cached data for both the live matrix and system stats.
- This ensures that any resources associated with these caches are properly released when an instance of the class is deleted.

### Purpose:
This destructor ensures that when an instance of the `AdaptiveSearchBackend` class is about to be garbage-collected or destroyed, it first calls the `close` method to clean up and clear any cached data. This helps in releasing memory and ensuring proper resource management before the object is removed from memory.

---

#### `get_performance_stats`

The function `get_performance_stats` returns a dictionary containing various performance metrics related to the search backend.

### Parameters:
- None

### Return Value:
- Returns a dictionary with keys representing different performance statistics:
  - `"search_count"`: Total number of searches performed.
  - `"live_hits"`: Number of successful hits from live data that met or exceeded the confidence threshold.
  - `"system_stats_calls"`: Number of times system stats were retrieved and cached.
  - `"memory_source"`: Source of memory management ("bridge" if `self.memory_bridge` is not None, otherwise "csr").
  - `"live_cache_age"`: Time elapsed since the live cache was last updated (in seconds), or `-1` if no live cache exists.

### Important Behaviour/Side-effects:
- The function calculates and returns performance metrics that help in monitoring the efficiency and usage of the search backend.
- It provides insights into how often searches are performed, successful hits from live data, system stats calls, memory source, and age of the live cache.

### Purpose:
This function provides a convenient way to retrieve various performance statistics related to the semantic search operations. The returned dictionary can be used for monitoring and analyzing the backend's efficiency and resource usage over time.

---



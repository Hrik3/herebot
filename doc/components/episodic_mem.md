## Doc for episodic_mem
---

This Python file defines an `EpisodicMemory` class that stores interaction episodes for reinforcement learning and offline consolidation. It reads the capacity from a configuration managed by `ConfigManager`, loads episode files from disk, and provides methods to add new episodes, sample batches of episodes for training, and replay sequences in chronological order. The episodic data includes user input, system response, factor vector, intrinsic reward, and metadata, stored as JSON files in a specified directory.

---

### EpisodicMemory
The `EpisodicMemory` class stores interaction episodes for reinforcement learning and offline consolidation. It manages a list of episode dictionaries, each containing user input, system response, factor vector, intrinsic reward, and metadata. Episodes are saved to disk in JSON format within the specified directory, with capacity dynamically managed based on configuration settings. The class ensures that only the most recent episodes up to the defined capacity are retained.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `EpisodicMemory` class.

Parameters:
- `capacity`: An optional integer specifying the maximum number of episodes to store (default: 10000).
- `config_manager`: An optional reference to a `ConfigManager` instance used for dynamic configuration management.

Return value: None

Behavior:
1. Sets up an internal reference to the provided `config_manager`.
2. If `config_manager` is not `None`, retrieves and sets the capacity from its configuration.
3. Otherwise, uses the provided or default `capacity`.
4. Ensures that the episode directory (`EPISODE_DIR`) exists by creating it if necessary.
5. Initializes an empty list to store episodes.
6. Calls `_load_episodes` to load existing episodes from disk.

Side-effects:
- Sets instance attributes such as `config_manager`, `capacity`, and `episodes`.
- Creates or confirms existence of the episode directory on disk.

Example usage:
```python
memory = EpisodicMemory(config_manager=config_manager_instance)
```

This initializes an episodic memory object with dynamic capacity management using a provided configuration manager.

---

#### `update_parameters_from_config`

The `update_parameters_from_config` function reloads the capacity from a configuration managed by `ConfigManager`.

Parameters: None

Return value: None

Behavior:
1. Checks if an instance of `config_manager` is available.
2. If present, retrieves and sets the new capacity using `self.config_manager.get('episodic_memory.capacity', 10000)`.
3. Trims episodes to ensure that only up to the specified capacity are retained.

Side-effects:
- Updates the internal attribute `capacity` with a value from configuration.
- Removes excess episodes from the list if necessary, starting from the oldest episode.

Example usage:
```python
memory.update_parameters_from_config()
```

This function ensures that the episodic memory's capacity is dynamically updated based on the latest configuration settings.

---

#### `_load_episodes`

The `_load_episodes` function loads all episode files from disk into the internal list of episodes.

Parameters: None

Return value: None

Behavior:
1. Iterates through each file in `EPISODE_DIR`.
2. For each JSON file, attempts to open and load its content using `json.load`.
3. Appends successfully loaded episodes to the internal list `self.episodes`.
4. Handles exceptions by printing a warning message if an episode cannot be loaded.
5. Trims the list of episodes to retain only the last `capacity` episodes.

Side-effects:
- Populates `self.episodes` with all valid episodes from disk, up to the specified capacity.

Example usage:
```python
memory._load_episodes()
```

This function ensures that all episode files are loaded into memory and maintains a limited history based on the configured capacity.

---

#### `add_episode`

The `add_episode` function adds a new episode to the episodic memory and saves it to disk.

Parameters:
- `user_input`: A string representing user input.
- `system_response`: A string representing system response.
- `factor_vector`: A dictionary containing factor values.
- `intrinsic_reward`: A float value for intrinsic reward.
- `metadata` (optional): An optional dictionary with additional metadata (default: {}).

Return value: None

Behavior:
1. Ensures that the provided `factor_vector` is a dictionary, defaulting to an empty dictionary if not.
2. Constructs a new episode dictionary containing timestamp, user input, system response, factor vector, dimension of factor vector, intrinsic reward, and metadata.
3. Appends the new episode to the internal list `self.episodes`.
4. Saves the episode as a JSON file in the specified directory with a unique filename based on the timestamp and random integer.
5. Trims episodes if necessary by removing the oldest one from the list.

Side-effects:
- Adds a new episode to the internal list of episodes.
- Persists the new episode to disk.
- Ensures that only up to `capacity` episodes are retained in memory.

Example usage:
```python
memory.add_episode("user input", "system response", {"factor1": 0.5, "factor2": 0.8}, 0.7)
```

This function adds a new episode with the specified details and saves it to disk while maintaining the configured capacity limit.

---

#### `sample_batch`

The `sample_batch` function randomly samples a batch of episodes from the stored list.

Parameters:
- `n`: An integer specifying the number of episodes to sample (default: 32).

Return value:
- A list of dictionaries, each representing an episode sampled from `self.episodes`.

Behavior:
1. Generates random indices using `np.random.choice` within the range of available episodes.
2. Ensures that the size of the batch is at most the number of available episodes to avoid sampling more than what exists.
3. Returns a list of episodes corresponding to the randomly selected indices.

Side-effects: None

Example usage:
```python
batch = memory.sample_batch(10)
```

This function returns a random sample of `n` episodes for offline learning or other purposes, ensuring that the batch size does not exceed the number of available episodes.

---

#### `replay_sequence`

The `replay_sequence` function returns the last 100 episodes in chronological order for offline consolidation.

Parameters: None

Return value:
- A list of dictionaries, each representing an episode from the most recent to older ones (up to a maximum of 100).

Behavior:
1. Returns the last 100 entries from `self.episodes`.

Side-effects: None

Example usage:
```python
recent_episodes = memory.replay_sequence()
```

This function provides access to the most recent episodes for purposes such as offline consolidation or analysis, ensuring that only up to the latest 100 are included.

---



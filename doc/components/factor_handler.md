## Doc for factor_handler
---

This Python file defines classes and functions to manage state vectors representing learned concepts and resolve conflicts between similar states in a machine learning system. Key components include the `CuriosityFactors` dataclass for storing curiosity scores, the `StateManager` class managing persistent storage of factor vectors with conflict detection and resolution features, and the `FactorEngine` class computing base control signals from the current system state using an adaptive control space.

The StateManager handles loading/saving states to disk, detecting conflicts based on cosine similarity between vector representations, and resolving conflicts through merging or permanent selection. It also manages probabilistic metadata for each state including usage statistics and confidence levels.

FactorEngine derives control signals like curiosity, threat, trust, uncertainty etc from the current system state by analyzing token analyses, diagnostic reports, phonetic scores, concept models, semantic learners, rewards, and more. These factors are used to guide adaptive behavior in the learning agent.

Overall design is modular with clear separation of concerns between managing learned states (StateManager) and deriving control signals based on those states (FactorEngine). The system allows dynamic expansion via AdaptiveControlSpace for additional factors as needed.

---

### CuriosityFactors
The `CuriosityFactors` class is a container for storing and managing curiosity scores associated with a specific word or token. It stores factors dynamically in an internal dictionary (`factor_dict`) to allow flexible addition of new types of curiosity metrics. The class provides methods to safely access and set attributes from this dictionary, ensuring compatibility with serialization mechanisms like pickling. Additionally, it can return the factor values as a numpy array for given factor names.

### Functions

#### `__getattr__`

The `__getattr__` function is used to safely access attributes from the internal dictionary (`factor_dict`) of a `CuriosityFactors` instance. This ensures compatibility with serialization mechanisms like pickling, where the dictionary may not yet be fully loaded.

Parameters:
- `self`: The instance of the class.
- `name`: A string representing the attribute name to retrieve.

Return value:
Returns the value associated with the given attribute name in the `factor_dict` if it exists. Otherwise, raises an AttributeError.

Important behavior:
- Checks if `factor_dict` is not None and contains the requested attribute before returning its value.
- Raises an AttributeError if the attribute does not exist in `factor_dict`.

This function prevents infinite recursion during deserialization when accessing attributes from a partially loaded dictionary.

---

#### `__setattr__`

The `__setattr__` function is used to set attributes for an instance of the `CuriosityFactors` class. It ensures that the internal dictionary (`factor_dict`) can be safely updated without interfering with other attribute assignments.

Parameters:
- `self`: The instance of the class.
- `name`: A string representing the name of the attribute to set.
- `value`: The value to assign to the specified attribute.

Return value:
None

Important behavior:
- If the attribute being set is named `'factor_dict'`, it calls the superclass's `__setattr__` method directly to handle this special case.
- Otherwise, it sets the given value in the `factor_dict`.

This function prevents unintended side effects when setting attributes and ensures that updates to `factor_dict` are handled appropriately.

---

#### `as_array`

The `as_array` function returns an array of curiosity scores corresponding to specified factor names.

Parameters:
- `factor_names`: A list of strings representing the names of factors to include in the output array.

Return value:
Returns a numpy array containing the values associated with each factor name provided, defaulting to 0.0 if a factor is not found.

Important behavior:
- Iterates over the given `factor_names` and retrieves their corresponding values from `self.factor_dict`.
- Constructs and returns an np.ndarray of these values in the same order as the input list.

This function ensures that curiosity scores are easily accessible as numerical arrays, which can be useful for further processing or analysis.

---


### StateManager
The `StateManager` class manages a collection of state vectors, aliases, and conflict resolutions. It stores factor vectors representing different states under unique labels, with metadata tracking usage statistics and probabilistic properties. The manager handles loading and saving these states from disk, as well as maintaining shadow states for temporary use during conflicts.

Key attributes include:
- `states`: A dictionary mapping label to state vector (np.array).
- `aliases`: Mapping old labels to new ones.
- `state_metadata`: Metadata about each state's usage, confidence, etc.
- `conflicts`: List of unresolved or resolved conflict records between states.
- `shadow_states`: Temporary shadow states used during resolution.

The class is responsible for loading and saving states from disk, detecting conflicts based on similarity thresholds, resolving conflicts through merging vectors into temporary shadows, and permanently choosing one state over another. It also manages groupings of related states to form head labels representing clusters of similar states.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `StateManager` class. It sets up various attributes and loads necessary data from disk.

Parameters:
- `state_dir: str = "states"`: The directory where state files are stored.
- `config_manager: Optional['ConfigManager'] = None`: An optional configuration manager for loading parameters dynamically.

Return value:
None

Important behavior and side-effects:

1. Sets up the instance attributes such as `self.config_manager`, `self.state_dir`, etc.
2. Creates necessary directories if they do not exist, catching any exceptions that occur during this process.
3. Initializes various dictionaries to store state vectors, aliases, metadata, conflict records, shadow states, and group relations.
4. Loads all existing states from disk using `_load_all_states()`.
5. Handles loading of conflicts, shadows, state relation observations, and groups from their respective files on disk.
6. Updates the reference dimension (`self.ref_dim`) based on stored vectors.
7. Ensures each loaded state has necessary probabilistic metadata by calling `_ensure_state_probability_metadata()`.

This function is responsible for setting up the initial state of a `StateManager` instance, ensuring all required data structures are populated and initialized correctly from disk or default values.

---

#### `_load_state_metadata`

The `_load_state_metadata` function loads the state metadata from a JSON file into an instance attribute.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Constructs the path to the metadata file using `STATE_DIR`.
2. Checks if the metadata file exists at that path.
3. If the file exists, attempts to open it in read mode.
4. Loads the JSON content from the file into `self.state_metadata` dictionary.
5. Catches any exceptions during file opening or loading and prints a warning message.

Purpose:
This function is responsible for reading state metadata from disk and populating the `state_metadata` attribute of the instance with this data, ensuring that all stored metadata about states are loaded correctly on initialization or reload.

---

#### `_save_state_metadata`

The `_save_state_metadata` function saves the state metadata to a JSON file.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Constructs the path to the metadata file using `STATE_DIR`.
2. Creates a temporary filename by appending `.tmp` to the metadata file path.
3. Opens the temporary file in write mode and dumps the contents of `self.state_metadata` into it as JSON, with an indentation of 2 for better readability.
4. Uses `os.replace` to atomically replace the original metadata file with the temporary one.

Purpose:
This function ensures that the state metadata stored in memory (`self.state_metadata`) is saved safely and reliably to disk by writing to a temporary file first and then replacing the original file, which helps prevent data corruption during write operations.

---

#### `_prob_config`

The `_prob_config` function retrieves the probabilistic labeling configuration from a config manager.

Parameters:
- None

Return value:
- `dict`: A dictionary containing the probabilistic labeling configuration, or an empty dictionary if no config manager is available.

Important behavior and side-effects:

1. Checks if `self.config_manager` exists.
2. If it does, retrieves the 'probabilistic_labeling' section from its configuration.
3. Returns this section as a dictionary.
4. If no config manager is present, returns an empty dictionary.

Purpose:
This function provides access to probabilistic labeling configurations stored in a separate `ConfigManager`, ensuring that relevant settings are easily accessible within the state management context.

---

#### `_state_config`

The `_state_config` function retrieves a configuration value from the `ConfigManager`.

Parameters:
- `key: str`: The configuration key to retrieve.
- `default`: The default value to return if the specified key is not found.

Return value:
- Any type depending on the stored configuration, or the provided default value if the key does not exist.

Important behavior and side-effects:

1. Checks if `self.config_manager` exists.
2. If it does, calls `config_manager.get(f'state_manager.{key}', default)` to retrieve the specified configuration value with a fallback to the given default.
3. Returns this retrieved value or the provided default if no config manager is present.

Purpose:
This function provides access to specific state management configurations stored in a separate `ConfigManager`, ensuring that relevant settings are easily accessible within the state management context.

---

#### `_legacy_interactive_conflicts_enabled`

The `_legacy_interactive_conflicts_enabled` function checks if legacy interactive conflicts are enabled based on the configuration.

Parameters:
- None

Return value:
- `bool`: A boolean indicating whether legacy interactive conflicts are enabled (`True`) or not (`False`).

Important behavior and side-effects:

1. Calls `_state_config('legacy_interactive_conflicts', False)` to retrieve the setting for 'legacy_interactive_conflicts'.
2. Converts the retrieved configuration value to a boolean.

Purpose:
This function determines if legacy interactive conflict detection is enabled by checking the relevant configuration setting, ensuring that appropriate behavior can be determined based on this flag.

---

#### `_ensure_state_probability_metadata`

The `_ensure_state_probability_metadata` function ensures that the probabilistic metadata for a given state label is properly initialized or updated.

Parameters:
- `label: str`: The label of the state.
- `interaction_index: int = 0`: An optional interaction index, defaulting to 0 if not provided.

Return value:
- `dict`: A dictionary containing the updated metadata for the specified state label.

Important behavior and side-effects:

1. Retrieves or initializes the metadata dictionary for the given label using `self.state_metadata.setdefault(label, {})`.
2. Extracts initial usage statistics (`recent_usage` or `usage`) from the metadata.
3. Extracts initial probability values (`probability` or `confidence`) from the metadata.
4. Calls `ensure_probabilistic_stats(meta, interaction_index=interaction_index, initial_usage=initial_usage, initial_probability=initial_probability)` to update the probabilistic statistics for the state label.
5. Returns the updated metadata dictionary.

Purpose:
This function ensures that each state's metadata is properly initialized with necessary probabilistic and usage information, making it easier to manage and track states over interactions.

---

#### `_normalize_state_probabilities`

The `_normalize_state_probabilities` function normalizes the probabilities of state labels based on provided parameters.

Parameters:
- `labels: Optional[List[str]] = None`: A list of state labels to normalize. If not specified, defaults to all keys in `self.states`.
- `interaction_index: int = 0`: An optional interaction index.
- `reward: float = 0.0`: The reward value for the current interaction.
- `touched_label: str = None`: A label that was "touched" or interacted with during this normalization process.
- `weight: float = 0.0`: Weight associated with the touched state.

Return value:
- `dict`: A dictionary mapping each label to its updated metadata after probability field normalization.

Important behavior and side-effects:

1. If no labels are provided, defaults to using all keys in `self.states`.
2. Initializes an empty dictionary `stats_map` to store metadata for each label.
3. Iterates over the specified labels, ensuring that each label's probabilistic metadata is properly initialized or updated by calling `_ensure_state_probability_metadata(label, interaction_index)`.
4. Calls `normalize_probability_field(stats_map, ...)` to normalize the probability field in the metadata dictionary based on provided parameters and configuration settings.
5. Updates the state metadata with normalized values from `stats_map`.

Purpose:
This function ensures that probabilities for specified states are properly normalized and updated according to usage statistics and interaction rewards, maintaining consistent probabilistic tracking of states over interactions.

Side-effects:
- Modifies `self.state_metadata` by updating probability fields in each label's metadata.

---

#### `get_state_probabilities`

The `get_state_probabilities` function retrieves the probabilities of states based on their metadata.

Parameters:
- `interaction_index: int = 0`: An optional interaction index.
- `include_archived: bool = False`: A boolean flag indicating whether to include archived states in the result.

Return value:
- `Dict[str, float]`: A dictionary mapping each state label to its probability (float).

Important behavior and side-effects:

1. Retrieves all labels from `self.states`.
2. Calls `_normalize_state_probabilities(labels, interaction_index)` to ensure that probabilities are normalized based on provided parameters.
3. Filters the resulting metadata entries to include only active states unless specified otherwise by `include_archived`.
4. Constructs a dictionary mapping each label to its probability value.

Purpose:
This function provides a way to get the current probabilistic estimates of all relevant state labels, ensuring they reflect recent interactions and usage statistics accurately.

Side-effects:
- Modifies internal metadata through `_normalize_state_probabilities` but does not alter external state beyond returning probabilities.

---

#### `_state_relation_key`

The `_state_relation_key` function generates a unique key based on sorted state labels.

Parameters:
- `labels: List[str]`: A list of state labels (strings).

Return value:
- `str`: A string representing the sorted and concatenated labels, separated by "||".

Important behavior and side-effects:

1. Converts each label in the provided list to a string.
2. Sorts the converted labels alphabetically.
3. Joins the sorted labels into a single string using "||" as a separator.

Purpose:
This function generates a unique key that represents a set of state labels, ensuring consistent and comparable keys for related states or groups of states.

Example:
```python
labels = ["state1", "state2"]
key = _state_relation_key(labels)
# Output: "state1||state2"
```

Side-effects:
- None. This function does not modify any external data structures.

---

#### `_align_pair`

The `_align_pair` function aligns two numpy arrays to the same length by padding or truncating them.

Parameters:
- `v1: np.ndarray`: The first input vector as a numpy array.
- `v2: np.ndarray`: The second input vector as a numpy array.

Return value:
- `Tuple[np.ndarray, np.ndarray]`: A tuple containing two aligned vectors (both padded/truncated to the same length).

Important behavior and side-effects:

1. Normalizes both input vectors using `_normalize_vector`.
2. Determines the maximum length between the normalized vectors.
3. Pads shorter vectors with zeros on the right until they match the maximum length, or truncates longer vectors to that length.

Purpose:
This function ensures two state vectors are of equal length for comparison purposes, such as calculating similarity metrics like cosine similarity.

Example:
```python
v1 = np.array([0.5, 0.3])
v2 = np.array([0.4, 0.6, 0.7])

aligned_v1, aligned_v2 = _align_pair(v1, v2)
# Output: (array([0.5, 0.3, 0. ]), array([0.4, 0.6, 0.7]))
```

Side-effects:
- Modifies the input vectors by padding or truncating them to ensure they have equal length before returning.

Mathematical formula classification: This function does not directly involve any specific mathematical formulas but is used in preparation for calculations like cosine similarity.

---

#### `_cosine_similarity`

The `_cosine_similarity` function calculates the cosine similarity between two numpy arrays.

Parameters:
- `v1: np.ndarray`: The first input vector as a numpy array.
- `v2: np.ndarray`: The second input vector as a numpy array.

Return value:
- `float`: A float representing the cosine similarity between the two vectors, ranging from 0.0 to 1.0.

Important behavior and side-effects:

1. Aligns both input vectors using `_align_pair` to ensure they have equal length.
2. Calculates the Euclidean norms (magnitudes) of the aligned vectors `v1` and `v2`.
3. Computes the dot product of the two aligned vectors.
4. Divides the dot product by the product of their magnitudes to obtain cosine similarity.
5. Returns 0.0 if the denominator is very small or zero, avoiding division by near-zero values.

Purpose:
This function computes a measure of similarity between two state vectors based on the angle between them in high-dimensional space.

Mathematical formula classification: The cosine similarity is calculated using the dot product and Euclidean norms:

\[
\text{cosine\_similarity} = \frac{\mathbf{v1} \cdot \mathbf{v2}}{\|\mathbf{v1}\| \|\mathbf{v2}\|}
\]

Side-effects:
- Modifies input vectors by padding or truncating them to ensure they have equal length before calculating similarity.

---

#### `_factor_delta`

The `_factor_delta` function calculates the element-wise difference between two numpy arrays and computes summary statistics on this difference.

Parameters:
- `v1: np.ndarray`: The first input vector as a numpy array.
- `v2: np.ndarray`: The second input vector as a numpy array.

Return value:
- `dict`: A dictionary containing three keys:
  - `'mean'`: The mean of the absolute differences between elements in `v1` and `v2`.
  - `'max'`: The maximum absolute difference between any pair of corresponding elements.
  - `'vector'`: A list representation of the element-wise absolute differences.

Important behavior and side-effects:

1. Aligns both input vectors using `_align_pair` to ensure they have equal length.
2. Computes the element-wise absolute difference between `v1` and `v2`.
3. Calculates the mean and maximum values from this difference vector, handling empty or zero-length arrays by returning 0.0 for these statistics.
4. Converts the difference vector to a list format.

Purpose:
This function provides detailed information about how two state vectors differ element-wise, which can be useful in various analyses such as detecting significant changes between states.

Mathematical formula classification: This function does not directly involve any specific mathematical formulas but performs basic arithmetic operations like absolute value and mean calculation.

Side-effects:
- Modifies input vectors by padding or truncating them to ensure they have equal length before calculating the difference.

---

#### `_load_state_relations`

The `_load_state_relations` function loads state relation observations from a JSON file.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Checks if the `relation_file` exists.
2. If it does, attempts to open the file in read mode.
3. Loads the content of the file as JSON data.
4. Sets `self.state_relation_observations` to the loaded data if it is a dictionary; otherwise, sets it to an empty dictionary.
5. Catches any exceptions during file opening or loading and prints a warning message.

Purpose:
This function ensures that state relation observations stored in a JSON file are properly loaded into memory for further processing.

Side-effects:
- Modifies `self.state_relation_observations` by populating it with the data from the relations file if successful, otherwise sets it to an empty dictionary.

---

#### `_save_state_relations`

The `_save_state_relations` function saves state relation observations to a JSON file.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Ensures the directory containing `self.relation_file` exists by creating it if necessary.
2. Constructs a temporary filename by appending `.tmp` to the relations file path.
3. Opens the temporary file in write mode and dumps the content of `self.state_relation_observations` as JSON, with an indentation of 2 for better readability.
4. Uses `os.replace` to atomically replace the original relations file with the temporary one.

Purpose:
This function ensures that state relation observations stored in memory are safely saved to disk by writing to a temporary file first and then replacing the original file, which helps prevent data corruption during write operations.

Side-effects:
- Modifies files on disk by creating necessary directories, writing JSON content to a temporary file, and replacing the original relations file with this temporary file.

---

#### `_load_state_groups`

The `_load_state_groups` function loads state groups from a JSON file and sets up aliases between states.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Checks if the `group_file` exists.
2. If it does, attempts to open the file in read mode.
3. Loads the content of the file as JSON data.
4. Sets `self.state_groups` to the loaded data if it is a dictionary; otherwise, sets it to an empty dictionary.
5. Catches any exceptions during file opening or loading and prints a warning message.
6. Initializes `self.state_group_aliases` as an empty dictionary.
7. Iterates over each group in `self.state_groups`, skipping archived groups.
8. For each non-archived group, sets up aliases between the head label (if present) and its child labels.

Purpose:
This function ensures that state groups stored in a JSON file are properly loaded into memory for further processing and sets up necessary alias mappings to track relationships between states.

Side-effects:
- Modifies `self.state_groups` by populating it with data from the groups file if successful, otherwise sets it to an empty dictionary.
- Populates `self.state_group_aliases` based on the state group information.

---

#### `_save_state_groups`

The `_save_state_groups` function saves state groups to a JSON file.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Ensures the directory containing `self.group_file` exists by creating it if necessary.
2. Constructs a temporary filename by appending `.tmp` to the group file path.
3. Opens the temporary file in write mode and dumps the content of `self.state_groups` as JSON, with an indentation of 2 for better readability.
4. Uses `os.replace` to atomically replace the original groups file with the temporary one.

Purpose:
This function ensures that state groups stored in memory are safely saved to disk by writing to a temporary file first and then replacing the original file, which helps prevent data corruption during write operations.

Side-effects:
- Modifies files on disk by creating necessary directories, writing JSON content to a temporary file, and replacing the original groups file with this temporary file.

---

#### `get_group_heads_for_labels`

The `get_group_heads_for_labels` function retrieves the group heads (canonical states) associated with a list of state labels.

Parameters:
- `labels: List[str]`: A list of state labels for which to find corresponding group heads.

Return value:
- `List[str]`: A list of unique group head labels that correspond to any of the input labels.

Important behavior and side-effects:

1. Initializes an empty list called `heads` to store the resulting group heads.
2. Iterates over each label in the provided list (or an empty list if no labels are given).
3. For each label, retrieves its corresponding head from `self.state_group_aliases`.
4. If a valid head is found and it is not already in the `heads` list, appends it to the list.
5. Returns the final list of unique group heads.

Purpose:
This function provides a way to map multiple state labels to their respective canonical (head) states within predefined groups, ensuring that only distinct group heads are returned.

Side-effects:
- None. This function does not modify any external data structures beyond returning a result based on existing mappings in `self.state_group_aliases`.

---

#### `resolve_label`

The `resolve_label` function resolves a given state label to its final canonical form by following any alias chains.

Parameters:
- `label: str`: The input state label to resolve.

Return value:
- `str`: The resolved, canonical state label after following all aliases.

Important behavior and side-effects:

1. Initializes an empty set called `visited` to keep track of labels that have been processed.
2. Enters a loop where it checks if the current `label` exists in `self.aliases`.
3. If the label is found in `self.aliases`, adds it to the `visited` set and updates `label` to its aliased value.
4. The loop breaks if an alias forms a cycle (i.e., revisits a previously seen label) or when no more aliases are available for the current label.
5. Returns the final resolved canonical label.

Purpose:
This function ensures that any given state label is mapped to its ultimate, non-aliased form by following all existing alias chains.

Side-effects:
- None. This function does not modify external data structures beyond returning a result based on existing mappings in `self.aliases`.

Mathematical formula classification: No specific mathematical formulas are involved; the function primarily handles string manipulation and set operations to resolve aliases.

---

#### `get_state_vector`

The `get_state_vector` function retrieves the state vector associated with a given label by resolving aliases and considering both regular states and shadow states.

Parameters:
- `label: str`: The input state label to resolve.
- `include_shadows: bool = True`: A boolean flag indicating whether to include shadow states in the search (default is `True`).

Return value:
- `Tuple[str, np.ndarray]`: A tuple containing the resolved label and its corresponding vector. If no matching vector is found, returns `(None, None)`.

Important behavior and side-effects:

1. Initializes an empty list called `candidates` to store potential labels.
2. Defines a helper function `add_candidate(value)` that adds `value` to `candidates` if it is not already present.
3. Adds the input label, resolved label (using `resolve_label(label)`), and aliased label (`self.aliases.get(label)`) as candidates.
4. Iterates over all alias pairs in `self.aliases`, adding any old labels that resolve to the input label as additional candidates.
5. Constructs a list of stores to search for state vectors, including both regular states (`self.states`) and shadow states (`self.shadow_states`), based on the `include_shadows` flag.
6. Iterates over each candidate label and checks if it exists in any store.
7. Returns the first matching label and its corresponding vector found; otherwise, returns `(None, None)`.

Purpose:
This function safely resolves a state label to its canonical form and retrieves the associated state vector from either regular states or shadow states based on provided parameters.

Side-effects:
- Modifies `candidates` list by adding potential labels.
- Does not modify external data structures beyond returning results based on existing mappings in `self.aliases`, `self.states`, and `self.shadow_states`.

Mathematical formula classification: No specific mathematical formulas are involved; the function primarily handles string manipulation, set operations, and dictionary lookups to resolve aliases and retrieve state vectors.

---

#### `mark_conflict_stale`

The `mark_conflict_stale` function marks a given conflict as resolved and stale, updating the conflict record with relevant metadata.

Parameters:
- `conflict: Dict`: A dictionary representing the conflict to be marked.
- `reason: str`: The reason for marking the conflict as stale.

Return value:
None

Important behavior and side-effects:

1. Sets `resolved` in the conflict dictionary to `True`, indicating that the conflict is resolved.
2. Marks the conflict as "stale" by setting `"stale"` to `True`.
3. Specifies the resolution type as `"stale"` in the conflict dictionary.
4. Updates the `"resolution_label"` and `"stale_reason"` fields with the provided reason string.
5. Calls `_save_conflicts()` to persist these changes to disk.

Purpose:
This function updates a conflict record to indicate that it is resolved but no longer relevant (stale), ensuring that stale conflicts are properly marked for archival or removal in future operations.

Side-effects:
- Modifies the input `conflict` dictionary by updating its fields.
- Persists updated conflict records to disk via `_save_conflicts()`.

Mathematical formula classification: No specific mathematical formulas are involved; the function primarily handles dictionary updates and file I/O.

---

#### `_update_ref_dim`

The `_update_ref_dim` function sets the reference dimension to the mode (most common length) of all stored state vectors.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Constructs a list called `dims` containing the lengths of each vector in `self.states.values()`.
2. If there are any dimensions in `dims`, sets `ref_dim` to the most common dimension (mode) using `max(set(dims), key=dims.count)`.
3. If no vectors exist, sets `ref_dim` to a default value of 10.

Purpose:
This function ensures that all state vectors have a consistent reference dimension based on their lengths, which is useful for alignment and comparison purposes.

Side-effects:
- Modifies the instance attribute `self.ref_dim`.

Mathematical formula classification: No specific mathematical formulas are involved; the function primarily handles list operations to determine the mode of vector dimensions.

---

#### `_normalize_vector`

The `_normalize_vector` function normalizes a given numpy array to unit norm and optionally aligns it to the reference dimension.

Parameters:
- `vector: np.ndarray`: The input vector to normalize.
- `align: bool = False`: A boolean flag indicating whether to pad or truncate the normalized vector to match the reference dimension (default is `False`).

Return value:
- `np.ndarray`: The normalized and optionally aligned vector.

Important behavior and side-effects:

1. Checks if the input vector is empty (`None` or length 0) and returns an empty numpy array in such cases.
2. Converts the input to a float dtype numpy array for consistency.
3. Normalizes the vector by dividing it with its Euclidean norm (unit norm).
4. If `align=True`, ensures that the reference dimension (`self.ref_dim`) is set:
   - Sets `ref_dim` to the length of the vector if not already defined.
   - Pads or truncates the normalized vector to match the reference dimension.

Purpose:
This function normalizes a given vector and optionally aligns it to ensure consistent dimensions, which is useful for comparison and alignment purposes in state management.

Side-effects:
- Modifies `self.ref_dim` if necessary during alignment.
- Returns a normalized (and possibly aligned) numpy array.

Mathematical formula classification: The normalization process involves the Euclidean norm calculation:

\[
\text{normalized\_vector} = \frac{\mathbf{vector}}{\|\mathbf{vector}\|}
\]

where \(\|\mathbf{vector}\|\) is the Euclidean norm of the vector.

---

#### `update_parameters_from_config`

The `update_parameters_from_config` function is intended to reload configuration parameters but currently does nothing.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. The function body contains only the keyword `pass`, which means it has no active code or functionality.
2. It is designed for future implementation where parameters can be reloaded from a configuration source dynamically (hot reload).

Purpose:
This placeholder function is intended to facilitate dynamic reloading of configuration parameters, though currently does not perform any action.

Side-effects:
- None; the function performs no operations and has no side effects.

Mathematical formula classification: No mathematical formulas are involved since there is no active code in this function.

---

#### `_load_all_states`

The `_load_all_states` function loads all existing state files from a specified directory into memory.

Parameters:
- None

Return value:
None

Important behavior and side-effects:

1. Clears the `self.states` dictionary to remove any previously loaded states.
2. Checks if the `state_dir` exists; returns early if it does not.
3. Iterates over all filenames in the state directory, filtering for `.json` files that do not start with a specified prefix (`SHADOW_PREFIX`).
4. For each valid file:
   - Constructs the full path to the file.
   - Opens and reads the JSON content from the file.
   - Extracts label, vector, and metadata from the loaded data.
   - Normalizes the vector using `_normalize_vector`.
   - Stores the normalized vector in `self.states` with the extracted label as the key.
   - Initializes `state_metadata` if not already present and stores the metadata for each state.
5. Handles any exceptions during file operations by printing an error message.
6. Updates the reference dimension (`ref_dim`) based on all loaded vectors using `_update_ref_dim`.

Purpose:
This function loads all existing state files from a directory into memory, ensuring that they are properly normalized and stored in dictionaries for later use.

Side-effects:
- Modifies `self.states` by populating it with loaded states.
- Initializes or modifies `state_metadata` to store metadata about each state.
- Updates the reference dimension based on loaded vectors.

Mathematical formula classification: No specific mathematical formulas are involved; the function primarily handles file I/O and dictionary operations.

---

#### `save_state`

The `save_state` function saves a factor vector under the given label, optionally as a shadow state.

Parameters:
- `label: str`: The label for the new state.
- `vector: np.ndarray`: The numpy array representing the state's factor vector.
- `metadata: dict = None`: Optional metadata dictionary for the state (default is an empty dictionary).
- `is_shadow: bool = False`: A boolean flag indicating whether to save this as a shadow state (temporary) or regular state (default is `False`).
- `response: str = None`: An optional response string, not used in current implementation.
- `interaction_index: int = 0`: The interaction index for the new state.

Return value:
- `str`: The final safe label of the saved state after sanitizing and normalizing it.

Important behavior and side-effects:

1. Ensures that the specified directory exists by creating it if necessary using `os.makedirs`.
2. Sanitizes the input label to remove non-alphanumeric characters, except for underscores and hyphens.
3. If sanitized label is empty, sets it to "unnamed_state".
4. Constructs a filename based on whether this state is a shadow or regular state, appending a timestamp.
5. Creates the full path by joining the directory with the constructed filename.
6. Initializes data dictionary containing:
   - Label
   - Vector as list (original length)
   - Dimension of vector
   - Timestamp
   - Provided metadata
   - Flag indicating if it's a shadow state
7. Adds metabolic metadata for new states, including placeholders and default values.
8. Writes the JSON content to a temporary file with `.tmp` extension.
9. Renames the temporary file to its final name using `os.replace`.
10. Normalizes the vector using `_normalize_vector`.
11. Stores the normalized state in either `self.states` or `self.shadow_states` based on whether it's a shadow state.
12. Updates the reference dimension (`ref_dim`) based on all stored vectors using `_update_ref_dim`.
13. Saves updated metadata to disk via `_save_state_metadata`.

Purpose:
This function saves a factor vector under a specified label, ensuring that states are properly normalized and stored in memory as well as persisted to disk.

Side-effects:
- Modifies `self.states` or `self.shadow_states` by adding the new state.
- Initializes or modifies `state_metadata` with metadata for each saved state.
- Updates the reference dimension based on all

---

#### `update_state`

### Function Documentation

**Purpose:**
Update an existing state with a new vector by deleting the old files associated with that label and saving a new one.

**Parameters:**
- `label` (str): The identifier for the state to be updated.
- `new_vector` (np.ndarray): A numpy array representing the new state vector.
- `metadata` (dict, optional): Additional metadata related to the state. Defaults to None if not provided.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Deletion of Old Files:** Deletes all files in the specified directory (`self.state_dir`) that start with the given label and have a `.json` extension.
2. **Saving New State:** Saves the new state vector to disk using `save_state()`. If an error occurs during deletion or saving, it is ignored (handled by passing).
3. **In-Memory Update:** Updates the in-memory dictionary (`self.states`) with the normalized version of the new vector.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs file operations and updates a state based on provided inputs.

---

#### `delete_state`

### Function Documentation

**Purpose:**
Remove a state file and its corresponding entries from memory.

**Parameters:**
- `label` (str): The identifier for the state to be deleted.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Deletion in Memory:** Deletes the specified label's entry from both `self.states` and `self.shadow_states`.
2. **File Deletion:** Removes all files in the directory (`self.state_dir`) that start with the given label and have a `.json` extension.
3. **Metadata Update:** If the state has associated metadata, it is deleted from `self.state_metadata`, and this change is saved to disk using `_save_state_metadata()`.
4. **Reference Dimension Update:** Calls `_update_ref_dim()` to update any reference dimensions or related data structures.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs file and memory operations based on the provided label.

---

#### `find_similar_state`

### Function Documentation

**Purpose:**
Return the label of the most similar known state if the cosine similarity between the query vector and stored vectors is greater than or equal to a specified threshold.

**Parameters:**
- `vector` (np.ndarray): A numpy array representing the input vector.
- `threshold` (float, optional): The minimum required cosine similarity for considering states as similar. Defaults to 0.85 if not provided.
- `include_shadows` (bool, optional): Whether to include shadow states in the search. Defaults to False if not specified.
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not provided.
- `reward` (float, optional): A reward value used for updating state probabilities. Defaults to 0.0 if not specified.

**Return Value:**
Optional[str]: The label of the most similar known state that meets the threshold criteria; None otherwise.

**Important Behaviour and Side-effects:**
1. **Threshold Retrieval:** If `threshold` is not provided, it defaults to a value retrieved from configuration (`self.config_manager.get('state_manager.conflict_threshold', 0.85)`), or falls back to 0.85 if no config manager exists.
2. **Normalization of Vectors:** The input vector and stored vectors are normalized using `_normalize_vector()`.
3. **State Filtering:** Only active states (not archived) are considered for similarity comparison based on metadata (`self.state_metadata`).
4. **Probability Calculation:** Probabilities associated with each state are retrieved from `get_state_probabilities(interaction_index=interaction_index)`.
5. **Similarity Computation:**
   - Cosine similarity is calculated using the formula:
     \[
     \text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
     \]
6. **Tie Resolution:**
   - If multiple states have the same highest similarity score, a tie tolerance (`self.config_manager.get('state_manager.selection_tie_tolerance', 1e-9)`) is used to break ties.
7. **State Probability Update:** Updates state probabilities based on the chosen label and interaction index using `_normalize_state_probabilities()`.
8. **Metadata Saving:** Metadata changes are saved to disk with `self._save_state_metadata()`.

**Mathematical Formula:**

---

#### `find_similar_states`

### Function Documentation

**Purpose:**
Return a list of tuples containing the labels and similarity scores of all states with cosine similarity greater than or equal to a specified threshold.

**Parameters:**
- `vector` (np.ndarray): A numpy array representing the input vector.
- `threshold` (float, optional): The minimum required cosine similarity for considering states as similar. Defaults to 0.85 if not provided.
- `include_shadows` (bool, optional): Whether to include shadow states in the search. Defaults to False if not specified.
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.
- `reward` (float, optional): A reward value used for updating state probabilities. Defaults to 0.0 if not specified.

**Return Value:**
List[Tuple[str, float]]: A list of tuples where each tuple contains the label and similarity score of a matching state.

**Important Behaviour and Side-effects:**
1. **Threshold Retrieval:** If `threshold` is not provided, it defaults to a value retrieved from configuration (`self.config_manager.get('state_manager.conflict_threshold', 0.85)`), or falls back to 0.85 if no config manager exists.
2. **Normalization of Vectors:** The input vector and stored vectors are normalized using `_normalize_vector()`.
3. **State Filtering:**
   - Only active states (not archived) are considered for similarity comparison based on metadata (`self.state_metadata`).
4. **Probability Calculation:** Probabilities associated with each state are retrieved from `get_state_probabilities(interaction_index=interaction_index)` and a probability threshold is set using the configuration.
5. **Similarity Computation:**
   - Cosine similarity is calculated using the formula:
     \[
     \text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
     \]
6. **State Probability Update:** Updates state probabilities based on the chosen label and interaction index using `_normalize_state_probabilities()`.
7. **Tie Resolution:**
   - If multiple states have the same highest similarity score, a tie tolerance (`self.config_manager.get('state_manager.selection_tie_tolerance', 1e-9)`) is used to break ties.
8. **Metadata Saving:** Metadata changes are saved to disk with `

---

#### `_active_state_labels`

### Function Documentation

**Purpose:**
Return a list of active state labels based on the current probabilities and metadata.

**Parameters:**
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.

**Return Value:**
List[str]: A list containing the labels of all active states that are not archived.

**Important Behaviour and Side-effects:**
1. **Probability Retrieval:** Calls `get_state_probabilities(interaction_index=interaction_index)` to ensure up-to-date probabilities.
2. **State Filtering:** Iterates through keys in `self.states` and checks if each state is marked as 'active' or not archived based on metadata (`self.state_metadata.get(label, {}).get('phase', 'active') != 'archived'`). If a state meets the criteria, its label is added to the list of active labels.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs filtering and retrieval based on provided parameters.

---

#### `record_state_relation_observation`

### Function Documentation

**Purpose:**
Record high-similarity state pairs as relation observations. This function identifies and records pairs of states with cosine similarity above a specified threshold.

**Parameters:**
- `labels` (List[str]): A list of labels for the states to be considered.
- `tokens` (List[str], optional): List of tokens associated with the observation. Defaults to an empty list if not provided.
- `factor_vector` (np.ndarray, optional): An additional factor vector used in some contexts. Not directly utilized in this function but may affect related operations.
- `reward` (float, optional): A reward value used for updating state probabilities and relations. Defaults to 0.0 if not specified.
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not provided.
- `reasoner`: An object or function responsible for reasoning about the states and their relationships.

**Return Value:**
List[Dict]: A list of dictionaries containing details of recorded state relation observations.

**Important Behaviour and Side-effects:**
1. **Threshold Retrieval:** The similarity threshold is retrieved from configuration (`self._state_config('state_relation_similarity_threshold', 0.85)`), defaulting to 0.85 if not specified.
2. **Label Cleaning:** Resolves labels, filters out archived states, and removes duplicates.
3. **Probability Calculation:** Retrieves state probabilities using `get_state_probabilities(interaction_index=interaction_index)`.
4. **Similarity Computation:**
   - Cosine similarity is calculated between pairs of states using `_cosine_similarity()`, which uses the formula:
     \[
     \text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
     \]
5. **Observation Recording:** Records pairs of states with similarity above the threshold using `_record_state_relation_pair()`.
6. **Metadata Saving:** Saves recorded state relations to disk with `self._save_state_relations()`.
7. **State Regrouping Refresh:**
   - Calls `refresh_state_regrouping(interaction_index=interaction_index, reasoner=reasoner)` if any observations are recorded.

**Mathematical Formula:**
The cosine similarity formula is used:
\[
\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b

---

#### `refresh_state_relations`

### Function Documentation

**Purpose:**
Scan all active states and record high-similarity relation observations.

**Parameters:**
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.
- `tokens` (List[str], optional): List of tokens associated with the observation. Defaults to an empty list if not provided.
- `reward` (float, optional): A reward value used for updating state probabilities and relations. Defaults to 0.0 if not specified.
- `reasoner`: An object or function responsible for reasoning about the states and their relationships.

**Return Value:**
List[Dict]: A list of dictionaries containing details of recorded state relation observations.

**Important Behaviour and Side-effects:**
1. **Active State Retrieval:** Retrieves a list of active state labels using `_active_state_labels(interaction_index=interaction_index)`.
2. **Threshold Retrieval:** The similarity threshold is retrieved from configuration (`self._state_config('state_relation_similarity_threshold', 0.85)`), defaulting to 0.85 if not specified.
3. **Probability Calculation:** Retrieves state probabilities using `get_state_probabilities(interaction_index=interaction_index)`.
4. **Similarity Computation:**
   - Cosine similarity is calculated between pairs of states using `_cosine_similarity()`, which uses the formula:
     \[
     \text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
     \]
5. **Observation Recording:** Records pairs of states with similarity above the threshold using `_record_state_relation_pair()`.
6. **Metadata Saving:** Saves recorded state relations to disk with `self._save_state_relations()`.
7. **State Regrouping Refresh:**
   - Calls `refresh_state_regrouping(interaction_index=interaction_index, reasoner=reasoner)` if any observations are recorded.

**Mathematical Formula:**
The cosine similarity formula is used:
\[
\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
\]

### Classification
This function involves the use of a mathematical formula for cosine similarity.

---

#### `_record_state_relation_pair`

### Function Documentation

**Purpose:**
Record a pair of states and their observed relationship, including similarity scores and related metadata.

**Parameters:**
- `l1` (str): The label of the first state.
- `l2` (str): The label of the second state.
- `similarity` (float): The cosine similarity score between the two states.
- `probabilities` (Dict[str, float]): A dictionary containing probability values for each state.
- `tokens` (List[str], optional): List of tokens associated with the observation. Defaults to an empty list if not provided.
- `reward` (float, optional): A reward value used for updating state probabilities and relations. Defaults to 0.0 if not specified.
- `interaction_index` (int): An index representing a specific interaction or context.

**Return Value:**
Dict: A dictionary containing details of the recorded relationship between the two states.

**Important Behaviour and Side-effects:**
1. **Key Generation:** Generates a unique key for the state pair using `_state_relation_key([l1, l2])`.
2. **Relation Initialization:** Initializes or retrieves an existing relation object from `self.state_relation_observations` with default values if it does not exist.
3. **Update Metadata:**
   - Updates metadata fields such as interaction index, observation count, similarity history, factor delta history, and probability delta history.
4. **Token Weighting:**
   - Adds tokens to the shared_tokens dictionary with a weighted value based on reward.
5. **Reward Accumulation:** Tracks cumulative rewards (`reward_trace`) and positive rewards (`positive_reward_trace`).
6. **Regroup Score Calculation:** Calculates a regroup score using `_relation_regroup_score(relation)`.
7. **Metadata Update:** Updates the relation object in `self.state_relation_observations`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs metadata and relationship recording based on provided inputs.

---

#### `_relation_regroup_score`

### Function Documentation

**Purpose:**
Calculate a regroup score for the given state relation based on similarity history and probability delta.

**Parameters:**
- `relation` (Dict): A dictionary containing details of the recorded relationship between two states.

**Return Value:**
float: The calculated regroup score.

**Important Behaviour and Side-effects:**
1. **Similarity Extraction:** Extracts similarity values from the `similarity_history`.
2. **Probability Delta Extraction:** Extracts probability delta values from the `probability_delta_history`.
3. **Recent Observations Filtering:**
   - Filters out recent observations based on a minimum number of observations (`recent_n`), which is retrieved from configuration (`self._state_config('state_regroup_min_observations', 3)`).
4. **Average Similarity Calculation:** Computes the average similarity score for the most recent `recent_n` observations.
5. **Average Probability Delta Calculation:**
   - If available, computes the average probability delta for the most recent `recent_n` observations.
6. **Reward Evidence Calculation:** Calculates a reward evidence value using the hyperbolic tangent of positive rewards (`positive_reward_trace`) and ensures it is non-negative.
7. **Regroup Score Computation:** Computes the regroup score as the product of:
   - Average similarity
   - Maximum of 0 or (1 minus average probability delta)
   - Reward evidence

**Mathematical Formula:**
The reward evidence calculation uses the hyperbolic tangent function:
\[
\text{reward\_evidence} = \max(0.0, \tanh(\text{positive\_reward\_trace}))
\]

### Classification
This function involves a mathematical formula for calculating reward evidence using the hyperbolic tangent function.

---

#### `refresh_state_regrouping`

### Function Documentation

**Purpose:**
Create or update head-label groups from repeated state relation observations based on specified criteria.

**Parameters:**
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.
- `reasoner`: An object or function responsible for reasoning about the states and their relationships.
- `force` (bool, optional): A flag indicating whether to force regrouping regardless of interval conditions. Defaults to False if not specified.

**Return Value:**
Dict[str, Dict]: The updated dictionary containing head-label groups (`self.state_groups`).

**Important Behaviour and Side-effects:**
1. **Interval Check:** Determines if the current interaction index meets the required interval for state regrouping based on configuration (`state_regroup_interval_interactions`). If not forced and the interval condition is not met, returns `self.state_groups`.
2. **Minimum Observations Retrieval:** Retrieves the minimum number of observations needed to consider a relation eligible from configuration (`state_regroup_min_observations`).
3. **Similarity Threshold Retrieval:** Retrieves the similarity threshold for eligibility from configuration (`state_regroup_similarity_threshold`).
4. **Eligible Pairs Identification:**
   - Iterates through state relations and identifies pairs that meet criteria such as non-archived status, sufficient observation count, average recent similarity above a threshold, positive regroup score.
5. **Union-Find Algorithm:** Uses the union-find algorithm to group eligible labels into connected components.
6. **Component Grouping:**
   - For each component (set of related states), calls `_upsert_state_group()` to create or update head-label groups in `self.state_groups`.
7. **Metadata Saving:** Saves updated state groups to disk with `self._save_state_groups()`.
8. **Return Updated Groups:** Returns the updated dictionary containing head-label groups (`self.state_groups`).

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs complex operations for grouping and updating state relations based on provided parameters.

---

#### `_upsert_state_group`

### Function Documentation

**Purpose:**
Upsert (insert or update) a state group with the specified labels, updating metadata and related structures.

**Parameters:**
- `labels` (List[str]): A list of labels for states to be grouped.
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.
- `reasoner`: An object or function responsible for reasoning about the states and their relationships.

**Return Value:**
Dict: The updated dictionary containing details of the state group.

**Important Behaviour and Side-effects:**
1. **Key Generation:** Generates a unique key for the state group using `_state_relation_key(labels)`.
2. **Relation Keys Extraction:** Extracts relation keys from pairs of labels.
3. **Token Scores Calculation:**
   - Aggregates token scores across related states into `token_scores` using `Counter()`.
4. **Dominant Tokens Identification:** Identifies the most common tokens based on aggregated scores.
5. **Head Label Selection:** Selects a head label for the group using `_select_group_head_label(labels, token_scores, reasoner=reasoner)`.
6. **Centroid Calculation:** Computes the centroid vector of the state group using `_compute_group_centroid(labels, interaction_index=interaction_index)`.
7. **Factor Signature Computation:** Calculates the factor signature of the group using `_group_factor_signature(relation_keys)`.
8. **Probability and Reward Trace Calculation:**
   - Aggregates probabilities for each label.
   - Sums reward traces from metadata associated with labels.
9. **Label Set Creation:** Creates a set of child labels (`label_set`).
10. **Existing Group Archiving:**
    - Iterates through existing groups to archive any that are fully contained within the current group's labels.
11. **Group Update or Insertion:**
    - Updates an existing state group with new details if it exists, otherwise inserts a new one into `self.state_groups`.
12. **Metadata and Alias Updating:** 
   - Sets up metadata for each label to include its head label in the 'state_groups' list.
   - Saves updated metadata using `_save_state_metadata()`.
13. **Alias Mapping:** Maps child labels to their respective head labels.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs complex operations for grouping and updating state relations based on provided parameters.

---

#### `_select_group_head_label`

### Function Documentation

**Purpose:**
Select a head label for the state group based on dominant tokens, predefined phrases/definitions from a reasoner, or default fallback.

**Parameters:**
- `labels` (List[str]): A list of labels for states to be grouped.
- `token_scores` (Counter): A Counter object containing token scores.
- `reasoner`: An optional object or function responsible for reasoning about the states and their relationships.

**Return Value:**
str: The selected head label for the state group.

**Important Behaviour and Side-effects:**
1. **Token Evidence Floor Retrieval:** Retrieves a floor value for token evidence from configuration (`state_group_token_evidence_floor`).
2. **Max Words Retrieval:** Retrieves the maximum number of words allowed in the head label from configuration (`state_group_head_label_max_words`).
3. **Dominant Tokens Identification:**
   - Filters tokens with scores above or equal to `floor`.
4. **Head Label Selection Based on Dominant Tokens:**
   - If dominant tokens are available, selects up to `max_words` tokens and joins them into a single string.
5. **Reasoner Phrase Definitions Check:** 
   - Checks predefined phrases from the reasoner's `phrase_definitions`. Selects phrase if it matches any combination of tokens.
6. **Word Definition Check:**
   - Checks word definitions in the reasoner's `definitions` for matching tokens and returns the first match.
7. **Knowledge Graph Node Selection:**
   - If a knowledge graph is available, selects the node with the highest degree (number of edges) that matches any token.
8. **Fallback Head Label:** 
   - Returns the label of the top child state if no other criteria are met or returns a default fallback string `f"group_{top_child}"`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs complex operations for selecting head labels based on provided parameters and reasoner data.

---

#### `_dominant_phrases_for_tokens`

### Function Documentation

**Purpose:**
Generate a list of dominant phrases based on the provided tokens and predefined phrases from a reasoner.

**Parameters:**
- `dominant_tokens` (List[str]): A list of dominant tokens.
- `reasoner`: An optional object or function responsible for reasoning about the states and their relationships.

**Return Value:**
List[str]: A list of up to 5 dominant phrases that match any combination of provided tokens.

**Important Behaviour and Side-effects:**
1. **Token Set Creation:** Converts the list of dominant tokens into a set (`token_set`) for efficient intersection operations.
2. **Phrase Definitions Check:** Iterates through predefined phrases from `reasoner.phrase_definitions`.
3. **Intersection with Tokens:**
   - Splits each phrase into individual tokens and converts them to lowercase.
   - Checks if there is any overlap between the token set and the split phrase tokens using set intersection (`&`).
4. **Phrase Addition:** Adds matching phrases to a list of dominant phrases (`phrases`).
5. **Limiting Phrases:** Returns up to 5 dominant phrases.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for identifying and selecting relevant phrases based on provided tokens and reasoner data.

---

#### `_compute_group_centroid`

### Function Documentation

**Purpose:**
Compute the weighted centroid vector of a group of states based on their probabilities and reward traces.

**Parameters:**
- `labels` (List[str]): A list of labels for states to be grouped.
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.

**Return Value:**
np.ndarray: The normalized centroid vector of the group.

**Important Behaviour and Side-effects:**
1. **Probability Retrieval:** Retrieves state probabilities using `get_state_probabilities(interaction_index=interaction_index, include_archived=True)`.
2. **Vector and Weight Calculation:**
   - Iterates through each label to ensure it exists in states.
   - For each valid label:
     - Ensures metadata is available with `_ensure_state_probability_metadata(label, interaction_index)`.
     - Computes a weight based on the probability (`p`) and reward trace (`q`):
       \[
       \text{weight} = p \times (1 + \max(0.0, \tanh(q)))
       \]
   - Normalizes each state vector using `_normalize_vector(self.states[label])`.
3. **Empty Vectors Handling:** If no valid vectors are found, returns an empty numpy array.
4. **Vector Alignment:**
   - Ensures all vectors have the same length by padding shorter ones and truncating longer ones to match the maximum length among them.
5. **Weighted Centroid Calculation:**
   - Computes a weighted sum of aligned vectors using their respective weights.
6. **Normalization:** Normalizes the resulting centroid vector before returning it.

**Mathematical Formula:**
The weight calculation uses:
\[
\text{weight} = p \times (1 + \max(0.0, \tanh(q)))
\]

### Classification
This function involves a mathematical formula for calculating weights based on hyperbolic tangent of the reward trace.

---

#### `_group_factor_signature`

### Function Documentation

**Purpose:**
Compute the factor signature of a state group based on the average delta values from related states.

**Parameters:**
- `relation_keys` (List[str]): A list of keys representing related state pairs.

**Return Value:**
List[Dict]: A list of dictionaries containing top factors with their mean deltas and indices.

**Important Behaviour and Side-effects:**
1. **Delta Extraction:** Iterates through each relation key to extract factor delta histories.
2. **Vector Collection:** Collects non-empty vectors from the extracted delta entries into a list (`deltas`).
3. **Empty Deltas Handling:** If no valid deltas are found, returns an empty list.
4. **Alignment of Vectors:**
   - Ensures all collected vector arrays have the same length by padding shorter ones and truncating longer ones to match the maximum length among them.
5. **Average Delta Calculation:** Computes the average delta across aligned vectors (`avg_delta`).
6. **Top Factor Selection:**
   - Sorts indices of `avg_delta` in descending order based on mean delta values.
   - Selects top 5 factors with positive mean deltas and constructs a list of dictionaries containing their details.

**Mathematical Formula:**
No mathematical formula involved here, but the function involves averaging vectors and sorting indices.

### Classification
This function does not involve any specific mathematical formulas beyond basic operations like vector padding, averaging, and sorting.

---

#### `archive_legacy_conflicts`

### Function Documentation

**Purpose:**
Archive legacy conflicts by marking them as resolved and archived when automatic grouping is active.

**Parameters:**
- `interaction_index` (int, optional): An index representing a specific interaction or context. Defaults to 0 if not specified.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Conflict Iteration:** Iterates through each conflict in the list of conflicts (`self.conflicts`).
2. **Marking Resolved Conflicts:** For unresolved conflicts, marks them as resolved by setting `resolved=True`, updating their phase to 'archived', adding a resolution type ('legacy_archived'), and recording the interaction index when archived.
3. **Change Tracking:** Tracks whether any changes were made (`changed`).
4. **Saving Changes:** If any changes are detected, saves updated conflict records using `_save_conflicts()`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for archiving legacy conflicts and saving the updates to disk.

---

#### `detect_conflicts`

### Function Documentation

**Purpose:**
Scan all active states and find pairs with cosine similarity above a specified threshold. Returns detected conflicts including associated words from metadata.

**Parameters:**
- `threshold` (float, optional): The minimum required cosine similarity for considering states as conflicting. Defaults to 0.85 if not provided.
  
**Return Value:**
List[Dict]: A list of dictionaries containing details about each conflict found.

**Important Behaviour and Side-effects:**
1. **Legacy Interactive Conflicts Check:** If legacy interactive conflicts are enabled, proceed with the detection process; otherwise, return an empty list.
2. **Threshold Retrieval:** Retrieves the similarity threshold from configuration (`self.config_manager.get('state_manager.conflict_threshold', 0.85)`) if not provided directly.
3. **Active State Labels Retrieval:** Obtains a list of active state labels using `_active_state_labels()`.
4. **Conflict Detection:**
   - Iterates through each pair of states, ensuring pairs are processed only once by tracking them in `processed_pairs` set.
   - Normalizes the vector for the first label (`l1`) and retrieves associated words from metadata if available.
5. **Similarity Computation:** Computes cosine similarity between vectors of state pairs using `_cosine_similarity()`.
6. **Conflict Recording:**
   - If a pair has a similarity above the threshold, records details about the conflict including labels, similarity score, associated words (if any), and timestamp.
7. **Metadata Inclusion:** Includes metadata tokens from both states in the recorded conflicts.

**Mathematical Formula:**
The cosine similarity formula is used:
\[
\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
\]

### Classification
This function involves the use of a mathematical formula for cosine similarity.

---

#### `refresh_conflicts`

### Function Documentation

**Purpose:**
Re-detect conflicts and update the list of detected conflicts, preserving any existing resolution status if possible.

**Parameters:**
- `threshold` (float, optional): The minimum required cosine similarity for considering states as conflicting. Defaults to 0.85 if not provided.

**Return Value:**
List[Dict]: A list of dictionaries containing details about each conflict found and updated.

**Important Behaviour and Side-effects:**
1. **Legacy Interactive Conflicts Check:** If legacy interactive conflicts are enabled, proceed with the detection process; otherwise, archive existing conflicts using `archive_legacy_conflicts()` and return an empty list.
2. **New Conflict Detection:** Detect new conflicts by calling `detect_conflicts(threshold)`.
3. **Existing Conflicts Mapping:**
   - Create a mapping of existing conflicts (`existing_map`) where keys are tuples of sorted labels for each conflict.
4. **Conflict Merging:**
   - Iterate through the newly detected conflicts and check if they exist in the existing map.
   - If an existing conflict is found:
     - Preserve its resolution status, type, label, and stale information (if applicable).
5. **Update Conflicts List:** Replace `self.conflicts` with the merged list of new and updated conflicts (`merged`).
6. **Save Updates:** Save the updated conflicts to disk using `_save_conflicts()`.

**Mathematical Formula:**
No mathematical formula involved here, but it uses cosine similarity in the underlying conflict detection process.

### Classification
This function does not involve any specific mathematical formulas beyond those used in the `detect_conflicts` method.

---

#### `_save_conflicts`

### Function Documentation

**Purpose:**
Save the list of conflicts to a JSON file.

**Parameters:**
None

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Temporary File Creation:** Writes conflict data to a temporary file with the extension `.tmp`.
2. **JSON Dump:** Uses `json.dump()` to write the list of conflicts (`self.conflicts`) into the JSON format in the temporary file.
3. **File Replacement:** Replaces the existing conflict file with the newly created temporary file using `os.replace(tmp, self.CONFLICT_FILE)`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for saving conflicts to a JSON file.

---

#### `_load_conflicts`

### Function Documentation

**Purpose:**
Load the list of conflicts from a JSON file on disk.

**Parameters:**
None

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **File Existence Check:** Checks if the conflict file exists using `os.path.exists(self.CONFLICT_FILE)`.
2. **Loading Conflicts:**
   - If the file exists, attempts to open it in read mode.
   - Uses `json.load()` to parse the JSON content and store it in `self.conflicts`.
3. **Exception Handling:** Catches any exceptions that occur during loading (e.g., file corruption or invalid JSON format) and prints an error message.
4. **Fallback Initialization:** If an exception occurs, initializes `self.conflicts` as an empty list.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for loading conflicts from a JSON file on disk.

---

#### `_load_shadows`

### Function Documentation

**Purpose:**
Load all shadow states from the specified directory and store them in `self.shadow_states`.

**Parameters:**
None

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Clear Existing Shadows:** Clears any existing entries in `self.shadow_states`.
2. **Directory Existence Check:** Checks if the state directory exists using `os.path.exists(self.state_dir)`. If not, returns immediately.
3. **File Iteration:** Iterates through each file in the state directory.
4. **Shadow State Identification:**
   - Filters files that start with a specific prefix (`self.SHADOW_PREFIX`) and end with `.json`.
5. **Loading Shadow States:**
   - Opens each identified shadow state file for reading.
   - Uses `json.load()` to parse the JSON content from the file.
6. **Label Extraction:** Extracts the label of the shadow state, defaulting to a filename without extension if no explicit label is provided in the data.
7. **Vector Normalization:**
   - Retrieves and normalizes the vector associated with each shadow state using `_normalize_vector(np.array(vector))`.
8. **Exception Handling:** Catches any exceptions that occur during loading (e.g., file corruption or invalid JSON format) and prints an error message.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for loading shadow states from a directory.

---

#### `create_shadow_resolution`

### Function Documentation

**Purpose:**
Create a temporary shadow state from the probability-weighted group centroid of conflicting states.

**Parameters:**
- `conflict` (Dict): A dictionary containing details about the conflict, including labels and interaction index.

**Return Value:**
str: The label of the created shadow state.

**Important Behaviour and Side-effects:**
1. **Label Extraction:** Extracts the two conflicting labels (`l1`, `l2`) from the provided conflict.
2. **Centroid Computation:** Computes the probability-weighted group centroid vector for the given states using `_compute_group_centroid([l1, l2], interaction_index=int(conflict.get('interaction_index', 0) or 0))`.
3. **Shadow Label Creation:**
   - Creates a unique shadow label based on conflicting labels and current timestamp.
4. **Metadata Construction:** Constructs metadata for the new shadow state including merged_from (conflicting states), similarity score, merge type, and timestamp.
5. **Save Shadow State:** Saves the new shadow state using `save_state(shadow_label, merged_vector, metadata, is_shadow=True)`.
6. **Exception Handling:**
   - Catches any exceptions that occur during the process and prints an error message.
7. **Fallback Label:** Returns a fallback label if an exception occurs.

**Mathematical Formula:**
No mathematical formula involved here beyond those used in `_compute_group_centroid`.

### Classification
This function does not involve any specific mathematical formulas but performs operations for creating shadow states from conflicting state centroids.

---

#### `apply_shadow_to_conflict`

### Function Documentation

**Purpose:**
Resolve a detected conflict by marking it as resolved and associating it with the given shadow label.

**Parameters:**
- `conflict` (Dict): A dictionary containing details about the conflict.
- `shadow_label` (str): The label of the shadow state used to resolve the conflict.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Mark Conflict as Resolved:** Sets the resolved status of the conflict to True (`conflict["resolved"] = True`).
2. **Resolution Label Assignment:** Assigns the provided `shadow_label` to the resolution label field (`conflict["resolution_label"] = shadow_label`).
3. **Resolution Type Setting:** Specifies the type of resolution as "shadow" (`conflict["resolution_type"] = "shadow"`).
4. **Save Conflicts:** Saves the updated conflicts list using `_save_conflicts()`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for marking a conflict as resolved and saving updates to disk.

---

#### `resolve_conflict_permanently`

### Function Documentation

**Purpose:**
Resolve a detected conflict by permanently choosing one label as the canonical state and optionally deleting the other conflicting state.

**Parameters:**
- `conflict` (Dict): A dictionary containing details about the conflict.
- `winning_label` (str): The label of the chosen winning state that will be preserved.
- `delete_other` (bool, optional): Whether to delete the losing state. Defaults to True if not specified.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Label Extraction:** Extracts both conflicting labels (`l1`, `l2`) from the conflict dictionary.
2. **Determine Label to Delete:** Determines which label should be deleted based on the provided winning_label.
3. **Preserve Identity Before Deletion:** Updates aliases to map the losing state's label to the winning label (`self.aliases[to_delete] = winning_label`).
4. **Delete Losing State:**
   - If `delete_other` is True, deletes the conflicting state using `self.delete_state(to_delete)`.
5. **Mark Conflict as Resolved:** Sets the resolved status of the conflict to True (`conflict["resolved"] = True`).
6. **Resolution Label Assignment:** Assigns the winning label to the resolution label field (`conflict["resolution_label"] = winning_label`).
7. **Resolution Type Setting:** Specifies the type of resolution as "permanent" (`conflict["resolution_type"] = "permanent"`).
8. **Save Conflicts:** Saves the updated conflicts list using `_save_conflicts()`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for resolving a conflict and saving updates to disk.

---

#### `get_unresolved_conflicts`

### Function Documentation

**Purpose:**
Retrieve a list of unresolved conflicts from the stored conflict records.

**Parameters:**
None

**Return Value:**
List[Dict]: A list of dictionaries containing details about each unresolved conflict.

**Important Behaviour and Side-effects:**
1. **Legacy Interactive Conflicts Check:** If legacy interactive conflicts are not enabled, return an empty list.
2. **Filter Unresolved Conflicts:** Filter the stored conflicts to include only those that have not been marked as resolved (`not c.get("resolved", False)`).

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs operations for filtering and returning unresolved conflict records.

---


### FactorEngine
### Class Documentation

**Purpose:**
The `FactorEngine` class is responsible for deriving base control signals from the current system state and computing dynamic factors using an adaptive control space. It ensures all outputs are within the range [0,1].

**Key Attributes:**
- `control_space`: An instance of `AdaptiveControlSpace` used to manage and compute both static and dynamic control signals.
- `config_manager`: Optional configuration manager for handling parameter updates dynamically.

The class is responsible for calculating various factors such as curiosity, threat, trust, confidence, uncertainty, energy, attention, comprehension, fluency, and cognitive load based on input data like token analyses, diagnostic reports, phonetic scores, stability metrics, semantic similarity, CPU usage, and more. It also handles dynamic factor computation using the adaptive control space.

### Classification
This class does not involve any specific mathematical formulas but performs operations for deriving and computing various system factors and signals.

### Functions

#### `__init__`

### Function Documentation

**Purpose:**
Initialize the `FactorEngine` instance with a given adaptive control space and an optional configuration manager.

**Parameters:**
- `control_space` (AdaptiveControlSpace): An instance of the adaptive control space used to manage and compute factors.
- `config_manager` (`Optional[ConfigManager]`, optional): Optional configuration manager for handling parameter updates dynamically. Defaults to None if not specified.

**Return Value:**
None

**Important Behaviour and Side-effects:**
1. **Control Space Assignment:** Assigns the provided adaptive control space instance to `self.control_space`.
2. **Configuration Manager Assignment:** Optionally assigns a configuration manager instance to `self.config_manager`.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas but performs initialization operations for setting up key attributes of the class.

---

#### `update_parameters_from_config`

### Function Documentation

**Purpose:**
This function serves as a placeholder for updating parameters from the configuration during runtime (hot reloading).

**Parameters:**
None

**Return Value:**
None

**Important Behaviour and Side-effects:**
- This method currently does nothing (`pass` statement) but is intended to be implemented in the future to handle dynamic parameter updates.

**Mathematical Formula:**
No mathematical formula involved here.

### Classification
This function does not involve any specific mathematical formulas or computations. It serves as a placeholder for potential future functionality.

---

#### `compute`

### Function Documentation

**Purpose:**
Compute and return a dictionary containing all control signals (base factors + dynamic factors).

**Parameters:**
- `token_analyses` (List[Dict]): A list of dictionaries, each representing an analysis of a token.
- `ci_report` (Dict): A report from the confidence indicator system.
- `phon_score` (float): Phonetic score indicating pronunciation quality or related metrics.
- `diag` (Dict): Diagnostic information about system performance and issues.
- `concept_model`: An object representing the concept model used for semantic analysis.
- `meta_learner`: The meta learner instance, likely responsible for learning patterns in data.
- `reward_total` (float): Total accumulated reward value from interactions or tasks.
- `semantic_learner`: A semantic learner instance that provides word embeddings and similarity measures.

**Return Value:**
Dict[str, float]: A dictionary containing all control signals with keys as factor names and values as corresponding scores between 0 and 1.

**Important Behaviour and Side-effects:**
1. **Curiosity Calculation:** Determines the maximum curiosity score among tokens.
2. **Threat Calculation:** Computes a weighted sum of negative diagnostics count, confidence inverse, and phonetic score inverse, then clips it to [0, 1].
3. **Trust Initialization:** Initializes trust as zero (computed later).
4. **Confidence Assignment:** Directly assigns the CI report's confidence value.
5. **Uncertainty Calculation:** Computes average entropy of known words from concept model and normalizes it between 0 and 1.
6. **Energy Computation:** Calculates energy based on CPU usage, clipping to [0, 1].
7. **Attention Assignment:** Determines attention score based on the number of tokens in input, clipped to [0, 1].
8. **Comprehension Calculation:** Computes average semantic similarity among words using a semantic learner.
9. **Fluency Placeholder:** Assigns fluency as a placeholder value (defaulted to 0.7).
10. **Cognitive Load Computation:** Calculates cognitive load as the inverse of energy * attention, clipped to [0, 1].
11. **Trust Recomputation:** Updates trust based on confidence, stability from CI report, and comprehension.
12. **Dynamic Factors Calculation:**
    - Computes dynamic factors using `self.control_space.compute_factor(factor_name, state)`.
    - Records each computed factor value in the control space.
13. **Numeric Stability Check:** Ens

---



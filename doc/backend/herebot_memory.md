## Doc for herebot_memory
---

This Python file provides a unified memory facade (`HerebotMemoryBridge`) for backend engines to manage and synchronize runtime data with stable corpus storage. The module ensures that turn-based text is synced inline while episodic/reasoner/graph data is synchronized periodically or explicitly through snapshots. Key components include methods for adding runtime texts, syncing turns and snapshots, verifying engine alignment, and collecting various types of documents from the engine's memory.

Overall design involves using an `IncrementalCSR` backend to manage document storage with deduplication and source-aware handling, ensuring stable views over potentially unstable runtime data. The module also includes utility functions for cleaning text, generating stable hashes, and reporting synchronization status.

---

### SyncReport
The `SyncReport` class is responsible for generating and storing synchronization reports. It tracks various metrics such as the success status (`ok`), mode of sync (`mode`), number of added and skipped documents, errors encountered, source counts, corpus size, snapshot syncing status, and the number of documents seen during synchronization.

Key attributes include boolean flags like `ok`, `snapshot_synced`, lists for storing errors and sources with their respective counts, as well as integers tracking added, skipped, and document count metrics. The class provides a method to convert its state into a dictionary format for easy serialization or logging purposes.

### Functions

#### `to_dict`

The `to_dict` function converts the current state of a `SyncReport` object into a dictionary format.

Parameters:
- No parameters.

Return value:
A dictionary containing various synchronization metrics and status information, including:
- `"ok"`: Boolean indicating success status.
- `"mode"`: String representing the mode of sync (e.g., "turn", "snapshot").
- `"added"`: Integer count of added documents.
- `"skipped"`: Integer count of skipped documents.
- `"errors"`: List of errors encountered during synchronization.
- `"source_counts"`: Dictionary mapping source names to their respective counts.
- `"corpus_size"`: Integer representing the size of the corpus.
- `"snapshot_synced"`: Boolean indicating if a snapshot was synced.
- `"documents_seen"`: Integer count of documents seen during synchronization.

Important behavior:
- Converts each attribute of the `SyncReport` object into appropriate dictionary entries, ensuring that lists and dictionaries are properly converted to their respective types in the output dictionary.

Side effects:
- No side effects; it only reads attributes from the current instance and constructs a dictionary.

---


### VerificationReport
The `VerificationReport` class is responsible for generating and storing verification reports. It tracks various metrics such as the success status (`ok`), corpus size, source counts, missing sources, errors encountered, alignment of current input with the backend corpus, CI corpus size, and phonology profile activity.

Key attributes include boolean flags like `ok`, `current_input_aligned`, lists for storing missing sources and errors, dictionaries for source counts, and integers or booleans representing other metrics. The class provides a method to convert its state into a dictionary format for easy serialization or logging purposes.

### Functions

#### `to_dict`

The `to_dict` function converts the current state of a `VerificationReport` object into a dictionary format.

Parameters:
- No parameters.

Return value:
A dictionary containing various verification metrics and status information, including:
- `"ok"`: Boolean indicating success status.
- `"corpus_size"`: Integer representing the size of the corpus.
- `"source_counts"`: Dictionary mapping source names to their respective counts.
- `"missing_sources"`: List of missing sources.
- `"errors"`: List of errors encountered during verification.
- `"current_input_aligned"`: Boolean indicating if the current input is aligned with the backend corpus.
- `"ci_corpus_size"`: Optional integer representing the size of the CI (Core Inference) corpus.
- `"phon_profile_active"`: Optional boolean indicating if a phonology profile is active.

Important behavior:
- Converts each attribute of the `VerificationReport` object into appropriate dictionary entries, ensuring that lists and dictionaries are properly converted to their respective types in the output dictionary.

Side effects:
- No side effects; it only reads attributes from the current instance and constructs a dictionary.

---


### HerebotMemoryBridge
The `HerebotMemoryBridge` class serves as a unified memory facade for backend engines to manage and synchronize runtime data with stable corpus storage. It ensures that current-turn text is synced inline while episodic/reasoner/graph data is synchronized periodically or explicitly through snapshots.

Key attributes include:
- `csr`: An instance of the core state representation (CSR) engine.
- `snapshot_interval_turns` and `snapshot_interval_seconds`: Intervals for snapshot synchronization based on turns or time, respectively.
- `_lock`: A threading lock to ensure thread-safe operations.
- `_turn_count`, `_last_snapshot_turn`, `_last_snapshot_time`: Track turn count and timing of snapshots.
- `_last_sync_report` and `_last_verification_report`: Store the last sync and verification reports.

The class is responsible for adding runtime texts, syncing turns and snapshots, retrieving the corpus with optional limits, verifying engine alignment, collecting documents from various sources (interaction history, episodes, reasoner memory, knowledge graph), deduping collected documents, and handling errors during these operations.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `HerebotMemoryBridge` class, setting up various attributes and configurations for managing memory synchronization.

Parameters:
- `csr`: An optional instance of the core state representation (CSR) engine.
- `csr_config_path`: Path to the CSR configuration file (default: "config_listner.json").
- `config_manager`: A configuration manager object.
- `snapshot_interval_turns`: Optional interval in turns for snapshot synchronization.
- `snapshot_interval_seconds`: Optional interval in seconds for snapshot synchronization.
- `corpus_limit`: An optional limit on corpus size.

Return value:
None

Important behavior:
- Initializes the CSR engine using either a provided instance or by creating one with configuration from `_config`.
- Sets up intervals (`snapshot_interval_turns` and `snapshot_interval_seconds`) based on provided values or default configurations.
- Configures various attributes such as lock, turn count, snapshot timing variables, and last reports.

Side effects:
- Initializes internal state variables like locks, counters, and report storages.

---

#### `_config`

The `_config` function retrieves a configuration value using a provided key and returns it or a default value if the key is not found.

Parameters:
- `key`: The configuration key (type: str).
- `default`: The default value to return if the key is not found (any type).

Return value:
A configuration value of any type, or the specified default value if an exception occurs.

Important behavior:
- Uses a configuration manager (`self.config_manager`) to retrieve values.
- Returns the default value if no configuration manager is available or if an exception occurs during retrieval.

Side effects:
- None; it only reads from the configuration and does not modify any external state.

---

#### `add_runtime_text`

The `add_runtime_text` function adds a cleaned text to the core state representation (CSR) engine with optional metadata and returns a report of the operation.

Parameters:
- `text`: The input text (type: str).
- `source`: The source identifier for the text (default: "runtime", type: str).
- `doc_id`: An optional document ID (type: Optional[str]).
- `metadata`: Optional additional metadata dictionary (type: Optional[Dict[str, Any]]).

Return value:
A dictionary containing a report of whether the addition was successful and any relevant details such as added status, skipped status, reason for skipping, and document key.

Important behavior:
- Cleans the input text using `_clean_text`.
- Generates a stable hash ID if no `doc_id` is provided.
- Adds the cleaned text to the CSR engine with specified source and metadata.
- Returns a report indicating success or failure of adding the text.

Side effects:
- Modifies the internal state by potentially adding new documents to the CSR engine.

---

#### `sync_turn`

The `sync_turn` function synchronizes a single turn of text with the backend memory, optionally triggering a full snapshot synchronization based on specified intervals.

Parameters:
- `text`: The input text to be synchronized (type: str).
- `engine`: An optional engine object for additional operations during synchronization (type: Any).

Return value:
A `SyncReport` instance containing details about the sync operation, including added and skipped documents, errors encountered, snapshot status, etc.

Important behavior:
- Increments the turn count using a lock to ensure thread safety.
- Synchronizes the provided text with metadata indicating it is user input from this turn.
- Optionally triggers a full snapshot synchronization if an engine object is provided and conditions for snapshots are met.
- Aggregates results of both inline sync and potential snapshot into a single `SyncReport`.
- Finishes the report by updating source counts, corpus size, etc.

Side effects:
- Modifies internal state variables such as turn count, last snapshot details, and synchronization reports.

---

#### `sync_snapshot`

The `sync_snapshot` function synchronizes a snapshot of documents collected from an engine to the backend memory and returns a synchronization report.

Parameters:
- `engine`: An engine object used for collecting documents (type: Any).

Return value:
A `SyncReport` instance containing details about the sync operation, including added and skipped documents, errors encountered, etc.

Important behavior:
- Collects documents from the provided engine using `_collect_engine_documents`.
- Synchronizes each collected document with metadata indicating its source.
- Updates internal state variables such as last snapshot turn count and time under a lock to ensure thread safety.
- Finishes the report by updating source counts, corpus size, etc.

Side effects:
- Modifies internal state variables like `last_snapshot_turn`, `last_snapshot_time`, and synchronization reports.

---

#### `get_corpus`

The `get_corpus` function retrieves the backend memory corpus with an optional limit on the number of texts.

Parameters:
- `limit`: An optional integer specifying a maximum number of texts to retrieve (type: Optional[int]).

Return value:
A list of strings representing the retrieved text documents from the corpus.

Important behavior:
- Uses either `csr.get_all_texts` or `csr.get_document_texts` based on available methods.
- Applies an optional limit when retrieving texts, using either a provided limit or the internal `corpus_limit`.
- Prints an error message if an exception occurs during retrieval and returns an empty list in such cases.

Side effects:
- Logs any exceptions that occur during corpus retrieval.

---

#### `verify_engine_alignment`

The `verify_engine_alignment` function verifies the alignment between an engine's state and the backend memory corpus, returning a verification report.

Parameters:
- `engine`: An engine object used for verification (type: Any).

Return value:
A `VerificationReport` instance containing details about the verification process, including success status (`ok`), corpus size, source counts, missing sources, errors encountered, etc.

Important behavior:
- Retrieves and processes the backend memory corpus.
- Checks if the last input from the engine's interaction history is aligned with the current corpus.
- Adds relevant information such as CI corpus size and phonology profile activity to the report.
- Updates internal state variables like `_last_verification_report`.

Side effects:
- Logs any exceptions that occur during verification.

---

#### `_should_snapshot`

The `_should_snapshot` function determines whether a snapshot synchronization should be performed based on turn count and elapsed time intervals.

Parameters:
- `turn`: The current turn number (type: int).

Return value:
A boolean indicating whether a snapshot sync is due (`True`) or not (`False`).

Important behavior:
- Checks if both the interval in turns and seconds are non-positive, returning `False` immediately if so.
- Uses a lock to ensure thread safety when accessing shared state variables `_last_snapshot_turn` and `_last_snapshot_time`.
- Calculates whether it's time for a snapshot based on turn count difference or elapsed time since the last snapshot.

Side effects:
- Ensures that access to internal state is thread-safe by using a lock.

---

#### `_sync_one`

The `_sync_one` function synchronizes a single text document with the backend memory and updates a synchronization report.

Parameters:
- `report`: A `SyncReport` instance to track sync results (type: SyncReport).
- `text`: The input text or other object containing text data (type: Any).
- `source`: The source identifier for the text (type: str).
- `doc_id`: An optional document ID (type: Optional[str]).
- `metadata`: Optional additional metadata dictionary (default is an empty dict, type: Optional[Dict[str, Any]]).

Return value:
None

Important behavior:
- Cleans the input text using `_clean_text`.
- Adds the cleaned text to the backend memory via `add_runtime_text` with specified source and metadata.
- Updates the sync report based on whether adding the document was successful or not (increases added/skipped counts accordingly).
- Appends any encountered errors to the report's error list.

Side effects:
- Modifies the synchronization report by updating its attributes.

---

#### `_finish_report`

The `_finish_report` function finalizes a synchronization report by updating its status and other attributes.

Parameters:
- `report`: A `SyncReport` instance to be finalized (type: SyncReport).

Return value:
None

Important behavior:
- Sets the success status (`ok`) of the report based on whether any errors were encountered.
- Updates the source counts in the report using `_source_counts`.
- Calculates and sets the corpus size by calling `get_corpus()`.
- Stores the finalized report as the last sync report.

Side effects:
- Modifies the provided synchronization report and updates internal state variables like `_last_sync_report`.

---

#### `_source_counts`

The `_source_counts` function retrieves and returns the counts of documents from each source in the backend memory.

Parameters:
- No parameters.

Return value:
A dictionary mapping source names to their respective document counts (type: Dict[str, int]).

Important behavior:
- Calls `csr.get_source_counts()` if available.
- Catches any exceptions that occur during retrieval and prints an error message before returning an empty dictionary.

Side effects:
- Logs any exceptions that occur during the process of retrieving source counts.

---

#### `_collect_engine_documents`

The `_collect_engine_documents` function collects documents from various sources within an engine and returns a deduplicated list of document dictionaries.

Parameters:
- `engine`: An object representing the engine to collect documents from (type: Any).

Return value:
A list of dictionaries, each containing details about a collected document (type: List[Dict[str, Any]]).

Important behavior:
- Calls helper functions `_collect_interaction_history`, `_collect_episodes`, `_collect_reasoner_memory`, and `_collect_knowledge_graph` to gather documents from different parts of the engine.
- Deduplicates the list of collected documents using `_dedupe_docs`.

Side effects:
- Modifies internal state by potentially adding new document entries.

---

#### `_collect_interaction_history`

The `_collect_interaction_history` function collects interaction history from an engine and appends cleaned text documents to the provided list of document dictionaries.

Parameters:
- `engine`: An object representing the engine containing interaction history (type: Any).
- `docs`: A list of dictionaries where collected documents will be appended (type: List[Dict[str, Any]]).

Return value:
None

Important behavior:
- Retrieves up to 250 recent entries from the engine's interaction history.
- Cleans each text entry using `_clean_text`.
- Appends cleaned texts as new document entries with source and metadata information.
- Handles exceptions by appending an error document if an exception occurs.

Side effects:
- Modifies the provided list of documents (`docs`) by adding collected interaction history entries.

---

#### `_collect_episodes`

The `_collect_episodes` function collects episodes from an engine's episodic memory and appends cleaned text documents to the provided list of document dictionaries.

Parameters:
- `engine`: An object representing the engine containing episodic memory (type: Any).
- `docs`: A list of dictionaries where collected documents will be appended (type: List[Dict[str, Any]]).

Return value:
None

Important behavior:
- Retrieves up to 500 recent episodes from the engine's episodic memory.
- Cleans each user input and system response using `_clean_text`.
- Appends cleaned texts as new document entries with source and metadata information for both user inputs and system responses.
- Handles exceptions by appending an error document if an exception occurs.

Side effects:
- Modifies the provided list of documents (`docs`) by adding collected episode entries.

---

#### `_collect_reasoner_memory`

The `_collect_reasoner_memory` function collects documents from an engine's reasoner memory and appends cleaned text documents to the provided list of document dictionaries.

Parameters:
- `engine`: An object representing the engine containing reasoner memory (type: Any).
- `docs`: A list of dictionaries where collected documents will be appended (type: List[Dict[str, Any]]).

Return value:
None

Important behavior:
- Retrieves and processes various types of data from the reasoner's definitions, phrase definitions, relations, and sentence metadata.
- Cleans each text entry using `_clean_text`.
- Appends cleaned texts as new document entries with source and metadata information for different types of reasoner memory documents.
- Handles exceptions by appending an error document if an exception occurs.

Side effects:
- Modifies the provided list of documents (`docs`) by adding collected reasoner memory entries.

---

#### `_collect_knowledge_graph`

The `_collect_knowledge_graph` function collects documents from an engine's knowledge graph and appends cleaned text documents to the provided list of document dictionaries.

Parameters:
- `engine`: An object representing the engine containing a reasoner with a knowledge graph (type: Any).
- `docs`: A list of dictionaries where collected documents will be appended (type: List[Dict[str, Any]]).

Return value:
None

Important behavior:
- Retrieves and processes nodes and relations from the knowledge graph.
- Cleans each text entry using `_clean_text`.
- Appends cleaned texts as new document entries with source and metadata information for both node labels and relation descriptions.
- Handles exceptions by appending an error document if an exception occurs.

Side effects:
- Modifies the provided list of documents (`docs`) by adding collected knowledge graph entries.

---

#### `_iter_graph_edges`

The `_iter_graph_edges` function retrieves all edges from a knowledge graph and returns them as an iterable.

Parameters:
- `kg`: An object representing the knowledge graph (type: Any).

Return value:
An iterable containing all edges in the knowledge graph (type: Iterable).

Important behavior:
- Calls `get_all_edges()` on the provided knowledge graph to retrieve its edges.
- Returns an empty list if an exception occurs during edge retrieval.

Side effects:
- None; it only retrieves and returns data without modifying external state.

---

#### `_collect_dict_texts`

The `_collect_dict_texts` function collects and appends cleaned text documents from a dictionary of values to the provided list of document dictionaries.

Parameters:
- `docs`: A list of dictionaries where collected documents will be appended (type: List[Dict[str, Any]]).
- `values`: A dictionary containing key-value pairs for processing (type: Dict[Any, Any]).
- `source`: The source identifier for the text entries (type: str).
- `template`: A string template used to format each value entry as a text document (type: str).

Return value:
None

Important behavior:
- Iterates over each key-value pair in the provided dictionary.
- Formats and cleans each value using the specified template and `_clean_text`.
- Appends cleaned texts as new document entries with source, metadata, and stable hash ID if the resulting text is not empty.

Side effects:
- Modifies the provided list of documents (`docs`) by adding collected dictionary entry documents.

---

#### `_dedupe_docs`

The `_dedupe_docs` function removes duplicate documents from a list of document dictionaries.

Parameters:
- `docs`: A list of dictionaries representing the input documents (type: List[Dict[str, Any]]).

Return value:
A list of de-duplicated dictionaries representing cleaned and unique documents (type: List[Dict[str, Any]]).

Important behavior:
- Cleans each text entry using `_clean_text`.
- Uses a set to track seen document keys based on source and stable hash of the clean text.
- Skips adding duplicates or empty/error entries to the deduped list.
- Modifies the input documents by cleaning their texts before appending them to the deduplicated list.

Side effects:
- Modifies the provided list of documents (`docs`) by potentially removing duplicate entries.

---

#### `_error_doc`

The `_error_doc` function creates a dictionary representing an error document.

Parameters:
- `source`: The source identifier for the error (type: str).
- `exc`: An exception object containing details about the error (type: Exception).

Return value:
A dictionary representing the error document with keys `"text"`, `"source"`, `"doc_id"`, and `"metadata"` (type: Dict[str, Any]).

Important behavior:
- Sets the text to an empty string.
- Marks the source as "bridge:error".
- Generates a stable hash ID using `_stable_hash` based on the error type and message.
- Includes metadata with the original source and formatted exception message.

Side effects:
- None; it only constructs and returns a dictionary representing an error document.

---


---

## Global Functions

### Functions

#### `_clean_text`

### Function Documentation:

**Purpose:**  
The `_clean_text` function takes a given input `text`, converts it to a string (if not already), and then removes any extra whitespace, including leading and trailing spaces. It returns the cleaned text as a single line with words separated by exactly one space.

**Parameters:**
- `text`: Any type (`Any`) - The input value that needs to be cleaned up into a properly formatted string.

**Return Value:**  
The function returns a `str` (string) which is the cleaned version of the input text, where all extra whitespace has been removed and words are separated by exactly one space. If the input was originally `None`, an empty string (`""`) is returned instead.

**Important Behaviour/Side-effects:**
- The function converts any non-string inputs to a string before processing.
- It ensures that there is only one space between each word in the final output, and no leading or trailing spaces are present.

---

#### `_stable_hash`

### Function Documentation:

**Purpose:**  
The `_stable_hash` function generates a stable hash value from the given input `parts`. It concatenates cleaned versions of each part with null bytes (`\0`) in between, then computes and returns an SHA-256 hash of this concatenated string.

**Parameters:**
- `*parts`: Any type (`Any`) - Variable number of parts that need to be included in the hashing process. Each part will be individually cleaned before concatenation.

**Return Value:**  
The function returns a `str` (string) representing the hexadecimal digest of an SHA-256 hash of the concatenated, cleaned input parts.

**Important Behaviour/Side-effects:**
- The `_clean_text` helper function is called for each part to ensure that they are properly formatted before concatenation.
- Parts are joined with null bytes (`\0`) between them rather than spaces or other delimiters.
- An SHA-256 hash (a cryptographic hashing algorithm) is computed from the concatenated string, and its hexadecimal digest is returned.

**Mathematical Formula Classification:**  
No specific mathematical formula is mentioned in this function. The primary operation involves a cryptographic hash function (`SHA-256`).

---


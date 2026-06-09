## Doc for incremental_csr_lib
---

This Python file defines an `IncrementalCSR` class for building and managing a large sparse matrix (Compressed Sparse Row format) in a memory-efficient manner. The system is designed to handle streaming data with self-optimizing features like dynamic configuration reloading, background health monitoring, and true streaming rebuild capabilities.

Key components include:
1. **Configuration Management**: Parameters are loaded from a JSON file that can be updated dynamically.
2. **HashingVectorizer**: Utilizes `sklearn`'s HashingVectorizer to convert text data into sparse matrix rows without vocabulary explosion.
3. **Sharding Mechanism**: Sentences are batched and written as multi-row shards, which helps in managing memory usage efficiently.
4. **Background Monitoring Thread**: Continuously monitors the system for configuration changes, updates statistics, and triggers auto-rebuilds when necessary.
5. **True Streaming Rebuild**: Merges all shard files into a single main matrix using memory-mapped arrays to minimize peak RAM usage.

The overall design ensures zero-downtime ingestion of data while maintaining efficient storage and processing capabilities through dynamic adjustments based on system load and configuration updates.

---

### IncrementalCSR
The `IncrementalCSR` class is designed to build and manage a large sparse matrix in Compressed Sparse Row format incrementally. It reads parameters from a JSON configuration file, which can be updated dynamically during runtime. The system ensures thread safety through the use of locks and handles memory-efficient batch processing by accumulating sentences into multi-row shards before writing them to disk.

Key attributes include:
- `config_path`: Path to the JSON config file.
- `_pending_sentences` and `_pending_doc_refs`: Buffers for pending text data and document references that are yet to be flushed as a shard.
- `lock`: A thread lock to ensure synchronization during concurrent operations.
- `_stats`: Statistics cache updated by a background monitoring thread, tracking various metrics like total rows, non-zero entries (nnz), number of shards, etc.

The class is responsible for adding sentences and documents efficiently while maintaining memory constraints. It also handles periodic rebuilds that merge all shard files into the main matrix using memory-mapped arrays to minimize peak RAM usage. Additionally, it provides utility methods for monitoring system health and retrieving statistics about the sparse matrix and indexed documents.

### Functions

#### `__init__`

### Function Documentation:

**Purpose:**  
The `__init__` method initializes an instance of the `IncrementalCSR` class. It sets up necessary attributes, loads or creates a configuration file, and starts a background monitoring thread.

**Parameters:**
- `config_path`: A string (`str`) representing the path to the JSON config file (default is `"config.json"`).

**Return Value:**  
The method does not return any value; it initializes an instance of the class.

**Important Behaviour/Side-effects:**
1. **Configuration Path Resolution**: 
   - If `config_path` is absolute, sets `self.base_dir` to its directory and uses `config_path`.
   - Otherwise, sets `self.base_dir` to the current file's directory and constructs an appropriate path for `config_path`.

2. **Default Configuration Creation**:
   - If the specified configuration file does not exist, a default configuration is written there using `DEFAULT_CONFIG`.

3. **Configuration Loading**: 
   - Reads the JSON config file into a dictionary named `config`.
   
4. **Applying Configuration**:
   - Calls `_update_config` to set instance attributes based on the loaded configuration.
   - Loads an existing document index or creates one if it does not exist.

5. **HashingVectorizer Initialization**:
   - If a pickled HashingVectorizer exists, loads and checks its `n_features`.
   - Otherwise, initializes a new HashingVectorizer with specified parameters (`self.n_features`, binary=True, ngram_range=(1, 2), norm=None, alternate_sign=False) and saves it to disk.

6. **Pending Batch Attributes**:
   - Initializes attributes for pending sentences and document references.
   
7. **Statistics Cache Initialization**: 
   - Sets up a dictionary `_stats` with various statistics counters (e.g., total rows, non-zero entries).
   
8. **Background Monitor Thread Start**:
   - Starts a background thread that runs the `_background_monitor` method to handle periodic tasks like configuration reloading and stats updating.

This initialization ensures that all necessary components are set up correctly before any operations can be performed on the instance of `IncrementalCSR`.

---

#### `_update_config`

### Function Documentation:

**Purpose:**  
The `_update_config` method updates the instance attributes of an `IncrementalCSR` object based on a provided configuration dictionary. It ensures that necessary directories exist and converts string data types to actual numpy dtypes if needed.

**Parameters:**
- `config`: A dictionary (`Dict[str, Any]`) containing key-value pairs for updating instance attributes.
- `initial`: A boolean (`bool`, default is `False`). If set to `True`, it indicates an initial setup where all attributes are created from the configuration.

**Return Value:**  
The method does not return any value; it updates instance attributes in place.

**Important Behaviour/Side-effects:**
1. **Attribute Update**: 
   - Iterates through each key-value pair in the provided config dictionary and sets corresponding instance attributes using `setattr`.

2. **Directory Creation**:
   - Constructs paths for directories like `main_npz`, `vectorizer_pkl`, `shard_dir`, and `archive_dir` by joining them with `self.base_dir`.
   - Ensures that these directories exist by creating them if they do not, using `os.makedirs`.

3. **Data Type Conversion**:
   - Converts the instance's `dtype` attribute from a string to an actual numpy data type (`np.dtype`) if it is still in string format.

4. **Document Index JSON Path Handling**: 
   - Sets default value for `document_index_json` as `"document_index.json"` if not already set.
   - Ensures that the path provided or constructed for `document_index_json` is absolute by joining with `self.base_dir`.

This method ensures that all necessary attributes are updated correctly and directories exist, making it thread-safe when called within a lock context.

---

#### `_empty_document_index`

### Function Documentation:

**Purpose:**  
The `_empty_document_index` method returns a default empty structure for the document index. This structure includes fields to store version information, documents, and source content keys.

**Parameters:**
- No parameters are required or used in this function.

**Return Value:**  
Returns a dictionary (`Dict[str, Any]`) with predefined keys:
- `"version"`: An integer representing the current version of the document index (default is `1`).
- `"documents"`: A nested dictionary to store documents.
- `"source_content_keys"`: Another nested dictionary for source content keys.

**Important Behaviour/Side-effects:**  
The function does not have any side effects; it simply returns a predefined empty structure used as an initial or default document index. This ensures that the document index always has consistent fields, even when first initialized.

No mathematical formula is involved in this function.

This method initializes a clean and structured dictionary to represent the document index, ensuring consistency across different operations involving documents and their metadata.

---

#### `_load_document_index`

### Function Documentation:

**Purpose:**  
The `_load_document_index` method loads the document index from a JSON file and returns it. If there is an issue with loading or accessing the file, it returns an empty default structure.

**Parameters:**
- No parameters are required or used in this function.

**Return Value:**  
Returns a dictionary (`Dict[str, Any]`) representing the loaded document index:
- `"version"`: An integer indicating the version of the document index.
- `"documents"`: A nested dictionary containing documents.
- `"source_content_keys"`: Another nested dictionary for source content keys.

**Important Behaviour/Side-effects:**  
1. **Check if `document_index_json` is defined**: 
   - If `self.document_index_json` does not exist as an attribute, the method returns a default empty document index structure from `_empty_document_index`.

2. **File Existence Check**:
   - If the JSON file specified by `self.document_index_json` does not exist, it also returns the default empty document index.

3. **Load and Validate Data**: 
   - Opens and reads the JSON file to load data into a dictionary.
   - Ensures that the loaded data is of type `dict`. Otherwise, it returns an empty document index structure.
   
4. **Set Defaults**:
   - Sets defaults for `"version"`, `"documents"`, and `"source_content_keys"` keys if they do not already exist in the loaded data.

5. **Exception Handling**: 
   - Catches any exceptions that occur during file reading or JSON loading, prints a warning message with details of the exception, and returns an empty document index structure.

This method ensures that even if there are issues with accessing or parsing the document index file, it still provides a consistent default structure for further operations.

---

#### `_save_document_index`

### Function Documentation:

**Purpose:**  
The `_save_document_index` method saves the document index to a JSON file. It ensures that necessary directories are created if they do not exist and uses a temporary file for writing before replacing the original file.

**Parameters:**
- No parameters are required or used in this function.

**Return Value:**  
This method does not return any value; it performs side effects such as saving the document index to disk.

**Important Behaviour/Side-effects:**

1. **Check if `document_index_json` is defined**: 
   - If `self.document_index_json` does not exist as an attribute, the function returns immediately without performing further actions.
   
2. **Create Directories**:
   - Ensures that the directory path for the document index file exists by using `os.makedirs`, creating intermediate directories if necessary.

3. **Write to Temporary File**: 
   - Creates a temporary JSON file with a `.tmp` extension in the same location as the final destination.
   - Opens and writes the current `_document_index` dictionary to this temporary file in UTF-8 encoding, formatted with indentation for readability using `json.dump`.

4. **Replace Original File**:
   - Renames the temporary file to replace the original document index file using `os.replace`, ensuring atomicity of the write operation.

This method ensures that the document index is saved safely and atomically, preventing partial writes or corruption in case of interruptions during writing.

---

#### `_normalize_document_text`

### Function Documentation:

**Purpose:**  
The `_normalize_document_text` method normalizes a given input string by removing extra whitespace and ensuring that words are separated by exactly one space.

**Parameters:**
- `text`: A string (`str`) representing the text to be normalized. If it is `None`, an empty string is returned.

**Return Value:**  
Returns a normalized version of the input text as a `str`.

**Important Behaviour/Side-effects:**

1. **Check for None Input**: 
   - If the input `text` is `None`, returns an empty string (`""`).

2. **Normalization Process**:
   - Converts any non-string inputs to strings.
   - Splits the text into words, removes extra whitespace between them, and then joins the words back together with a single space.

This method ensures that all input texts are consistently formatted without excessive or inconsistent spacing. No mathematical formula is involved in this function.

---

#### `_hash_text`

### Function Documentation:

**Purpose:**  
The `_hash_text` method generates a SHA-256 hash of the input string.

**Parameters:**
- `text`: A string (`str`) representing the text to be hashed. Any non-string inputs will result in an error since only strings are accepted as valid input.

**Return Value:**  
Returns a hexadecimal representation of the SHA-256 hash as a `str`.

**Important Behaviour/Side-effects:**

1. **Encoding**: 
   - The input string is encoded to bytes using UTF-8 encoding, ignoring any characters that cannot be encoded (errors="ignore").

2. **Hash Generation**:
   - Computes the SHA-256 hash of the byte-encoded text.
   
3. **Return Hexdigest**:
   - Returns the hexadecimal representation of the computed hash.

This method ensures consistent and secure hashing of input strings, which can be used for various purposes such as deduplication or content verification. No mathematical formula is involved in this function.

---

#### `_doc_key`

### Function Documentation:

**Purpose:**  
The `_doc_key` method generates a SHA-256 hash of a concatenated string formed by combining the `source`, null byte (`\0`), and `doc_id`.

**Parameters:**
- `doc_id`: A string (`str`) representing the document ID.
- `source`: A string (`str`) representing the source identifier.

**Return Value:**  
Returns a hexadecimal representation of the SHA-256 hash as a `str`.

**Important Behaviour/Side-effects:**

1. **Concatenation**: 
   - Combines the `source`, null byte (`\0`), and `doc_id` into a single string.
   
2. **Encoding**: 
   - Encodes the concatenated string to bytes using UTF-8 encoding, ignoring any characters that cannot be encoded (errors="ignore").

3. **Hash Generation**:
   - Computes the SHA-256 hash of the byte-encoded concatenated string.

4. **Return Hexdigest**:
   - Returns the hexadecimal representation of the computed hash.

This method ensures consistent and secure hashing for uniquely identifying documents based on their source and ID, which can be used in various applications such as deduplication or indexing. No mathematical formula is involved in this function.

---

#### `_source_content_key`

### Function Documentation:

**Purpose:**  
The `_source_content_key` method generates a SHA-256 hash of a concatenated string formed by combining the `source`, null byte (`\0`), and `text_hash`.

**Parameters:**
- `source`: A string (`str`) representing the source identifier.
- `text_hash`: A string (`str`) representing an existing text hash.

**Return Value:**  
Returns a hexadecimal representation of the SHA-256 hash as a `str`.

**Important Behaviour/Side-effects:**

1. **Concatenation**: 
   - Combines the `source`, null byte (`\0`), and `text_hash` into a single string.
   
2. **Encoding**: 
   - Encodes the concatenated string to bytes using UTF-8 encoding, ignoring any characters that cannot be encoded (errors="ignore").

3. **Hash Generation**:
   - Computes the SHA-256 hash of the byte-encoded concatenated string.

4. **Return Hexdigest**:
   - Returns the hexadecimal representation of the computed hash.

This method ensures consistent and secure hashing for uniquely identifying content based on its source and text hash, which can be used in various applications such as deduplication or indexing. No mathematical formula is involved in this function.

---

#### `add`

### Function Documentation:

**Purpose:**  
The `add` method adds a single sentence to the pending shard queue. The sentence is added as part of a batch and will eventually be written to disk when the batch size threshold is met.

**Parameters:**
- `sentence`: A string (`str`) representing the text to add.

**Return Value:**  
Returns a dictionary (`Dict[str, Any]`), which contains metadata about the added sentence. The returned dictionary typically includes details such as whether it was successfully added and any relevant identifiers or status information.

**Important Behaviour/Side-effects:**

1. **Batching**: 
   - Calls `self.add_batch([sentence])`, adding the single sentence to a batch.
   
2. **Return Metadata**:
   - Returns the first element of the list returned by `add_batch`, which is metadata about the added sentence.

This method ensures that individual sentences are efficiently managed and processed in batches, reducing the overhead associated with frequent disk writes. No mathematical formula is involved in this function.

---

#### `_queue_sentences`

### Function Documentation:

**Purpose:**  
The `_queue_sentences` method adds multiple sentences to the pending shard queue. It also handles document references, stores them in memory buffers, and triggers a flush if necessary based on batch size.

**Parameters:**
- `sentences`: A list of strings (`List[str]`) representing the text sentences to add.
- `doc_refs`: An optional list of strings (`Optional[List[str]]`). If provided, it contains document references corresponding to each sentence. Defaults to `[None] * len(sentences)` if not specified.

**Return Value:**  
Returns a list of dictionaries (`List[Dict[str, Any]]`), where each dictionary represents metadata for the added sentences:
- `"shard"`: The shard identifier (set to `None` initially).
- `"row_index"`: The row index in the CSR matrix (unknown until flushed and vectorized).
- `"nnz"`: Non-zero entries count (unknown until flushed and vectorized).
- `"doc_key"`: Document reference key.

**Important Behaviour/Side-effects:**

1. **Input Validation**: 
   - If `sentences` is empty, returns an empty list immediately.
   
2. **Document References Handling**:
   - Ensures that the length of `doc_refs` matches the number of sentences by defaulting to `[None] * len(sentences)` if not provided or mismatched.

3. **Lock Management**: 
   - Uses a lock (`self.lock`) to ensure thread safety while modifying shared attributes.
   
4. **Pending Buffers Update**:
   - Initializes `_pending_sentences` and `_pending_doc_refs` lists if they do not exist yet.
   - Extends the pending buffers with new sentences and document references.

5. **Update Row Count**: 
   - Increments `_pending_rows` by the number of added sentences.

6. **Metadata Generation**:
   - Creates metadata dictionaries for each sentence, including placeholders for shard information that will be filled later.

7. **Auto-Flush Check**:
   - If `auto_flush` is enabled and the pending row count reaches or exceeds `batch_size`, calls `_flush_batch()` to write the current batch of sentences to a shard file.

This method ensures efficient batching and management of text sentences, while maintaining thread safety during concurrent operations. No mathematical formula is involved in this function.

---

#### `add_batch`

### Function Documentation

**Purpose**: 
The function `add_batch` is used to add multiple sentences to the internal queue of an object. These sentences accumulate until they reach the specified batch size (`batch_size`). Once this threshold is met, a multi-row shard (a collection of rows) containing these accumulated sentences will be flushed.

**Parameters**:
- `sentences`: A list of strings representing individual sentences that need to be added to the queue.
  - Type: List[str]

**Return Value**: 
The function returns a list of dictionaries where each dictionary represents metadata or information about one of the input sentences. The exact structure and content of these dictionaries depend on how `_queue_sentences` is implemented, but they typically contain details such as whether processing was successful for that sentence.
  - Type: List[Dict[str, Any]]

**Important Behaviour/Side-Effects**: 
The function does not modify or return any direct information about the sentences themselves. Instead, it adds them to an internal queue and flushes a batch of these sentences when necessary (when `batch_size` is reached). The returned list provides metadata for each input sentence.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `add_document`

### Function Documentation

**Purpose**: 
The function `add_document` is used to add a document with source-aware deduplication. It ensures that duplicate documents (based on content and source) are not added multiple times.

**Parameters**:
- `text`: A string representing the text of the document.
  - Type: str
- `doc_id`: An optional identifier for the document, typically used to uniquely identify it within a system.
  - Type: Optional[str]
- `source`: The source from which the document was obtained. This is important for deduplication purposes as documents with different sources are considered distinct even if their content matches.
  - Type: str
- `metadata`: An optional dictionary containing additional metadata about the document.
  - Type: dict (default value is None)

**Return Value**: 
The function returns a dictionary that indicates whether the document was added successfully, any skipped operations due to existing documents or duplicate content, and relevant details such as the reason for skipping or the unique identifier (`doc_key`) of the document.

- Added keys in return dictionary:
  - `added`: A boolean indicating if the document was actually added (True) or not (False).
  - `skipped`: A boolean indicating whether the operation was skipped due to an existing document with the same content and source.
  - `reason`: The reason for skipping, either "doc_exists" or "content_exists".
  - `doc_key`: The unique key assigned to the document.

**Important Behaviour/Side-Effects**: 
1. **Normalization of Text**: The input text is normalized using `_normalize_document_text`.
2. **Deduplication Based on Source and Content**: If a document with identical content from the same source already exists, it will be skipped.
3. **Updating Existing Documents**: If an existing document's metadata needs to be updated (e.g., due to new information), this function updates that entry in the index.
4. **Saving Document Index**: The `_save_document_index` method is called to persist changes made to the internal document index.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `add_documents`

### Function Documentation

**Purpose**: 
The function `add_documents` is used to add a batch of document dictionaries. Each dictionary in the input list should contain keys for text, doc_id, and source. The function processes each document individually using the `add_document` method.

**Parameters**:
- `docs`: A list of dictionaries where each dictionary represents a document with specific fields such as "text", "doc_id", "source", and optional metadata.
  - Type: List[Dict[str, Any]]

**Return Value**: 
The function returns a list of dictionaries that provide the result for each individual document added. Each dictionary in the returned list contains information about whether the document was successfully added or skipped due to existing documents with identical content from the same source.

- Added keys in return dictionary:
  - `added`: A boolean indicating if the document was actually added (True) or not (False).
  - `skipped`: A boolean indicating whether the operation was skipped.
  - `reason`: The reason for skipping, either "doc_exists", "content_exists", or an error message prefixed with "error:".
  - `doc_key`: The unique key assigned to the document.

**Important Behaviour/Side-Effects**: 
1. **Processing Each Document Individually**: For each document in the input list, the function calls `add_document` and appends its result to a results list.
2. **Error Handling**: If an exception occurs during processing of any individual document, it is caught, logged as an error reason, and appended to the results list with relevant information.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `flush`

### Function Documentation

**Purpose**: 
The function `flush` forces the writing of any currently pending batch of documents to a shard file. This ensures that all accumulated data is saved immediately, bypassing normal batching behavior.

**Parameters**: 
- None

**Return Value**: 
The function does not return anything.
  - Type: None

**Important Behaviour/Side-Effects**: 
1. **Locking Mechanism**: The function acquires a lock (`self.lock`) to ensure that the flushing operation is performed atomically and without interference from other concurrent operations.
2. **Internal Method Call**: It calls an internal method `_flush_batch` which handles the actual writing of the pending batch to a shard file.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `_flush_batch`

### Function Documentation

**Purpose**: 
The function `_flush_batch` writes all pending sentences to a single shard file in `.npz` format along with associated metadata. This ensures that the accumulated data is saved and ready for further processing.

**Parameters**: 
- None

**Return Value**: 
The function does not return anything.
  - Type: None

**Important Behaviour/Side-Effects**: 
1. **Vectorization of Sentences**: The pending sentences are vectorized using `self.vectorizer.transform` to convert them into a sparse matrix format (CSR).
2. **Shard File Creation**: A shard file with a unique name based on the current timestamp and batch counter is created in the specified directory (`shard_dir`). This file contains the sparse matrix representation of the sentences.
3. **Metadata Storage**: Metadata about the shard, including number of rows, non-zero entries (nnz), timestamps, batch counters, original texts, and document references are stored as a JSON file alongside the `.npz` file.
4. **Clearing Pending Buffers**: After writing to the shard files, all pending buffers (`_pending_sentences`, `_pending_doc_refs`) are cleared.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `rebuild`

### Function Documentation

**Purpose**: 
The function `rebuild` merges all existing shard files and the main `.npz` file into a new consolidated main `.npz` file. This process uses memory-mapped CSR arrays to efficiently append rows from each shard one-by-one, minimizing peak RAM usage.

**Parameters**:
- `keep_shards`: A boolean flag indicating whether to keep the processed shards or archive them after merging.
  - Type: bool (default value is False)

**Return Value**: 
The function returns a dictionary containing various statistics about the rebuild process and final state of the consolidated matrix. The keys in this dictionary include:
- `main_rows_before`: Number of rows before rebuilding the main.npz file
- `main_nnz_before`: Non-zero elements count before rebuilding the main.npz file
- `shards_processed`: Total number of shard files processed during rebuild
- `shard_nnz_total`: Sum total non-zero elements from all shards
- `status`: Status indicating success or failure of the operation ("success")
- `final_rows`: Final number of rows in the consolidated matrix after rebuilding
- `final_nnz`: Final count of non-zero elements in the consolidated matrix after rebuilding
- `shards_moved`: Number of shard files moved to archive directory (if applicable)
- `last_rebuild_time`: Timestamp indicating when the rebuild was completed

**Important Behaviour/Side-Effects**: 
1. **Locking Mechanism**: The function acquires a lock (`self.lock`) to ensure that the rebuild operation is performed atomically and without interference from other concurrent operations.
2. **Temporary Directory Creation**: A temporary directory is created for storing intermediate memory-mapped arrays during the merge process.
3. **Memory-Mapped CSR Arrays**: Memory-mapping techniques are used to efficiently append rows from each shard file into a consolidated matrix, minimizing peak RAM usage.
4. **Shard Processing and Archiving**:
   - Each shard file is processed sequentially by loading it as a sparse matrix (`csr_matrix`) and appending its contents to the main matrix using `append_csr`.
   - After processing all shards, any existing main.npz file is also appended if present.
5. **Archival of Processed Shards**: If `keep_shards` is False (default), processed shard files are moved into an archive directory with a timestamp-based subdirectory name.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `get_stats`

### Function Documentation

**Purpose**: 
The function `get_stats` returns a copy of the latest system statistics. This ensures that any ongoing modifications to the internal state are not visible in the returned dictionary.

**Parameters**: 
- None

**Return Value**: 
The function returns a dictionary containing the current system statistics.
  - Type: Dict[str, Any]

**Important Behaviour/Side-Effects**: 
1. **Locking Mechanism**: The function acquires a lock (`self.lock`) to ensure that the retrieval of statistics is performed atomically and without interference from other concurrent operations.
2. **Copying Statistics**: A copy of `self._stats` (the internal dictionary storing system statistics) is returned, ensuring thread safety.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `_compute_stats_fast`

### Function Documentation

**Purpose**: 
The function `_compute_stats_fast` quickly computes system statistics using pre-written shard metadata. This is called frequently (every few seconds) by a background thread to provide up-to-date statistics without loading large datasets into memory.

**Parameters**: 
- None

**Return Value**: 
The function returns a dictionary containing various statistical information about the current state of the system.
  - Type: Dict[str, Any]

**Important Behaviour/Side-Effects**: 
1. **Pending Data Calculation**: The number of pending rows and non-zero entries (`nnz`) are calculated using internal attributes `_pending_rows` and `_pending_nnz`.
2. **Shard Metadata Processing**: Shard metadata is read from `.meta.json` files in the shard directory to compute total rows, non-zero elements (nnz), and count of shards.
3. **Main Matrix Statistics**:
   - The shape and nnz values for the main matrix are obtained by reading relevant entries from the `main_npz` file without fully loading it into memory.
4. **Total Counts**: Aggregates total rows, non-zero elements (nnz), document count, and other statistics based on pending data, shard metadata, and main matrix information.
5. **Timestamps**:
   - The current timestamp is recorded as `last_stats_update`.
   - If available, the last rebuild time from `_stats` dictionary is included.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `_background_monitor`

### Function Documentation

**Purpose**: 
The function `_background_monitor` runs in a background thread to perform periodic tasks such as reloading configuration files, updating dynamic parameters, refreshing statistics cache, and triggering automatic rebuilds if necessary.

**Parameters**: 
- None

**Return Value**: 
The function does not return anything.
  - Type: None

**Important Behaviour/Side-Effects**: 
1. **Configuration Reload**:
   - The function checks the modification time of the configuration file (`config_path`). If it has changed since the last check, the new configuration is loaded and applied with `_update_config`.
2. **Statistics Update**:
   - Statistics are updated periodically based on `stats_update_interval`. This involves calling `_compute_stats_fast` to compute system statistics.
3. **Auto-Rebuild Triggering**:
   - If a rebuild threshold (`rebuild_threshold_shards`) is set and the number of shards exceeds this threshold, an automatic rebuild is triggered in a separate thread.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### Detailed Breakdown

1. **Configuration Reload**:
    ```python
    if os.path.exists(self.config_path):
        mtime = os.path.getmtime(self.config_path)
        if mtime != last_config_mtime:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            with self.lock:
                self._update_config(config, initial=False)
            last_config_mtime = mtime
    ```
   - This block checks for changes in the configuration file and reloads it if necessary.

2. **Statistics Update**:
    ```python
    if now - last_stats_time >= self.stats_update_interval:
        stats = self._compute_stats_fast()
        with self.lock:
            self._stats.update(stats)
        last_stats_time = now
    ```
   - This block updates the system statistics by calling `_compute_stats_fast` and updating `self._stats`.

3. **Auto-Rebuild Triggering**:
    ```python
    if hasattr(self, 'rebuild_threshold_shards'):
        threshold = self.rebuild_threshold_shards
        if threshold > 0:
            meta_pattern = os.path.join(self.shard_dir, "shard_*.meta.json")
            shard_count = len(glob.glob(meta_pattern))
            if shard_count >= threshold:
                print(f"Auto-rebuild triggered: {shard_count} shards >= {threshold}")
                threading.Thread(target=self.rebuild, daemon=True).start()

---

#### `count_shards`

### Function Documentation

**Purpose**: 
The function `count_shards` returns the number of unprocessed shard files in a specified directory without loading them into memory.

**Parameters**: 
- None

**Return Value**: 
The function returns an integer representing the count of shard files.
  - Type: int

**Important Behaviour/Side-Effects**: 
1. **Pattern Matching**: The function uses `glob.glob` to find all `.npz` files in the specified directory (`self.shard_dir`) that match a specific pattern ("shard_*.npz").
2. **Count Calculation**: It returns the length of the list obtained from globbing, which corresponds to the number of shard files found.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `get_document_texts`

### Function Documentation

**Purpose**: 
The function `get_document_texts` returns the text of indexed documents, preserving their insertion order. If a limit parameter is provided, it will return only the most recently added document texts up to that limit.

**Parameters**:
- `limit`: An optional integer specifying how many recent documents' texts should be returned.
  - Type: Optional[int]

**Return Value**: 
The function returns a list of strings representing the text content of indexed documents.
  - Type: List[str]

**Important Behaviour/Side-Effects**: 
1. **Document Retrieval and Sorting**:
   - Documents are retrieved from `self._document_index.get("documents", {})`.
   - The documents are sorted based on their creation time (`created_at`), ensuring the order is preserved.
2. **Text Extraction**:
   - For each document, if it contains a non-empty text field, its text content is extracted and added to the list of texts.
3. **Limit Application**: 
   - If `limit` is provided and greater than 0, only the most recent documents' texts are returned.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `get_source_counts`

### Function Documentation

**Purpose**: 
The function `get_source_counts` returns a dictionary containing the count of documents indexed by their source.

**Parameters**: 
- None

**Return Value**: 
The function returns a dictionary where keys are unique sources and values are the corresponding document counts.
  - Type: Dict[str, int]

**Important Behaviour/Side-Effects**: 
1. **Document Iteration**: The function iterates over all documents in `self._document_index.get("documents", {})`.
2. **Source Counting**: For each document, it extracts the source and updates a count dictionary to keep track of how many times each source appears.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `_read_texts_from_meta_files`

### Function Documentation

**Purpose**: 
The function `_read_texts_from_meta_files` reads and processes text data from JSON files matching a specified pattern. It extracts the text content, normalizes it if necessary, and appends valid texts to a list.

**Parameters**:
- `pattern`: A string representing the glob pattern used to match relevant metadata files.
  - Type: str

**Return Value**: 
The function returns a list of strings containing normalized text data extracted from the matched JSON files.
  - Type: List[str]

**Important Behaviour/Side-Effects**: 
1. **Glob Matching**: The function uses `glob.glob` to find all metadata files matching the specified pattern (`pattern`, recursive=True).
2. **Metadata File Processing**:
   - For each file, it attempts to open and load JSON content.
   - It extracts text data from the loaded metadata dictionary under the key `"texts"`.
3. **Text Normalization**: Each extracted text is normalized using `_normalize_document_text`. If normalization results in a non-empty string, the cleaned text is appended to the list of texts.
4. **Error Handling**:
   - Any exceptions encountered during file processing are caught and logged as warnings.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `get_shard_texts`

### Function Documentation

**Purpose**: 
The function `get_shard_texts` reads text payloads from both active shard metadata files in the current directory (`shard_dir`) and optionally archived metadata files. It returns a list of these texts, potentially limited by an input parameter.

**Parameters**:
- `limit`: An optional integer specifying how many recent documents' texts should be returned.
  - Type: Optional[int]
- `include_archive`: A boolean flag indicating whether to include text payloads from archived shard metadata files in addition to active ones. Defaults to True.
  - Type: bool (default value is True)

**Return Value**: 
The function returns a list of strings containing the extracted and normalized texts from both active and optionally archived shard metadata files.
  - Type: List[str]

**Important Behaviour/Side-Effects**: 
1. **Active Shard Metadata Reading**:
   - The function calls `_read_texts_from_meta_files` to read text payloads from active shard metadata files in `self.shard_dir`.
2. **Archived Shard Metadata Reading (if applicable)**:
   - If `include_archive` is True, the function also reads and processes texts from archived shard metadata files.
3. **Text Limitation**:
   - If a positive limit value is provided (`limit > 0`), only the most recent documents' texts are returned based on their insertion order.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `get_all_texts`

### Function Documentation

**Purpose**: 
The function `get_all_texts` returns a deduplicated list of all indexed, shard, and pending text documents. The returned list can be used for corpus purposes.

**Parameters**:
- `limit`: An optional integer specifying how many recent documents' texts should be returned.
  - Type: Optional[int]
- `include_archive`: A boolean flag indicating whether to include text payloads from archived shard metadata files in addition to active ones. Defaults to True.
  - Type: bool (default value is True)

**Return Value**: 
The function returns a list of strings containing deduplicated texts from indexed documents, shards, and pending sentences.
  - Type: List[str]

**Important Behaviour/Side-Effects**: 
1. **Text Deduplication**:
   - A set `seen` is used to keep track of unique text hashes (using `_hash_text`) to ensure no duplicate texts are added to the final list.
2. **Indexed Document Texts**:
   - The function calls `self.get_document_texts()` and adds non-duplicate texts from indexed documents to the result list.
3. **Shard Metadata Texts**:
   - If `include_archive` is True, it also includes texts from shard metadata files (both active and archived) that are not already in the set of seen hashes.
4. **Pending Sentences**:
   - Pending sentences are normalized using `_normalize_document_text`, hashed, and added to the result list if they are unique and non-empty.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### `load_full`

### Function Documentation

**Purpose**: 
The function `load_full` loads the entire main matrix stored in a `.npz` file into memory as a sparse CSR (Compressed Sparse Row) matrix. If no such file exists, it returns an empty sparse matrix.

**Parameters**: 
- None

**Return Value**: 
The function returns a sparse CSR matrix representing the loaded data.
  - Type: `scipy.sparse.csr_matrix`

**Important Behaviour/Side-Effects**: 
1. **File Existence Check**:
   - The function checks if the main `.npz` file exists at the specified path (`self.main_npz`). If it does not exist, an empty sparse matrix is returned.
2. **Matrix Loading**:
   - If the file exists, `sp.load_npz(self.main_npz)` loads and returns the full matrix as a CSR matrix.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### Detailed Breakdown

1. **File Existence Check**:
    ```python
    if not os.path.exists(self.main_npz):
        return sp.csr_matrix((0, self.n_features), dtype=self.dtype)
    ```
   - This block checks whether the main `.npz` file exists at `self.main_npz`. If it does not exist, an empty sparse matrix is returned.

2. **Matrix Loading**:
    ```python
    return sp.load_npz(self.main_npz)
    ```
   - If the file exists, this line loads and returns the full main matrix as a CSR matrix using `sp.load_npz`.

### Example

If there is no `.npz` file at `self.main_npz`, an empty sparse matrix with 0 rows and `n_features` columns will be returned:
```python
return sp.csr_matrix((0, self.n_features), dtype=self.dtype)
```

Otherwise, the function loads the full main matrix from the specified `.npz` file.

---

#### `get_vectorizer`

### Function Documentation

**Purpose**: 
The function `get_vectorizer` returns the `HashingVectorizer` instance used for transforming text data into a vector representation.

**Parameters**: 
- None

**Return Value**: 
The function returns an instance of `HashingVectorizer`.
  - Type: HashingVectorizer

**Important Behaviour/Side-Effects**: 
1. **No Side Effects**: The function simply retrieves and returns the internal `vectorizer` attribute without modifying it or causing any side effects.

### Mathematical Formula
No mathematical formula is involved in this function.

---

#### Detailed Breakdown

The function `get_vectorizer` performs a straightforward task of returning an instance of the `HashingVectorizer` used for transforming text data into vector representations. This allows other parts of the code to access and use the same vectorization method that was originally configured or trained with the object.

### Example
If the internal attribute `self.vectorizer` is set, it will be returned directly:
```python
return self.vectorizer
```

This function ensures that any component requiring the vectorizer can obtain a reference to the correct instance.

---



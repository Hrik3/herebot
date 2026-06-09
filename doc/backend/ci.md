## Doc for ci
---

This Python file defines a comprehensive evaluation suite for assessing the confidence and realism of text inputs, primarily in transliteration scenarios. It includes multiple modules such as CIEngine for evaluating text against a corpus using TF-IDF similarity and other linguistic features, PhonotacticEngine for scoring phonetic structure based on syllable patterns and consonant clusters, and DiagnosticEngine for generating human-readable explanations about the evaluation results.

Key components include:
- FeatureExtractor: Extracts various statistical and linguistic features from input texts.
- CIEngine: Evaluates text confidence using a probabilistic model trained on corpus data.
- PhonotacticEngine: Analyzes phonetic structure to ensure natural-sounding words.
- DiagnosticEngine: Converts technical diagnostics into understandable explanations.

The overall design is modular, with separate classes for each evaluation aspect. It supports robustness testing through perturbation and comparison against real text variants provided by the user. The file also includes utility functions for cleaning texts, adding sentences to memory, and loading corpus data from shards or a database.

At its core, this suite aims to provide detailed insights into why certain transliterations are deemed more confident or realistic than others based on both statistical similarity to known patterns and phonetic plausibility according to learned norms.

---

### FeatureExtractor
The `FeatureExtractor` class is responsible for extracting numerical linguistic and statistical features from a transliteration string. It initializes with a given corpus, preparing it by removing duplicates and fitting TF-IDF vectorization models on character n-grams. Key attributes include the cleaned corpus (`self.corpus`), a TF-IDF vectorizer (`self.vectorizer`) trained on this corpus, an array of text lengths (`self.lengths`), and an optional Isolation Forest model for detecting outliers based on length.

This class is responsible for calculating various features such as character validity, length score, vowel structure, TF-IDF similarity to the corpus, and outlier detection. These features are combined into a dictionary when calling the `extract` method, providing a comprehensive set of metrics for evaluating transliteration strings.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `FeatureExtractor` class.

**Purpose:** 
Sets up the feature extractor by preparing and cleaning a given corpus and building necessary models for extracting features from text inputs.

**Parameters:**
- `corpus`: A list or iterable containing textual data. The type is typically a list of strings, but it can be any iterable that yields textual elements.

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Calls `_prepare_corpus(corpus)` to clean and deduplicate the provided corpus.
2. Fits a TF-IDF vectorizer on the cleaned corpus for character n-grams (n=1 to 5).
3. Builds an Isolation Forest model if there are at least two texts in the prepared corpus, used later for outlier detection based on text length.

This function does not return any value but sets up essential attributes (`self.corpus`, `self.vectorizer`, and optionally `self.outlier_model`) that will be used by other methods of the class.

---

#### `_prepare_corpus`

The `_prepare_corpus` function cleans and deduplicates a given corpus.

**Purpose:** 
Prepares the input corpus by removing duplicates, normalizing to lowercase, and joining words with spaces.

**Parameters:**
- `corpus`: A list or iterable containing textual data. The type is typically a list of strings, but it can be any iterable that yields textual elements.

**Return Value:** 
A cleaned list of unique, normalized texts (`clean`).

**Important Behaviour/Side-effects:**
1. Converts each item in the corpus to a string and joins words with spaces.
2. Skips empty or whitespace-only items.
3. Ensures uniqueness by checking lowercase versions of strings against a set.

This function does not have any side effects outside its scope and returns a cleaned list of unique, normalized texts.

---

#### `_build_models`

The `_build_models` function initializes and fits necessary models for feature extraction.

**Purpose:** 
Sets up the TF-IDF vectorizer and outlier detection model based on text lengths.

**Parameters:**
None

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Raises a `ValueError` if the corpus is empty.
2. Initializes and fits a `TfidfVectorizer` with character-level n-grams (n=1 to 5) on the cleaned corpus.
3. Creates an array of text lengths from the cleaned corpus.
4. Optionally builds an Isolation Forest model for outlier detection based on text length if there are at least two texts in the corpus.

This function does not return any value but sets up essential attributes (`self.vectorizer`, `self.lengths`, and optionally `self.outlier_model`) that will be used by other methods of the class.

---

#### `character_validity`

The `character_validity` function calculates the proportion of valid characters in a given text.

**Purpose:** 
Evaluates how many characters in the input text are considered valid based on predefined character sets and returns this ratio as a normalized score.

**Parameters:**
- `text`: A string representing the input text to be evaluated. (Type: str)

**Return Value:** 
A float value between 0.0 and 1.0, where higher values indicate more characters are considered valid according to `VALID_CHARS`.

**Important Behaviour/Side-effects:**
1. Converts each character in the text to lowercase.
2. Counts how many of these lowercase characters are present in the predefined set `VALID_CHARS`.
3. Divides this count by the total length of the input text using `_safe_div` to avoid division by zero.

This function does not have any side effects and returns a normalized score representing character validity.

---

#### `length_score`

The `length_score` function evaluates the score based on the length of a given text.

**Purpose:** 
Assigns a score to an input text based on its length, with specific thresholds for short and long texts.

**Parameters:**
- `text`: A string representing the input text. (Type: str)

**Return Value:** 
A float value between 0.3 and 1.0:
- Returns 0.3 if the length of the text is less than 3 characters.
- Returns 0.6 if the length of the text is greater than 300 characters.
- Otherwise, returns 1.0.

**Important Behaviour/Side-effects:**
1. Calculates the length `n` of the input text using `len(text)`.
2. Assigns a score based on predefined thresholds for short and long texts:
   - If `n < 3`, the score is set to 0.3.
   - If `n > 300`, the score is set to 0.6.
   - Otherwise, the score is set to 1.0.

This function does not have any side effects and returns a normalized length score for the input text.

---

#### `vowel_structure`

The `vowel_structure` function evaluates the proportion of vowel characters in a given text and normalizes this ratio.

**Purpose:** 
Calculates the normalized score representing the balance between vowels and consonants in an input text.

**Parameters:**
- `text`: A string representing the input text. (Type: str)

**Return Value:** 
A float value between 0.0 and 1.0, where higher values indicate a more balanced vowel-to-consonant ratio according to predefined criteria.

**Important Behaviour/Side-effects:**
1. Converts each character in the input text to lowercase.
2. Counts how many of these characters are vowels by checking membership in the `vowels` set.
3. Divides this count by the total length of the input text using `_safe_div`.
4. Scales the resulting ratio by dividing it by 0.5 and then clips the value between 0.0 and 1.0 using `_clip01`.

This function does not have any side effects and returns a normalized score representing vowel structure in the input text.

Mathematical formula:
\[ \text{score} = _\text{clip01}\left(\frac{\sum_{c \in \text{text.lower()}} c \in \text{vowels}}{\text{len(text)}} / 0.5\right) \]

---

#### `tfidf_similarity`

The `tfidf_similarity` function calculates the TF-IDF similarity of a given text to each document in the corpus and returns the highest similarity score.

**Purpose:** 
Computes the cosine similarity between the input text and all documents in the corpus, then identifies the most similar document based on this measure.

**Parameters:**
- `text`: A string representing the input text. (Type: str)

**Return Value:** 
A dictionary containing:
- `"score"`: The highest TF-IDF cosine similarity score.
- `"index"`: The index of the most similar document in the corpus.
- `"matched_text"`: The actual text content of the most similar document.

**Important Behaviour/Side-effects:**
1. Converts the input `text` to a string and appends it to the list of documents (`self.corpus + [str(text)]`).
2. Transforms all texts into TF-IDF vectors using `self.vectorizer.transform`.
3. Computes cosine similarity between the vector of the input text (last in the transformed list) and each other document's vector.
4. Identifies the index of the highest similarity score (`best_idx`) and retrieves the corresponding document from the corpus.
5. Returns a dictionary containing the best match’s similarity score, its index, and the matched text.

This function does not have any side effects but performs necessary computations to determine how similar an input text is to known documents in the corpus based on TF-IDF vectors.

Mathematical formula:
\[ \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} \]
where \(\mathbf{A}\) and \(\mathbf{B}\) are the TF-IDF vectors of two documents.

---

#### `outlier_score`

The `outlier_score` function evaluates whether a given text length is an outlier and returns a score based on this evaluation.

**Purpose:** 
Determines if the input text's length is considered an outlier using an Isolation Forest model, returning a normalized score accordingly.

**Parameters:**
- `text`: A string representing the input text. (Type: str)

**Return Value:** 
A float value between 0.4 and 1.0:
- Returns 1.0 if the length of the text is not an outlier.
- Returns 0.4 if the length of the text is identified as an outlier.

**Important Behaviour/Side-effects:**
1. Checks if `self.outlier_model` (an Isolation Forest model) has been initialized.
2. If `outlier_model` is None, returns a score of 1.0 by default.
3. Otherwise, predicts whether the length of the input text is an outlier using the Isolation Forest model.
4. Converts the prediction to a normalized score: 
   - Returns 1.0 if the text length is not predicted as an outlier (`pred == 1`).
   - Returns 0.4 if the text length is identified as an outlier.

This function does not have any side effects and returns a normalized score indicating whether the input text's length is considered typical or unusual based on previously learned patterns.

Mathematical formula:
None

Behavior Classification: This function primarily involves conditional checks and model predictions to determine outliers.

---

#### `extract`

The `extract` function computes a set of linguistic and statistical features for an input text.

**Purpose:** 
Extracts multiple numerical features from the input text, including character validity, length score, vowel structure, outlier detection, and TF-IDF similarity to the corpus.

**Parameters:**
- `text`: A string representing the input text. (Type: str)

**Return Value:** 
A dictionary containing various feature scores:
- `"char_valid"`: Character validity score.
- `"length"`: Length score of the text.
- `"structure"`: Vowel structure score.
- `"outlier"`: Outlier detection score based on length.
- `"similarity"`: TF-IDF similarity to the most similar document in the corpus.
- `"_match_index"`: Index of the most similar document in the corpus (for debugging purposes).
- `"_matched_text"`: The actual text content of the most similar document.

**Important Behaviour/Side-effects:**
1. Converts the input `text` to a string, defaulting to an empty string if it is None.
2. Calls `self.tfidf_similarity(text)` to compute TF-IDF similarity and store results in `sim`.
3. Computes character validity using `self.character_validity(text)`.
4. Calculates length score with `self.length_score(text)`.
5. Evaluates vowel structure via `self.vowel_structure(text)`.
6. Determines outlier score through `self.outlier_score(text)`.
7. Returns a dictionary containing all the computed scores and additional metadata.

This function does not have any side effects but performs necessary computations to extract multiple features from an input text, providing insights into its linguistic properties relative to a known corpus.

Mathematical formula:
None

Behavior Classification: This function primarily involves calling other methods to compute various feature scores and returning them in a structured format.

---


### ProbabilisticCIModel
The `ProbabilisticCIModel` class is responsible for converting feature vectors into a probabilistic confidence estimate. It initializes with predefined weights for different features and provides methods to score the features, measure uncertainty as variance, and convert scores into a normal distribution.

Key attributes include:
- `weights`: A dictionary containing weight values for each feature type (e.g., character validity, length, structure, similarity, outlier).

This class is responsible for aggregating weighted feature scores to produce an overall confidence estimate, measuring the variability of multiple estimates using standard deviation, and converting these estimates into a mean and variance representing a normal distribution.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `ProbabilisticCIModel` class.

**Purpose:** 
Sets up the model with predefined weights for different feature types to be used in scoring and confidence estimation.

**Parameters:**
None

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Initializes a dictionary `self.weights` containing weight values for each feature type:
   - `"char_valid"`: Weight of 0.20
   - `"length"`: Weight of 0.15
   - `"structure"`: Weight of 0.20
   - `"similarity"`: Weight of 0.30
   - `"outlier"`: Weight of 0.15

This function does not return any value but sets up the essential attribute `self.weights` that will be used by other methods in the class to compute weighted scores.

Mathematical formula:
None

Behavior Classification: This function primarily involves initializing a dictionary with predefined weights for different feature types.

---

#### `score`

The `score` function computes a weighted sum of feature values to produce an overall confidence estimate.

**Purpose:** 
Aggregates the scores from different features using predefined weights and returns a normalized confidence score between 0.0 and 1.0.

**Parameters:**
- `features`: A dictionary containing feature names as keys and their corresponding scores as values. (Type: dict)

**Return Value:** 
A float value between 0.0 and 1.0 representing the overall confidence estimate after applying weights to each feature score.

**Important Behaviour/Side-effects:**
1. Initializes a sum `s` set to 0.0.
2. Iterates over each key-value pair in `self.weights`.
3. For each weight, multiplies it by the corresponding value from `features.get(k, 0.0)` (defaulting to 0.0 if the feature is not present).
4. Accumulates these weighted scores into `s`.
5. Returns `_clip01(s)`, which clips the final sum between 0.0 and 1.0.

This function does not have any side effects but performs necessary computations to produce a normalized confidence score based on input features.

Mathematical formula:
\[ s = \sum_{k} w_k \cdot f_k \]
where \( w_k \) is the weight for feature \( k \), and \( f_k \) is the value of feature \( k \).

Behavior Classification: This function primarily involves iterating over a dictionary to compute a weighted sum, which is then normalized.

---

#### `uncertainty`

The `uncertainty` function measures the variability of a set of confidence estimates using standard deviation.

**Purpose:** 
Computes the standard deviation (variance) of an array of scores to quantify their dispersion or uncertainty.

**Parameters:**
- `scores`: A list or iterable containing multiple score values. (Type: list or iterable)

**Return Value:** 
A float value representing the standard deviation of the input scores, which quantifies how spread out these scores are from their mean.

**Important Behaviour/Side-effects:**
1. Computes the standard deviation of the `scores` using `np.std(scores)`.
2. Converts the result to a float and returns it.

This function does not have any side effects but calculates the variability (uncertainty) in an array of confidence scores, providing insight into their consistency or dispersion.

Mathematical formula:
\[ \text{std} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2} \]
where \( x_i \) are the individual scores and \( \mu \) is the mean of the scores.

Behavior Classification: This function primarily involves calculating standard deviation to measure uncertainty.

---

#### `to_distribution`

The `to_distribution` function converts a set of confidence estimates into a normal distribution by computing the mean and standard deviation.

**Purpose:** 
Computes the mean (mu) and standard deviation (sigma) from an array of scores to represent them as a normal distribution.

**Parameters:**
- `scores`: A list or iterable containing multiple score values. (Type: list or iterable)

**Return Value:** 
A tuple containing two float values:
1. The mean (`mu`) of the input scores.
2. The standard deviation (`sigma`) of the input scores, with a small constant added to avoid division by zero.

**Important Behaviour/Side-effects:**
1. Computes the mean (average) of `scores` using `np.mean(scores)` and converts it to a float.
2. Computes the standard deviation of `scores` using `np.std(scores)`, adds \( 1 \times 10^{-6} \) to avoid division by zero, and converts it to a float.
3. Returns a tuple containing both mean (`mu`) and adjusted standard deviation (`sigma`).

This function does not have any side effects but performs necessary computations to represent the scores as a normal distribution.

Mathematical formula:
- Mean: \( \mu = \frac{1}{N} \sum_{i=1}^{N} x_i \)
  where \( N \) is the number of scores and \( x_i \) are individual score values.
- Standard Deviation: \( \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 + 10^{-6}} \)

Behavior Classification: This function primarily involves calculating the mean and standard deviation to represent scores as a normal distribution.

---


### MonteCarloEngine
The `MonteCarloEngine` class is responsible for testing the stability of text confidence estimates through random perturbation. It initializes with an extractor and a model, which are used to evaluate perturbed texts.

Key attributes include:
- `extractor`: An instance that extracts features from input texts.
- `model`: A model that scores the extracted features.

This class is responsible for applying random character deletions to text (`perturb` method) and evaluating multiple perturbed versions of a given text to measure confidence stability (`sample` method).

Mathematical formula:
None

Behavior Classification: This class primarily involves perturbing texts randomly, extracting features from these perturbed texts, scoring the extracted features, and aggregating scores to assess stability.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `MonteCarloEngine` class.

**Purpose:** 
Sets up the engine with necessary components for perturbation and evaluation.

**Parameters:**
- `extractor`: An object responsible for extracting features from input texts. (Type: object)
- `model`: A model used to score the extracted features. (Type: object)

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Initializes an attribute `self.extractor` with the provided extractor.
2. Initializes another attribute `self.model` with the provided model.

This function does not return any value but sets up essential attributes that will be used by other methods in the class to perform perturbation and evaluation tasks.

Mathematical formula:
None

Behavior Classification: This function primarily involves initializing instance variables with provided objects.

---

#### `perturb`

The `perturb` function applies random deletions to an input text.

**Purpose:** 
Randomly deletes characters from the given text with a specified probability, simulating perturbation effects.

**Parameters:**
- `text`: A string representing the input text. (Type: str)
- `p`: Probability of deleting each character in the text. Default value is 0.08. (Type: float)

**Return Value:** 
A new string with randomly deleted characters based on the specified probability.

**Important Behaviour/Side-effects:**
1. Converts the input `text` into a list of characters.
2. Iterates over each character in the list and, for each one, decides whether to delete it (set to an empty string) based on the random number generated by `np.random.rand()` being less than `p`.
3. Joins the modified list back into a single string with deleted characters.

This function does not have any side effects but performs necessary operations to perturb the input text randomly for evaluation purposes.

Mathematical formula:
None

Behavior Classification: This function primarily involves random character deletion based on a specified probability.

---

#### `sample`

The `sample` function evaluates the stability of a given text by perturbing it multiple times and scoring each perturbed version.

**Purpose:** 
Perturbs an input text randomly, extracts features from each perturbed version, and scores these features to measure confidence stability.

**Parameters:**
- `text`: A string representing the input text. (Type: str)
- `n`: Number of samples or perturbations to generate. Default value is 50. (Type: int)

**Return Value:** 
A list of float values, each representing a score from evaluating one of the perturbed texts.

**Important Behaviour/Side-effects:**
1. Initializes an empty list `scores` to store the scores.
2. Iterates `n` times:
   - Perturbs the input text using `self.perturb(text)`.
   - Extracts features from the perturbed text with `self.extractor.extract(t)`.
   - Scores the extracted features using `self.model.score(feats)` and appends the score to `scores`.
3. Returns the list of scores.

This function does not have any side effects but performs necessary operations to generate multiple perturbations, extract features from each one, and compute their respective scores for stability analysis.

Mathematical formula:
None

Behavior Classification: This function primarily involves iterative perturbation, feature extraction, and scoring to assess confidence stability.

---


### CIEngine
The `CIEngine` class provides a high-level interface for evaluating the confidence of input texts based on their similarity and structure compared to a corpus.

**Purpose:** 
Evaluates text confidence by perturbing it, extracting features from each perturbed version, scoring these features, and aggregating results to measure stability and overall confidence.

**Key Attributes:**
- `memory_bridge`: An object for accessing external memory or data sources.
- `config_manager`: Configuration manager (optional).
- `corpus`: A list of cleaned texts used as a reference corpus.
- `_lock`: Thread lock for ensuring thread safety during corpus updates.
- `extractor`: Feature extractor initialized with the corpus.
- `model`: Probabilistic confidence model.
- `mc`: Monte Carlo engine for perturbation and evaluation.

**Responsibilities:**
1. Initializes and manages an internal corpus of texts, either provided explicitly or loaded from external sources.
2. Builds necessary engines (feature extraction, probabilistic scoring, and perturbation) based on the available corpus.
3. Refreshes the internal corpus if needed to ensure it remains up-to-date.
4. Evaluates input text by generating multiple perturbed versions, extracting features, scoring them, and aggregating results to compute confidence metrics.

This class is designed for high-level evaluation in a competition setting, providing detailed diagnostics on text confidence based on learned patterns from the reference corpus.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `CIEngine` class.

**Purpose:** 
Sets up the engine with a memory bridge, configuration manager, and corpus for evaluating text confidence.

**Parameters:**
- `memory_bridge`: An object for accessing external memory or data sources. (Type: object)
- `corpus`: A list of texts used as a reference corpus. If not provided explicitly, it is loaded from the memory bridge or other sources. (Type: list)
- `config_manager`: Configuration manager (optional). (Type: object)

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Checks if an explicit corpus was provided and adjusts `memory_bridge` accordingly.
2. Initializes attributes:
   - `self.memory_bridge`: The memory bridge for accessing external data.
   - `self.config_manager`: Configuration manager (if provided).
   - `self._explicit_corpus`: A boolean indicating whether the corpus is explicitly provided or not.
   - `self._lock`: Thread lock to ensure thread safety during corpus updates.
3. Initializes an empty list `self.corpus` for storing cleaned texts.
4. If a corpus is explicitly provided, cleans and prepares it using `_prepare_corpus(corpus)`. Otherwise, reads the corpus from external sources or memory bridge using `_read_corpus()`.
5. Computes a fingerprint of the prepared corpus to detect changes later.
6. Builds necessary engines (feature extraction, probabilistic scoring, and perturbation) based on the available corpus.

This function does not return any value but sets up essential attributes that will be used by other methods in the class for evaluating text confidence.

Mathematical formula:
None

Behavior Classification: This function primarily involves initializing instance variables and setting up necessary components for text evaluation.

---

#### `_prepare_corpus`

The `_prepare_corpus` function cleans and deduplicates a given corpus.

**Purpose:** 
Prepares the input corpus by removing duplicates, normalizing to lowercase, and joining words with spaces.

**Parameters:**
- `corpus`: A list or iterable containing textual data. (Type: list)

**Return Value:** 
A cleaned list of unique texts (`clean`).

**Important Behaviour/Side-effects:**
1. Converts each item in the corpus to a string and joins words with spaces.
2. Skips empty or whitespace-only items.
3. Ensures uniqueness by checking lowercase versions of strings against a set.

This function does not have any side effects but returns a cleaned list of unique, normalized texts.

Mathematical formula:
None

Behavior Classification: This function primarily involves cleaning and deduplicating text data to ensure it is in the desired format for further processing.

---

#### `_fingerprint`

The `_fingerprint` function generates a unique fingerprint (hash) for the given corpus.

**Purpose:** 
Computes a hash value that uniquely represents the provided corpus, ensuring it can be quickly compared to detect changes in the corpus.

**Parameters:**
- `corpus`: A list or iterable containing textual data. (Type: list)

**Return Value:** 
A hexadecimal string representing the SHA256 fingerprint of the input corpus.

**Important Behaviour/Side-effects:**
1. Initializes a new SHA256 hash object.
2. Iterates over each text in the corpus:
   - Converts each text to bytes, encoding it as UTF-8 with error handling set to "ignore".
   - Updates the hash object with both the encoded text and a null byte (`b"\0"`).
3. Returns the hexadecimal digest of the hash object.

This function does not have any side effects but generates a unique fingerprint for the input corpus, which can be used to detect changes or updates in the corpus efficiently.

Mathematical formula:
None

Behavior Classification: This function primarily involves generating a cryptographic hash (SHA256) to uniquely represent and compare corpora.

---

#### `_read_corpus`

The `_read_corpus` function loads the corpus from either a memory bridge or shard texts.

**Purpose:** 
Reads and returns the corpus of text data, attempting to use a memory bridge if available. If not, it falls back to loading all shard texts.

**Parameters:**
None

**Return Value:** 
A list containing textual data representing the loaded corpus.

**Important Behaviour/Side-effects:**
1. Checks if `self.memory_bridge` is not None.
   - If true, attempts to get the corpus from the memory bridge using `memory_bridge.get_corpus()`.
   - Catches any exceptions and returns an empty list in case of failure.
2. If no memory bridge or if it fails, tries loading all shard texts using `load_all_shard_texts()`.
3. Catches any exceptions during this process and also returns an empty list.

This function does not have significant side effects but handles potential errors gracefully by returning an empty list when corpus retrieval fails.

Mathematical formula:
None

Behavior Classification: This function primarily involves loading text data from a memory bridge or shard texts, with error handling to ensure it always returns a valid (possibly empty) list.

---

#### `_build_engines`

The `_build_engines` function initializes the necessary engines for evaluating text confidence.

**Purpose:** 
Sets up feature extraction, probabilistic scoring, and perturbation components based on the available corpus.

**Parameters:**
None

**Return Value:** 
None

**Important Behaviour/Side-effects:**
1. Initializes a `FeatureExtractor` instance with the prepared corpus if it is not empty.
2. Initializes a `ProbabilisticCIModel` instance for scoring features probabilistically.
3. If an extractor was successfully initialized, sets up a `MonteCarloEngine` using the feature extractor and probabilistic model.

This function does not return any value but initializes essential components required for evaluating text confidence based on the provided corpus.

Mathematical formula:
None

Behavior Classification: This function primarily involves initializing instance variables with necessary objects to perform feature extraction, scoring, and perturbation.

---

#### `refresh_corpus`

The `refresh_corpus` function updates the internal corpus if necessary.

**Purpose:** 
Ensures that the internal corpus is up-to-date by refreshing it from external sources or shard texts, and rebuilding associated engines if needed.

**Parameters:**
- `force`: A boolean flag indicating whether to force a refresh of the corpus regardless of changes. (Type: bool)

**Return Value:** 
A boolean value:
- Returns False if no update is necessary.
- Returns True if the corpus was successfully refreshed and updated.

**Important Behaviour/Side-effects:**
1. Checks if an explicit corpus was provided (`self._explicit_corpus`) and whether to force a refresh (`force`).
   - If both conditions are met, returns False without refreshing.
2. Prepares a new copy of the corpus using `_prepare_corpus(self._read_corpus())`.
3. Computes a fingerprint for the new corpus using `_fingerprint(new_corpus)`.
4. Compares fingerprints:
   - If not forcing and fingerprints match (`fp == self._corpus_fingerprint`), returns False without updating.
5. Otherwise, acquires a thread lock to ensure thread safety during updates.
6. Updates `self.corpus`, the fingerprint of the corpus, and rebuilds necessary engines using `_build_engines()`.
7. Returns True after successful update.

This function ensures that the internal state is consistent with external changes in the corpus while handling concurrency issues through threading locks.

Mathematical formula:
None

Behavior Classification: This function primarily involves checking for updates to the corpus, refreshing it if needed, and rebuilding associated engines to ensure consistency and accuracy.

---

#### `evaluate`

The `evaluate` function evaluates the confidence of an input text based on its similarity and structure compared to a reference corpus.

**Purpose:** 
Evaluates the given text by perturbing it, extracting features from each perturbed version, scoring these features, and aggregating results to measure stability and overall confidence.

**Parameters:**
- `text`: A string representing the input text. (Type: str)
- `samples`: Number of samples or perturbations to generate for evaluation. Default value is 60. (Type: int)

**Return Value:** 
A dictionary containing various metrics related to the evaluated text:
- `"confidence"`: Overall confidence score.
- `"mean"`: Mean score from multiple evaluations.
- `"std"`: Standard deviation of scores indicating uncertainty.
- `"stability"`: Stability measure based on standard deviation.
- `"matched_index"`: Index of the most frequently matched document in the corpus.
- `"matched_text"`: Content of the most frequently matched document in the corpus.
- `"corpus_size"`: Size of the reference corpus.
- `"backend_source"`: Source of the memory bridge or corpus ("bridge" if available, otherwise "csr").
- `"mode"`: Evaluation mode (always set to "corpus").
- `"diagnostics"`: Additional diagnostic information including reason for evaluation and corpus size.
- `"distribution"`: Normal distribution parameters (`mu` and `sigma`) from aggregated scores.

**Important Behaviour/Side-effects:**
1. Refreshes the internal corpus if necessary using `self.refresh_corpus()`.
2. Checks if the corpus, feature extractor, or Monte Carlo engine is not available; returns a neutral report in such cases.
3. Generates multiple perturbed versions of the input text and extracts features for each version.
4. Scores extracted features and collects scores, match indices, and matched texts from these evaluations.
5. Computes mean (`mu`) and standard deviation (`sigma`) of collected scores to represent them as a normal distribution.
6. Calculates stability based on `1 - sigma` and overall confidence by multiplying mean score with stability measure.
7. Determines the most frequent match index and text among all perturbed versions.
8. Returns a dictionary containing detailed metrics for evaluation, including confidence, mean, standard deviation, stability, matched document information, corpus size, backend source, diagnostics, and distribution parameters.

This function does not have any side effects but performs necessary computations to evaluate input texts based on learned patterns from the reference corpus.

Mathematical formula:
None

Behavior

---

#### `_neutral_report`

The `_neutral_report` function generates a neutral report for evaluation scenarios where the corpus is insufficient or other necessary components are missing.

**Purpose:** 
Provides a default set of metrics and diagnostics when evaluating text confidence under conditions that prevent proper evaluation, such as an empty or insufficiently large corpus.

**Parameters:**
- `mode`: A string indicating the mode in which the neutral report should be generated. Default value is "insufficient_corpus". (Type: str)

**Return Value:** 
A dictionary containing default metrics and diagnostics:
- `"confidence"`: Overall confidence score set to 0.5.
- `"mean"`: Mean score from multiple evaluations, also set to 0.5.
- `"std"`: Standard deviation of scores indicating uncertainty, set to 0.0.
- `"stability"`: Stability measure based on standard deviation, also set to 0.5.
- `"matched_index"`: Index of the most frequently matched document in the corpus, set to -1 (indicating no match).
- `"matched_text"`: Content of the most frequently matched document in the corpus, an empty string ("").
- `"corpus_size"`: Size of the reference corpus.
- `"backend_source"`: Source of the memory bridge or corpus ("bridge" if available, otherwise "csr").
- `"mode"`: Evaluation mode (set to `mode` parameter).
- `"diagnostics"`: Additional diagnostic information including reason for evaluation and corpus size.
- `"distribution"`: Normal distribution parameters (`mu` and `sigma`) from aggregated scores, both set to default values.

**Important Behaviour/Side-effects:**
1. Sets the confidence score to 0.5 as a neutral value when no proper evaluation can be performed.
2. Sets standard deviation to 0.0 indicating minimal uncertainty in this context.
3. Returns an empty string for matched text and -1 for match index, signifying no valid matches found.
4. Provides additional diagnostics including the reason for generating a neutral report (defaulting to "insufficient_real_corpus") and corpus size.

This function does not have any side effects but returns a structured dictionary with default values suitable for scenarios where proper evaluation is impossible due to insufficient data or other constraints.

Mathematical formula:
None

Behavior Classification: This function primarily involves constructing a predefined set of metrics and diagnostics for neutral reports in cases of insufficient input or missing necessary components.

---


### SyllableAnalyzer
The `SyllableAnalyzer` class is designed to identify and evaluate the structure of syllables within a word based on vowel (V) and consonant (C) patterns. It detects common syllable shapes such as CV, CVC, VC, or V, and calculates a score representing the proportion of valid syllable structures in the given word.

### Functions

#### `syllabify`

The `syllabify` function identifies and segments the syllable structures in a given word based on vowel (V) and consonant (C) patterns. It takes one parameter:

- `word`: A string representing the input word.

It returns a list of strings, where each string represents a detected syllable pattern consisting of "V" for vowels and "C" for consonants.

The function processes each character in the word to build these patterns and appends them to the `patterns` list. If there are remaining characters at the end of processing, they are also added as a final pattern.

---

#### `score`

The `score` function calculates a score representing the proportion of valid syllable structures within a given word. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 0 and 1, where higher values indicate more valid syllable patterns in the word.

The function first calls `syllabify` to obtain the list of detected syllables. If no syllables are found, it returns 0.0. Otherwise, it counts how many of these syllables match valid patterns ("V", "CV", "CVC", or "VC"). The score is then calculated as the ratio of valid syllable structures to the total number of syllables using `_safe_div`.

The function does not have any side effects and relies on an external helper function, `_safe_div`, for safe division.

---


### ClusterAnalyzer
The `ClusterAnalyzer` class evaluates and penalizes the presence of unnatural consonant clusters in a word. It tracks the maximum consecutive sequence of consonants and assigns scores based on their length, with shorter sequences receiving higher scores. The purpose is to discourage overly complex consonant clustering by reducing the score for longer clusters.

### Functions

#### `score`

The `score` function evaluates the maximum consecutive sequence of consonants in a given word and assigns a score based on that length. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 0.3 and 1.0, where higher values indicate fewer or shorter clusters of consonants.

The function iterates through each character in the word to count consecutive consonants, updating the maximum cluster length found so far. If the longest cluster is three or less, it returns 1.0; if the longest cluster is four, it returns 0.6; otherwise, it returns 0.3.

This function does not have any side effects and relies on an external helper function `_is_consonant` to determine whether a character is a consonant.

---


### VowelAnalyzer
The `VowelAnalyzer` class evaluates the balance and distribution of vowels in a word to ensure they are neither too sparse nor too dense and have an appropriate level of diversity. It calculates a score based on the ratio of vowels to total characters and the entropy of vowel types within the word, clipping scores to maintain values between 0 and 1.

The class is responsible for assessing the overall quality of vowel usage in words by combining these two metrics into a single score that reflects both balance and diversity.

### Functions

#### `score`

The `score` function evaluates the balance and diversity of vowels in a given word and returns an overall score based on these metrics. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 0 and 1, where higher values indicate better vowel distribution and diversity.

The function first filters out all consonants to get a list of vowels in the word. If no vowels are found, it returns 0.0. Otherwise, it calculates the ratio of vowels to total characters and computes the entropy of the different types of vowels using `pd.Series.value_counts` and `entropy`.

The score is then computed as follows:
1. The vowel ratio score is calculated as \(1 - \text{abs}(ratio - 0.45)\).
2. The entropy score is clipped to a range between 0 and 1 by dividing the entropy value by 2.5.
3. Finally, the overall score is determined by averaging the ratio score and the entropy score, with each contributing equally (both weighted at 0.5), and then clipping this final score to ensure it remains within [0, 1].

This function does not have any side effects and relies on external helper functions `_is_vowel`, `pd.Series.value_counts`, and `entropy`.

---


### ForbiddenSequenceAnalyzer
The `ForbiddenSequenceAnalyzer` class evaluates and penalizes the presence of specific unstable character sequences or low-signal clusters in a word. It maintains a list of forbidden sequences (`FORBIDDEN`) that are considered undesirable.

The purpose is to assign scores based on how many of these forbidden sequences appear in the given word, with higher penalties resulting in lower scores. The class ensures that words containing no forbidden sequences receive a perfect score (1.0), while those with one or more forbidden sequences receive progressively lower scores (0.5 for one sequence and 0.2 for two or more).

### Functions

#### `score`

The `score` function evaluates the presence of forbidden character sequences in a given word and returns a score based on how many such sequences are found. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 1.0 and 0.2, where higher values indicate fewer or no forbidden sequences.

The function iterates through each forbidden sequence in `self.FORBIDDEN` to count how many times any of these sequences appear in the word. If there are no forbidden sequences found, it returns 1.0. If one forbidden sequence is found, it returns 0.5. For two or more forbidden sequences, it returns 0.2.

This function does not have any side effects and relies on the `FORBIDDEN` attribute of the class to determine which sequences are considered undesirable.

---


### MorphemeAnalyzer
The `MorphemeAnalyzer` class evaluates the realism of hyphen usage in a given word by assessing whether hyphens are used appropriately and meaningfully. It is responsible for assigning scores based on the presence and correct use of hyphens.

If no hyphens are present, it returns a score of 1.0. If any part resulting from splitting the word at hyphens has fewer than two characters, it returns a lower score of 0.4. Otherwise, if all parts have at least two characters, it returns a higher score of 0.9.

The purpose is to ensure that words with hyphenated segments are realistic and well-formed according to standard usage guidelines.

### Functions

#### `score`

The `score` function evaluates the realism of hyphen usage in a given word and returns a score based on whether the hyphens are used appropriately. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 1.0 and 0.9, where higher values indicate more appropriate hyphen usage.

The function checks if there is at least one hyphen in the word:
- If no hyphens are present, it returns 1.0.
- If any part resulting from splitting the word by hyphens has fewer than two characters, it returns a score of 0.4.
- Otherwise, it returns a score of 0.9.

This function does not have any side effects and relies solely on the input `word` to determine its output.

---


### DataAdaptivePhonProfile
The `DataAdaptivePhonProfile` class learns and applies phonotactic norms from a given corpus to evaluate the realism of new words. It tracks n-grams (bigrams and trigrams), vowel ratios, consonant clusters, and syllable patterns based on a minimum word count threshold.

Key attributes include:
- `min_words`: The minimum number of words required for active profiling.
- `active`: A boolean indicating whether enough data has been learned to make evaluations.
- `word_count`: The total number of words used in learning the norms.
- `ngrams`: A counter tracking occurrences of bigrams and trigrams.
- `patterns`: A counter tracking occurrences of syllable patterns.

The class is responsible for rebuilding its internal models from a corpus, scoring new words based on learned norms, and cleaning input words to ensure they match expected formats.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `DataAdaptivePhonProfile` class and initializes its attributes with default values or parameters provided at instantiation.

Parameters:
- `min_words`: An integer representing the minimum number of words required to consider the model active (default value: 10).

Attributes initialized:
- `self.min_words`: The minimum word count threshold.
- `self.active`: A boolean indicating whether enough data has been learned to make evaluations, initially set to False.
- `self.word_count`: An integer representing the total number of words used in learning the norms, initially set to 0.
- `self.ngrams`: A `Counter` object tracking occurrences of bigrams and trigrams, initialized as an empty counter.
- `self.patterns`: A `Counter` object tracking occurrences of syllable patterns, initialized as an empty counter.
- `self.vowel_ratio_mean`: The mean vowel ratio used for scoring, initially set to 0.45.
- `self.vowel_ratio_std`: The standard deviation of the vowel ratios used for scoring, initially set to 0.2.
- `self.max_cluster_norm`: An integer representing the maximum allowed consonant cluster length based on learned norms, initially set to 3.

This function does not return a value and has no side effects beyond initializing class attributes.

---

#### `rebuild`

The `rebuild` function updates the internal counters and statistics of the `DataAdaptivePhonProfile` class based on a given corpus. It takes two parameters:

- `corpus`: A list or iterable containing text data from which words will be extracted.
- `syllable_analyzer`: An optional object used to syllabify words (default value: None).

The function does not return a value but returns the boolean attribute `self.active`, indicating whether enough words have been processed to make evaluations.

Key behaviors:
1. Extracts valid words from the corpus using `_words_from_corpus`.
2. Counts and stores bigrams, trigrams, vowel ratios, consonant clusters, and syllable patterns for each word.
3. Updates statistical measures such as mean and standard deviation of vowel ratios (`vowel_ratio_mean` and `vowel_ratio_std`) if available data exists.
4. Sets the maximum allowed consonant cluster length based on learned norms (`max_cluster_norm`).

Side effects:
- Resets internal counters (`self.ngrams`, `self.patterns`) before updating them with new counts from the corpus.
- Updates statistical attributes like mean and standard deviation of vowel ratios, as well as the maximum consonant cluster norm.

This function ensures that the model is active only if at least `min_words` have been processed.

---

#### `score`

The `score` function evaluates the realism of a given word based on learned phonotactic norms. It takes two parameters:

- `word`: A string representing the input word.
- `syllable_patterns`: An optional list of syllable patterns (default value: None).

The function returns a dictionary containing various scores and attributes related to the evaluation, including:
- `"score"`: The combined score for the word's realism based on multiple factors.
- `"ngram"`, `"vowel_ratio"`, `"syllable_pattern"`, `"cluster_norm"`: Individual scores for ngrams, vowel ratio balance, syllable patterns, and consonant clusters respectively.

Key behaviors:
1. Checks if the model is active (i.e., enough words have been processed).
2. Cleans the input word using `_clean_word`.
3. Calculates individual scores for n-gram frequency (`ngram_score`), vowel ratio balance (`ratio_score`), syllable pattern occurrence (`pattern_score`), and consonant cluster length compliance (`cluster_score`).
4. Combines these scores to produce a final score.

Side effects:
- Cleans the input word before scoring.
- Uses internal attributes like `vowel_ratio_mean`, `vowel_ratio_std`, and `max_cluster_norm`.

If the model is not active, it returns placeholder values for all scores with `"profile_words"` indicating the current number of processed words. If the cleaned word is empty after cleaning, it assigns a score of 0.0 to each component.

---

#### `_words_from_corpus`

The `_words_from_corpus` function extracts valid words from a given corpus for further processing. It takes one parameter:

- `corpus`: A list or iterable containing text data.

It returns a list of cleaned and validated words, each at least two characters long.

Key behaviors:
1. Iterates through the provided `corpus`, converting each element to lowercase.
2. Uses regular expressions to find sequences of letters (including hyphenated words).
3. Cleans each extracted word using `_clean_word`.
4. Filters out any cleaned words that are less than two characters in length before adding them to the list.

Side effects:
- The function does not modify external variables or state but returns a filtered and processed list of words.

This function ensures that only valid, hyphenated (if applicable), and sufficiently long words are included for further analysis.

---

#### `_clean_word`

The `_clean_word` function cleans and normalizes an input word by removing disallowed characters and stripping whitespace. It takes one parameter:

- `word`: A string representing the input word.

It returns a cleaned and normalized version of the input word as a new string.

Key behaviors:
1. Converts the input word to lowercase.
2. Filters out any character not in `ALLOWED_CHARS`.
3. Strips leading and trailing whitespace from the resulting string.

Side effects:
- The function does not modify external variables or state but returns a cleaned version of the input word.

This function ensures that only valid characters are included in the processed word, making it suitable for further analysis.

---

#### `_ngrams`

The `_ngrams` function generates a list of n-grams (substrings of length `n`) from an input word. It takes two parameters:

- `word`: A string representing the input word.
- `n`: An integer specifying the size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).

It returns a list of substrings, each of length `n`, extracted from the input word.

Key behaviors:
1. Checks if the word is long enough to generate at least one n-gram.
2. Slices the word into consecutive substrings of length `n` and appends them to a list.

Side effects:
- The function does not modify external variables or state but returns a list of n-grams.

If the input word is shorter than the specified `n`, an empty list is returned. Otherwise, it generates and returns all possible n-grams from the word.

---

#### `_ngram_score`

The `_ngram_score` function calculates the score of a word based on how many n-grams (bigrams and trigrams) from the word are found in previously learned data. It takes one parameter:

- `word`: A string representing the input word.

It returns a float value between 0 and 1, where higher values indicate that more n-grams from the word have been observed before.

Key behaviors:
1. Generates bigrams (n=2) and trigrams (n=3) for the input word.
2. Counts how many of these generated n-grams are present in `self.ngrams` (a counter tracking occurrences of n-grams).
3. Calculates a score as the ratio of known n-grams to the total number of generated n-grams.

Side effects:
- The function does not modify external variables or state but returns an n-gram score based on learned data.

If no valid n-grams are found, it returns 0.0. Otherwise, it calculates and returns a normalized score using `_safe_div`.

---

#### `_pattern_score`

The `_pattern_score` function calculates the score of a word based on how many of its syllable patterns have been observed before. It takes one parameter:

- `syllable_patterns`: A list of strings representing the input word's syllable patterns.

It returns a float value between 0 and 1, where higher values indicate that more syllable patterns from the word have been observed before.

Key behaviors:
1. Counts how many of the given syllable patterns are present in `self.patterns` (a counter tracking occurrences of syllable patterns).
2. Calculates a score as the ratio of known syllable patterns to the total number of provided syllable patterns.

Side effects:
- The function does not modify external variables or state but returns a pattern score based on learned data.

If no valid syllable patterns are found, it returns 0.0. Otherwise, it calculates and returns a normalized score using `_safe_div`.

---

#### `_max_cluster`

The `_max_cluster` function calculates the length of the longest consecutive sequence of consonants in a given word. It takes one parameter:

- `word`: A string representing the input word.

It returns an integer value indicating the maximum number of consecutive consonants found in the word.

Key behaviors:
1. Iterates through each character in the word.
2. Counts the length of consecutive sequences of consonants using `_is_consonant`.
3. Tracks and updates the longest sequence encountered during iteration.

Side effects:
- The function does not modify external variables or state but returns an integer representing the maximum cluster size.

This function ensures that only valid consonants are considered in determining the longest cluster, making it useful for evaluating word structure based on consonant patterns.

---


### PhonotacticEngine
The `PhonotacticEngine` class aggregates and evaluates the phonotactic realism of words based on multiple criteria. It initializes several analyzers (SyllableAnalyzer, ClusterAnalyzer, VowelAnalyzer, ForbiddenSequenceAnalyzer, MorphemeAnalyzer) to assess different aspects of word structure and uses a DataAdaptivePhonProfile to learn from a corpus.

Key attributes include:
- `memory_bridge`: An optional object for accessing the corpus.
- `fallback_mode`: A boolean indicating whether to use fallback heuristics if adaptive scoring is not available.
- `syllable`, `cluster`, `vowel`, `forbidden`, and `morpheme`: Instances of analyzers responsible for evaluating specific phonotactic features.
- `profile`: An instance of DataAdaptivePhonProfile that learns from the corpus to adaptively score words.
- `profile_active`: A boolean indicating whether adaptive scoring is available.

The class is responsible for refreshing its internal profile based on a provided corpus and calculating combined scores for individual words or entire texts by aggregating multiple evaluation metrics.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `PhonotacticEngine` class and initializes its attributes with default values or parameters provided at instantiation. It takes two parameters:

- `memory_bridge`: An optional object used to access a corpus (default value: None).
- `fallback_mode`: A boolean indicating whether to use fallback heuristics if adaptive scoring is not available (default value: True).

Attributes initialized:
- `self.memory_bridge`: The memory bridge for accessing the corpus.
- `self.fallback_mode`: Whether to use fallback mode or not.
- `self.syllable`, `self.cluster`, `self.vowel`, `self.forbidden`, and `self.morpheme`: Instances of analyzers responsible for evaluating different phonotactic features.
- `self.profile`: An instance of DataAdaptivePhonProfile used for adaptive scoring based on learned data from the corpus.
- `self.profile_active`: A boolean indicating whether adaptive scoring is available, initially set to False.
- `self._corpus_fingerprint`: A hash value representing the current state of the corpus, initialized as None.

Key behaviors:
1. Initializes instances of various analyzers (`SyllableAnalyzer`, `ClusterAnalyzer`, etc.) for evaluating different phonotactic features.
2. Sets up an instance of DataAdaptivePhonProfile to learn from and adaptively score words based on a provided corpus.
3. Calls the `refresh_profile` method with force=True to ensure that the profile is refreshed immediately upon initialization.

Side effects:
- The function does not return a value but initializes several attributes and calls `refresh_profile`.

This ensures that the class starts in an appropriate state for evaluating phonotactic realism of words based on learned data or fallback heuristics as needed.

---

#### `_base_score`

The `_base_score` function calculates a weighted average score based on multiple phonotactic evaluation metrics for a given word or text. It takes one parameter:

- `scores`: A dictionary containing individual scores for different phonotactic features (syllable, cluster, vowel, forbidden, and morpheme).

It returns a float value representing the combined base score.

Key behaviors:
1. Weights each of the input scores by predefined coefficients.
2. Sums these weighted scores to produce an overall base score.

Side effects:
- The function does not modify external variables or state but calculates and returns a single aggregated score based on multiple evaluation metrics.

The weights used are as follows:
- Syllable: 0.25
- Cluster: 0.20
- Vowel: 0.20
- Forbidden: 0.20
- Morpheme: 0.15

This function ensures that each phonotactic feature contributes appropriately to the overall score, reflecting a balanced evaluation of word or text realism based on learned norms and heuristics.

---

#### `refresh_profile`

The `refresh_profile` function updates the internal profile of phonotactic norms based on a provided corpus. It takes one parameter:

- `force`: A boolean indicating whether to force refreshing the profile even if no changes are detected (default value: False).

It returns a boolean value indicating whether the profile was refreshed.

Key behaviors:
1. Checks if a memory bridge is available for accessing the corpus.
2. Retrieves the corpus from the memory bridge or sets it to an empty list in case of exceptions.
3. Computes a fingerprint hash (`fp`) representing the current state of the corpus.
4. Compares the new fingerprint with the stored one and decides whether to refresh based on `force` parameter or changes detected.
5. Updates the internal profile using the provided corpus if necessary, setting `self.profile_active` accordingly.

Side effects:
- The function updates the internal attributes (`profile_active`, `_corpus_fingerprint`) and may call methods of other objects (e.g., `memory_bridge.get_corpus()`).

This ensures that the phonotactic norms are regularly updated based on new or changed corpus data, maintaining accurate adaptive scoring capabilities.

---

#### `score_word`

The `score_word` function evaluates the phonotactic realism of a given word by aggregating multiple evaluation metrics. It takes one parameter:

- `word`: A string representing the input word.

It returns a dictionary containing various scores and metadata related to the evaluation.

Key behaviors:
1. Syllabifies the word using `self.syllable` and calculates individual scores for different phonotactic features (syllable, cluster, vowel, forbidden, morpheme).
2. Evaluates adaptive scores based on learned data from the profile.
3. Combines base scores with adaptive scores if available or uses fallback heuristics if not.
4. Updates the dictionary of scores to include additional metadata such as mode used and individual components of the adaptive score.

Side effects:
- The function updates internal dictionaries and attributes but does not modify external variables.

This ensures that each word is evaluated comprehensively, combining learned norms with heuristic evaluations for a balanced assessment of phonotactic realism.

---

#### `score_text`

The `score_text` function evaluates the phonotactic realism of each word within a given text and returns an overall score for the entire text. It takes one parameter:

- `text`: A string representing the input text.

It returns a tuple containing:
1. A float value representing the mean combined score of all words in the text.
2. A list of dictionaries, where each dictionary contains detailed scores and metadata for individual words.

Key behaviors:
1. Refreshes the internal profile using `refresh_profile` to ensure up-to-date norms.
2. Splits the input text into a list of non-empty words.
3. Evaluates each word's phonotactic realism by calling `score_word`.
4. Collects detailed scores and metadata for each evaluated word in the `diagnostics` list.
5. Computes the mean combined score across all evaluated words.

Side effects:
- The function calls `refresh_profile`, which may update internal attributes, but does not modify external variables or state beyond returning the results.

This ensures that the entire text is assessed comprehensively by aggregating individual word scores and providing detailed diagnostics for each evaluation.

---


### TextPerturbator
The `TextPerturbator` class applies controlled noise to a given text by deleting and swapping characters based on specified probabilities. It initializes with two key attributes:

- `deletion_prob`: The probability of deleting each character in the text (default value: 0.1).
- `swap_prob`: The probability of swapping adjacent characters in the text (default value: 0.05).

The class is responsible for applying controlled noise to a given input text by first deleting characters according to the deletion probability and then swapping adjacent characters based on the swap probability, resulting in perturbed versions of the original text.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `TextPerturbator` class and initializes its attributes with default values or parameters provided at instantiation. It takes two parameters:

- `deletion_prob`: A float representing the probability of deleting each character in the text (default value: 0.1).
- `swap_prob`: A float representing the probability of swapping adjacent characters in the text (default value: 0.05).

It does not return a value but initializes two attributes:
- `self.deletion_prob`: The deletion probability.
- `self.swap_prob`: The swap probability.

Side effects:
- The function sets up instance variables for controlling character deletions and swaps, which are used in subsequent methods to apply perturbations to text.

This ensures that the class starts with appropriate probabilities for deleting and swapping characters when applying controlled noise to input texts.

---

#### `delete_chars`

The `delete_chars` function applies character deletions to a given text based on a specified deletion probability. It takes one parameter:

- `text`: A string representing the input text.

It returns a new string with characters deleted according to the deletion probability (`self.deletion_prob`).

Key behaviors:
1. Iterates through each character in the input text.
2. Randomly decides whether to include or delete each character based on the deletion probability using `np.random.rand()`.
3. Collects included characters into a list and joins them back into a string.

Side effects:
- The function does not modify external variables but returns a modified version of the input text with controlled deletions applied.

This ensures that the returned text has some characters deleted according to the specified probability, introducing controlled noise to the original text.

---

#### `swap_chars`

The `swap_chars` function applies character swaps to a given text based on a specified swap probability. It takes one parameter:

- `text`: A string representing the input text.

It returns a new string with adjacent characters swapped according to the swap probability (`self.swap_prob`).

Key behaviors:
1. Converts the input text into a list of characters.
2. Iterates through each pair of adjacent characters in the list.
3. Randomly decides whether to swap each pair based on the swap probability using `np.random.rand()`.
4. Swaps the positions of adjacent characters if the random decision is favorable.

Side effects:
- The function does not modify external variables but returns a modified version of the input text with controlled swaps applied.

This ensures that the returned text has some pairs of adjacent characters swapped according to the specified probability, introducing controlled noise to the original text.

---

#### `perturb`

The `perturb` function applies controlled noise to a given text by first deleting characters and then swapping adjacent characters based on specified probabilities. It takes one parameter:

- `text`: A string representing the input text.

It returns a new string with both character deletions and swaps applied according to the deletion probability (`self.deletion_prob`) and swap probability (`self.swap_prob`).

Key behaviors:
1. Calls `delete_chars` to apply character deletions based on the specified deletion probability.
2. Calls `swap_chars` to apply adjacent character swaps based on the specified swap probability.

Side effects:
- The function does not modify external variables but returns a modified version of the input text with both controlled deletions and swaps applied.

This ensures that the returned text has been perturbed by applying both types of noise (deletions and swaps) in sequence, introducing controlled variations to the original text.

---


### StabilityAnalyzer
The `StabilityAnalyzer` class measures the confidence stability of a given text under perturbation. It initializes with two key attributes:

- `ci_engine`: An object for evaluating confidence intervals.
- `phonotactic_engine`: An engine for phonotactic evaluation.

It also includes an instance of `TextPerturbator` to apply controlled noise to input texts.

The class is responsible for running multiple trials where it perturbs the text, evaluates its confidence and phonotactic scores, and calculates stability metrics such as mean, standard deviation, minimum, maximum, and robustness based on these evaluations.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `StabilityAnalyzer` class and initializes its attributes with default values or parameters provided at instantiation. It takes two parameters:

- `ci_engine`: An object responsible for evaluating confidence intervals.
- `phonotactic_engine`: An engine used for phonotactic evaluation.

It does not return a value but initializes three key attributes:
- `self.ci_engine`: The CI (confidence interval) engine.
- `self.phonotactic_engine`: The phonotactic engine.
- `self.perturbator`: An instance of the `TextPerturbator` class for applying controlled noise to text.

Side effects:
- The function sets up necessary components required for evaluating stability metrics, ensuring that both confidence interval and phonotactic evaluations can be performed on perturbed texts.

This ensures that the class starts with appropriate engines and a perturbation tool ready to measure confidence stability under various levels of text corruption.

---

#### `run_trials`

The `run_trials` function performs multiple trials of text perturbation and evaluation to measure confidence intervals (CI) and phonotactic scores. It takes two parameters:

- `text`: A string representing the input text.
- `trials`: An integer specifying the number of perturbation trials to run (default value: 50).

It returns a tuple containing:
1. A NumPy array of CI confidence values from each trial.
2. A NumPy array of phonotactic scores from each trial.

Key behaviors:
1. Iterates `trials` times, applying controlled noise to the input text using `self.perturbator`.
2. Evaluates the perturbed text with both the CI engine and phonotactic engine.
3. Appends the confidence score from the CI report and the phonotactic score for each trial.

Side effects:
- The function does not modify external variables but returns arrays of scores collected over multiple trials.

This ensures that the returned values represent stability metrics derived from repeated perturbations and evaluations, providing insights into how robust the text's evaluation is under noise.

---

#### `stability_metrics`

The `stability_metrics` function calculates stability metrics for a given text based on multiple perturbation trials. It takes two parameters:

- `text`: A string representing the input text.
- `trials`: An integer specifying the number of perturbation trials to run (default value: 50).

It returns a dictionary containing various stability metrics derived from the confidence interval scores and phonotactic scores collected during the trials.

Key behaviors:
1. Calls `run_trials` to perform multiple perturbations on the input text, collecting CI confidence values (`ci_scores`) and phonotactic scores (`phon_scores`).
2. Computes statistical measures for both sets of scores:
   - Mean, standard deviation, minimum, and maximum of CI scores.
   - Mean and standard deviation of phonotactic scores.
3. Calculates robustness as \(1 - \text{std}(ci\_scores)\).

Return value includes the following metrics:
- `"ci_mean"`: The mean confidence score from all trials.
- `"ci_std"`: The standard deviation of confidence scores from all trials.
- `"ci_min"`: The minimum confidence score from all trials.
- `"ci_max"`: The maximum confidence score from all trials.
- `"phon_mean"`: The mean phonotactic score from all trials.
- `"phon_std"`: The standard deviation of phonotactic scores from all trials.
- `"robustness"`: A measure of robustness calculated as \(1 - \text{std}(ci\_scores)\).

Side effects:
- The function does not modify external variables but returns a dictionary containing stability metrics.

This ensures that the returned values provide comprehensive insights into how stable and consistent the evaluations are under perturbations.

---


### FeatureAttribution
The `FeatureAttribution` class determines which features contributed positively or negatively to confidence by comparing their values against predefined thresholds. It initializes with one key attribute:

- `thresholds`: A dictionary specifying the threshold values for positive and negative contributions (default value: {"good": 0.7, "bad": 0.4}).

The class is responsible for categorizing features into three groups based on their scores:
1. Positive contributors if their score meets or exceeds the "good" threshold.
2. Negative contributors if their score falls below the "bad" threshold.
3. Neutral contributors otherwise.

This ensures that each feature's impact on confidence can be clearly identified and categorized as positive, negative, or neutral.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `FeatureAttribution` class and initializes its attributes with default values or parameters provided at instantiation. It takes one parameter:

- `thresholds`: A dictionary specifying threshold values for positive ("good") and negative ("bad") contributions (default value: {"good": 0.7, "bad": 0.4}).

It does not return a value but initializes the attribute:
- `self.thresholds`: The thresholds used to categorize feature scores.

Side effects:
- The function sets up instance variables for determining positive and negative contributions based on predefined or provided threshold values.

This ensures that the class starts with appropriate threshold settings to evaluate which features contribute positively, negatively, or neutrally to confidence.

---

#### `attribute`

The `attribute` function categorizes feature scores into positive, negative, or neutral contributions based on predefined thresholds. It takes one parameter:

- `features`: A dictionary where keys are feature names and values are their corresponding scores.

It returns a tuple containing three lists:
1. A list of features that contributed positively.
2. A list of features that contributed negatively.
3. A list of features with neutral contributions.

Key behaviors:
1. Iterates through each feature in the input `features` dictionary.
2. Compares each feature's score against the "good" and "bad" thresholds stored in `self.thresholds`.
3. Categorizes each feature into positive, negative, or neutral based on its score relative to these thresholds.

Side effects:
- The function does not modify external variables but returns lists of features categorized by their contribution type.

This ensures that the returned values provide a clear breakdown of which features are contributing positively, negatively, or neutrally to confidence.

---


### FailureModeAnalyzer
The `FailureModeAnalyzer` class infers the reasons for degraded confidence by evaluating various feature scores and overall stability metrics. It is responsible for identifying potential issues such as character corruption, abnormal structure, divergence from known patterns, phonotactic violations, or high sensitivity to noise.

Key attributes:
- `features`: A dictionary containing feature scores.
- `stability`: A float representing the stability of confidence under perturbation.

The class categorizes and returns a list of reasons for degraded confidence based on predefined thresholds for each feature score and overall stability level. If no dominant failure mode is detected, it indicates this as well.

### Functions

#### `analyze`

The `analyze` function identifies potential reasons for degraded confidence by evaluating various feature scores and overall stability metrics. It takes two parameters:

- `features`: A dictionary containing feature scores (e.g., "char_valid", "structure", "similarity", "phonotactics").
- `stability`: A float representing the stability of confidence under perturbation.

It returns a list of strings, each describing a potential reason for degraded confidence based on predefined thresholds and conditions.

Key behaviors:
1. Checks if character corruption is detected by evaluating `"char_valid"` score.
2. Identifies abnormal vowel-consonant structure by checking the `"structure"` score.
3. Detects divergence from known memory patterns using the `"similarity"` score.
4. Determines phonotactic violations based on the `"phonotactics"` score.
5. Evaluates if confidence is highly sensitive to noise by comparing `stability` against a threshold.

If no dominant failure mode is detected, it appends "No dominant failure mode detected" to the list of reasons.

Side effects:
- The function does not modify external variables but returns a list of potential issues contributing to degraded confidence.

This ensures that the returned values provide clear insights into why confidence may have decreased based on various feature scores and stability metrics.

---


### ExplanationGenerator
The `ExplanationGenerator` class converts diagnostic metrics into natural-language explanations for why a given text has high or low confidence. It initializes with two key attributes:

- `attributor`: An instance of the `FeatureAttribution` class to categorize feature contributions.
- `failure_analyzer`: An instance of the `FailureModeAnalyzer` class to identify failure modes.

The purpose is to generate detailed explanations by analyzing positive, negative, and neutral features as well as potential failure modes. The class constructs a summary sentence based on identified issues or strong evidence supporting high confidence.

This ensures that diagnostic metrics are translated into clear, natural-language explanations for better understanding of the factors affecting text confidence.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `ExplanationGenerator` class and initializes its attributes with default values or parameters provided at instantiation. It does not take any parameters.

It sets up two key attributes:
- `self.attributor`: An instance of the `FeatureAttribution` class to categorize feature contributions.
- `self.failure_analyzer`: An instance of the `FailureModeAnalyzer` class to identify failure modes.

This function does not return a value but initializes necessary components for generating natural-language explanations based on diagnostic metrics and identified issues.

---

#### `generate`

The `generate` function converts diagnostic metrics into a natural-language explanation for why a given text has high or low confidence. It takes two parameters:

- `features`: A dictionary containing feature scores (e.g., "char_valid", "structure", etc.).
- `stability`: A float representing the stability of confidence under perturbation.

It returns a dictionary containing detailed explanations and summary sentences based on identified issues or strong evidence supporting high confidence.

Key behaviors:
1. Categorizes features into positive, negative, and neutral contributions using `self.attributor.attribute`.
2. Constructs natural-language descriptions for each category (positives, negatives, neutral) by referencing predefined reasons (`POSITIVE_REASONS`, `NEGATIVE_REASONS`) or generating default messages.
3. Analyzes potential failure modes using `self.failure_analyzer.analyze` and appends them to the explanation dictionary.
4. Generates a summary sentence based on whether there are any negative contributions.

Side effects:
- The function does not modify external variables but returns a detailed explanation dictionary containing categorized feature contributions, identified issues, and a summary sentence.

This ensures that diagnostic metrics are translated into clear, natural-language explanations for better understanding of the factors affecting text confidence.

---


### DiagnosticEngine
The `DiagnosticEngine` class provides a high-level interface for generating detailed explanations based on confidence interval (CI) reports and phonotactic scores. It initializes with one key attribute:

- `generator`: An instance of the `ExplanationGenerator` class to convert diagnostics into natural-language explanations.

The purpose is to generate comprehensive explanations by combining CI report metrics, phonotactic scores, and other relevant features. The class constructs a detailed explanation dictionary that includes positive, negative, neutral contributions, failure modes, and a summary sentence.

This ensures that diagnostic outputs are translated into clear, understandable explanations for better interpretability of text confidence evaluations.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `DiagnosticEngine` class and initializes its attributes with default values or parameters provided at instantiation. It does not take any parameters.

It sets up one key attribute:
- `self.generator`: An instance of the `ExplanationGenerator` class to convert diagnostics into natural-language explanations.

This function does not return a value but initializes necessary components for generating detailed explanations based on diagnostic metrics and identified issues.

---

#### `explain`

The `explain` function generates a detailed explanation based on the provided confidence interval (CI) report and phonotactic score. It takes two parameters:

- `ci_report`: A dictionary containing output from CIEngine.evaluate().
- `phon_score`: A float representing the phonotactic score from Module 2.

It returns a dictionary containing a comprehensive natural-language explanation of why a given text has high or low confidence, including positive and negative contributions, neutral features, failure modes, and a summary sentence.

Key behaviors:
1. Constructs a feature dictionary (`features`) with relevant metrics derived from `ci_report` and `phon_score`.
2. Calls the `generate` method of the `self.generator` to produce detailed explanations based on these features.
3. Returns the generated explanation dictionary.

Side effects:
- The function does not modify external variables but returns a detailed explanation dictionary containing categorized feature contributions, identified issues, and a summary sentence.

This ensures that diagnostic metrics are translated into clear, natural-language explanations for better understanding of the factors affecting text confidence.

---


### BenchmarkEvaluator
The `BenchmarkEvaluator` class compares confidence scores for provided sets of real positive and negative texts. It initializes with one key attribute:

- `ci_engine`: An instance responsible for evaluating confidence intervals.

The purpose is to evaluate the performance by comparing mean and standard deviation of confidence scores between positive and negative text sets, providing a separation score that indicates how well they are distinguished.

This ensures that the class can assess the effectiveness of the CI engine in distinguishing between real positive and negative texts based on their confidence scores.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `BenchmarkEvaluator` class and initializes its attributes with default values or parameters provided at instantiation. It takes one parameter:

- `ci_engine`: An object responsible for evaluating confidence intervals.

It does not return a value but sets up the attribute:
- `self.ci_engine`: The CI engine used to evaluate confidence scores for positive and negative text sets.

Side effects:
- The function initializes necessary components required for benchmarking by setting up the CI engine instance.

This ensures that the class starts with an appropriate CI engine ready to perform evaluations on provided real positive and negative text sets.

---

#### `evaluate`

The `evaluate` function compares confidence scores for provided sets of real positive and negative texts. It takes two parameters:

- `positives`: A list or iterable containing strings representing the input positive text samples.
- `negatives`: A list or iterable containing strings representing the input negative text samples.

It returns a dictionary with evaluation results, including mean and standard deviation of confidence scores for both sets and their separation score. The function also handles cases where either set is empty by returning an error message.

Key behaviors:
1. Cleans the provided positive and negative texts using `_clean_eval_texts`.
2. Checks if either cleaned text list is empty; returns a failure response with details if so.
3. Evaluates confidence scores for each text in both sets using `self.ci_engine.evaluate`.
4. Computes mean and standard deviation of confidence scores for positive and negative sets.
5. Calculates the separation score as the difference between the means of positive and negative scores.

Side effects:
- The function cleans input texts but does not modify external variables beyond returning a dictionary with evaluation results.

This ensures that the function provides comprehensive performance metrics to assess how well the CI engine distinguishes real positive from negative text sets based on confidence scores.

---


### RobustnessAnalyzer
The `RobustnessAnalyzer` class measures confidence across caller-provided real text variants at different levels of perturbation. It initializes with one key attribute:

- `ci_engine`: An instance responsible for evaluating confidence intervals.

The purpose is to generate a robustness curve by evaluating the mean and standard deviation of confidence scores for each level of provided text variants, assessing how well the CI engine maintains consistent performance under varying conditions.

This ensures that the class can evaluate the stability and reliability of confidence evaluations across different levels of perturbation in input texts.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `RobustnessAnalyzer` class and initializes its attributes with default values or parameters provided at instantiation. It takes one parameter:

- `ci_engine`: An object responsible for evaluating confidence intervals.

It does not return a value but sets up the attribute:
- `self.ci_engine`: The CI engine used to evaluate confidence scores for text variants.

Side effects:
- The function initializes necessary components required for robustness analysis by setting up the CI engine instance.

This ensures that the class starts with an appropriate CI engine ready to perform evaluations on provided real text variants.

---

#### `robustness_curve`

The `robustness_curve` function measures confidence across caller-provided real text variants at different levels of perturbation. It takes one parameter:

- `variants_by_level`: A dictionary where keys are perturbation levels and values are lists or iterables containing strings representing the input text samples for each level (default value: None).

It returns a dictionary with evaluation results, including mean confidence scores, standard deviation, and count of evaluated variants at each level. The function handles cases where no variants are provided by returning an error message.

Key behaviors:
1. Checks if `variants_by_level` is empty; returns a failure response with details if so.
2. Iterates through each perturbation level and its corresponding text samples.
3. Cleans the input texts using `_clean_eval_texts`.
4. Evaluates confidence scores for each variant in the list using `self.ci_engine.evaluate`.
5. Computes mean and standard deviation of confidence scores for variants at each level.

Side effects:
- The function cleans input texts but does not modify external variables beyond returning a dictionary with evaluation results.

This ensures that the function provides comprehensive performance metrics to assess how well the CI engine maintains consistent confidence evaluations across different levels of perturbation in text samples.

---


### AblationStudy
The `AblationStudy` class is designed to evaluate the confidence score of a given text using a core inference engine, optionally incorporating phonotactic scoring from an additional phonology engine. It reports component scores without simulating removed components and indicates that ablation study simulations are not available due to lack of alternate engines for comparison.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `AblationStudy` class. It takes two parameters:
- `ci_engine`: The core inference engine used for evaluating text confidence.
- `phon_engine` (optional): An additional phonology engine used to score the phonotactics of a given text.

This method sets up the attributes `self.ci_engine` and `self.phon_engine`. There is no return value, and it does not have any side effects beyond initializing instance variables.

---

#### `run`

The `run` function evaluates the given text using the core inference engine and, if available, a phonology engine. It returns a dictionary containing evaluation results.

Parameters:
- `text`: The input text to be evaluated (type: str).

Return value:
A dictionary (`results`) with the following keys:
- `"full"`: Confidence score from the core inference engine.
- `"phonotactics"`: Optional confidence score for phonotactic scoring if a phonology engine is provided.
- `"mode"`: Set to "component_report".
- `"ablation_available"`: Always set to `False`.
- `"reason"`: Reason why ablation study simulations are not available, typically set to "no_alternate_real_engine_supplied".

Important behavior:
- If the phonology engine (`self.phon_engine`) is provided, it scores the text and adds the score under the key `"phonotactics"`.
- The function always indicates that ablation studies are unavailable by setting `"ablation_available"` to `False` and providing a reason.

---


### EvaluationSuite
The `EvaluationSuite` class provides a high-level interface for comprehensive text evaluation. It initializes with core inference and phonology engines, along with an optional corpus of texts. Key attributes include the cleaned corpus, benchmark evaluator, robustness analyzer, and ablation study instance. The class is responsible for coordinating various evaluations—benchmarking against positive and negative samples, analyzing robustness across variants, and reporting component scores without simulating removed components.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `EvaluationSuite` class. It takes three parameters:
- `ci_engine`: The core inference engine used for evaluations.
- `phon_engine` (optional): An additional phonology engine used for ablation studies.
- `corpus` (optional): A corpus of texts to be cleaned and stored.

This method sets up the following attributes:
- `self.corpus`: Cleaned version of the provided corpus or an empty list if no corpus is given.
- `self.benchmark`: An instance of `BenchmarkEvaluator` initialized with the core inference engine.
- `self.robustness`: An instance of `RobustnessAnalyzer` initialized with the core inference engine.
- `self.ablation`: An instance of `AblationStudy` initialized with the core and phonology engines.

There is no return value, and it does not have any side effects beyond initializing these attributes.

---

#### `full_evaluation`

The `full_evaluation` function performs a comprehensive evaluation of the given text, including benchmarking against positive and negative samples, analyzing robustness across variants, and running an ablation study. It returns a dictionary containing results from each evaluation component.

Parameters:
- `text`: The input text to be evaluated (type: str).
- `corpus` (optional): A corpus of texts used for benchmarking (default is the initialized corpus).
- `negatives` (optional): Negative samples for benchmarking.
- `robustness_variants` (optional): Variants used for robustness analysis.

Return value:
A dictionary with keys `"benchmark"`, `"robustness"`, and `"ablation"` containing results from each respective evaluation component.

Important behavior:
- Cleans the provided corpus or uses the initialized corpus if no corpus is given.
- Calls `self.benchmark.evaluate` to generate benchmarking results using positive samples (`positives`) and negative samples (`negatives`).
- Calls `self.robustness.robustness_curve` to analyze robustness across specified variants.
- Runs an ablation study on the input text using `self.ablation.run`.
- Returns a dictionary aggregating all evaluation results.

Side effects:
- No side effects beyond performing evaluations and returning aggregated results.

---


---

## Global Functions

### Functions

#### `_safe_div`

The `_safe_div` function safely performs division of two numbers, ensuring that division by zero does not occur.

Parameters:
- `a`: The numerator (type: float or int).
- `b`: The denominator (type: float or int).

Return value:
A float representing the result of dividing `a` by `b`, or 0.0 if `b` is zero.

Important behavior:
- Returns 0.0 if the denominator (`b`) is zero to avoid division by zero errors.
- Otherwise, returns the result of `a / b`.

Side effects:
- No side effects; it only performs a safe division operation and does not modify any external state.

---

#### `_clip01`

The `_clip01` function clips a given value to ensure it falls within the range [0.0, 1.0].

Parameters:
- `x`: The input value (type: float or int).

Return value:
A float representing the clipped value of `x`, constrained between 0.0 and 1.0.

Important behavior:
- Uses `np.clip` to clip the input value `x` to ensure it is not less than 0.0 and not greater than 1.0.
- Converts the result to a float before returning.

Side effects:
- No side effects; it only performs clipping and does not modify any external state.

---

#### `get_default_memory_bridge`

The `get_default_memory_bridge` function retrieves the default memory bridge instance, ensuring it is initialized only once.

Parameters:
- No parameters.

Return value:
A singleton instance of `HerebotMemoryBridge`.

Important behavior:
- Uses a global lock `_DEFAULT_MEMORY_LOCK` to ensure thread-safe initialization.
- Initializes `_DEFAULT_MEMORY_BRIDGE` with an instance of `HerebotMemoryBridge(csr=CSR_MEMORY)` if it is not already initialized.
- Returns the default memory bridge instance.

Side effects:
- Ensures that `_DEFAULT_MEMORY_BRIDGE` is only created once, making this function idempotent.

---

#### `add_sentence_to_memory`

The `add_sentence_to_memory` function adds a sentence to the default memory bridge or, if an exception occurs, directly to `CSR_MEMORY`.

Parameters:
- `text`: The input text (type: str).

Return value:
A boolean indicating whether the addition was successful.

Important behavior:
- Uses `get_default_memory_bridge()` to retrieve the singleton instance of the memory bridge.
- Adds the provided sentence to the memory using `add_runtime_text` with specified source and document ID derived from a hash of the text.
- If an exception occurs during this process, it falls back to adding the text directly to `CSR_MEMORY`.

Side effects:
- Modifies the default memory bridge or `CSR_MEMORY` by adding the provided sentence.

---

#### `load_all_shard_texts`

The `load_all_shard_texts` function loads all text shards from the memory bridge or fallback storage.

Parameters:
- `memory_bridge`: An optional parameter for specifying a custom memory bridge (type: object).

Return value:
A list of strings representing all loaded shard texts.

Important behavior:
- If a valid `memory_bridge` is provided, it attempts to retrieve the corpus using `get_corpus()`.
- Fallbacks to loading text shards from `CSR_MEMORY.get_all_texts()` if available.
- As a last resort, loads shard texts by iterating over metadata files in `CSR_MEMORY.shard_dir`, extracting and extending the list of texts.

Side effects:
- No side effects; it only reads and returns data without modifying external state.

---

#### `_is_vowel`

The `_is_vowel` function checks whether a given character is a vowel.

Parameters:
- `c`: The input character (type: str).

Return value:
A boolean indicating whether the character is a vowel.

Important behavior:
- Checks if the character `c` is in the predefined set of vowels (`VOWELS`).
- Returns `True` if the character is a vowel, otherwise returns `False`.

Side effects:
- No side effects; it only performs a membership check and does not modify any external state.

---

#### `_is_consonant`

The `_is_consonant` function checks whether a given character is a consonant.

Parameters:
- `c`: The input character (type: str).

Return value:
A boolean indicating whether the character is a consonant.

Important behavior:
- Checks if the character `c` is in the predefined set of consonants (`CONSONANTS`).
- Returns `True` if the character is a consonant, otherwise returns `False`.

Side effects:
- No side effects; it only performs a membership check and does not modify any external state.

---

#### `_safe_div`

The `_safe_div` function safely performs division of two numbers, ensuring that division by zero does not occur.

Parameters:
- `a`: The numerator (type: float or int).
- `b`: The denominator (type: float or int).

Return value:
A float representing the result of dividing `a` by `b`, or 0.0 if `b` is zero.

Important behavior:
- Returns 0.0 if the denominator (`b`) is zero to avoid division by zero errors.
- Otherwise, returns the result of `a / b`.

Side effects:
- No side effects; it only performs a safe division operation and does not modify any external state.

---

#### `_clean_eval_texts`

The `_clean_eval_texts` function cleans a list of input texts by removing extra whitespace and filtering out empty strings.

Parameters:
- `texts`: A list of text inputs (type: List[str]) or None, which defaults to an empty list if not provided.

Return value:
A cleaned list of non-empty, space-normalized strings.

Important behavior:
- Joins each string in the input list with a single space and removes extra whitespace.
- Filters out any resulting empty strings from the processed texts.

Side effects:
- No side effects; it only processes and returns a cleaned version of the input list.

---


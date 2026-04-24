## class HierarchicalConceptModel:
Represents words as distributions over an unbounded set of latent concepts.
Uses a Dirichlet Process prior (via Chinese Restaurant Process) to allow
new concepts to be created as needed.
Now config‑aware: reads alpha, beta from ConfigManager.

---

### functions

---

1) ### init 
    The `__init__` function is the constructor for a class, which initializes various attributes when an object of that class is created:

    ### Variables Initialized and Used:
    - **config_manager**: An optional parameter of type `Optional['ConfigManager']`. If provided, it will be stored as an instance variable.
    - **alpha**: An optional float value for the hyperparameter alpha. It is either derived from `config_manager` or directly passed in as a parameter.
    - **beta**: An optional float value for the hyperparameter beta. Similar to alpha, it can come from `config_manager` or be provided as a direct argument.
    - **concept_counts**: A `Counter` object that tracks the total number of assignments per concept.
    - **word_concept_counts**: A nested dictionary where each key is a word and its value is a `Counter` tracking how many times each concept was assigned to that word.
    - **concept_features**: A nested dictionary where each key is a concept, and its value is a `Counter` representing the count of features associated with that concept.
    - **feature_vocab**: A set used to store all unique feature words encountered during processing.
    - **prev_posterior**: A dictionary storing previous posterior probabilities for each word (word -> {concept: probability}).
    - **meta_concept_counts**: A `Counter` object tracking counts per meta-concept.
    - **concept_to_meta**: A dictionary mapping concepts to their respective meta-concepts.
    - **meta_alpha**: A float representing a hyperparameter value set to 0.1.

    ### Functionality:
    1. **Initialization of config_manager**:
    - The function starts by setting `self.config_manager` to the provided `config_manager`.
    
    2. **Setting alpha and beta based on config_manager or direct arguments**:
    - If `config_manager` is not None, it uses the values specified in the configuration manager for both `alpha` and `beta`, otherwise, it defaults these parameters to 0.5 and 1.0 respectively.

    3. **Initializing Counters and Dictionaries**:
    - The function initializes several counters (`concept_counts`, `meta_concept_counts`) and dictionaries (`word_concept_counts`, `concept_features`, `prev_posterior`, `concept_to_meta`), setting them up to store relevant counts, features, and previous posterior probabilities.
    
    4. **Setting Meta-Concept Hyperparameter**:
    - The hyperparameter `meta_alpha` is set to 0.1.

---

2) ### update_parameters_from_config
    Reload parameters from the config manager (for hot reload).
    ass

    The function `update_parameters_from_config` updates the instance variables `alpha` and `beta` based on values retrieved from a configuration manager (`self.config_manager`). If `self.config_manager` is not `None`, it calls the method `get` of `self.config_manager` to retrieve the value associated with `'hierarchical_concept_model.alpha'` and assigns it to `self.alpha`. Similarly, it retrieves the value associated with `'hierarchical_concept_model.beta'` and assigns it to `self.beta`. Both retrieval operations provide default values: `0.5` for `alpha` if no specific value is found in the configuration manager, and `1.0` for `beta` under similar conditions.

    The variables this function initializes or updates are:
    - `self.alpha`: initialized with the value from `'hierarchical_concept_model.alpha'` in the config manager, defaulting to `0.5`.
    - `self.beta`: initialized with the value from `'hierarchical_concept_model.beta'` in the config manager, defaulting to `1.0`.

    The function uses the following variables:
    - `self.config_manager`: an instance variable assumed to be a configuration manager that provides methods for retrieving values based on keys.
    - `self.alpha`: an attribute of the class which is updated with the value retrieved from the configuration manager or defaults to `0.5`.
    - `self.beta`: another attribute of the class which is updated similarly, defaulting to `1.0`.

    The function does not initialize any variables beyond updating existing instance attributes and does not create any new local variables within its scope.
    
---

3) ### add_word_observation
    The function `add_word_observation` performs the following operations:

    1. **Initializes and Updates Previous Posterior:**
    - Checks if `word` exists in `self.word_concept_counts`.
    - If it does, calculates the previous posterior probability distribution over concepts for this word.
    - Stores the result in `self.prev_posterior[word]`.

    2. **Computes CRP Probabilities:**
    - Computes probabilities using Chinese Restaurant Process (CRP) for assigning a concept to the given word.
    - Includes existing concepts based on their counts and a probability for a new concept.

    3. **Selects Concept via Sampling:**
    - Samples a concept from the computed CRP probabilities.

    4. **Updates Counts:**
    - Increments the count of the selected concept in `self.concept_counts`.
    - Updates the word-concept count in `self.word_concept_counts[word][concept]`.

    5. **Initializes Concept Features (if new):**
    - If the sampled concept is new, initializes its feature distribution using `_initialize_concept_features(concept, features)`.

    6. **Updates Feature Counts:**
    - Updates the counts of each feature associated with the selected concept.
    - Adds each feature to `self.feature_vocab` if not already present.

    7. **Returns Concept ID:**
    - Returns the assigned concept ID as an integer.

    Variables Initialised and Used:
    - `total`: Sum of counts for concepts or word-concepts.
    - `probs`, `concepts`: Arrays/lists used in CRP probability calculation.
    - `concept`: Integer representing the selected concept ID.
    - `k, v`: Keys and values from the input `features` dictionary.

    The function does not modify `word`, `features`, nor return any other variables.

---

4) ### _initialize_concept_features
    The function `_initialize_concept_features` performs the following operations:

    1. Iterates through each key-value pair `(k, v)` in the `features` dictionary.
    2. Initializes or updates the feature count for the given concept with the value scaled by a prior beta factor.

    Variables Used:
    - `concept`: Integer representing the concept ID.
    - `features`: Dictionary of features and their corresponding values.
    - `beta`: Prior scaling factor.

    The function does not return any variables.

---

5) ### predict_features
    The function `predict_features` computes a predictive distribution over features for a given word by marginalizing over concepts. Here is the precise description of its behavior:

    1. **Input Parameters**:
    - `word`: A string representing the input word.

    2. **Variables Initialized and Used**:
    - `total`: An integer or float that stores the total count of concepts associated with the input word.
    - `probs`: A dictionary where keys are concept strings and values are probabilities (floats) representing the normalized counts of each concept for the given word.
    - `features`: A `defaultdict` initialized to store the cumulative probability distribution over features.

    3. **Function Steps**:
    - Checks if the input word is present in `self.word_concept_counts`. If not, returns an empty dictionary `{}`.
    - Computes `total`, which is the sum of counts for all concepts associated with the given word.
    - Constructs a dictionary `probs` where each key-value pair represents a concept and its probability (normalized count) derived from `word_concept_counts[word]`.
    - Initializes an empty `defaultdict` named `features` to store feature probabilities.
    - Iterates over each concept-probability pair in `probs`. For each concept:
        - Computes the total count of features associated with that concept, stored in `concept_total`.
        - Skips processing if `concept_total == 0`.
        - Iterates over each feature-count pair within `self.concept_features[c]` (where `c` is a concept):
        - Updates `features[f]` by adding the product of `p` and `(cnt / concept_total)`, where `f` is a feature, `p` is the probability of the current concept, and `cnt` is the count of the feature within that concept.
    - Converts `defaultdict` `features` to an ordinary dictionary before returning it.

    4. **Output**:
    - Returns a dictionary where keys are features (strings) and values are their cumulative probabilities (floats) derived from marginalizing over concepts associated with the input word.

    This function ensures formal, deterministic computation of feature distributions by strictly following the provided steps without any uncertainties or approximations.

---

6) ### entropy_word
    The function `entropy_word` computes the entropy of the concept distribution for a given word. Here is the precise description of its behavior:

    1. **Input Parameters**:
    - `word`: A string representing the input word.

    2. **Variables Initialized and Used**:
    - `counts`: A list containing the counts of each concept associated with the given word.
    - `total`: An integer or float that stores the total count of concepts associated with the given word.
    - `probs`: A list where each element is a probability (float) representing the normalized count of a concept for the given word.
    - `entropy`: The computed entropy value, which is a float.

    3. **Function Steps**:
    - Checks if the input word is present in `self.word_concept_counts`. If not, returns 0.0.
    - Constructs a list `counts` containing the counts of all concepts associated with the given word.
    - Computes `total`, which is the sum of these counts.
    - Normalizes the counts to obtain probabilities by dividing each count by `total` and stores them in a list `probs`.
    - Initializes `entropy` as 0.0.
    - Iterates over each probability `p` in `probs`:
        - Computes `- p * math.log(p + 1e-12)`, where `math.log` is the natural logarithm, and adds this value to `entropy`.
    - Returns the computed entropy.

    4. **Output**:
    - Returns a float representing the entropy of the concept distribution for the given word.

    This function ensures formal, deterministic computation of the entropy by strictly following the provided steps without any uncertainties or approximations.

---

7) ### bayesian_surprise
    The function `bayesian_surprise` computes the Kullback-Leibler (KL) divergence between the prior and posterior distributions over concepts for a given word. Here is a detailed breakdown of what it does:

    - **Parameters:**
        - `word`: A string representing the input word.

    - **Variables Initialised and Used:**
        - `self.word_concept_counts`: A dictionary storing counts of concepts associated with each word.
        - `self.prev_posterior`: A dictionary storing prior probabilities over concepts for each word.
        - `total`: The total count of all concept occurrences for the given word. It is computed as the sum of values in `self.word_concept_counts[word]`.
        - `posterior`: A dictionary that normalises the counts in `self.word_concept_counts[word]` to probabilities by dividing each count by `total`.

    - **Functionality:**
    1. The function first checks if the word is present in both `self.word_concept_counts` and `self.prev_posterior`. If not, it returns 0.0.
    2. It extracts the prior distribution over concepts for the given word from `self.prev_posterior[word]`.
    3. It calculates the total number of concept occurrences associated with the input word by summing up all values in `self.word_concept_counts[word]` and assigns this value to `total`.
    4. It then normalises these counts into a probability distribution, storing the result in `posterior`. Each key-value pair in `posterior` represents a concept and its corresponding posterior probability.
    5. The KL divergence is computed by iterating over each concept-cum-probability pair in `posterior`. For each concept:
        - It retrieves the prior probability from `prior.get(c, 1e-10)`, using a small epsilon value (1e-10) to avoid division by zero.
        - It calculates the contribution of that concept to the KL divergence and accumulates it in `kl`.
    6. Finally, it clips the computed KL divergence between 0 and 10 to prevent numerical instability or "explosion".

    The function returns the clipped KL divergence value as a float.

---

8) ### save
    The function `save` serializes and saves the state of an object to a specified file path using pickle. Here is a precise description:

    - **Parameters:**
        - `path`: A string representing the target file path where the serialized data will be saved.

    - **Functionality:**
    1. It constructs a temporary file path by appending ".tmp" to the provided `path`.
    2. Opens the temporary file in write-binary mode.
    3. Dumps a dictionary containing several instance variables (`alpha`, `beta`, `concept_counts`, `word_concept_counts`, `concept_features`, `feature_vocab`, and `prev_posterior`) into the opened file using pickle.
    4. Closes the file after dumping the data.
    5. Renames the temporary file to the specified target path using `os.replace` to ensure atomicity, preventing partial writes.

    The function does not return any value.

---

9) ### register_definaition_event
    The `register_definition_event` function performs the following actions:

    - **Input Parameters:**
        - `word`: A string representing a word that has received an explicit definition.

    - **Initializations and Usage of Variables:**
        - `self.word_concept_counts`: Dictionary where keys are words and values are dictionaries containing counts of different concepts for those words.
        - `counts`: Local variable initialized to the value of `self.word_concept_counts[word]`, representing concept counts associated with the given word.
        - `dominant`: Variable set to the key in `counts` corresponding to the highest count, if `counts` is a non-empty dictionary.
        - `self.stabilized_words`: Set containing words that have been marked as stabilized. If it does not exist, it is initialized as an empty set.
        - `self.definition_events`: Dictionary tracking definition events for each word. If it does not exist, it is initialized as an empty dictionary.

    - **Actions Performed:**
    1. The function checks if the provided `word` exists in `self.word_concept_counts`. If not, the function returns immediately without performing any further actions.
    2. If `counts` (i.e., concept counts for the word) is a non-empty dictionary, the dominant concept is determined by finding the key with the highest count value and incrementing this count by 10 to reinforce the dominant concept.
    3. The provided `word` is added to `self.stabilized_words`, marking it as stabilized if it was not already present in the set.
    4. A new entry is created or updated in `self.definition_events[word]`. This entry contains:
        - `"timestamp"`: Current Unix timestamp, obtained using `time.time()`.
        - `"stability"`: Value of 1.0 indicating full stability after receiving a definition event.
        - `"type"`: String "explicit_definition" to indicate the type of stabilization event.

    In summary, this function updates concept counts for the provided word by reinforcing its dominant concept, marks it as stabilized, and logs an explicit definition event with timestamp and type.

---

10) ### form_meta_concepts
    The function `form_meta_concepts(self, min_concepts_per_meta=3)` clusters concepts into meta-concepts based on their feature distributions using a simple Chinese Restaurant Process (CRP). The function operates as follows:

    1. **Initialization and Input Validation**:
    - It initializes the variable `concept_ids` with all keys from `self.concept_features`.
    - If the number of concept IDs is less than `min_concepts_per_meta`, the function returns immediately without performing any clustering.

    2. **Feature Vector Normalization**:
    - For each concept ID, it calculates a feature vector by normalizing the counts in `self.concept_features[c]` over their sum.
    - If the sum of features for a concept is zero, that concept is skipped.
    - These normalized vectors are stored in `concept_vectors`.

    3. **Input Validation After Normalization**:
    - If the number of concepts with valid feature vectors (`len(concept_vectors)`) is less than `min_concepts_per_meta`, the function returns immediately.

    4. **Chinese Restaurant Process (CRP)**:
    - Initializes counters for meta-concepts (`meta_counts`), a dictionary to map each concept to its assigned meta-concept (`concept_to_meta`).
    - For each concept in `concept_vectors`:
        1. Computes probabilities for existing meta-concepts based on their average feature similarity with the current concept.
            - Retrieves members of each meta-concept and calculates the average feature vector of these members.
            - Computes cosine similarity between the concept's normalized feature vector (`vec`) and the average feature vector of the meta-concept.
            - Multiplies this similarity by the count (popularity) of concepts in the meta-concept to get a weighted probability.
        2. Adds a new meta-concept probability based on `self.meta_alpha`.
    - Converts probabilities into an array, normalizes them, and uses `np.random.choice` with these probabilities to select a meta-concept for the current concept.
    - Updates the counts of the chosen meta-concept in `meta_counts`, and maps the current concept to this meta-concept.

    5. **Output**:
    - Stores the final meta-concepts' counts in `self.meta_concept_counts`.
    - Maps each base concept to its assigned meta-concept in `self.concept_to_meta`.
    - Prints a summary statement indicating how many meta-concepts were formed and from how many base concepts.

    The function initializes and uses several variables:
    - `concept_ids`: A list of all concept IDs.
    - `total`: The sum of feature counts for each concept, used to normalize the feature vectors.
    - `vec`: Normalized feature vector for each concept.
    - `concept_vectors`: A dictionary mapping each concept ID to its normalized feature vector.
    - `meta_counts`: A counter tracking the number of concepts in each meta-concept.
    - `concept_to_meta`: A dictionary mapping each base concept to its assigned meta-concept.
    - `members`, `avg_vec`, `sim`, `probs`, and `metas`: Intermediate variables used during the CRP process.

    The function does not modify or use any external variables beyond those provided as input (`min_concepts_per_meta`) or defined within itself.

---

11) ### load
    The function `load(cls, path: str, config_manager: Optional['ConfigManager'] = None)` is a class method that loads a previously saved model from a specified file. The function operates as follows:

    1. **File Opening and Data Loading**:
    - Opens the file at the given `path` in binary read mode.
    - Uses `pickle.load(f)` to deserialize the data stored in the file, storing it in the variable `data`.

    2. **Model Instantiation**:
    - Creates an instance of the class using the constructor (`cls`) and initializes it with the parameters `alpha`, `beta`, and optionally `config_manager` from the deserialized `data`.
    
    3. **Data Assignment**:
    - Assigns the loaded data to various attributes of the model instance:
        - `model.concept_counts`: The concept counts from the serialized data.
        - `model.word_concept_counts`: The word-concept counts from the serialized data.
        - `model.concept_features`: The concept features from the serialized data.
        - `model.feature_vocab`: The feature vocabulary from the serialized data.
        - `model.prev_posterior`: The previous posterior probability distribution from the serialized data.

    4. **Return**:
    - Returns the fully initialized model instance.

    The function initializes and uses several variables:
    - `data`: A dictionary containing all the deserialized data loaded from the file.
    - `f`: File object opened in binary read mode.

    The function does not modify or use any external variables beyond those provided as input (`path`, `config_manager`) or defined within itself.
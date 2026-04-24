## heart.py

Integrates all components: HDP, GP uncertainty, Bayesian surprise,
intrinsic reward, meta-learner, reasoning, AND now factor-based
state snapshots with interactive human labeling.

---

### functions 

1) ### init
    The provided function `__init__(self, load_state: bool = True, config_manager: Optional[ConfigManager] = None)` is the constructor for an unspecified class. It initializes several variables and performs various tasks as detailed below:

    1. **Configuration Manager Initialization**:
    - If `config_manager` is not provided (`None`), a new instance of `ConfigManager` is created.
    - Otherwise, the provided `config_manager` is used.

    2. **Real Engines Initialization**:
    - Initializes three engines: `ci_engine`, `phon_engine`, and `diag_engine`.

    3. **Probabilistic Concept Model**:
    - Initializes `concept_model` with a `HierarchicalConceptModel`, using the configuration manager.

    4. **Gaussian Process Uncertainty Model**:
    - Initializes `gp_model` as an instance of `GaussianProcessUncertainty`, using the configuration manager.

    5. **Curiosity Meta-Learner and Intrinsic Reward Tracker**:
    - Initializes `meta_learner` with a `MetaCuriosityLearner`.
    - Initializes `reward_tracker` with an `IntrinsicRewardTracker`.

    6. **Reasoning Engine**:
    - Imports `ReasoningEngine` from `components.resoning` and initializes `reasoner`.
    - Sets `_last_state_learning_time` to 0.

    7. **Language & Consciousness Modules**:
    - Initializes `semantic_learner`, `syntax_learner`, and `action_generator`.
    - Initializes `consciousness` with a `ParameterRevision`.
    - Initializes `episodic_memory`.

    8. **Parallel Reasoning Engine**:
    - Imports `ParallelReasoner` from `components.resoning` and initializes `parallel_reasoner`.

    9. **Adaptive Control Space**:
    - Initializes `control_space` as an instance of `AdaptiveControlSpace`.
    - Adds a listener `_on_new_control_factor` for new control factors.

    10. **Factor Engine & State Manager**:
        - Initializes `factor_engine` with the adaptive control space and configuration manager.
        - Sets `state_dir` to 'states' or retrieves it from the config manager.
        - Initializes `state_manager`.

    11. **Identity Persistence Module**:
        - Retrieves `emb_dim` from the configuration manager, defaulting to 64 if not present.
        - Initializes `identity_manager` with `emb_dim`.

    12. **Persistent Latent State Concept**:
        - Initializes `current_state_labels`, an empty list for state labels.
        - Sets `current_state_embedding` to `None`.

    13. **Memory of Past Observations**:
        - Initializes `word_history` as a dictionary mapping words to lists of dictionaries.

    14. **Interaction History**:
        - Initializes `interaction_history` as an empty list.
        - Sets `last_interaction_time` to the current time in seconds.
        - Initializes `last_asked_word` to `None`.

    15. **Self-Reconfiguration Counter**:
        - Sets `interaction_counter` to 0.
        - Retrieves or sets `reconfigure_interval` from the configuration manager, defaulting to 20.

    16. **State Persistence and Feedback Tracking**:
        - Initializes several attributes for feedback tracking: `expecting_feedback`, `feedback_attempt_word`, `feedback_attempt_response`, `feedback_timestamp`.
    
    17. **Behavioral Rule Confirmation Tracking**:
        - Sets `expecting_rule_confirmation` to `False`.

    18. **Response Feedback Request Tracking**:
        - Sets `expecting_response_feedback` and initializes `pending_feedback_context`.
        - Sets `feedback_ask_probability` to 0.3.

    19. **Last Factor Dictionary and Dynamic Curiosity Factors**:
        - Initializes `_last_factor_dict` to `None`.
        - Adds a dynamic curiosity factor for state relevance if not already present in the meta-learner's factors, setting initial mu to 0.0 and using `_compute_state_relevance`.

    20. **Advanced Idle-Time Capabilities**:
        - Initializes various attributes related to idle-time capabilities such as world model (`world_model`), rehearsal buffer (`rehearsal_buffer`), policy model (`policy_model`), user model (`user_model`), research queue (`research_queue`), identity history (`identity_history`), episodic embeddings (`episodic_embeddings`), state manifold (`state_manifold`).

    21. **Structural Entropy Gate**:
        - Initializes `token_profile`, a dictionary for token behavioral profiles.
        - Sets initial stats for the token profile statistics: count mean, count std, context set mean, context set std, surprise mean, surprise std, drift mean, drift std.

    22. **Procedure Learning Attributes**:
        - Initializes several attributes related to procedure learning: `expecting_procedure_steps`, `pending_problem_expr`, `recorded_steps`, `pending_teaching_request`, and `pending_teaching_expr`.

    23. **Chat Mode and Behavioral Rule Learner**:
        - Sets `chat_mode_active` to `False`.
        - Initializes `chat_history` as an empty list.
        - Imports and initializes `BehavioralRuleLearner`.

    24. **Conscience Manager, Code Generation & Guilt Tracking**:
        - Initializes `conscience_manager`, `code_generator`, and `shadow_executor`.
        - Initializes or reassigns `guilt_tracker` if it is not present or set to `None`.
        - Ensures the conscience manager uses the same guilt tracker.
        - Adds "conscience_active" to base control factors if not already present.

    25. **Identity Aliases and Logic Inheritance Manager**:
        - Initializes `identity_aliases` as an empty set for storing learned aliases for self-reference.
        - Imports and initializes `LogicInheritanceManager`.

    26. **Language Model Initialization**:
        - Creates a default language model (`language_model`) using the configuration manager.
        - Sets up various configurations related to the language model.

    27. **Cognition Watchdog**:
        - Initializes `_cycle_counter` to 0 and updates dynamic limits based on system memory availability.

    28. **State Loading**:
    - Calls `_load_state()` if `load_state` is `True`.

    29. **Structured World Model and Curiosity Tracker Initialization**:
        - Initializes `structured_world_model` using the knowledge graph from `reasoner.knowledge_graph` and state manager.
        - Imports and initializes `CuriosityTracker`.

    In summary, this function initializes a complex system involving multiple components and attributes related to reasoning, language learning, consciousness management, adaptive control, feedback mechanisms, and advanced idle-time capabilities.

---

2) ### _update_dynamic_limits
    The function `_update_dynamic_limits` adapts internal limits to the available system RAM. It performs the following specific actions:

    1. Initializes `mem`, an object containing memory statistics obtained from the `psutil.virtual_memory()` call.
    2. Calculates the available memory in gigabytes (`avail_gb`) by dividing `mem.available` (which is in bytes) by $\{1024^3 \}$.
    3. Computes a scale factor `scale` using the formula:
    $$\{
    \text{scale} = \max(0.5, \min(8.0, \frac{\text{avail\_gb}}{4.0}))
    \} $$
    This ensures that `scale` is between 0.5 and 8.0, normalized around a baseline of ~4GB.
    4. Updates the internal variables:
    - `_max_cycles_per_session`: set to $\{100000 \times \text{scale}\} $ (rounded down to an integer).
    - `_max_hypotheses`: set to $\{5000 \times \text{scale}\} $ (rounded down to an integer).
    - `_max_logic_rules`: set to $\{2000 \times \text{scale}\} $(rounded down to an integer).

    If any exception occurs during the execution of these steps, it catches the exception and sets the internal variables to their fallback default values:
    - `_max_cycles_per_session` is set to 100000.
    - `_max_hypotheses` is set to 5000.
    - `_max_logic_rules` is set to 2000.

---

3) ### _metabolic_maintenance
    The function `_metabolic_maintenance` performs the following operations:

    1. It initializes `now`, which is set to the current time in seconds since the epoch using `time.time()`.

    2. The function iterates over all behavioral rules stored in `self.behavioral_learner.rules.values()`. For each rule, it calls `_apply_metabolic_to_item(rule, now)`.

    3. It iterates over all conscience beliefs stored in `self.conscience_manager.beliefs.values()`. For each belief, it calls `_apply_metabolic_to_item(belief, now)`.

    4. The function loops through each hypothesis (`hyp`) within the list `self.reasoner.hypotheses` and applies the metabolic cost by calling `_apply_metabolic_to_dict_item(hyp, now)`.

    5. It iterates over dynamic curiosity factors stored in `self.meta_learner.factor_metadata.items()`. For each factor metadata item (referred to as `meta`), it calls `_apply_metabolic_to_dict_item(meta, now, key_prefix=("factor", factor_name))`.

    6. The function loops through states within `self.state_manager.state_metadata.items()`. For each state metadata item (`meta`), it applies the metabolic cost by calling `_apply_metabolic_to_dict_item(meta, now, key_prefix=("state", label))`.

    7. It iterates over confirmed logic rules stored in `self.logic_inheritance.confirmed_logic.items()`, applying the metabolic cost to each rule via a call to `_apply_metabolic_to_dict_item(rule, now, key_prefix=("logic", logic_id))`. 

    8. For each pending doubt within `self.logic_inheritance.pending_doubts`, it retrieves the associated rule and applies the metabolic cost by calling `_apply_metabolic_to_dict_item(rule, now, key_prefix=("doubt", doubt['doubt_id']))`.

    The function does not initialize any variables other than `now` which is set to the current time. It uses `self.behavioral_learner.rules`, `self.conscience_manager.beliefs`, `self.reasoner.hypotheses`, `self.meta_learner.factor_metadata`, `self.state_manager.state_metadata`, and `self.logic_inheritance.confirmed_logic` as inputs, along with iterating over pending doubts from `self.logic_inheritance.pending_doubts`.

    The function `_apply_metabolic_to_item` or `_apply_metabolic_to_dict_item` is called for each item in the respective collections. The specific values of these items (rules, beliefs, hypotheses, factors, states, and logic rules) are not specified within this function; they are managed elsewhere in the codebase.

    The exact nature of `self._apply_metabolic_to_item(item, now)` or `self._apply_metabolic_to_dict_item(dict_item, now, key_prefix=(prefix1, prefix2))` is not defined here but assumed to apply a metabolic cost (or phase transition) based on current time (`now`) and the type of item being processed.

---

4) ### _apply_metabolic_to_item
    The function `_apply_metabolic_to_item` performs the following operations:

    1. It initializes `now`, which is passed as an argument and represents the current time in seconds since the epoch.
    2. It checks if the item's phase is 'deleted'. If it is, the function returns without further processing.

    3. It calculates the elapsed time `dt`:
    - If `item.last_used` exists, `dt` is calculated as `now - item.last_used`.
    - Otherwise, `dt` is calculated as `now - item.created`.

    4. The decay factor for recent usage is computed using the formula `decay_factor = 0.99 ** dt`. This exponentially decays the recent usage.

    5. The recent usage of the item (`item.recent_usage`) is updated by multiplying it with the decay factor: `item.recent_usage *= decay_factor`.

    6. It computes the effective gain (`effective_gain`) using the formula:
    $$\{
    \text{effective\_gain} = \frac{\text{item.predictive\_gain} \times \text{item.recent\_usage}}{1 + dt}
    \} $$

    7. The maintenance cost of the item (`cost`) is set to `item.energy_cost`.

    8. If the effective gain is less than the cost, it moves the item to the next phase:
    - If the current phase is 'active', it transitions to 'weak' and prints: "   [Metabolic] {item} moved to weak".
    - If the current phase is 'weak', it transitions to 'dormant' and prints: "   [Metabolic] {item} moved to dormant".
    - If the current phase is 'dormant', it transitions to 'archived' and prints: "   [Metabolic] {item} archived".
    - If the current phase is 'archived', it transitions to 'deleted' and prints: "   [Metabolic] {item} deleted".

    The function initializes no variables other than `dt` and uses the following attributes of `item`: `phase`, `last_used`, `created`, `recent_usage`, `predictive_gain`, and `energy_cost`.

    Specific values are not provided for these attributes, as they depend on the specific instance of the item. However, the function clearly defines how it processes these attributes based on their current values.

    The function does not explicitly remove items from their container when transitioning to 'deleted'. Instead, it prints a message indicating that the item has been deleted and leaves the actual removal to be handled separately.

---

5) ### _apply_metabolic_to_dict_item
    The function `_apply_metabolic_to_dict_item` performs the following operations:

    1. It initializes `now`, which is passed as an argument and represents the current time in seconds since the epoch.
    2. If the dictionary item `d` contains a key 'phase' with value 'deleted', the function returns without further processing.

    3. The variable `last_used` is initialized to the value of `d.get('last_used')`. If this key does not exist, it remains `None`.
    4. The variable `created` is initialized to the value of `d.get('created')`, with a fallback value of `now - 86400` (representing one day ago if 'created' is not specified).
    5. It calculates the elapsed time `dt` as `now - (last_used if last_used else created)`.
    6. The variable `recent_usage` is initialized to the value of `d.get('recent_usage', 0.0)`. If this key does not exist, it defaults to `0.0`.
    7. It updates `recent_usage` by multiplying it with the decay factor `(0.99 ** dt)` and stores the result back in `d['recent_usage']`.

    8. The variable `predictive_gain` is initialized to the value of `d.get('predictive_gain', 0.0)`. If this key does not exist, it defaults to `0.0`.
    9. The variable `cost` is initialized to the value of `d.get('energy_cost', 0.5)`. If this key does not exist, it defaults to `0.5`.

    10. It checks if the product of `predictive_gain * 2` exceeds `cost`. If true, it sets `d['phase']` to 'active' and returns.

    11. The effective gain (`effective_gain`) is computed as the maximum value between:
        - `predictive_gain * 0.5`
        - `(predictive_gain * recent_usage) / (1 + dt)`

    12. If `effective_gain` is less than `cost`, it transitions the phase of the item based on its current phase:
        - If the current phase is 'active', it sets `d['phase']` to 'weak'.
        - If the current phase is 'weak', it sets `d['phase']` to 'dormant'.
        - If the current phase is 'dormant', it sets `d['phase']` to 'archived'.
        - If the current phase is 'archived', it sets `d['phase']` to 'deleted'.

    The function initializes no variables other than `dt`, `last_used`, `created`, `recent_usage`, and `effective_gain`. It uses the following keys of dictionary `d`: `'phase'`, `'last_used'`, `'created'`, `'recent_usage'`, `'predictive_gain'`, and `'energy_cost'`.

    Specific values are provided for fallbacks:
    - If `'created'` is not specified, it defaults to `now - 86400`.
    - If `'recent_usage'` or `'predictive_gain'` are not specified, they default to `0.0`.
    - If `'energy_cost'` is not specified, it defaults to `0.5`.

    The function does not explicitly remove items from their container when transitioning to 'deleted'. Instead, it sets the phase to 'deleted' and leaves the actual removal to be handled separately in later cleanup operations.

---

6) ### _align_factor_vector, resolve_state_label, _memory_pressure_backoff
    The function `_align_factor_vector` performs the following operations:

    1. It takes an input array `vec` of type `np.ndarray`.
    2. It calls the function `align_vector(vec, len(self.meta_learner.factor_names))` to align the vector based on the length of `self.meta_learner.factor_names`.
    3. The result of `align_vector` is returned.

    The variables initialized and used are:
    - `vec`: An input array of type `np.ndarray`.

    The function does not initialize any additional variables within itself, but it uses the length of `self.meta_learner.factor_names` as an argument for the `align_vector` function.

    ---

    The function `resolve_state_label` performs the following operations:

    1. It takes a label as an input.
    2. If the label is `None`, it returns `None`.
    3. Otherwise, it calls `self.state_manager.resolve_label(label)` and returns the result.

    The variables initialized and used are:
    - `label`: An input parameter representing a state label.

    No additional variables are initialized within this function.

    ---

    The function `_memory_pressure_backoff` performs the following operations:

    1. It attempts to retrieve system memory information using `psutil.virtual_memory()`.
    2. If less than 12% of the total available memory is free (`mem.available / mem.total < 0.12`), it:
    - Calls `time.sleep(0.15)` to pause execution for 0.15 seconds.
    - Returns `True`.

    3. If any exception occurs during the retrieval of system memory information, the function catches the exception and continues without performing further actions.
    4. If less than 12% of available memory is not free, or if an exception occurs, it returns `False`.

    The variables initialized are:
    - `mem`: An object representing system memory statistics from `psutil.virtual_memory()`.
    - `now`: The current time in seconds since the epoch (implicitly used by `time.sleep`).

    No additional variables are explicitly initialized within this function.

    Specific values and fallbacks provided include:
    - Memory threshold: `<0.12` (less than 12% free memory).
    - Sleep duration: `0.15` seconds.

7) ### _load_state
    The function `_load_state` performs the following actions:

    1. Initializes and uses the following variables:
    - `concept_path`: Set to `"STATE_DIR/concept_model.pkl"`
    - `meta_path`: Set to `"STATE_DIR/meta_learner.npz"`
    - `gp_path`: Set to `"STATE_DIR/gp_model.pkl"`
    - `reward_path`: Set to `"STATE_DIR/reward_tracker.pkl"`
    - `reasoner_path`: Set to `"STATE_DIR/reasoner.pkl"`
    - `history_path`: Set to `"STATE_DIR/history.pkl"`
    - `semantic_path`: Set to `"STATE_DIR/semantic_learner.pkl"`
    - `syntax_path`: Set to `"STATE_DIR/syntax_learner.pkl"`
    - `conscience_path`: Set to `"STATE_DIR/conscience_manager.pkl"`
    - `world_model_path`: Set to `"STATE_DIR/world_model.pkl"`
    - `policy_path`: Set to `"STATE_DIR/policy_model.pkl"`
    - `user_model_path`: Set to `"STATE_DIR/user_model.pkl"`
    - `research_queue_path`: Set to `"STATE_DIR/research_queue.pkl"`
    - `identity_history_path`: Set to `"STATE_DIR/identity_history.pkl"`
    - `state_manifold_path`: Set to `"STATE_DIR/state_manifold.pkl"`
    - `aliases_path`: Set to `"STATE_DIR/identity_aliases.json"`
    - `guilt_path`: Set to `"STATE_DIR/guilt_tracker.pkl"`
    - `alias_path`: Set to `"STATE_DIR/state_aliases.json"`
    - `lm_path`: Set to `"STATE_DIR/language_model.pkl"`
    - `current_state_path`: Set to `"STATE_DIR/current_state.json"`
    - `token_profile_path`: Set to `"STATE_DIR/token_profile.pkl"`

    2. Loads the conscience manager from file if it exists, and sets its configuration manager.
    3. Loads the logic inheritance model from file.
    4. If `concept_path` exists:
    - Initializes `self.concept_model` by loading a HierarchicalConceptModel from `concept_path`.
    5. If `meta_path` exists:
    - Loads `self.meta_learner` from `meta_path`.
    6. If `gp_path` exists:
    - Opens the file and loads `self.gp_model` using pickle.
    - Sets `self.gp_model.config_manager` to `self.config_manager`.
    7. If `reward_path` exists:
    - Loads `self.reward_tracker` from `reward_path`.
    - Sets `self.reward_tracker.config_manager` to `self.config_manager`.
    8. If `reasoner_path` exists:
    - Loads `self.reasoner` from `reasoner_path`.
    - Sets `self.reasoner.config_manager` to `self.config_manager`.
    9. If `history_path` exists:
    - Opens the file and loads `self.word_history` using pickle.
    10. If `semantic_path` exists:
        - Loads `self.semantic_learner` from `semantic_path`.
        - Sets `self.semantic_learner.config_manager` to `self.config_manager`.
    11. If `syntax_path` exists:
        - Loads `self.syntax_learner` from `syntax_path`.
        - Sets `self.syntax_learner.config_manager` to `self.config_manager`.
    12. If `world_model_path` exists:
        - Opens the file and loads `self.world_model` using pickle.
    13. If `policy_path` exists:
        - Opens the file and loads `self.policy_model` using pickle.
    14. If `user_model_path` exists:
        - Opens the file and loads `self.user_model` using pickle.
    15. If `research_queue_path` exists:
        - Opens the file and loads `self.research_queue` using pickle.
    16. If `identity_history_path` exists:
        - Opens the file and loads `self.identity_history` using pickle.
    17. If `state_manifold_path` exists:
        - Opens the file and loads `self.state_manifold` using pickle.
    18. Loads identity aliases from `aliases_path`.
    19. Sets `self.state_manager.aliases` by loading a JSON object from `alias_path`.
    20. Loads `self.guilt_tracker` from `guilt_path`.
    21. Ensures the conscience manager shares the same guilt tracker instance.
    22. Loads generated code feedback.
    23. If `lm_path` exists:
        - Loads `self.language_model` from `lm_path`, prints a message with vocabulary size and steps, otherwise builds initial vocabulary from existing word embeddings in `semantic_learner.word_embeddings`.
    24. If `current_state_path` exists:
        - Opens the file and loads `self.current_state_labels` and `self.current_state_embedding` based on data inside.
    25. Loads token profile and related stats from `token_profile_path`, otherwise sets default values for these variables.

    This function ensures that all necessary models, trackers, and profiles are loaded into memory from disk, initializing them with appropriate configurations and defaults where files do not exist or loading fails.

---

8) ### _save_state
    The function `_save_state` performs the following actions:

    1. **Initialization and Time Check**:
    - It initializes `now` with the current time using `time.time()`.
    - It checks if the difference between `now` and `_last_save_time`, which defaults to 0, is less than `_min_save_interval`, which defaults to 20 seconds.
    - If the condition is met (i.e., less than 20 seconds since last save), it returns without performing any further actions.

    2. **Saving Object States**:
    - Sets `_last_save_time` to `now`.
    - Saves the `concept_model` object using `save` method and writes to `'concept_model.pkl'` file in `STATE_DIR`.
    - Saves the `meta_learner` object using `save` method and writes to `'meta_learner.npz'` file in `STATE_DIR`.
    - Dumps `gp_model` object into a pickle file named `'gp_model.pkl'` in `STATE_DIR`.
    - Saves the `reward_tracker` object using `save` method and writes to `'reward_tracker.pkl'` file in `STATE_DIR`.
    - Saves the `reasoner` object using `save` method and writes to `'reasoner.pkl'` file in `STATE_DIR`.
    - Dumps the `word_history` list into a pickle file named `'history.pkl'` in `STATE_DIR`.
    - Saves the `semantic_learner` object using `save` method and writes to `'semantic_learner.pkl'` file in `STATE_DIR`.
    - Saves the `syntax_learner` object using `save` method and writes to `'syntax_learner.pkl'` file in `STATE_DIR`.
    - Calls the `save` method on `identity_manager`.

    3. **Saving Behavioral Rule Learner**:
    - Calls the `save` method on `behavioral_learner`.

    4. **Saving Identity Aliases**:
    - Writes a JSON list of `identity_aliases` to `'identity_aliases.json'` in `STATE_DIR`.
    
    5. **Saving New Idle-Task Related Data**:
    - Dumps `world_model`, `policy_model`, and `user_model` objects into pickle files named `'world_model.pkl'`, `'policy_model.pkl'`, and `'user_model.pkl'` respectively, all in `STATE_DIR`.
    - Dumps `research_queue` object into a pickle file named `'research_queue.pkl'` in `STATE_DIR`.
    - Dumps `identity_history` list into a pickle file named `'identity_history.pkl'` in `STATE_DIR`.
    - Dumps `state_manifold` object into a pickle file named `'state_manifold.pkl'` in `STATE_DIR`.

    6. **Saving Additional Data**:
    - Calls the `save` method on `conscience_manager`, `logic_inheritance`, and `guilt_tracker`.
    - Saves the `generated_code_feedback` object.
    - Dumps the `language_model` object into a pickle file named `'language_model.pkl'` in `STATE_DIR`.

    7. **Saving Current State Concept**:
    - Creates a dictionary with keys `'labels'` and `'embedding'`, where `labels` is set to `current_state_labels` and `embedding` is converted to list if it exists, else sets `None`.
    - Writes the current state concept data into a JSON file named `'current_state.json'` in `STATE_DIR`.

    8. **Saving Token Profiles**:
    - Creates a dictionary containing `token_profile`, `token_profile_stats`, and `_cold_start_threshold`.
    - Dumps this dictionary into a pickle file named `'token_profile.pkl'` in `STATE_DIR`.

    9. **Saving State Aliases**:
    - Writes the aliases from `state_manager` to a JSON file named `'state_aliases.json'` in `STATE_DIR`.

    10. **Exception Handling**:
        - If any error occurs during saving, it prints an error message indicating that state may be incomplete and logs the traceback for debugging purposes.

    The function uses several variables including `_last_save_time`, `_min_save_interval`, `concept_model`, `meta_learner`, `gp_model`, `reward_tracker`, `reasoner`, `word_history`, `semantic_learner`, `syntax_learner`, `identity_manager`, `behavioral_learner`, `identity_aliases`, `world_model`, `policy_model`, `user_model`, `research_queue`, `identity_history`, `state_manifold`, `conscience_manager`, `logic_inheritance`, `guilt_tracker`, `generated_code_feedback`, `language_model`, `current_state_labels`, `current_state_embedding`, `token_profile`, `token_profile_stats`, `_cold_start_threshold`, and `state_manager`.

---

9) ### reload_config
    The function `reload_config` performs the following specific actions:

    1. Calls the method `reload()` on the object `self.config_manager`, which presumably reloads a configuration file.
    2. Updates the parameters of each component listed below using their respective `update_parameters_from_config()` methods:
        - `self.concept_model`
        - `self.gp_model`
        - `self.meta_learner`
        - `self.reward_tracker`
        - `self.semantic_learner`
        - `self.syntax_learner`
        - `self.consciousness`
        - `self.action_generator`
        - `self.reasoner`
        - `self.factor_engine`
        - `self.state_manager`
        - `self.episodic_memory`
    3. If the object `self` has an attribute `_idle_reasoner`, it updates its parameters using `update_parameters_from_config()`.
    4. Sets the instance variable `reconfigure_interval` to the value retrieved from the configuration manager by calling `get('engine.reconfigure_interval', 20)`. This will retrieve a specific value from the configuration file, defaulting to `20` if the specified key is not found.

    The function initializes and uses no variables explicitly. However, it updates several instance variables of the components listed above through their respective methods. The only new variable it sets directly is `reconfigure_interval`.

---

10) ### _maybe_reconfigure
    The function `_maybe_reconfigure` performs the following specific actions:

    1. Checks if the object `self` has an attribute `_last_factor_vector`.
    2. If `_last_factor_vector` exists, it assigns its value to the variable `factor_vector`.
    3. Calls the method `generate_modulated_config(factor_vector)` on the object `self.config_manager`, which presumably generates a new configuration based on the given factor vector.
    4. Calls the method `reload_config()` on the same object `self`, updating all component parameters as described in the previous function.
    5. Calls the method `evolve_parameters()` on the object `self.conscience_manager`, allowing it to adjust its own thresholds.
    6. Prints the message "🧬 [Emotional Reconfiguration] Configuration updated based on internal state."

    The function initializes and uses no variables explicitly but assigns a value to the variable `factor_vector` if `_last_factor_vector` exists.

    It updates several instance variables of the components through their respective methods, specifically:
    - The configuration managed by `self.config_manager`
    - Parameters of all sub-components through `reload_config()`

    No new variables are directly initialized or set within the function, except for `factor_vector`.

---

11) ### _extract_features
    The function `_extract_features` performs the following specific actions:

    1. Takes four parameters:
    - `token`: a string representing the input token.
    - `ci_report`: a dictionary containing confidence and stability information.
    - `phon_score`: a float representing a phonetic score.
    - `diag`: a dictionary containing diagnostic information.

    2. Initializes an empty dictionary `features` to store the extracted features.

    3. Populates the `features` dictionary with the following key-value pairs:
        - `'confidence'`: The value from `ci_report['confidence']`, defaulting to 0.5 if not present.
        - `'stability'`: The value from `ci_report['stability']`, defaulting to 0.5 if not present.
        - `'phon_score'`: The provided `phon_score`.
        - `'char_valid'`: The value from `ci_report['mean']`, defaulting to 0.5 if not present.
        - `'structure'`: The provided `phon_score`.

    4. Iterates over the list of negative items in `diag.get('negatives', [])`. For each negative item:
    - If 'character' is found in the lowercase version of the negative item, set `features['char_valid']` to 0.1.
    - If 'vowel' is found in the lowercase version of the negative item, set `features['structure']` to 0.2.

    5. Returns the populated `features` dictionary.

    The function initializes and uses no variables explicitly other than `features`. It updates the values within the `features` dictionary based on the provided input parameters and specific conditions.

---

12) ### _features_to_array
    The function `_features_to_array` performs the following specific actions:

    1. Takes one parameter:
    - `features`: a dictionary containing feature values.

    2. Initializes a list of keys: `['confidence', 'stability', 'phon_score', 'char_valid', 'structure']`.

    3. Constructs an array using a list comprehension where each element is obtained from the `features` dictionary based on the key, defaulting to 0.5 if the key is not present.

    4. Converts the constructed list into a NumPy array and returns it.

    The function initializes no variables explicitly other than the list of keys. It uses the input dictionary `features` to populate the array with specific values or defaults (0.5) for missing keys.

    Here is the precise description:

    1. The parameter `features` is a dictionary containing feature values.
    2. A fixed list of keys `['confidence', 'stability', 'phon_score', 'char_valid', 'structure']` is defined.
    3. For each key in the list, the function retrieves the corresponding value from the `features` dictionary using `.get(k, 0.5)`, which returns the value associated with key `k` or defaults to 0.5 if the key is not present.
    4. The retrieved values are converted into a NumPy array and returned.

    The function initializes no variables explicitly other than the list of keys. It uses the input dictionary `features` to populate the array with specific values or defaults (0.5) for missing keys.

---

13) ### _apply_knowledge_graph_curiosity
    The function `_apply_knowledge_graph_curiosity` performs the following specific actions:

    1. Takes two parameters:
    - `token`: a string representing the input token.
    - `base_factors`: a dictionary containing initial curiosity factors.

    2. Calls the method `get_node_id(token, "word")` on the object `self.reasoner.knowledge_graph` to retrieve the node ID for the given token. If no node ID is found (`nid is None`), it returns the `base_factors` unchanged.

    3. Retrieves the neighbors of the node with ID `nid` using `get_neighbors(nid)` and calculates the degree (number of neighbors) as `len(neighbors)`.

    4. Normalizes the degree by dividing it by 50 and ensures the value does not exceed 1.0, setting `degree_factor = min(1.0, degree / 50.0)`.

    5. Updates the curiosity factors in `base_factors`:
    - For `'novelty'`: Sets to `min(1.0, base_factors.get('novelty', 0.5) + 0.2 * degree_factor)`.
    - For `'uncertainty'`: Sets to `min(1.0, base_factors.get('uncertainty', 0.5) + 0.15 * degree_factor)`.
    - For `'surprise'`: Sets to `min(1.0, base_factors.get('surprise', 0.5) + 0.1 * degree_factor)`.

    6. Optionally boosts the complexity factor based on the diversity of relation types:
    - Collects unique relation types from the neighbors into a set.
    - Calculates the relative diversity as `len(rel_types) / 5.0` (maximum value is 1.0 if there are 5 different relation types).
    - Updates the `'complexity'` factor in `base_factors` to `min(1.0, base_factors.get('complexity', 0.5) + 0.1 * rel_diversity)`.

    7. Returns the updated `base_factors`.

    The function initializes no variables explicitly other than `nid`, `neighbors`, `degree`, and `rel_types`. It uses the input dictionary `base_factors` to update specific curiosity factors based on the degree of the node in the knowledge graph and the diversity of relation types.

---

14) ### _update_token_profile
    The function `_update_token_profile` updates the behavioral statistics for a given token based on provided parameters. It initializes and uses several variables as follows:

    - **Parameters:**
    - `token`: A string representing the token to update.
    - `context_tokens`: A list of strings containing context tokens associated with the token.
    - `bayes_surprise`: A float representing Bayesian surprise value for the token.
    - `info_gain`: A float representing information gain value for the token.
    - `embedding`: An optional numpy array representing the embedding vector of the token.

    - **Variables Initialised and Used:**
    - `self.token_profile[token]`: Initializes a dictionary if the token does not exist in `token_profile`, otherwise uses existing profile for the given token.
        - `count`: Integer count initialized to 0.
        - `context_set`: Set of context tokens, initialized as an empty set.
        - `avg_surprise`: Float representing average Bayesian surprise, initialized to 0.0.
        - `avg_info_gain`: Float representing average information gain, initialized to 0.0.
        - `embedding_drift`: Float representing the average embedding drift, initialized to 0.0.
        - `embedding_centroid`: Numpy array representing centroid of embeddings for the token, initially set to None.

    - **Specific Actions Performed:**
    - Increases the count of occurrences (`prof['count']`) by 1.
    - Adds context tokens to a capped set of up to 200 unique context tokens (`prof['context_set']`).
    - Updates moving averages for Bayesian surprise and information gain using an exponential smoothing factor `alpha = 0.1`.
    - If the token has an embedding:
        - Initializes or updates the centroid of embeddings.
        - Computes cosine similarity between current embedding and centroid, then calculates drift as 1 minus this similarity.
        - Updates the moving average of embedding drift with the computed drift value.
    - For global statistics in `self.token_profile_stats`:
        - Keeps track of running means and standard deviations for count, context set size, average surprise, and embedding drift using `_update_running_stat` function.

    Specific values used in calculations include:
    - Exponential smoothing factor: `alpha = 0.1`.
    - Context set cap limit: 200.
    - Cosine similarity normalization factors based on norms of vectors (`norm_cent`, `norm_emb`).

---

15) ### _normalize_stat
The function `_normalize_stat` performs sigmoid normalization on a given value based on its mean and standard deviation. It initializes and uses the following variables:

- **Parameters:**
  - `value`: A float representing the value to be normalized.
  - `mean`: A float representing the mean of the values.
  - `std`: A float representing the standard deviation of the values.

- **Specific Actions Performed:**
  - Computes a z-score (`z`) for the given value using the formula:
    $$ \{
    z = \frac{value - mean}{std + 1e-8}
    \} $$
    where `1e-8` is added to the standard deviation to avoid division by zero.
  - Applies the sigmoid function to the computed z-score to normalize the value:
    $$ \{
    \text{normalized\_value} = \frac{1}{1 + e^{-z}}
    \} $$

The function returns a normalized value between 0 and 1.

---

16) ### structural_score
    The function `structural_score` computes a structural score for a given token (where high scores indicate tokens similar to glue-like structures). It takes one parameter:

    - `token`: A string representing the input token.

    **Variables Initialization and Usage:**
    - `prof`: The profile of the provided token, retrieved from `self.token_profile`. If no profile is found (`not prof`), the function returns a neutral score of 0.5.
    - `stats`: Profile statistics for all tokens stored in `self.token_profile_stats`.
    - `freq`: Normalized frequency of the token's appearance.
    - `ctx`: Normalized context set length, representing how many unique contexts the token appears in.
    - `surp`: Normalized average surprise value, indicating how unexpected the token is given its context.
    - `drift`: Normalized embedding drift value, which measures the dissimilarity between the token’s current and initial embeddings.
    - `score`: A weighted sum of the normalized frequency (`freq`), context set length (`ctx`), average surprise inverse (`surp`), and embedding drift inverse (`drift`). The weights are 0.35 for `freq`, 0.35 for `ctx`, 0.15 for `surp`, and 0.15 for `drift`.
    - `unique_tokens`: The number of unique tokens in the token profile.
    - `_cold_start_threshold`: A threshold value from `self._cold_start_threshold` that is used to determine if cold start conditions are met.

    **Function Behavior:**
    - If no profile exists for the given token, it returns a neutral score of 0.5.
    - Otherwise, it calculates four normalized statistics: frequency (`freq`), context set length (`ctx`), average surprise inverse (`surp`), and embedding drift inverse (`drift`). These values are calculated using `_normalize_stat`.
    - The structural score is then computed as the weighted sum of these normalized statistics.
    - If fewer than `self._cold_start_threshold` unique tokens exist in the token profile, and if the length of the input token is less than or equal to 2 characters, an additional boost of 0.2 is added to the score (with a cap at 1.0).
    - Finally, the computed score is clipped between 0 and 1, ensuring it falls within this range.

    The specific values for `self._cold_start_threshold` are not provided in the code snippet, but it is an integer used as a threshold condition for determining if cold start conditions apply.

---

17) ### _compute_curiosity_factors
    The function `_compute_curiosity_factors` computes all curiosity factors for a single token and returns a dynamic `CuriosityFactors` object. It takes the following parameters:

    - `token`: A string representing the input token.
    - `token_ci`: A dictionary containing confidence interval information for the token.
    - `token_phon`: A float value representing phonotactic anomaly of the token.
    - `diag`: A dictionary containing diagnostic information about the token.
    - `features_array`: A numpy array of features associated with the token.
    - `global_factor_dict`: An optional dictionary used by the meta-learner for dynamic factor computation.

    **Variables Initialization and Usage:**
    - `uncertainty`: The entropy value calculated from the concept model for the input token.
    - `learning_progress`: A float representing the progress in learning based on previous entropies stored in `self.word_history`.
    - `novelty`: Calculated as 1 minus the confidence of the token, defaulting to 0.5 if no confidence is provided.
    - `stability`: The stability value from `token_ci`, defaulting to 0.5 if not provided.
    - `complexity`: A float representing complexity based on length and stability.
    - `predicted`: A dictionary containing predicted features for the token according to the concept model.
    - `observed`: A dictionary containing observed features extracted from the token using `_extract_features`.
    - `surprise`: Calculated as a weighted sum of logarithmic differences between observed and predicted feature probabilities.
    - `phon_anomaly`: The phonotactic anomaly, calculated as 1 minus `token_phon`.
    - `neg_count`: Number of negative examples in diagnostic information (`diag`).
    - `diag_neg`: A float representing the normalized number of negatives in diagnostic information.
    - `bayes_surprise`: Bayesian surprise value from the concept model for the token.
    - `gp_uncertainty`: Gaussian process uncertainty calculated using `features_array`.
    - `factor_dict`: A dictionary containing all computed factors.

    **Function Behavior:**
    1. **Uncertainty Calculation**:
    - Computes the entropy (uncertainty) of the input token using `self.concept_model.entropy_word(token)`.

    2. **Learning Progress Calculation**:
    - If the token exists in `self.word_history` and has more than one entry, calculates learning progress as the difference between previous entropy and current entropy.
    
    3. **Novelty Calculation**:
    - Computes novelty as 1 minus the confidence value from `token_ci`, defaulting to 0.5 if no confidence is provided.

    4. **Complexity Calculation**:
    - Calculates complexity based on stability and token length, ensuring it falls within [0, 1] using `np.clip`.

    5. **Surprise Calculation**:
    - Predicts features for the token.
    - Extracts observed features from the token.
    - Computes surprise as a weighted sum of logarithmic differences between predicted and observed feature probabilities.

    6. **Phonotactic Anomaly Calculation**:
    - Calculates phonotactic anomaly as 1 minus `token_phon`.

    7. **Diagnostic Negatives Calculation**:
    - Determines the number of negative examples in diagnostic information.
    - Normalizes this count to a value between [0, 1].

    8. **Bayesian Surprise and GP Uncertainty Calculation**:
    - Computes Bayesian surprise using `self.concept_model.bayesian_surprise(token)`.
    - Calculates Gaussian process uncertainty using `self.gp_model.uncertainty(features_array)`.

    9. **Factor Dictionary Construction**:
    - Constructs a dictionary of factors including: `uncertainty`, `learning_progress`, `novelty`, `complexity`, `surprise`, `phonotactic_anomaly`, `diagnostic_negatives`, `bayesian_surprise`, and `gp_uncertainty`.

    10. **Knowledge Graph Curiosity Boost**:
        - Applies knowledge graph-based curiosity boost to the factor dictionary using `_apply_knowledge_graph_curiosity`.

    11. **Definition Adjustment**:
        - If a definition exists for the token in `self.reasoner.definitions`, reduces novelty, surprise, bayesian surprise, gp uncertainty, and learning progress by 0.1.

    12. **Dynamic Factor Computation**:
        - Computes dynamic factors using `self.meta_learner.compute_dynamic_factor` for each factor name specified in `self.meta_learner.dynamic_factor_names`.

    13. **CuriosityFactors Object Creation**:
        - Creates a `CuriosityFactors` object from the computed factor dictionary.

    14. **Top-Down Modulation Application**:
        - Applies top-down modulation to the factors using `_apply_top_down_modulation`.

    Finally, the function returns the `factors` object containing all computed curiosity factors.

    In case of an exception during any step of this process, the function catches it, prints the error and traceback, and returns neutral factors with a value of 0.5 for each factor name specified in `self.meta_learner.factor_names`.

---

18) ### _apply_top_down_modulation
    The function `_apply_top_down_modulation` modifies the `CuriosityFactors` object based on top-down signals such as the current state embedding, identity embedding, and conscience beliefs. Here are the specific actions it performs:

    1. **Initialization of Variables:**
    - The function takes two parameters:
        - `token`: A string representing a token (word).
        - `factors`: An instance of `CuriosityFactors`, which is an object containing curiosity-related factors.

    2. **Modulation by Current State Embedding and Identity Consistency:**
    - If `self.current_state_embedding` is not None, the function proceeds to compute similarity between the embedding of the token and the current state.
    - It retrieves the word embedding for the given `token` using `semantic_learner.word_embeddings.get(token)`.
    - If the token embedding exists (`token_emb is not None`), it aligns both `self.current_state_embedding` and `token_emb` to match their dimensions with `_align_vector_to_emb_dim`.
    - It calculates a similarity score between the aligned state embedding and the aligned token embedding using cosine similarity: \( \text{sim} = \frac{\mathbf{s}^T \mathbf{t}}{\|\mathbf{s}\| \|\mathbf{t}\| + 10^{-12}} \), where \(\mathbf{s}\) is `state_emb` and \(\mathbf{t}\) is `token_emb_aligned`.
    - It computes the identity consistency similarity using `identity_manager.get_stability_bonus(token_emb_aligned)` which returns a value `identity_sim`. 
    - If the token aligns well with the agent's identity (high `identity_sim`), it reduces novelty by multiplying the current novelty factor (`factors.factor_dict['novelty']`) by \(1.0 - 0.5 \times \text{identity\_sim}\).

    3. **Modulation by Conscience Beliefs:**
    - The function iterates over each active conscience belief in `self.conscience_manager.get_active_conscience()`.
    - For each belief, if the rule type is 'ethical' and its action effect is 'reduce_curiosity':
        - It checks if the token contains a specific keyword defined by the condition's keyword (`belief.condition.get('keyword')`).
        - If the condition matches, it reduces both `curiosity` and `novelty` factors in `factors.factor_dict`, setting them to 30% and 50%, respectively.

    4. **Return Statement:**
    - The function returns the modified `CuriosityFactors` object (`return factors`).

    In summary:
    - If a token aligns well with the current state, it does not directly boost curiosity but may indirectly affect novelty through identity consistency.
    - If any conscience belief's condition matches the token and its effect is to reduce curiosity, both curiosity and novelty are reduced.

---

19) ### _information_gain
    The function `_information_gain` calculates the information gain for a given token based on its entropy values from previous occurrences and a new entropy value. Here are the specific actions it performs:

    1. **Initialization of Variables:**
    - The function takes two parameters:
        - `token`: A string representing a token (word).
        - `new_entropy`: A float representing the new entropy value for the token.

    2. **Check Word History Conditions:**
    - If the `token` is not in `self.word_history` or if the length of the list associated with `token` in `self.word_history` is less than 2, the function returns `0.0`.

    3. **Retrieve Old Entropy Value:**
    - If the token has at least two entries in its history (`len(self.word_history[token]) >= 2`), it retrieves the old entropy value from the second-to-last entry of `self.word_history[token]`.
    - The old entropy is accessed using `old_entropy = self.word_history[token][-2].get('entropy', 1.0)`. If the 'entropy' key does not exist in the dictionary at the specified index, it defaults to `1.0`.

    4. **Calculate Information Gain:**
    - It calculates the information gain as the difference between the old entropy and the new entropy (`old_entropy - new_entropy`).
    - The function ensures that the information gain is non-negative by taking the maximum of 0 and the calculated difference, i.e., `max(0, old_entropy - new_entropy)`.

    5. **Return Statement:**
    - The function returns the calculated information gain as a float.

    In summary:
    - If the token does not have at least two entries in its history, the function returns `0.0`.
    - Otherwise, it calculates and returns the maximum of 0 or the difference between the old entropy value (from the second-to-last entry) and the new entropy value.

---

20) ### _handle_state_learning
    The function `_handle_state_learning` performs the following operations:

    1. **Input Validation**:
    - Checks if `factor_vector` is `None` or empty.
        ```python
        if factor_vector is None or len(factor_vector) == 0:
            print("   [State Learning] Empty factor vector – skipping.")
            return
        ```
    If the condition is true, it prints a message and returns immediately.

    2. **Time Interval Check**:
    - Checks if less than 30 seconds have passed since the last state learning.
        ```python
        if time.time() - self._last_state_learning_time < 30:
            return
        ```
    If less than 30 seconds have passed, it returns immediately.

    3. **Configuration and Thresholds**:
    - Retrieves `sim_threshold` from configuration with a default value of 0.97.
    - Calculates the number of states (`n_states`) as the maximum between 1 and the length of known states.
        ```python
        n_states = max(1, len(self.state_manager.states))
        ```
    - Computes `maturity` using a logarithmic function with bounds [0.2, 1.0].
        ```python
        maturity = np.clip(np.log1p(n_states) / 5.0, 0.2, 1.0)
        ```
    - Defines `novelty_dist_threshold` based on `maturity`.
        ```python
        novelty_dist_threshold = 0.35 * (1.2 - maturity)
        ```

    4. **Finding Similar States**:
    - Finds states above the similarity threshold using `state_manager.find_similar_states()`.
        ```python
        similar = self.state_manager.find_similar_states(factor_vector, threshold=sim_threshold)
        ```
    
    5. **Meta Novelty Gate Application**:
    - Iterates through each state found in `similar` to check if it meets the novelty distance threshold.
        ```python
        for label, sim in similar:
            stored_vec = self.state_manager.states.get(label)
            if stored_vec is not None:
                v1 = self.state_manager._normalize_vector(factor_vector)
                v2 = self.state_manager._normalize_vector(stored_vec)
                max_len = max(len(v1), len(v2))
                v1 = np.pad(v1, (0, max_len - len(v1)))
                v2 = np.pad(v2, (0, max_len - len(v2)))
                dist = np.linalg.norm(v1 - v2)
                if dist <= novelty_dist_threshold:
                    active_labels.append(label)
                    active_embeddings.append(self._align_vector_to_emb_dim(v2))
        ```

    6. **Active States Update**:
    - If there are any `active_labels`, it prints the matching states and updates the state embedding by averaging the embeddings.
        ```python
        if active_labels:
            print(f"\n🔍 [State Recognition] This state matches: {', '.join(active_labels)}.")
            combined = np.mean(active_embeddings, axis=0) if active_embeddings else None
            self._update_state_embedding(active_labels, combined)
            return
        ```

    7. **New State Creation**:
    - If no close match is found and the factor vector variance is below 0.05, it skips state creation.
        ```python
        vec_variance = np.var(factor_vector)
        if vec_variance < 0.05:
            print("   [State Learning] Factor vector too uniform – skipping state creation.")
            return
        ```
    - If no close match and variance is acceptable, it prints a message indicating a new internal state.
        ```python
        print("\n [State Learning] I encountered a new internal state.")
        print("    Factor vector: " + " ".join(f"{v:.2f}" for v in factor_vector))
        ```
    - Prompts the user to label the new state, defaulting to "unnamed_state" if no input is provided.
        ```python
        metadata['tokens'] = self._tokenize(metadata.get('input_text', ''))
        self.state_manager.save_state(label, factor_vector, metadata)
        print(f"    ✅ State saved as '{label}'.")
        ```

    8. **Additional Operations**:
    - Updates the last state learning time.
        ```python
        self._last_state_learning_time = time.time()
        ```
    - Reinforces identity and updates the state embedding for the new state.
        ```python
        self._reinforce_identity(factor_vector, reward_signal=0.2)
        self._update_state_embedding([label], self._align_vector_to_emb_dim(factor_vector))
        ```

    **Variables Initialised/Used:**
    - `sim_threshold`: Configuration value or default 0.97.
    - `n_states`: Maximum of 1 and the number of states in `self.state_manager.states`.
    - `maturity`: Computed using a logarithmic function with bounds [0.2, 1.0].
    - `novelty_dist_threshold`: Calculated as `0.35 * (1.2 - maturity)`.
    - `similar`: List of tuples containing state labels and similarity scores.
    - `active_labels`: List of active state labels meeting the novelty distance threshold.
    - `active_embeddings`: List of aligned embeddings for each active label.
    - `vec_variance`: Variance of the factor vector.
    - `label`: User-provided or default label for a new state.

    This function performs state recognition and learning by comparing the current factor vector with known states, updating active states if matches are found, and creating a new state otherwise.

---

21) ### _detect_semantic_relations
    The function `_detect_semantic_relations` performs the following operations:

    1. **Processing Tokens**:
    - Iterates through each dictionary in `token_analyses`, extracting the token (word) from each dictionary.
        ```python
        for ta in token_analyses:
            word = ta['token']
        ```

    2. **Checking Definitions**:
    - Checks if the extracted word has a stored definition in `self.reasoner.definitions`.
        ```python
        if word in self.reasoner.definitions:
        ```
    If the word is found in `self.reasoner.definitions`, it proceeds to generate a synonym hypothesis for that word.

    3. **Generating Synonym Hypothesis**:
    - Calls `self.reasoner.generate_synonym_hypothesis(word)` to generate a hypothesis for synonymy.
        ```python
        hypothesis = self.reasoner.generate_synonym_hypothesis(word)
        ```
    If the generated hypothesis is not empty, it prints the hypothesis.

    4. **Printing Hypothesis**:
    - Prints the semantic hypothesis if one is generated.
        ```python
        if hypothesis:
            print(f"💡 [Semantic Hypothesis] {hypothesis}")
        ```

    **Variables Initialised/Used:**
    - `word`: Extracted from each dictionary in `token_analyses`.
    - `ta`: Each dictionary in `token_analyses` containing token information.
    - `hypothesis`: Result of calling `self.reasoner.generate_synonym_hypothesis(word)`.

    This function processes a list of tokens and their analyses, checking for words that have stored definitions. For each such word, it generates hypotheses for synonymy and prints the hypothesis if one is generated.

    In summary:
    - The function iterates over `token_analyses`.
    - It checks if each token has a definition in `self.reasoner.definitions`.
    - If a definition exists, it generates a synonym hypothesis.
    - If a hypothesis is generated, it prints the hypothesis.

---

22) ### _reinforce_identity
    The function `_reinforce_identity` performs the following actions:

    1. It accepts two parameters: `factor_vector`, which is a numpy array, and `reward_signal`, which is a float.
    2. It initializes no variables within its scope.
    3. It calls the method `self.identity_manager.reinforce(factor_vector, reward_signal)` to reinforce the self-identity based on the provided `factor_vector` and `reward_signal`.
    4. After reinforcing the identity, it optionally logs the identity drift if it exceeds 0.5:
    - It calls `self.identity_manager.detect_drift(factor_vector)` to calculate the drift value.
    - If the calculated drift is greater than 0.5, it prints a message indicating that the current state diverges from the learned identity with a specific drift value formatted to three decimal places.

    The function does not return any value and performs exactly these steps without any additional operations or conditions beyond those specified.

---

23) ### _align_vector_to_emb_dim
    The function `_align_vector_to_emb_dim` performs the following actions:

    1. It accepts one parameter: `vector`, which is a numpy array.
    2. It initializes no variables within its scope other than `target_dim`.
    3. It determines the target dimension (`target_dim`) by accessing `self.semantic_learner.emb_dim`.
    4. It checks the length of the input vector:
    - If the length of `vector` is equal to `target_dim`, it returns a copy of the vector.
    - If the length of `vector` is greater than `target_dim`, it truncates the vector to the first `target_dim` elements and returns a copy of this truncated vector.
    - If the length of `vector` is less than `target_dim`, it pads the vector with zeros up to the target dimension:
        - It initializes an array `padded` filled with zeros of size `target_dim`.
        - It copies the content of `vector` into the beginning of `padded`.
        - It returns `padded`.

    The function does not return any value beyond the processed numpy array and performs exactly these steps without additional operations or conditions.

---

24) ### _compute_state_relevance
    The function `_compute_state_relevance` computes the similarity between a token's embedding and the current state embedding, returning a value scaled from \([-1, 1]\) to \([0, 1]\). Here are the specific details:

    **Input Parameters:**
    - `token`: A string representing the token.
    - `token_ci`: A dictionary related to the token curiosity (not used in the function).
    - `token_phon`: A float representing phonetic similarity of the token (not used in the function).
    - `diag`: A dictionary containing diagnostic information (not used in the function).
    - `features_array`: An array of features (not used in the function).
    - `global_factor_dict`: An optional dictionary for global factors (default is None, not used in the function).

    **Variables Initialised and Used:**
    1. **`token_emb`**: The embedding vector corresponding to the token obtained from `self.semantic_learner.word_embeddings`.
    2. **`state_emb`**: The current state embedding aligned to the same dimension as `token_emb` using `_align_vector_to_emb_dim`.
    3. **`norm_state`**: The Euclidean norm of `state_emb`.
    4. **`norm_token`**: The Euclidean norm of `token_emb`.

    **Function Steps:**
    1. If `self.current_state_embedding` is `None`, the function returns 0.0.
    2. Retrieves the token embedding (`token_emb`) from `self.semantic_learner.word_embeddings`.
    - If no token embedding exists, the function returns 0.0.
    3. Aligns both `state_emb` and `token_emb` to the same dimension using `_align_vector_to_emb_dim`.
    4. Computes the Euclidean norms of `state_emb` and `token_emb`.
    5. Checks if either norm is zero; if so, returns 0.0.
    6. Calculates the cosine similarity (`sim`) between `state_emb` and `token_emb`, scaled from \([-1, 1]\) to \([0, 1]\).
    7. If `self.current_state_labels` contains elements, updates metadata for the first active state label:
    - Retrieves metadata for the resolved state label.
    - Updates the 'recent_usage' value with a decay factor of \(0.99^{\Delta t}\), where \(\Delta t\) is the time since last use.
    - Increases 'recent_usage' by 0.05 and sets the 'last_used' timestamp to the current time.
    8. Returns the final scaled similarity as \((sim + 1.0) * 0.5\).

    Return Value:
    The function returns a float value between 0.0 and 1.0, representing the scaled cosine similarity between `state_emb` and `token_emb`.

---

25) ### _update_state_embedding
    The function `_update_state_embedding` performs the following actions:

    1. **Initialization and Input Variables**:
        - `labels`: A list of strings representing state labels.
        - `combined_vector`: An optional NumPy array representing a combined embedding for all labels.

    2. **Combined Vector Computation**:
        - If `combined_vector` is not provided (`None`):
            1. It initializes an empty list `embs`.
            2. Iterates through each label in `labels`. For each label, if the label exists in `self.semantic_learner.word_embeddings`, it appends its embedding to `embs`; otherwise, it calls `semantic_learner.initialize_word(lbl)` to initialize the word and then appends its new embedding.
            3. If `embs` is not empty, it computes `combined` as the mean of all embeddings in `embs`.
            4. If `embs` is empty, it sets `combined` to a NumPy array of zeros with shape `(self.semantic_learner.emb_dim,)`.

        - If `combined_vector` is provided:
            It directly assigns `combined_vector` to `combined`.

    3. **Ensure Embeddings for All Labels**:
        - Iterates through each label in `labels`. For each label, if the label does not exist in `self.semantic_learner.word_embeddings`, it initializes the word and sets its embedding using a vector from `self.state_manager.states[lbl]` (if available) after aligning to the correct dimension size via `_align_vector_to_emb_dim`.

    4. **Update State Variables**:
        - Sets `self.current_state_labels` to `labels`.
        - Sets `self.current_state_embedding` to the result of `_align_vector_to_emb_dim(combined)`.

    In summary, this function updates or initializes embeddings for labels in a semantic learner based on provided state labels and optionally an embedding vector, ensuring each label has its own embedding. It then sets the current active state labels and their combined embedding after alignment to the correct dimension size.

---

26) ### _process_sentence_reasoning
    The function `_process_sentence_reasoning` performs the following actions:

    **Input Variables:**
    - `sentence`: A string representing a sentence.
    - `tokens`: A list of strings representing tokens (words) in the sentence.
    - `token_analyses`: A list of dictionaries containing analyses for each token.
    - `sentence_emb`: A NumPy array representing an embedding for the sentence.

    **Detailed Steps:**

    1. **Store Sentence with Embedding**:
        - The function calls `self.reasoner.store_sentence(sentence, tokens, embedding=sentence_emb, metadata={'timestamp': time.time(), 'source': 'user'})`.
        - This stores the sentence in the reasoner's database along with its tokens and embedding.
        - Metadata includes a timestamp (`time.time()`) and source ('user').

    2. **Pattern-Based Relation Detection**:
        - Feeds `tokens` to the pattern observer (`self.reasoner.pattern_observer.observe(tokens)`), which updates structural word statistics and returns triplets of (head, connector, tail).

    3. **Math Sentence Parsing**:
        - Calls `self.reasoner.pattern_observer.try_parse_math(tokens, self.reasoner.operators)`.
        - If a math result is obtained, it calls `_handle_math_sentence(math_result, tokens)`.

    4. **Generate Hypotheses for Connector Roles**:
        - Iterates through pending questions from the pattern observer (`get_pending_questions()`).
        - For each question, if not already in hypotheses, adds it to `self.reasoner.hypotheses` with specific details like key, type, connector, text, asked status, energy cost, predictive gain, phase, and last used.

    5. **Process Each Triplet**:
        - Iterates through each triplet `(subject, connector, object)`.
        - Maps the connector to a relation type using `self.reasoner.pattern_observer.connector_to_reltype(connector)`; if not found, sets rel_type to 'unknown'.
        - Stores the relation in the reasoner (`store_phrase_relation(subject, object, rel_type, confidence=1.0)`).
        - Generates a confirmation hypothesis for each new triplet:
            ```python
            key = f"relation:{rel_type}:{subject}:{object}"
            if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                self.reasoner.hypotheses.append({
                    'key': key,
                    'type': 'relation',
                    'subject': subject,
                    'predicate': rel_type,
                    'object': object,
                    'text': f"Did I understand correctly: '{subject}' {connector} '{object}'?",
                    'asked': False
                })
            ```
        - Updates word-state associations:
            - For `rel_type` in ('means', 'is_a'), adds association between subject and object, then propagates labels from object to subject.
            - For `rel_type == 'similar'`, propagates labels between subject and object.

    6. **Add to Knowledge Graph**:
        - Adds nodes for the subject and object if they do not exist already.
        - Adds an edge in the knowledge graph connecting the subject and object with a confidence of 1.0.

    7. **Generate Sentence Hypothesis**:
        - Calls `self.reasoner.generate_sentence_hypothesis(sentence, tokens, sentence_emb)`.
        - If a hypothesis is generated, it prints the hypothesis (`print(f"💡 [Sentence Hypothesis] {hypothesis}")`).

    8. **Store Phrase Embedding**:
        - Stores the phrase embedding for future similarity checks if `sentence_emb` is not None.

    9. **Check Comprehension and Generate Query**:
        - If token analyses are provided, calculates average novelty of tokens.
        - Computes comprehension as 1 minus average novelty.
        - If comprehension is below a threshold (0.5), plans a sentence query (`self.reasoner.plan_sentence_query(sentence, uncertainty=1.0 - comprehension)`).
        - Stores the pending query if one is generated.

    **Summary:**
    - The function processes a sentence by storing it with its embedding and tokens in the reasoner.
    - It detects relations using pattern-based methods and updates word-state associations accordingly.
    - Generates hypotheses for confirmation of detected relations and other sentence-level hypotheses.
    - Adds identified relationships to the knowledge graph.
    - Evaluates comprehension based on token novelty and plans queries if necessary.

---

27) ### _derive_inferences_from_definition
    The function `_derive_inferences_from_definition` takes two parameters: `subject`, which is a string representing a word or phrase, and `definition_text`, another string providing the definition of the subject. The function does not initialize any variables beyond those provided as arguments.

    Here are the steps this function performs precisely:

    1. It receives the input parameters `subject` (a string) and `definition_text` (another string).
    2. Currently, it relies on a reasoner's relation graphs for deriving transitive inferences.
    3. The actual inference process is triggered when a relation is added via another method called `store_phrase_relation`.
    4. This function logs the addition of the definition but does not perform any direct action beyond logging.

    The variables used are:
    - `subject`: A string representing the word or phrase whose definition is being considered.
    - `definition_text`: A string containing the definition text for the subject.

    No specific values are given in this function; it relies on the calling context to provide these strings.

---

28) ### _normalize_vector, _all_tokens_known
    ### Function `_normalize_vector`

    1. Receives one parameter `vector`, which is an instance of `np.ndarray` (NumPy array).
    2. Calls the method `_normalize_vector(vector)` from `self.state_manager`.
    3. Returns the result of the call, which is also a NumPy array.

    Variables used:
    - `vector`: A NumPy array representing the vector to be normalized.
    - `state_manager`: An instance that contains the method `_normalize_vector`.

    No specific values are given in this function; it relies on the calling context to provide the `vector` and the `state_manager`.

    ### Function `_all_tokens_known`

    1. Receives one parameter `tokens`, which is a list of strings.
    2. Checks if every string (token) in the list exists in `self.semantic_learne`.
    3. Returns `True` if all tokens are present, otherwise returns `False`.

    Variables used:
    - `tokens`: A list of strings representing the tokens to be checked.
    - `semantic_learne`: An object or collection that contains semantic embeddings for known tokens.

    No specific values are given in this function; it relies on the calling context to provide the `tokens` and the `self.semantic_learne`.

    Note: The provided code snippet for `_all_tokens_known` is incomplete, but based on the description, it is clear what the intended behavior is.

---

29) ### _integrate_behavioral_rule
    The function `_integrate_behavioral_rule` takes one parameter `rule_id`, which is a string representing the ID of a confirmed behavioral rule. The function performs the following steps:

    1. Retrieves the rule with the given `rule_id` from `self.behavioral_learner.rules`.
    2. Checks if the retrieved rule exists and is confirmed:
    - If either condition fails, it returns immediately without further processing.
    3. Evaluates the `rule_type` of the rule:
    - If the `rule_type` is `'identity_alias'`, it proceeds with specific actions:
        1. Extracts the alias from the rule's condition.
        2. Adds the alias to `self.identity_aliases`.
        3. Checks if the word "i" exists in `self.semantic_learner.word_embeddings`:
            - If "i" is present, copies its embedding and initializes the word alias in `self.semantic_learner.word_embeddings`.
            - Logs a message indicating that the self-alias has been learned.
            - Stores a definition for the alias using `self.reasoner.store_definition`, associating it with an embedding if available or `None` otherwise.
    4. Integrates the rule into the behavioral learner by calling `self.behavioral_learner.integrate_rule(rule_id)`.

    Variables used:
    - `rule`: A dictionary representing the behavioral rule retrieved from `self.behavioral_learner.rules`.
    - `alias`: A string representing an alias extracted from the rule's condition.
    - `i_emb`: A NumPy array or similar data structure representing the embedding of "I".
    - `identity_aliases`: A set or collection to store identity aliases.
    - `semantic_learner.word_embeddings`: A dictionary containing word embeddings, where keys are strings and values are embeddings (e.g., NumPy arrays).
    - `reasoner`: An instance that contains the method `store_definition` for storing definitions.

    No specific values other than the input parameter `rule_id` are given in this function; it relies on the calling context to provide these structures.

---

30) ### _should_ask_feedback
    The function `_should_ask_feedback` takes three parameters: `tokens`, a list of strings representing tokens; `factor_dict`, a dictionary containing factors; and an optional parameter `template_id`, a string representing a template ID. The function performs the following steps:

    1. Retrieves the base probability for asking feedback from `self.feedback_ask_probability`.
    2. Determines the context confidence:
    - If `template_id` is provided, retrieves the context confidence using `self.reasoner.get_context_confidence(f"template:{template_id}")`.
    - Otherwise, sets the context confidence to 0.0.
    3. Adjusts the base probability if the context confidence is greater than 0.8:
    - Multiplies the base probability by 0.2 (reduces it significantly).
    4. Checks if the time since the last feedback request (`self._last_feedback_ask_time`) is less than 60 seconds:
    - If true, returns `False` immediately.
    5. Generates a random number between 0 and 1 using `np.random.rand()`:
    - If this random number is greater than the base probability, returns `False`.
    6. Checks if feedback is needed based on the tokens, factor dictionary, and template ID by calling `self.reasoner.should_ask_feedback(tokens, factor_dict, template_id)`.
    7. Updates the last feedback ask time to the current time (`time.time()`) if feedback is needed.
    8. Returns whether feedback is needed.

    Variables used:
    - `base_prob`: A float representing the base probability for asking feedback.
    - `context_confidence`: A float representing the context confidence.
    - `template_id`: A string representing a template ID, which may be `None`.
    - `_last_feedback_ask_time`: A float or integer representing the last time feedback was asked (in seconds since the epoch).

    Specific values:
    - The base probability is retrieved from `self.feedback_ask_probability`.
    - If context confidence is greater than 0.8, the base probability is reduced to 20% of its original value.
    - Cooldown period is set to 60 seconds.

    No specific values are given for `tokens`, `factor_dict`, or other internal variables; they rely on the calling context to provide these inputs.

---

31) ### _generate_feedback_question
    The function `_generate_feedback_question` generates a neutral question asking for user feedback. The function performs the following steps:

    1. Defines a list of templates, each representing a different question to ask for user feedback:
    - `"How would you rate my response? (positive/negative/neutral)"`
    - `"Was my answer helpful? (positive/negative/neutral)"`
    - `"Did that make sense? Please rate: positive, negative, or neutral."`
    - `"How do you feel about my reply? (positive/negative/neutral)"`
    2. Selects one of these templates randomly using `np.random.choice(templates)`.
    3. Returns the selected template as a string.

    Variables used:
    - `templates`: A list of strings containing different feedback question templates.
    - The specific values in the templates are predefined and deterministic.

    No additional variables or parameters are initialized within this function; it relies solely on the internal list of templates and the random selection process provided by `np.random.choice`.

    The specific values in the templates are:
    1. `"How would you rate my response? (positive/negative/neutral)"`
    2. `"Was my answer helpful? (positive/negative/neutral)"`
    3. `"Did that make sense? Please rate: positive, negative, or neutral."`
    4. `"How do you feel about my reply? (positive/negative/neutral)"`

    The function returns one of these templates randomly selected from the list.

---

32) ### _on_new_control_factor
    The function `_on_new_control_factor` is designed to be called when a new dynamic control factor is added. It performs the following actions:

    1. Initializes `curiosity_compute_function`, which computes the curiosity value for each token based on the specified control factor and the token's own novelty.
    - **Input Parameters**:
        - `token`: A string representing a token.
        - `token_ci`: A dictionary containing confidence information about the token, specifically `confidence`.
        - `token_phon`: A float representing phonetic information of the token (unused in this function).
        - `diag`: A dictionary for diagnostic purposes (unused in this function).
        - `features_array`: An array of features related to the token (unused in this function).
        - `global_factor_dict` (optional): A dictionary containing global values for control factors, with a default value of `None`.
    - **Output**:
        - Returns a float representing the computed curiosity value.
    - **Computation Steps**:
        1. Retrieves the current global value (`ctrl_val`) of the specified control factor from `global_factor_dict`, defaulting to 0.5 if not provided.
        2. Calculates the deviation of `ctrl_val` from neutral (0.5) and scales it to a range [0, 1].
        3. Computes the token's novelty as `(1 - confidence)` where `confidence` is derived from `token_ci`.
        4. Combines the deviation with the novelty to compute the curiosity value.
        5. Clips the computed curiosity value between 0 and 1.

    2. Adds a new curiosity factor corresponding to the specified control factor to `self.meta_learner`. The added factor has:
    - **Factor Name**: Formatted as `"ctrl_{factor_name}"`.
    - **Initial Mu (Mean)**: Set to 0.0, indicating a neutral prior.
    - **Compute Function**: Uses `curiosity_compute_function` initialized above.
    - **Metadata**:
        - `'source'`: 'control_space'.
        - `'original_factor'`: The name of the original control factor.

    3. Ensures metadata for this new curiosity factor is tracked in `self.meta_learner.factor_metadata`. If the metadata does not exist, it initializes it with default values.
    4. Prints a confirmation message indicating the creation of the new curiosity factor.

    **Variables Initialized and Used:**
    - `curiosity_compute_function`: A nested function that calculates the token's curiosity value.
    - `ctrl_val`: Float representing the current global value of the control factor (default 0.5).
    - `deviation`: Float representing the scaled deviation from neutral for the control factor.
    - `novelty`: Float representing the token's novelty, computed as `(1 - confidence)`.
    - `curiosity`: Final float value representing the combined curiosity score.

    **Specific Values:**
    - Default global control factor value (`ctrl_val`) is 0.5.
    - Initial mean (`initial_mu`) for new factors is set to 0.0.
    - Confidence default value within `token_ci` is 0.5, leading to a maximum novelty of 1.0.

    This function ensures that each added dynamic control factor is associated with a corresponding curiosity factor designed to encourage exploration based on the deviation from neutral and token novelty.

---

33) ### build_world_model_transitions
    The function `build_world_model_transitions` is designed to scan the episodic memory to build transition counts for a structured world model. It performs the following actions:

    1. Retrieves the last 500 episodes from `self.episodic_memory.episodes`.
    2. Iterates through these episodes, comparing each episode with the next one.
    3. For each pair of consecutive episodes (`ep` and `next_ep`), checks if both have a `factor_vector` key:
    - If present, converts the factor vectors to numpy arrays using `_align_factor_vector`.
    - Finds similar states in the state manager for these factor vectors, with a similarity threshold of 0.7.
    4. If both episodes are associated with a valid state label (`state_label` and `next_state_label`):
    - Extracts an action from the current episode as `ep.get('system_response', '')`.
    - Observes this transition in the structured world model by calling `self.structured_world_model.observe_episode(state_label, action, next_state_label)`.

    **Variables Initialized and Used:**
    - `episodes`: A list containing the last 500 episodes from `self.episodic_memory.episodes`.
    - `vec`: Numpy array representing the factor vector of the current episode after alignment.
    - `next_vec`: Numpy array representing the factor vector of the next episode after alignment.
    - `state_label`: String representing the state label for the current episode's factor vector, found using `_align_factor_vector` and `self.state_manager.find_similar_state`.
    - `next_state_label`: String representing the state label for the next episode's factor vector, similarly obtained.
    - `action`: A string representing a crude action representation from the current episode.

    **Specific Values:**
    - The function processes the last 500 episodes in `episodes`.
    - Similarity threshold for finding states is set to 0.7.
    - Action extraction uses `ep.get('system_response', '')`, defaulting to an empty string if no system response is present.

    This function ensures that transitions between states are recorded based on aligned factor vectors from consecutive episodes, thereby building a structured world model of state-action-next state transitions.

---

34) ### process_input

    Full pipeline with meta-learning, reasoning, AND factor-based state learning.
    Also increments interaction counter and triggers self-reconfiguration.

    FIXES applied vs previous version:
    - factor_dict initialized to {} at top to avoid NameError in section 8
    - expecting_word_conflict handler moved to the top-of-function waiting-state
    block (was previously buried after normal response generation, causing the
    full pipeline to run before the handler was ever reached)
    - 'section 2.5' handler body removed from bottom of function (it is now
    correctly placed as handler #7 at the top)
    - Chat-mode reward multiplier guarded so it only runs when factor_dict has
    been properly computed (i.e. not in early-return paths)
    - expecting_word_conflict flag defensively reset even if _pending_conflict
    is missing, to prevent permanent lock-up

    #### functions 

    1) #### Maintainance
        This function performs several operations related to managing and updating the state of an object, particularly focusing on tracking interactions, memory management, and behavioral rule testing. Here is a precise description of what it does:

        1. **Initialization of Variables:**
            - `self.last_interaction_time`: This variable is initialized with the current time (in seconds since the epoch) using `time.time()`.
            - `factor_dict`: A dictionary named `factor_dict` is pre-initialized as an empty dictionary `{}` to allow safe referencing in section 8 (behavioral rule testing).

        2. **Cycle Counter Management:**
            - The variable `_cycle_counter` is incremented by 1.
            - If `_cycle_counter` modulo 250 equals 0, the method `_update_dynamic_limits()` is called.

        3. **Session Cycle Limit Check:**
            - If `_cycle_counter` exceeds `_max_cycles_per_session`, a message "[WATCHDOG] Cycle limit reached — checkpointing." is printed.
            - The state of the object is saved using the method `_save_state()`.
            - `_cycle_counter` is reset to 0.

        4. **Memory Pressure Backoff:**
            - The method `_memory_pressure_backoff()` is called, which presumably manages memory usage based on certain criteria.

        5. **Idle Reasoner Management (if present):**
            - If the object has an attribute `_idle_reasoner`, this attribute's `reset_long_idle_flag` method is called to reset a flag indicating long-idle status.

        6. **Interaction History Update:**
            - The variable `text` (representing user input or interaction) is appended to the list `self.interaction_history`.
            - If the length of `interaction_history` exceeds `_max_interaction_history`, which defaults to 200,000 if not defined, `interaction_history` is truncated to its last `_max_interaction_history` elements.

        7. **Prediction Error Calculation:**
            - If both `_expected_next_state` and `self.current_state_labels` are non-null, the first element of `current_state_labels` (denoted as `actual_state`) is compared with `_expected_next_state`.
            - If there is a mismatch (`actual_state != _expected_next_state`), a prediction error of 1.0 is set.
            - A message in the format "   [Prediction Error] Expected {self._expected_next_state}, got {actual_state}" is printed.
            - Finally, `_expected_next_state` is reset to `None`.

        In summary, this function manages various aspects of state tracking and updates, including time stamping, cycle counting, memory management, interaction history maintenance, and prediction error handling.

    2) #### Response Handling

        1) #### Chat mode toggle
            This function processes a special command to toggle chat mode and performs several operations related to state management and output formatting. Here is a precise description of what it does:

            1. **Chat Mode Toggle:**
                - The input `text` is converted to lowercase, stripped of leading/trailing whitespace, and compared with `self.chat_mode_trigger`.
                - If the comparison matches, `self.chat_mode_active` is toggled between `True` and `False`.

            2. **Activation Actions (if chat mode becomes active):**
                - If `chat_mode_active` is now `True`, `self.chat_history` is set to an empty list `[]`.
                - A system reply `"Chat mode activated. I'll try to talk with you naturally. What shall we discuss?"` is generated.

            3. **Deactivation Actions (if chat mode becomes inactive):**
                - If `chat_mode_active` is now `False`, a system reply `"Chat mode deactivated. Returning to normal learning mode."` is generated.

            4. **System Reply Output:**
                - The system reply (`system_reply`) is printed with the prefix " Herebot>".

            5. **Return Dictionary:**
                - A dictionary is returned containing several specific values:
                    - `ci_report`: An empty dictionary `{}`.
                    - `phon_score`: Integer value 0.
                    - `diagnostic`: An empty dictionary `{}`.
                    - `token_analyses`: An empty list `[]`.
                    - `curiosity_artifacts`: A list with one element `[f"Chat mode: {'on' if self.chat_mode_active else 'off'}"]` indicating the current state of chat mode.
                    - `meta_weights`: The weights from the method `self.meta_learner.get_weights()`.
                    - `total_intrinsic_reward`: The total reward accumulated in `self.reward_tracker.total_reward`.
                    - `episode_reward`: Float value 0.0.
                    - `factor_vector`: An empty dictionary `{}`.

            In summary, this function toggles chat mode based on a special command (`text`), manages state variables accordingly, and returns a structured dictionary containing specific values for further processing.

        2) #### Response feedback
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `self.expecting_response_feedback` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Converts the input `text` to lowercase and stores it in the variable `feedback_lower`.

            3. Initializes a variable `rating` as `None`.

            4. Evaluates whether any of the keywords from the list `['positive', 'good', 'great', 'yes', 'correct', 'right', '👍']` are present in `feedback_lower`. If so, sets `rating = 'positive'`.

            5. Checks if any keyword from the list `['negative', 'bad', 'wrong', 'no', 'incorrect', '👎']` is in `feedback_lower`. If true, assigns `rating = 'negative'`.

            6. Determines if any of the keywords in `['neutral', 'ok', 'fine', 'average']` are found in `feedback_lower`, setting `rating = 'neutral'` accordingly.

            7. Verifies if `rating` is not `None` and `self.pending_feedback_context` is not `None`. If both conditions hold, it performs the following steps:
            - Retrieves `entry` from `self.pending_feedback_context`.
            - Sets the rating attribute of `entry` to the value stored in `rating`.
            - Adds a timestamp under key `'feedback_timestamp'` within `entry.metadata`, using `time.time()`.
            - Stores the original feedback text (`text`) as `'feedback_text'` inside `entry.metadata`.
            - Calls the method `self.reasoner.add_feedback(entry)` to add the feedback record.
            - Prints "📊 [Feedback Recorded] User rated response as: {rating}" where `{rating}` is replaced by actual value of `rating`.

            8. If a 'positive' rating is detected and `_last_factor_vector` attribute exists, it calls `_reinforce_identity(self._last_factor_vector, reward_signal=0.2)`.

            9. Sets `self.expecting_response_feedback = False` and clears `self.pending_feedback_context`.

            10. Returns a dictionary with the following structure:
                - `"ci_report"`: Empty dictionary `{}`.
                - `"phon_score"`: Integer value `0`.
                - `"diagnostic"`: Empty dictionary `{}`.
                - `"token_analyses"`: List containing no elements `[]`.
                - `"curiosity_artifacts"`: A list with a single string element `"Feedback recorded."`.
                - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
                - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
                - `"episode_reward"`: Float value set to `0.0`.
                - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.

        3) #### Rule confirmation response
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `self.expecting_rule_confirmation` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Converts the input `text` to lowercase and stores it in the variable `feedback_lower`.

            3. Initializes a boolean variable `is_positive` as `True` if any of the keywords from the list `['yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm']` are present in `feedback_lower`; otherwise, sets it to `False`.

            4. Initializes a boolean variable `is_negative` as `True` if any of the keywords from the list `['no', 'wrong', 'incorrect', 'not', 'nope', 'never']` are present in `feedback_lower`; otherwise, sets it to `False`.

            5. If either `is_positive` or `is_negative` is `True`, performs the following steps:
            - Retrieves `rule_id` from `self.pending_rule_id`.
            - Checks if `rule_id` exists in `self.behavioral_learner.rules`. If true, proceeds with further processing; otherwise, does nothing.
                - If `is_positive` is `True`:
                - Prints "✅ [Rule Confirmed] User accepted rule: {rule_id}" where `{rule_id}` is the actual value of `rule_id`.
                - Calls `self.behavioral_learner.update_confidence(rule_id, +0.3, feedback=True)` to update the confidence score of the rule by `+0.3` with feedback set to `True`.
                - If the updated confidence score of the rule is greater than or equal to `self.behavioral_learner.confirmation_threshold`, calls `_integrate_behavioral_rule(rule_id)`.
                - If `is_negative` is `True`:
                - Prints "❌ [Rule Rejected] User denied rule: {rule_id}" where `{rule_id}` is the actual value of `rule_id`.
                - Calls `self.behavioral_learner.update_confidence(rule_id, -0.5, feedback=True)` to update the confidence score of the rule by `-0.5` with feedback set to `True`.

            6. If neither `is_positive` nor `is_negative` is `True`, performs the following steps:
            - Prints "🌑 [Rule Shadow] No clear confirmation for rule: {self.pending_rule_id}" where `{self.pending_rule_id}` is the actual value of `self.pending_rule_id`.
            - Checks if `self.pending_rule_id` exists in `self.behavioral_learner.rules`. If true, calls `self.behavioral_learner.update_confidence(self.pending_rule_id, -0.1, feedback=False)` to update the confidence score by `-0.1` with feedback set to `False`.

            7. Sets `self.expecting_rule_confirmation = False` and clears `self.pending_rule_id`.

            8. Returns a dictionary with the following structure:
            - `"ci_report"`: Empty dictionary `{}`.
            - `"phon_score"`: Integer value `0`.
            - `"diagnostic"`: Empty dictionary `{}`.
            - `"token_analyses"`: List containing no elements `[]`.
            - `"curiosity_artifacts"`: A list with a single string element `"Rule confirmation processed."`.
            - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
            - `"episode_reward"`: Float value set to `0.0`.
            - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.

        4) #### Logic doubt feedback
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `self.expecting_logic_feedback` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Converts the input `text` to lowercase and stores it in the variable `feedback_lower`.

            3. Initializes a boolean variable `is_positive` as `True` if any of the keywords from the list `['yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm']` are present in `feedback_lower`; otherwise, sets it to `False`.

            4. Initializes a boolean variable `is_negative` as `True` if any of the keywords from the list `['no', 'wrong', 'incorrect', 'not', 'nope', 'never']` are present in `feedback_lower`; otherwise, sets it to `False`.

            5. If `self.pending_logic_doubt` is not `None`, performs the following steps:
            - Retrieves `logic_id` from `self.pending_logic_doubt['logic_id']`.
            - If `is_positive` is `True`:
                - Calls `self.logic_inheritance.confirm_logic(logic_id, is_correct=True)` to confirm the logic as correct.
                - Prints "✅ [Logic Confirmed] User accepted logic: {logic_id}" where `{logic_id}` is the actual value of `logic_id`.
            - If `is_negative` is `True`:
                - Calls `self.logic_inheritance.confirm_logic(logic_id, is_correct=False)` to confirm the logic as incorrect.
                - Prints "❌ [Logic Rejected] User denied logic: {logic_id}" where `{logic_id}` is the actual value of `logic_id`.

            6. If either `is_positive` or `is_negative` is `True`, performs the following steps:
            - Calls `self.generated_code_feedback.add_feedback(logic_id, "positive" if is_positive else "negative", {"factor_dict": self._last_factor_dict.copy() if self._last_factor_dict else {}})` to add feedback for the logic with the factor dictionary.
            - If `self._last_factor_dict` is not `None`, performs the following steps:
                - Sets `rating` as `"positive"` if `is_positive` is `True`; otherwise, sets it as `"negative"`.
                - Calls `self.guilt_tracker.update_guilt(rating, self._last_factor_dict)` to update guilt based on the rating and factor dictionary.

            7. If neither `is_positive` nor `is_negative` is `True`, prints "🌑 [Logic Shadow] No clear confirmation for logic: {logic_id}" where `{logic_id}` is the actual value of `logic_id`.

            8. Sets `self.expecting_logic_feedback = False` and clears `self.pending_logic_doubt`.

            9. Returns a dictionary with the following structure:
            - `"ci_report"`: Empty dictionary `{}`.
            - `"phon_score"`: Integer value `0`.
            - `"diagnostic"`: Empty dictionary `{}`.
            - `"token_analyses"`: List containing no elements `[]`.
            - `"curiosity_artifacts"`: A list with a single string element `"Logic feedback processed."`.
            - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
            - `"episode_reward"`: Float value set to `0.0`.
            - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.       

        5) #### Pattern label response
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `getattr(self, "expecting_pattern_label", False)` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Strips leading and trailing whitespace from `text`, converts the result to lowercase, and stores it in the variable `label`.

            3. Appends a tuple `("category", label, 1.0)` to the list stored under key `self.last_asked_word` in the dictionary `self.reasoner.relations`. If `self.last_asked_word` is not already a key in `self.reasoner.relations`, it creates the key with an empty list as its value.

            4. Checks if `self.last_asked_word` exists in `self.semantic_learner.word_embeddings`. If not, calls `self.semantic_learner.initialize_word(self.last_asked_word)` to initialize the word embedding for `self.last_asked_word`.

            5. Prints "📘 [Pattern Learning] '{self.last_asked_word}' categorized as '{label}'" where `{self.last_asked_word}` and `{label}` are replaced by their actual values.

            6. Sets `self.expecting_pattern_label = False` to indicate that it is no longer expecting a pattern label.

            7. Clears `self.last_asked_word`.

            8. Returns a dictionary with the following structure:
            - `"ci_report"`: Empty dictionary `{}`.
            - `"phon_score"`: Integer value `0`.
            - `"diagnostic"`: Empty dictionary `{}`.
            - `"token_analyses"`: List containing no elements `[]`.
            - `"curiosity_artifacts"`: A list with a single string element `"Pattern label learned."`.
            - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
            - `"episode_reward"`: Float value set to `0.0`.
            - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.   

        6) #### Trust feedback 
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `self.expecting_feedback` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Converts the input `text` to lowercase and stores it in the variable `feedback_lower`.

            3. Initializes a boolean variable `is_positive` as `True` if any of the keywords from the list `['yes', 'correct', 'right', 'yeah', 'true', 'yep']` are present in `feedback_lower`; otherwise, sets it to `False`.

            4. Initializes a boolean variable `is_negative` as `True` if any of the keywords from the list `['no', 'wrong', 'incorrect', 'not', 'nope']` are present in `feedback_lower`; otherwise, sets it to `False`.

            5. If either `is_positive` or `is_negative` is `True`, performs the following steps:
            - If `is_positive` is `True`:
                - Prints "✅ [Positive Feedback] User confirmed my attempt: '{self.feedback_attempt_response}'" where `{self.feedback_attempt_response}` is replaced by its actual value.
                - If `_last_factor_vector` attribute exists, calls `_reinforce_identity(self._last_factor_vector, reward_signal=0.4)` to reinforce the identity with a reward signal of `0.4`.
            - Otherwise (if `is_negative` is `True`):
                - Prints "❌ [Negative Feedback] User corrected my attempt: '{self.feedback_attempt_response}'" where `{self.feedback_attempt_response}` is replaced by its actual value.

            6. If neither `is_positive` nor `is_negative` is `True`, performs the following steps:
            - Prints "🌑 [Shadow Attempt] No feedback given for: '{self.feedback_attempt_response}'" where `{self.feedback_attempt_response}` is replaced by its actual value.
            - If `self.feedback_attempt_response` is not empty, creates a hypothesis dictionary `hyp` with the following structure:
                - `'key'`: A string formatted as `"shadow_trust_{int(time.time())}"`, where `{int(time.time())}` represents the current Unix timestamp converted to an integer.
                - `'type'`: String value `"shadow_trust"`.
                - `'utterance'`: The value of `self.feedback_attempt_response`.
                - `'timestamp'`: The value of `self.feedback_timestamp`.
            - If the length of `self.reasoner.hypotheses` is less than `_max_hypotheses` (defaulting to 5000 if not defined), appends `hyp` to `self.reasoner.hypotheses`.

            7. Sets `self.expecting_feedback = False` to indicate that it is no longer expecting feedback.

            8. Clears the following variables:
            - `self.feedback_attempt_word`.
            - `self.feedback_attempt_response`.
            - `self.feedback_timestamp`.

            9. Returns a dictionary with the following structure:
            - `"ci_report"`: Empty dictionary `{}`.
            - `"phon_score"`: Integer value `0`.
            - `"diagnostic"`: Empty dictionary `{}`.
            - `"token_analyses"`: List containing no elements `[]`.
            - `"curiosity_artifacts"`: A list with a single string element `"Feedback processed."`.
            - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
            - `"episode_reward"`: Float value set to `0.0`.
            - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.        

        7) #### Word-label conflict resolution
            The provided function performs the following actions:

            1. Checks if `self.chat_mode_active` is `False` and `getattr(self, "expecting_word_conflict", False)` is `True`. If both conditions are met, it proceeds with further processing; otherwise, it does nothing.

            2. Converts the input `text` to lowercase and stores it in the variable `feedback_lower`.

            3. Checks if the attribute `_pending_conflict` exists. If it does:
            - Retrieves the tuple stored in `_pending_conflict` as `(word, label1, label2)`.
            - Evaluates whether either `label1.lower()` or `label2.lower()` is present in `feedback_lower`. If true, performs the following steps:
                - Sets `keep` to `label1` if `label1.lower()` is found in `feedback_lower`; otherwise, sets it to `label2`.
                - Calls `self.reasoner.word_state.break_bridge(word, label1, label2, keep)` to break the bridge for the word with the chosen label.
                - Prints "✅ [Conflict Resolved] '{word}' is now primarily '{keep}'." where `{word}` and `{keep}` are replaced by their actual values.
            - If neither `label1.lower()` nor `label2.lower()` is found in `feedback_lower`, prints "🌑 No clear label chosen for '{word}'. Conflict remains."
            - Deletes `_pending_conflict` to remove the pending conflict state.

            4. If `_pending_conflict` does not exist, prints "🌑 [Word Conflict] Expected conflict state missing; resetting flag."

            5. Sets `self.expecting_word_conflict = False` to indicate that it is no longer expecting a word conflict.

            6. Returns a dictionary with the following structure:
            - `"ci_report"`: Empty dictionary `{}`.
            - `"phon_score"`: Integer value `0`.
            - `"diagnostic"`: Empty dictionary `{}`.
            - `"token_analyses"`: List containing no elements `[]`.
            - `"curiosity_artifacts"`: A list with a single string element `"Word conflict processed."`.
            - `"meta_weights"`: Retrieves weights from the method call to `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: Retrieves total reward from `self.reward_tracker.total_reward`.
            - `"episode_reward"`: Float value set to `0.0`.
            - `"factor_vector"`: Empty dictionary `{}`.

            This function does not modify or use variables outside the scope of those mentioned above, except for `text` which is an input parameter.

        8) ####  Definition response 
            The function described does the following:

            1. Checks if `self.chat_mode_active` is `False` and `self.last_asked_word` is not `None`.
            2. If condition in step 1 is met, it checks if any of the phrases "means", "is", or "refers to" are present in `text.lower()`.
            3. If one of those phrases is found, it:
            - Determines if `self.last_asked_word` contains a space by setting `is_phrase = ' ' in self.last_asked_word`.
            4. If `is_phrase` is `True`, it performs the following actions:
            - Sets `phrase = self.last_asked_word`.
            - Sets `definition = text`.
            - Splits `phrase` into tokens and stores them in `phrase_tokens`.
            - Initializes an empty list called `known_embs` to store embeddings for each token in `phrase_tokens` that exists in `self.semantic_learner.word_embeddings`.
            - Computes the average embedding of all known embeddings (`np.mean(known_embs, axis=0)`) and assigns it to `phrase_emb`, or sets `phrase_emb = None` if no embeddings are found.
            - Adds the phrase-definition pair `(phrase, definition)` to `self.reasoner.phrase_definitions`.
            - If `phrase_emb` is not `None`, adds the phrase-embedding pair `(phrase, phrase_emb)` to `self.reasoner.phrase_embeddings`.
            - Prints: `"📘 [Phrase Definition] Learned about '{phrase}': {definition}"`.
            - Calls `_derive_inferences_from_definition(phrase, definition)`.
            5. If `is_phrase` is `False`, it performs the following actions:
            - Sets `word = self.last_asked_word`.
            - Checks if `word` does not exist in `self.semantic_learner.word_embeddings`. If so, initializes the word using `self.semantic_learner.initialize_word(word)`.
            - Observes and stores the definition of `word` with respect to `text` via `self.semantic_learner.observe_definition(word, text, structural_detector=self.reasoner.pattern_observer.detector)`.
            - Converts `text` into an embedding (`def_emb`) using `self.semantic_learner.text_to_embedding(text)`.
            - If `def_emb` is not `None`, performs the following:
                - Stores the word-definition-embedding triple `(word, text, def_emb)` in `self.reasoner.store_definition(word, text, def_emb)`.
                - Registers a definition event for `word` using `self.concept_model.register_definition_event(word)`.
            - If `word` exists in `self.word_history`, performs the following:
                - Decreases the GP uncertainty and Bayesian Surprise by 0.7 (multiplied by 0.3) for each of the last three records in `self.word_history[word]`.
                - Updates the dominant concept count to be 100% certain for the most frequent concept if such a record exists in `self.concept_model.word_concept_counts[word]`.
            - Generates and prints a synonym hypothesis for `word` using `self.reasoner.generate_synonym_hypothesis(word)` if available.
            - Prints: `"📘 [Definition Learning] Learned about '{word}': {text}"`.
            - Calls `_derive_inferences_from_definition(word, text)`.
            6. If the attribute `_last_factor_vector` exists in the object, calls `self._reinforce_identity(self._last_factor_vector, reward_signal=0.3)`.
            7. Sets `self.last_asked_word = None`.
            8. Sets `handled_definition = True`.

            Variables initialized or used:
            - `is_phrase`: Boolean indicating whether `self.last_asked_word` contains a space.
            - `phrase`: The value of `self.last_asked_word` if it contains a space.
            - `definition`: Value of `text`, treated as the definition if "means", "is", or "refers to" is in lowercased `text`.
            - `phrase_tokens`: List containing tokens from `phrase`.
            - `known_embs`: List storing embeddings for each token in `phrase_tokens` that exists in `self.semantic_learner.word_embeddings`.
            - `phrase_emb`: Average of `known_embs`, or `None` if no embeddings are found.
            - `handled_definition`: Boolean set to `True`.

            The function does not modify any other variables outside the scope described above.
        
        9) #### procedure explanation
            The function described does the following:

            1. Checks if `self.chat_mode_active` is `False` and `self.expecting_procedure_steps` is `True`.
            2. If both conditions are met, it performs the following actions:
            - Strips any leading or trailing whitespace from `text` and assigns the result to `step_text`.
            - Appends `step_text` to `self.recorded_steps`.
            - Prints: `"📝 [Step Recorded] {step_text}"`.
            3. Checks if "done", "that's it", or "finish" is present in `step_text.lower()`. If any of these phrases are found, performs the following:
            - If `self.pending_problem_expr` is not `None`, learns a new procedure using `self.reasoner.procedural_learner.learn_procedure(self.pending_problem_expr, self.recorded_steps, None)`.
                - Prints: `" [Procedure Learned] I've learned a new procedure for {self.pending_problem_expr}"`
            - Sets `self.expecting_procedure_steps` to `False`.
            - Sets `self.pending_problem_expr` to `None`.
            - Clears `self.recorded_steps` by setting it to an empty list.
            4. Returns a dictionary with the following specific values:
            - `"ci_report"`: An empty dictionary `{}`.
            - `"phon_score"`: The integer value `0`.
            - `"diagnostic"`: An empty dictionary `{}`.
            - `"token_analyses"`: A list containing no elements `[]`.
            - `"curiosity_artifacts"`: A list containing the string `"Procedure recorded."`.
            - `"meta_weights"`: Obtained from calling `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: The value of `self.reward_tracker.total_reward`.
            - `"episode_reward"`: A floating-point number `0.0`.
            - `"factor_vector"`: An empty dictionary `{}`.

            5. If none of the phrases "done", "that's it", or "finish" are found in `step_text.lower()`, performs the following:
            6. Returns a dictionary with the following specific values:
            - `"ci_report"`: An empty dictionary `{}`.
            - `"phon_score"`: The integer value `0`.
            - `"diagnostic"`: An empty dictionary `{}`.
            - `"token_analyses"`: A list containing no elements `[]`.
            - `"curiosity_artifacts"`: A list containing the string `"Step recorded. Continue."`.
            - `"meta_weights"`: Obtained from calling `self.meta_learner.get_weights()`.
            - `"total_intrinsic_reward"`: The value of `self.reward_tracker.total_reward`.
            - `"episode_reward"`: A floating-point number `0.0`.
            - `"factor_vector"`: An empty dictionary `{}`.

            Variables initialized or used:
            - `step_text`: String obtained by stripping leading and trailing whitespace from `text`.
            - `proc`: The learned procedure from calling `self.reasoner.procedural_learner.learn_procedure(self.pending_problem_expr, self.recorded_steps, None)`.

            The function does not modify any other variables outside the scope described above.

        10) #### MAIN PROCESSING
            1) #### step 1-3
                The function described performs the following actions:

                1. **Increment Interaction Counter**:
                - Increments `self.interaction_counter` by 1.

                2. **Memory Ingestion**:
                - Calls `add_sentence_to_memory(text)` to add the provided text to memory.

                3. **Whole-Sentence Evaluation**:
                - Evaluates the whole sentence using `ci_report = self.ci_engine.evaluate(text)`.
                - Scores the phonetic quality of the text using `phon_score, phon_diagnostics = self.phon_engine.score_text(text)`.
                - Explains the diagnostic report and phonetic score using `diag_explanation = self.diag_engine.explain(ci_report, phon_score)`.

                4. **Token Analysis**:
                - Tokenizes the input text into a list of tokens via `tokens = self._tokenize(text)`.
                - Observes the sentence with its tokens by calling `self.reasoner.observe_sentence(text, tokens)`.
                - Resets the reward tracker using `self.reward_tracker.reset_episode()`.

                5. **Iterate Over Tokens**:
                - For each token in `tokens`, performs the following steps:
                    - Evaluates the confidence of the CI engine for the token: `token_ci = self.ci_engine.evaluate(token)`.
                    - Scores the phonetic quality of the token: `token_phon, _ = self.phon_engine.score_text(token)`.
                    - Extracts features from the token using `_extract_features(token, token_ci, token_phon, diag_explanation)` and converts them to an array with `_features_to_array(features)`.
                    - Adds a word observation for the token in the concept model: `concept = self.concept_model.add_word_observation(token, features)`.
                    - Computes curiosity factors using `_compute_curiosity_factors(token, token_ci, token_phon, diag_explanation, features_array, global_factor_dict=self._last_factor_dict)`.
                    - Calculates new entropy for the token in the concept model: `new_entropy = self.concept_model.entropy_word(token)`.
                    - Determines information gain (`igain`) using `_information_gain(token, new_entropy)` and adjusts it if necessary.
                    - Updates the GP model with the features array and confidence value from CI engine: `self.gp_model.update(features_array, token_ci['confidence'])`.
                    - Computes reward for the current token using `reward = self.reward_tracker.compute_reward(token_ci, igain, factors.bayesian_surprise)`.
                    - Appends a dictionary to `self.word_history[token]` containing:
                    - Confidence from CI engine.
                    - Phonetic score.
                    - Features extracted.
                    - Concept ID from concept model.
                    - New entropy value.
                    - Curiosity factors.
                    - Information gain.
                    - Intrinsic reward.

                    - Aligns the factor vector: `factor_array = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))`.
                    - Computes structural score for the token using `_structural_score(token)`.
                    - Adjusts information gain to avoid structural contamination: `igain_adjusted = igain * (1 - struct_score)`.
                    - Updates the meta-learner with the aligned factor vector and adjusted information gain.
                    - Predicts curiosity score based on the factor array and adjusts it for structural scoring.
                    - Appends token analysis details to `token_analyses`:
                    - Token string.
                    - Concept ID.
                    - Entropy value.
                    - Curiosity score.
                    - Factors dictionary.
                    - Information gain.
                    - Intrinsic reward.
                    - GP uncertainty.
                    - Bayesian surprise.

                6. **Update Token Profiles**:
                - Iterates over each token in `tokens` and performs the following steps:
                    - Retrieves embedding for the token from `self.semantic_learner.word_embeddings`, defaulting to a zero vector of dimension `self.semantic_learner.emb_dim` if not found.
                    - Finds corresponding token analysis entry using `next((t for t in token_analyses if t['token'] == token), None)`.
                    - Extracts Bayesian surprise and information gain from the token analysis entry.
                    - Updates the token profile with context tokens, Bayesian surprise, and information gain.

                7. **Sentence-Level Reasoning**:
                - If all tokens are known (`self._all_tokens_known(tokens)` is `True`):
                    - Converts the sentence to an embedding using `sentence_emb = self.semantic_learner.text_to_embedding(text)`.
                    - Processes sentence-level reasoning if the sentence embedding is not `None`.

                8. **Build Expression from Syntax Tree**:
                - Parses tokens into a syntax tree: `syntax_tree = self.syntax_learner.parse(tokens)`.
                - Converts syntax tree to an expression using `expr = syntax_tree.to_expression(self.reasoner.type_hierarchy, self.reasoner.operators)`.
                - If the root of the expression is an operator and not empty:
                    - Stores the expression in the reasoner: `self.reasoner.store_expression(text, expr)`.
                - Detects equations if top node is '=' with two children:
                    - Incorporates equation into schema learner using `self.reasoner.schema_learner.incorporate_equation(left, right)`.
                    - Attempts to solve the equation and builds a readable result string.
                    - If solution found, sets `self.pending_solution` to the result; otherwise, sets it to `None`.
                - If no solution is found and the input text contains a question mark ('?'):
                    - Finds a procedure for the expression if available: `proc = self.reasoner.procedural_learner.find_procedure(expr)`.
                    - Applies the procedure or requests teaching if no procedure is found.

                Variables initialized or used:
                - `ci_report`: Dictionary obtained from evaluating the input text using CI engine.
                - `phon_score`, `phon_diagnostics`: Tuple containing phonetic score and diagnostics for the input text.
                - `diag_explanation`: Explanation of diagnostic report and phonetic score.
                - `tokens`: List of tokens extracted from the input text via `_tokenize(text)`.
                - `concept`: Concept ID returned by `self.concept_model.add_word_observation(token, features)`.
                - `factors`: Curiosity factors computed using `_compute_curiosity_factors(token, token_ci, token_phon, diag_explanation, features_array, global_factor_dict=self._last_factor_dict)`.
                - `new_entropy`: Entropy value from the concept model for each token.
                - `igain`, `igain_adjusted`: Information gain values adjusted for structural contamination.
                - `factor_array`: Aligned factor vector used in meta-learner updates.
                - `struct_score`: Structural score of the token.
                - `reward`: Reward computed using `self.reward_tracker.compute_reward(token_ci, igain, factors.bayesian_surprise)`.
                - `expr`: Expression generated from syntax tree parsing.

                The function does not modify any other variables outside the scope described above.           

            2) #### step 4-6
                The function described performs the following actions:

                1. **Check for Teaching Request**:
                - If `self.chat_mode_active` is `False` and `self.pending_teaching_request` is `True`, it performs the following steps:
                    - Sets `self.pending_teaching_request` to `False`.
                    - Retrieves the pending teaching expression (`expr`) from `self.pending_teaching_expr` and sets `self.pending_teaching_expr` to `None`.
                    - Sets `self.expecting_procedure_steps` to `True`, initializes `self.pending_problem_expr` with `expr`, clears `self.recorded_steps`, and prepares a teach question: `teach_question = f"I don't know how to solve {expr}. Could you teach me step by step?"`.
                    - Prints the teaching request using `print(f"\n [Teaching Request] {teach_question}")` and sends it as a system reply.
                    - Returns a dictionary with specific values:
                    - `"ci_report"`: The current CI report (`ci_report`).
                    - `"phon_score"`: Phonetic score of the input text (`phon_score`).
                    - `"diagnostic"`: Diagnostic explanation (`diag_explanation`).
                    - `"token_analyses"`: List of token analyses (`token_analyses`).
                    - `"curiosity_artifacts"`: A list containing a single string `"Teaching mode activated."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: A floating-point number `0.0`.
                    - `"factor_vector"`: An empty dictionary `{}`.

                2. **Check for Logic Doubt**:
                - If `self.chat_mode_active` is `False`, checks if there is a next doubt generated by the logic inheritance system (`doubt = self.logic_inheritance.get_next_doubt()`):
                    - If `doubt` exists, retrieves the logical rule and description from it.
                    - Constructs a query: `query = f"I have a doubt about some logic I wrote: {description} Is this correct?"`.
                    - Prints the logic doubt using `print(f"\n [Logic Doubt] {query}")` and sends it as a system reply.
                    - Sets `self.expecting_logic_feedback` to `True`, initializes `self.pending_logic_doubt` with `doubt`.
                    - Returns a dictionary with specific values:
                    - `"ci_report"`: The current CI report (`ci_report`).
                    - `"phon_score"`: Phonetic score of the input text (`phon_score`).
                    - `"diagnostic"`: Diagnostic explanation (`diag_explanation`).
                    - `"token_analyses"`: List of token analyses (`token_analyses`).
                    - `"curiosity_artifacts"`: A list containing a single string `"Logic doubt – awaiting confirmation."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: A floating-point number `0.0`.
                    - `"factor_vector"`: An empty dictionary `{}`.

                3. **Check for Unknown Word Detection**:
                - If `self.chat_mode_active` is `False`, identifies unknown tokens not in the semantic learner's word embeddings (`unknown_tokens = [t for t in tokens if t not in self.semantic_learner.word_embeddings]`).
                - Computes the ratio of unknown tokens to total tokens: `unknown_ratio = len(unknown_tokens) / max(len(tokens), 1)`.
                - If `unknown_ratio > 0.5` and there are unknown tokens:
                    - Scores each unknown token using `self.structural_score(t)` and stores them in a list called `scored_unknowns`.
                    - Sorts the scored unknown tokens by their structural score in ascending order.
                    - Selects the first (most semantic) unknown token as the target (`target`) and its score (`struct_score`).
                    - If `struct_score <= 0.65`, prepares a question: `system_reply = f"What does '{target}' mean?"`.
                    - Prints the system reply using `print(f"\n [Knowledge Gap] {system_reply}")` and sends it as a system reply.
                    - Sets `self.last_asked_word` to `target`.
                    - Returns a dictionary with specific values:
                    - `"ci_report"`: The current CI report (`ci_report`).
                    - `"phon_score"`: Phonetic score of the input text (`phon_score`).
                    - `"diagnostic"`: Diagnostic explanation (`diag_explanation`).
                    - `"token_analyses"`: List of token analyses (`token_analyses`).
                    - `"curiosity_artifacts"`: A list containing a single string `"Knowledge gap detected — waiting for definition."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: A floating-point number `0.0`.
                    - `"factor_vector"`: An empty dictionary `{}`.

                4. **Check for Pattern Discovery**:
                - If `self.chat_mode_active` is `False`, iterates over unique tokens in the list of tokens (`tokens`) and checks if a pattern query can be detected for each token using `pattern_query = self.reasoner.detect_pattern_and_query(token)`.
                - If a pattern query exists, sets `self.last_asked_word` to `token`, initializes `self.expecting_pattern_label` to `True`, prints the pattern discovery question using `print(f"\n [Pattern Discovery] {pattern_query}")` and sends it as a system reply.
                - Returns a dictionary with specific values:
                    - `"ci_report"`: The current CI report (`ci_report`).
                    - `"phon_score"`: Phonetic score of the input text (`phon_score`).
                    - `"diagnostic"`: Diagnostic explanation (`diag_explanation`).
                    - `"token_analyses"`: List of token analyses (`token_analyses`).
                    - `"curiosity_artifacts"`: A list containing a single string `"Pattern discovered — awaiting label."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: The episode reward (`self.reward_tracker.episode_reward`).
                    - `"factor_vector"`: An empty dictionary `{}`.

                Variables initialized or used:
                - `doubt`: Dictionary obtained from the logic inheritance system using `get_next_doubt()`.
                - `rule`, `description`: Logical rule and its description.
                - `query`: Constructed query string for asking about a logical doubt.
                - `unknown_tokens`: List of tokens not found in semantic learner's word embeddings.
                - `unknown_ratio`: Ratio of unknown tokens to total tokens.
                - `scored_unknowns`: List containing tuples of unknown tokens and their structural scores.
                - `target`, `struct_score`: Most semantically relevant token and its score from the scored list.
                - `pattern_query`: Pattern query generated for a given token.

                The function does not modify any other variables outside the scope described above.   

            3) #### step 7-8
                The function described performs the following actions:

                1. **Conflict Resolution**:
                - If `self.chat_mode_active` is `False`, checks for unresolved conflicts using `unresolved = self.state_manager.get_unresolved_conflicts()`.
                - If there are unresolved conflicts, prints a message indicating unresolved state conflicts and iterates over the first two conflicts.
                - For each conflict, retrieves labels (`l1` and `l2`) and similarity score (`sim`). Resolves labels using `self.state_manager.resolve_label(l1)` and `self.state_manager.resolve_label(l2)`.
                - Prints details of each conflict, including the shadow state label if available.
                - Prompts user for input to resolve conflicts:
                    - If choice is `"2"`, asks which label to keep (`winner`), resolves the conflict permanently by keeping one label and deleting the other.
                    - If choice is `"3"`, prompts for new labels for both states, renames them, and saves the updated states. Deletes original states and marks conflicts as resolved with a "renamed" resolution label.
                    - If choice is `"4"` and a shadow state exists, asks for a permanent name, saves the merged state, deletes original states, and resolves conflicts by marking them as renamed.
                    - If choice is `"5"`, shows factor vectors for both states, allows editing of factor vectors, parses new input values, updates states with new vectors, and marks conflicts as resolved.

                2. **Behavioral Rule Testing**:
                - If `self.chat_mode_active` is `False`, constructs a context dictionary containing user input (`text`), tokens, token analyses, factor dictionary (`factor_dict`), and current state labels.
                - Retrieves testable behavioral rules using `testable = self.behavioral_learner.get_testable_rules(context)`.
                - If there are testable rules, selects the rule with highest confidence.
                - Depending on the type of rule (`rule.rule_type`):
                    - Constructs a query for identity alias, pronoun mapping, or general pattern-based hypotheses.
                - Sets `asked = True` and updates last tested time for the selected rule.
                - Saves behavioral rules to storage using `self.behavioral_learner.save()`.
                - Initializes `expecting_rule_confirmation = True`, sets `pending_rule_id = rule.rule_id`.
                - Prints a message indicating the rule hypothesis and sends it as a system reply.
                - Returns a dictionary with specific values:
                    - `"ci_report"`: The current CI report (`ci_report`).
                    - `"phon_score"`: Phonetic score of the input text (`phon_score`).
                    - `"diagnostic"`: Diagnostic explanation (`diag_explanation`).
                    - `"token_analyses"`: List of token analyses (`token_analyses`).
                    - `"curiosity_artifacts"`: A list containing a single string `"Behavioral rule hypothesis — awaiting confirmation."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: The episode reward (`self.reward_tracker.episode_reward`).
                    - `"factor_vector"`: Factor dictionary (`factor_dict`).

                Variables initialized or used:
                - `unresolved`: List of unresolved conflicts obtained from `self.state_manager.get_unresolved_conflicts()`.
                - `l1`, `l2`: Labels for conflicting states.
                - `sim`: Similarity score between conflicting states.
                - `shadow_label`: Temporary shadow state label if available.
                - `choice`: User input choice to resolve conflict.
                - `winner`: Label chosen by user to keep in case of option `"2"`.
                - `new_label1`, `new_label2`: New labels for states in case of option `"3"`.
                - `vec1`, `vec2`: State vectors for `l1` and `l2`.
                - `perm_label`: Permanent name for merged state if user chooses option `"4"`.
                - `factor_names`: List of factor names from the control space.
                - `v1_aligned`, `v2_aligned`: Aligned vector representations of states.
                - `context`: Dictionary containing context information for testing behavioral rules.
                - `testable`: List of testable behavioral rules obtained via `self.behavioral_learner.get_testable_rules(context)`.
                - `rule`: Selected rule with highest confidence from testable rules.
                - `query`: Constructed query string to ask user about the selected rule.
                - `alias`, `pronoun`: Aliases and pronouns used in identity alias and pronoun mapping rules.

                The function does not modify any other variables outside the scope described above.  

        11) #### NORMAL RESPONSE GENERATION   
            The function described performs the following actions:

            1. **Factor Computation & State Learning**:
            - Computes factors and state learning using `self.factor_engine.compute(token_analyses, ci_report, phon_score, diag_explanation, self.concept_model, self.meta_learner, self.reward_tracker.total_reward, self.semantic_learner)`.
            - Ensures `factor_dict` is a dictionary: if not, sets it to an empty dictionary `{}`.

            2. **Conscience Influence**:
            - Applies ethical influence from the conscience manager using `self.conscience_manager.apply_ethical_influence(factor_dict)`.
            - Ensures `factor_dict` remains a dictionary: if not, sets it to an empty dictionary `{}`.

            3. **Build Factor Vector**:
            - Constructs a list of factor values for each factor in `self.control_space.all_factors`, using the corresponding value from `factor_dict` or defaulting to `0.5`.
            - Converts this list into a NumPy array (`factor_vector`) and stores it in `self._last_factor_vector`.
            - Stores the latest factor dictionary in `self._last_factor_dict`.

            4. **Metadata Construction**:
            - Constructs metadata dictionary containing input text, timestamp, CI confidence from `ci_report`, and phonetic score.

            5. **State Learning Gate**:
            - If `not self.chat_mode_active` and the number of word embeddings in `self.semantic_learner.word_embeddings` exceeds 20, calls `_handle_state_learning(factor_vector, metadata)`.

            6. **Semantic Relation Discovery**:
            - Detects semantic relations between tokens using `_detect_semantic_relations(tokens, token_analyses)`.

            7. **Artifact Generation**:
            - Generates artifacts using `_generate_artifacts(tokens, token_analyses, ci_report, phon_score, diag_explanation, factor_dict)`.

            8. **System Reply Generation**:
            - Generates a system reply using `self.action_generator.generate_reply(text, {'token_analyses': token_analyses, 'factor_vector': factor_dict})`.
            - If there is a pending solution (`self.pending_solution`), prepends it to the system reply and resets `self.pending_solution`.

            9. **System Reply Storage**:
            - Stores the generated system reply in `_last_action`.

            10. **Top-Down State Prediction**:
                - If `self.structured_world_model` exists and `current_state_labels` are not empty, resolves the current state label.
                - Predicts the next state using the structured world model (`predicted_next = self.structured_world_model.predict_next_state(current_state, action)`).
                - Stores the prediction in `_expected_next_state`.

            11. **Last Asked Word Identification**:
                - Searches for a quoted word in `system_reply` and sets `self.last_asked_word` to the match if found.

            12. **System Reply Printing**:
                - Prints the system reply using `print(f" Herebot> {system_reply}")`.

            13. **Feedback Request Generation**:
                - If not in chat mode, expecting response feedback, expecting other types of feedback, and no pending conflicts:
                - Generates a template ID based on the hash of `system_reply`.
                - Registers the response template with the reasoner.
                - Checks if feedback should be asked for using `_should_ask_feedback(tokens, factor_dict, template_id)`.
                - If so, generates a feedback question and prints it.
                - Creates a `FeedbackEntry` object with relevant details.
                - Sets `self.pending_feedback_context` to this entry and sets `self.expecting_response_feedback = True`.

            14. **Word-Label Conflict Detection**:
                - Checks for word-label conflicts if not in chat mode, expecting response feedback, expecting other types of feedback, or handling a pending conflict.
                - Iterates over tokens and detects conflicts using `reasoner.word_state.detect_conflicts(token)`.
                - If a conflict is found, constructs a question about the correct label for the token and prints it.
                - Sets `self.expecting_word_conflict = True` and stores the conflict details in `_pending_conflict`, returning early with specific feedback:
                - Returns a dictionary containing:
                    - `"ci_report"`: The CI report.
                    - `"phon_score"`: Phonetic score of the input text.
                    - `"diagnostic"`: Diagnostic explanation.
                    - `"token_analyses"`: List of token analyses.
                    - `"curiosity_artifacts"`: A list containing a single string `"Word conflict – awaiting clarification."`.
                    - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                    - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                    - `"episode_reward"`: The episode reward (`self.reward_tracker.episode_reward`).
                    - `"factor_vector"`: Factor dictionary.

            15. **Chat Mode Reward Multiplication**:
                - Computes chat mode reward multiplier using `_compute_chat_reward_multiplier(factor_dict)`.
                - Multiplies episode reward by this factor if in chat mode.
                - Prints a message indicating the original and boosted rewards.

            16. **Episode Storage**:
                - Adds an episode to episodic memory with user input, system response, factor vector, intrinsic reward, and metadata containing CI confidence.

            17. **Chat History Update**:
                - Appends the current interaction (user input and system reply) to `self.chat_history`.
                - Ensures chat history does not exceed 20 interactions by removing the oldest entry if necessary.

            18. **State Saving**:
                - Calls `_save_state()` to save the internal state.

            19. **Periodic Reconfiguration Check**:
                - Checks if the interaction counter is a multiple of `self.reconfigure_interval` and calls `_maybe_reconfigure()` if so.

            20. **Tracker Recording**:
                - Records current object state using `self.tracker.record(self)`.

            21. **Return Values**:
                - Returns a dictionary containing:
                - `"ci_report"`: The CI report.
                - `"phon_score"`: Phonetic score of the input text.
                - `"diagnostic"`: Diagnostic explanation.
                - `"token_analyses"`: List of token analyses.
                - `"curiosity_artifacts"`: Generated artifacts from `_generate_artifacts`.
                - `"meta_weights"`: Weights from the meta-learner obtained via `self.meta_learner.get_weights()`.
                - `"total_intrinsic_reward"`: Total intrinsic reward tracked by `self.reward_tracker.total_reward`.
                - `"episode_reward"`: The episode reward (`self.reward_tracker.episode_reward`).
                - `"factor_vector"`: Factor dictionary.

            Variables initialized or used:
            - `factor_dict`: Dictionary of computed factors.
            - `factor_vector_list`, `factor_vector`: List and NumPy array representing the factor vector.
            - `current_state`: Current state label resolved from `self.current_state_labels`.
            - `predicted_next`: Predicted next state from structured world model.
            - `system_reply`: Generated system reply.
            - `template_id`: Template ID based on hash of `system_reply`.
            - `entry`: Feedback entry object containing context and response details.
            - `safe_factor_dict`: Safe factor dictionary ensuring it is a dictionary.
            - `multiplier`, `original_reward`: Reward multiplier and original episode reward for chat mode.

            The function does not modify any other variables outside the scope described above.                      
        
---

35) ### _tokenize
    The function `_tokenize` takes a single string argument `text` and returns a list of tokenized strings. Here are the specific actions it performs:

    1. **Imports**:
    - The module `re` is imported.

    2. **Base Tokenization**:
    - A regular expression (`r"[a-zA-Z']+|\d+\.?\d*|[+\-*/=<>%^()]"`) is used to find all matches in the input text.
    - Each match is converted to lowercase and filtered based on length, ensuring each token has at least 2 characters. The result is stored in `raw`.

    3. **State-based Min-Length Gate**:
    - If the attribute `semantic_learner` exists and has a non-empty `word_embeddings`, it calculates the vocabulary size.
    - Based on this vocabulary size, it sets a minimum length for tokens:
        - 1 if `vocab_size < 50`
        - 2 otherwise
    - Filters `raw` based on this minimum length to create `tokens`.

    4. **Self-developing: Learn Useful Token Patterns**:
    - If the attribute `semantic_learner` exists, it separates known tokens (`known`) from unknown tokens (`unknown`).
    - Known tokens are those that exist in `semantic_learner.word_embeddings`.
    - If there are no known tokens and there are unknown tokens, all elements of `raw` with a minimum length of 1 are added to `tokens`.

    5. **Track Tokenization Stats**:
    - Initializes `_tokenizer_stats` if it does not already exist, setting `'total'` to 0 and an empty list for `'known_ratio_history'`.
    - Increments the total count in `_tokenizer_stats['total']`.
    - If tokens are present, calculates the ratio of known tokens to total tokens and appends this ratio to `_tokenizer_stats['known_ratio_history']`.
    - Ensures that `known_ratio_history` does not exceed 200 elements by removing the oldest element if necessary.
    
    6. **Return**:
    - Returns `tokens` if it is non-empty, otherwise returns `raw`.

    Variables initialized and used within the function include:

    - `text`: Input string to be tokenized.
    - `re`: Regular expression module.
    - `vocab_size`: Integer representing the size of vocabulary in `semantic_learner.word_embeddings`.
    - `min_len`: Integer determining minimum length of tokens.
    - `tokens`: List of filtered tokens after applying min_length filter.
    - `known`: List of known tokens (present in `semantic_learner.word_embeddings`).
    - `unknown`: List of unknown tokens (not present in `semantic_learner.word_embeddings`).
    - `_tokenizer_stats`: Dictionary tracking tokenization statistics including total number of tokenizations and history of known token ratios.

---

36) ### _generate_artifacts
    The function `_generate_artifacts` takes several arguments and generates a list of string artifacts for reporting purposes. Here are the specific actions it performs:

    1. **Initialization**:
    - A list `artifacts` is initialized to store the generated artifacts.

    2. **Meta-Curiosity Analysis Header**:
    - Appends a header with version information: " [Meta-Curiosity Analysis v0.5 – Adaptive Control Space]".

    3. **Input Tokens**:
    - Appends the input tokens in a formatted string: " Input tokens: {tokens}".

    4. **Confidence and Stability Metrics**:
    - Retrieves the CI confidence and stability values from `ci_report` using default values of 0.5 if not present.
    - Appends these metrics in a formatted string: " CI confidence: {confidence:.3f} | Stability: {stability:.3f}".

    5. **Phonotactic Score**:
    - Appends the phonotactic score in a formatted string: " Phonotactic score: {phon_score:.3f}".

    6. **Diagnostic Summary**:
    - Retrieves and appends the diagnostic summary from `diag` with default value 'N/A': " Diagnostic: {summary}".

    7. **Meta-Learner Weights**:
    - Retrieves weights from `meta_learner`.
    - Sorts these weights by absolute values in descending order.
    - Appends a header for meta-learner weights and the top 5 weights in formatted strings: 
        - " Current curiosity priorities (learned):"
        - Each weight is appended as " {name}: {w:.3f}".

    8. **Intrinsic Reward**:
    - Retrieves total and episode intrinsic rewards from `reward_tracker`.
    - Appends these metrics in formatted strings: 
        - " Total intrinsic reward (lifetime): {total_reward:.3f}"
        - " Episode intrinsic reward: {episode_reward:.3f}"

    9. **Current Control-Signal State**:
    - Iterates over base and dynamic factors from `control_space` if present in `factor_dict`.
    - Appends each factor's state in formatted strings: 
        - For base factors: " {k}: {factor_dict[k]:.3f}"
        - For dynamic factors: " {k}: {factor_dict[k]:.3f} (dynamic)"

    10. **State Similarity Check**:
        - Constructs a vector from the current factor values.
        - Uses `state_manager` to find similar states with a threshold of 0.85.
        - If a similar state is found, appends it in a formatted string: " State recognized as: '{similar_label}'".

    11. **Token-Level Curiosity Analysis**:
        - Iterates over each token analysis (`ta`) from `token_analyses`.
        - For each token analysis, retrieves and formats the token's curiosity score, entropy, factors, Bayesian surprise, and GP uncertainty.
        - Appends these metrics in formatted strings: 
        - " {token}: curiosity={score:.3f} (entropy={entropy:.3f})"
        - First 6 factors are appended as " {name}:{val:.2f}"
        - Bayesian surprise and GP uncertainty are appended separately.

    12. **Generate Hypothesis**:
        - Uses `reasoner` to generate a hypothesis for the token.
        - Appends this hypothesis in a formatted string: " 💭 {hypothesis}"

    13. **Plan Query (if necessary)**:
        - If the curiosity score is greater than 0.4, uses `reasoner` to plan a query based on uncertainty.
        - Appends this query in a formatted string: " ❓ {query}"

    14. **Total Concepts**:
        - Retrieves and appends the total number of discovered latent concepts from `concept_model`: 
        - " Total latent concepts discovered: {total_concepts}"

    15. **Most Globally Uncertain Word**:
        - If `word_history` is not empty, finds the most uncertain word based on entropy.
        - Appends this information in a formatted string if available: 
        - " Most globally uncertain word: '{most_uncertain_word}' (entropy={entropy:.3f})"

    16. **Return**:
    - Returns the list of artifacts.

    Variables used within the function include:

    - `tokens`: List of input tokens.
    - `token_analyses`: List of dictionaries containing analysis results for each token.
    - `ci_report`: Dictionary containing confidence and stability metrics.
    - `phon_score`: Float representing the phonotactic score.
    - `diag`: Dictionary containing diagnostic information.
    - `factor_dict`: Dictionary mapping factor names to their values.
    - `self.meta_learner`: Meta-learner object from which weights are retrieved.
    - `weights`: Dictionary of current meta-learner weights.
    - `sorted_weights`: List of sorted weight tuples by absolute value.
    - `self.reward_tracker`: Reward tracker object providing lifetime and episode rewards.
    - `total_reward`: Float representing the total intrinsic reward (lifetime).
    - `episode_reward`: Float representing the episode intrinsic reward.
    - `self.control_space`: Control space object containing base and dynamic factors.
    - `base_factors`: List of base factor names from control space.
    - `dynamic_factors`: List of dynamic factor names from control space.
    - `vec`: Numpy array constructed from current factor values for similarity check.
    - `threshold`: Float representing the state similarity threshold (default 0.85).
    - `similar_label`: String label if a similar state is found.
    - `factors.factor_dict`: Dictionary mapping curiosity factors to their values for each token analysis.
    - `gp_unc`: Float representing GP uncertainty from token analysis.
    - `bayes_surp`: Float representing Bayesian surprise from token analysis.
    - `self.reasoner`: Reasoner object used to generate hypotheses and plan queries.
    - `hypothesis`: String hypothesis generated by the reasoner.
    - `query`: String query planned based on uncertainty if necessary.
    - `total_concepts`: Integer representing the total number of discovered latent concepts.
    - `most_uncertain_word`: String representing the most globally uncertain word based on entropy.

---

37) ### _handle_math_sentence
    The function `_handle_math_sentence` processes a detected math sentence and performs specific actions based on the presence of certain components. Here are the precise details:

    1. **Input Parameters**:
    - `math_result`: A dictionary containing information about the math sentence.
    - `tokens`: A list of tokens from which the math sentence was derived.

    2. **Extract Information**:
    - Extracts values for `left`, `right`, `op` (operator), and `expected` from `math_result`.
    - Optionally extracts a sentence string from `math_result`.

    3. **Validation Check**:
    - If any of `left`, `right`, or `op` are `None`, the function returns immediately without further processing.

    4. **Evaluate with Known Operators**:
    - Retrieves known operators from `reasoner.operators`.
    - Checks if the operator (`op`) is among the known operators.
        - If it is, attempts to evaluate the expression using the operator on `left` and `right`.
        - If an expected result is provided and matches the computed result within a tolerance of \(10^{-9}\):
            - Prints a confirmation message: "🔢 [Math] {sentence} → {left} {op} {right} = {computed} ✓".
        - Otherwise, prints the evaluated expression: "🔢 [Math] {sentence} → {left} {op} {right} = {computed}".
        - If an exception occurs during evaluation, it is caught and ignored.

    5. **Infer Unknown Operator**:
    - If the operator (`op`) is not among known operators and an expected result is provided:
        - Attempts to infer the unknown operator using `reasoner.pattern_observer.infer_operator(left, right, expected)`.
        - If inference is successful, teaches this operator to the pattern observer: 
        - Calls `reasoner.pattern_observer.teach_math_operator(op, inferred)`.
        - Prints a learned message: "🔢 [Math] Learned: '{op}' means '{inferred}' (from {left} {op} {right} = {expected})".

    Variables used within the function include:

    - `math_result`: Dictionary containing information about the math sentence.
    - `tokens`: List of tokens from which the math sentence was derived.
    - `left`: Value extracted from `math_result` representing the left operand.
    - `right`: Value extracted from `math_result` representing the right operand.
    - `op`: Operator extracted from `math_result`.
    - `expected`: Expected result value extracted from `math_result`.
    - `sentence`: Sentence string optionally extracted from `math_result`.
    - `known_ops`: Dictionary of known operators retrieved from `reasoner.operators`.
    - `computed`: Result of evaluating the expression using the known operator.
    - `inferred`: Inferred operator if the original operator is unknown.

    The function performs deterministic actions based on the provided inputs.

---

38) ### perform_state_introspection
    The function `perform_state_introspection` performs two main tasks: detecting conflicts among saved states and creating shadow resolutions for those conflicts, and offline learning from episodic memory. Here are the precise details:

    1. **Print Header**:
    - Prints a header indicating deep state introspection is starting: " [Deep State Introspection] Examining saved states for conflicts...".

    2. **Conflict Detection Threshold**:
    - Retrieves the conflict detection threshold from `config_manager` with a default value of 0.85.

    3. **Refresh Conflicts**:
    - Calls `self.state_manager.refresh_conflicts(threshold)` to refresh and detect state conflicts using the specified threshold.

    4. **Check Unresolved Conflicts**:
    - Retrieves unresolved conflicts from `state_manager.get_unresolved_conflicts()`.
    - If there are no unresolved conflicts, prints a message: "    No unresolved conflicts found."
    - Otherwise, for each of the first three unresolved conflicts (specifically the top 3):
        - Prints conflict details including labels and similarity score.
        - Lists words that led to each conflicting state if available.
        - Creates a shadow resolution using `self.state_manager.create_shadow_resolution(conflict)`.
        - Applies the shadow label to the conflict using `self.state_manager.apply_shadow_to_conflict(conflict, shadow_label)`.
        - Prints confirmation of shadow creation: "      → Created shadow state '{shadow_label}'. Will ask user at next interaction."
    - Saves conflicts after processing.

    5. **Offline Learning from Episodic Memory**:
    - Prints a header indicating offline learning is starting: "    Consolidating knowledge from recent episodes...".
    - Retrieves the last 20 episodes from `self.episodic_memory.replay_sequence()`.
    - For each episode in the last 20:
        - Extracts user input and tokenizes it using `_tokenize(user_input)`.
        - Observes context for each token in the tokens list with other tokens as context.
        - Adds the sentence to `self.syntax_learner`.

    6. **Completion Message**:
    - Prints a message indicating offline consolidation is complete: "    Offline consolidation complete.\n".

    Variables used within the function include:

    - `threshold`: Float representing the conflict detection threshold (default 0.85).
    - `unresolved`: List of unresolved conflicts from `state_manager.get_unresolved_conflicts()`.
    - `l1`, `l2`: Strings representing labels of conflicting states.
    - `sim`: Float representing similarity score between conflicting states.
    - `words1`, `words2`: Lists of words that led to each conflicting state.
    - `shadow_label`: String representing the shadow resolution label created for a conflict.

    The function performs deterministic actions based on specific inputs.

---

39) ### _get_resource_status
    The function `_get_resource_status` performs the following actions:

    - Initializes and returns five variables: `cpu_ok`, `mem_ok`, `throttle_factor`, `cpu_percent`, and `mem_percent`.
    - Attempts to retrieve CPU usage percentage (`cpu_percent`) using `psutil.cpu_percent(interval=0.1)`. This operation measures the CPU usage over a 0.1-second interval.
    - Retrieves memory information using `psutil.virtual_memory()` and extracts the memory usage percentage (`mem.percent`), storing it in `mem_percent`.
    - If an exception is encountered during any of these operations, it prints "psutil failed [throttle mode]" to indicate that there was a failure with the psutil library. It then returns `(True, True, 0.8, 0.0, 0.0)`, indicating that CPU and memory usage are assumed to be fine (below thresholds), but sets `throttle_factor` to 0.8.
    - If no exception is encountered, it calculates:
        - `cpu_ok`: a boolean value set to `True` if the `cpu_percent` is below 80%.
        - `mem_ok`: a boolean value set to `True` if the `mem_percent` is below 90%.
    - Sets `throttle_factor` based on whether `cpu_ok` and `mem_ok` are both `True`. If either `cpu_ok` or `mem_ok` is `False`, then `throttle_factor` is set to 0.3; otherwise, it is set to 1.0.

    The variables returned by the function are:
    - `cpu_ok`: Boolean indicating whether CPU usage is below 80%.
    - `mem_ok`: Boolean indicating whether memory usage is below 90%.
    - `throttle_factor`: A float in the range [0.3, 1.0] used to adjust workload based on resource strain (0.3 if resources are strained, 1.0 otherwise).
    - `cpu_percent`: Float representing CPU usage percentage measured over a 0.1-second interval.
    - `mem_percent`: Float representing memory usage percentage.

    Specific values returned:
    - If psutil fails: `(True, True, 0.8, 0.0, 0.0)`.
    - Otherwise, based on resource status: `(cpu_ok, mem_ok, throttle_factor, cpu_percent, mem_percent)` where `throttle_factor` is either 1.0 or 0.3 depending on the thresholds for CPU and memory usage.

---

40) ### _compute_chat_reward_multiplier
    The function `_compute_chat_reward_multiplier` performs the following actions:

    - Initializes four variables: `curiosity`, `trust`, `threat`, and `comprehension`. These variables are retrieved from a dictionary `factor_dict` using `.get()` method with default values of 0.5 for `curiosity`, 0.5 for `trust`, 0.0 for `threat`, and 0.5 for `comprehension`.
    - Computes the reward multiplier based on the formula:
    $$ \ \text{multiplier} = 1.0 + (0.5 \times \text{curiosity}) + (0.3 \times \text{trust}) - (0.4 \times \text{threat}) + (0.2 \times \text{comprehension}) \ $$
    - Clips the computed `multiplier` to ensure it falls within the range [0.5, 2.0] using `np.clip`.

    The function returns a float value that is the dynamic reward multiplier based on the emotional state, clamped between 0.5 and 2.0.

    Specific values used in the formula:
    - Curiosity scaled by 0.5.
    - Trust scaled by 0.3.
    - Threat scaled by -0.4 (negative scaling).
    - Comprehension scaled by 0.2.

    The final multiplier is clamped to ensure it does not fall below 0.5 or exceed 2.0.

---

41) ### world_model_dreaming
    The function `world_model_dreaming` performs the following actions:

    1. **Initial Checks:**
    - If `self.config_manager.get('idle_reasoner.enable_world_model_dreaming', True)` is `False`, it returns immediately.
    
    2. **Resource Status Check:**
    - Calls `_get_resource_status()` to get CPU and memory status, returning specific variables:
        - `cpu_ok`: Boolean indicating if CPU usage is below 80%.
        - `mem_ok`: Boolean indicating if memory usage is below 90%.
        - `throttle`: Float in the range [0.3, 1.0] used to adjust workload based on resource strain (0.3 if resources are strained, 1.0 otherwise).
        - `cpu_p`: CPU usage percentage.
        - `mem_p`: Memory usage percentage.
    - If either `cpu_ok` or `mem_ok` is `False`, it prints a message and returns.

    3. **Print Message:**
    - Prints "🌙 [World Model Dreaming] Simulating multi-term synthetic conversations..."

    4. **Build n-gram Model from Episodic Memory:**
    - Initializes `n` based on configuration with default value 3.
    - Creates a `defaultdict(Counter)` named `ngram_model`.
    - Retrieves the last 200 episodes from `self.episodic_memory.episodes` and processes each episode's user input:
        - Tokenizes the input using `_tokenize(sent)`.
        - Skips episodes where token length is less than `n` or if not all tokens are known.
        - For valid sequences, updates `ngram_model` with context and next word counts.

    5. **Check n-gram Data Availability:**
    - If `ngram_model` is empty, prints a message and returns.

    6. **Generate Synthetic Sentences:**
    - Initializes `base_num` based on configuration with default value 5.
    - Computes `num_syn = max(1, int(base_num * throttle))`.
    - Retrieves known vocabulary from `self.semantic_learner.word_embeddings.keys()` and skips if empty.
    
    7. **Sentence Generation Loop:**
    - For each synthetic sentence generation:
        - Chooses a random context of length `n-1` from `ngram_model.keys()`.
        - Generates up to 12 words, appending the next word based on probability distribution derived from `ngram_model[ctx]`.
        - If all words are known, simulates an interaction using `_simulate_interaction(sentence)`.

    8. **Print Final Message:**
    - Prints a summary message indicating the number of synthetic sentences processed, throttle factor, and n-gram order used.

    **Variables Used:**
    - `cpu_ok`: Boolean from `_get_resource_status()`.
    - `mem_ok`: Boolean from `_get_resource_status()`.
    - `throttle`: Float in range [0.3, 1.0] from `_get_resource_status()`.
    - `cpu_p`: CPU usage percentage from `_get_resource_status()`.
    - `mem_p`: Memory usage percentage from `_get_resource_status()`.
    - `n`: Integer (default 3) based on configuration.
    - `episodes`: List of recent episodes from episodic memory.
    - `sent`: User input string from each episode.
    - `tokens`: Tokenized user input list.
    - `context`: Tuple of words representing context in the n-gram model.
    - `next_word`: Next word in the sequence based on the context.
    - `ngram_model`: Dictionary where keys are tuples of contexts and values are Counters for next words.
    - `base_num`: Integer (default 5) from configuration.
    - `num_syn`: Number of synthetic sentences to generate, clamped between 1 and `int(base_num * throttle)`.
    - `vocab_known`: List of known vocabulary words.
    - `contexts`: List of keys in `ngram_model`.
    - `start_ctx`: Randomly selected context tuple for starting a sentence.
    - `words`: List of words generated from the n-gram model.
    - `sentence`: Generated synthetic sentence.

    **Summary:**
    This function generates synthetic sentences based on an n-gram model derived from recent user inputs, ensuring that only known words are used to maintain plausibility. The number of generated sentences is adjusted based on the current resource status and configuration settings.

42) ### _simulate_interaction
    The function `_simulate_interaction` performs the following actions:

    1. **Tokenization:**
    - Initializes `tokens`, a list of tokens obtained by tokenizing the input text using `_tokenize(text)`.

    2. **Observation and Contextual Evaluation:**
    - Calls `self.reasoner.observe_sentence(text, tokens)`.
    - For each token in `tokens`:
        - Evaluates the token's context independence (CI) score using `self.ci_engine.evaluate(token)`, storing it in `token_ci`.
        - Scores the phonetic representation of the token using `self.phon_engine.score_text(token)`, storing it in `token_phon`.
        - Generates a diagnostic explanation for the token using `self.diag_engine.explain(token_ci, token_phon)`.

    3. **Feature Extraction and Concept Model Update:**
    - Extracts features from the token using `_extract_features(token, token_ci, token_phon, diag_explanation)`, storing them in `features`.
    - Converts the extracted features to an array format using `_features_to_array(features)`, storing it in `features_array`.
    - Adds a word observation for the token in the concept model using `self.concept_model.add_word_observation(token, features)`, updating the concept of the token.
    - Computes new entropy for the token using `self.concept_model.entropy_word(token)` and stores it in `new_entropy`.
    - Calculates information gain (`igain`) for the token using `_information_gain(token, new_entropy)`.

    4. **Gaussian Process Model Update:**
    - Updates the Gaussian process model with the features array and CI confidence score using `self.gp_model.update(features_array, token_ci['confidence'])`.

    5. **Recording History:**
    - Appends a dictionary to `self.word_history[token]` containing:
        - `ci_confidence`: Confidence of context independence.
        - `phon_score`: Phonetic score.
        - `features`: Extracted features.
        - `concept`: Concept model update result.
        - `entropy`: New entropy value.
        - `information_gain`: Information gain.

    6. **Semantic Learner Update:**
    - Constructs a list `context` of tokens excluding the current token.
    - If `chat_mode_active` is true:
        - Updates the semantic learner three times with a higher learning rate by temporarily scaling down the word count denominator and restoring it afterward.
    - Otherwise, updates the semantic learner once using `self.semantic_learner.observe_context(token, context)`.

    7. **Episode Recording:**
    - Adds a dummy episode to the episodic memory containing:
        - `user_input`: The input text.
        - `system_response`: "[synthetic]".
        - `factor_vector`: An empty dictionary.
        - `intrinsic_reward`: 0.0.
        - `metadata`: A dictionary with key-value pair `'synthetic': True`.

    **Variables Used and Initialized:**
    - `tokens`: List of tokens obtained from tokenizing the input text.
    - `token_ci`: Dictionary containing context independence evaluation results for each token.
    - `token_phon`: Phonetic score for each token.
    - `diag_explanation`: Diagnostic explanation generated by `self.diag_engine.explain(token_ci, token_phon)`.
    - `features`: Features extracted from the token.
    - `features_array`: Array representation of the features.
    - `concept`: Concept model update result for the token.
    - `new_entropy`: New entropy value after updating the concept model.
    - `igain`: Information gain calculated for the token.
    - `context`: List of tokens excluding the current token.

    **Summary:**
    The function processes a synthetic input text by tokenizing it, evaluating context independence and phonetic scores, extracting features, updating the concept model and Gaussian process model, recording history, and updating the semantic learner. It also records a dummy episode in the episodic memory to indicate that the interaction was synthetic.

---

43) ### active_memory_rehearsal
    The function `active_memory_rehearsal` performs the following actions:

    1. **Initial Check:**
    - If `self.config_manager.get('idle_reasoner.enable_active_memory_rehearsal', True)` is `False`, it returns immediately.

    2. **Resource Status Check:**
    - Calls `_get_resource_status()` to get CPU and memory status, returning specific variables:
        - `cpu_ok`: Boolean indicating if CPU usage is below 80%.
        - `mem_ok`: Boolean indicating if memory usage is below 90%.
        - `throttle`: Float in the range [0.3, 1.0] used to adjust workload based on resource strain (0.3 if resources are strained, 1.0 otherwise).
        - `cpu_p`: CPU usage percentage.
        - `mem_p`: Memory usage percentage.
    - If either `cpu_ok` or `mem_ok` is `False`, it prints a message and returns.

    3. **Print Message:**
    - Prints "🔄 [Active Memory Rehearsal] Prioritizing episodes..."

    4. **Episode Processing:**
    - Retrieves the list of episodes from `self.episodic_memory.episodes`.
    - If the number of episodes is less than 5, it returns.

    5. **Compute Priority Scores for Each Episode:**
    - Initializes an empty list `scores` to store priority scores.
    - For each episode in the list:
        - Computes Bayesian surprise for words in the user input using `self.concept_model.bayesian_surprise(t)`.
        - Calculates rarity as the inverse of the frequency of the state vector if available and not empty.
        - Aligns factor vectors using `_align_factor_vector(np.array(list(ep['factor_vector'].values())))`.
        - Computes cosine similarity with stored states, then calculates rarity as `1.0 - np.mean(sims)`.
        - Calculates identity drift using `self.identity_manager.detect_drift(vec)` if available.
        - Adds an unresolved hypothesis bonus of 0.2 if any hypotheses in the reasoner have not been asked, otherwise adds 0.0.
        - Computes a priority score for each episode as:
        $$ \
        \text{score} = (\text{surprise} \times 0.4) + (\text{rarity} \times 0.3) + (\text{drift} \times 0.2) + (\text{hyp\_bonus} \times 0.1)
        \ $$
        - Appends the computed score to `scores`.

    6. **Select Top Episodes:**
    - Initializes `base_k` based on configuration with default value 20.
    - Computes `k = max(1, int(base_k * throttle))`.
    - Selects the top `k` episodes using `np.argsort(scores)[-k:]`, storing their indices in `top_idx`.

    7. **Print Final Message:**
    - Prints a message indicating the number of high-priority episodes to replay and the throttle factor used.

    8. **Replay Selected Episodes:**
    - For each index in `top_idx`:
        - Retrieves the corresponding episode.
        - Tokenizes the user input using `_tokenize(user_input)`.
        - Updates semantic learner with context for each token:
        - Constructs a list of tokens excluding the current one and observes context using `self.semantic_learner.observe_context(t, context)`.
        - Adds the sentence to syntax learner using `self.syntax_learner.add_sentence(user_input)`.

    **Variables Used and Initialized:**
    - `episodes`: List of episodes from `self.episodic_memory.episodes`.
    - `surprise`: Cumulative Bayesian surprise for words in user input.
    - `rarity`: Rarity score based on state vector frequency.
    - `drift`: Identity drift value if available.
    - `hyp_bonus`: Unresolved hypothesis bonus (0.2 or 0.0).
    - `scores`: List of computed priority scores for each episode.
    - `base_k`: Integer (default 20) from configuration.
    - `k`: Number of top episodes to replay, adjusted by throttle factor.
    - `top_idx`: Indices of the highest-priority episodes.

    **Mathematical Formulas:**
    The function uses a weighted sum formula to compute the priority score for each episode:
    $$ \
    \text{score} = (\text{surprise} \times 0.4) + (\text{rarity} \times 0.3) + (\text{drift} \times 0.2) + (\text{hyp\_bonus} \times 0.1)
    \ $$

    **Summary:**
    The function prioritizes episodes based on a weighted sum of Bayesian surprise, rarity, identity drift, and unresolved hypothesis bonus. It then selects the top `k` episodes (adjusted by throttle factor) to replay, updating semantic and syntax learners with the processed user inputs from these episodes.

---

44) ### curiosity_landscape_optimization
    The function `curiosity_landscape_optimization` performs the following actions:

    1. **Initial Check:**
    - If `self.config_manager.get('idle_reasoner.enable_curiosity_landscape_optimization', True)` is `False`, it returns immediately.

    2. **Resource Status Check:**
    - Calls `_get_resource_status()` to get CPU and memory status, returning specific variables:
        - `cpu_ok`: Boolean indicating if CPU usage is below 80%.
        - `mem_ok`: Boolean indicating if memory usage is below 90%.
        - `throttle`: Float in the range [0.3, 1.0] used to adjust workload based on resource strain (0.3 if resources are strained, 1.0 otherwise).
        - `cpu_p`: CPU usage percentage.
        - `mem_p`: Memory usage percentage.
    - If either `cpu_ok` or `mem_ok` is `False`, it prints a message and returns.

    3. **Print Message:**
    - Prints "📈 [Curiosity Landscape Optimization] Analyzing factor correlations..."

    4. **Collect Recent Factor Vectors and Information Gains:**
    - Initializes empty lists `factor_vals` and `ig_vals`.
    - Iterates through the last 20 entries in each word history to collect curiosity factors (`curiosity_factors`) and information gain (`information_gain`).
    - Converts collected factor vectors to arrays using `as_array(self.meta_learner.factor_names)`.
    - Appends these values to `factor_vals` and `ig_vals`.

    5. **Check Data Sufficiency:**
    - If the number of collected factor vectors is less than 10, it returns.

    6. **Form Factor Matrix:**
    - Converts `factor_vals` to a NumPy array named `factor_matrix`.

    7. **Remove Constant Columns:**
    - Computes standard deviations for each column in `factor_matrix`.
    - Identifies columns with standard deviation greater than $\{1 \times 10^{-6}\} $ and keeps them.
    - Removes constant columns from `factor_matrix`.

    8. **Check Column Sufficiency:**
    - If the number of valid columns is less than 2, it returns.

    9. **Correlation Analysis (If Throttled Below Threshold):**
    - If `throttle` is below 0.8, skips expensive correlation matrix and merging.
    - Otherwise, computes the correlation matrix using `np.corrcoef(factor_matrix.T)`.

    10. **Detect Highly Correlated Pairs:**
        - Initializes an empty list `redundant`.
        - Iterates through pairs of columns in the valid indices to find highly correlated pairs with a threshold of 0.9.
        - Appends pairs that exceed the correlation threshold to `redundant`.

    11. **Merge Weak Factors (Limited Per Run):**
        - Merges up to two weak factors per run by averaging their weights based on variance.
        - Updates the merged factor's weight and sets the redundant factor's weight to 0.0.
        - Prints a message indicating which factors were merged.

    12. **Spawn New Factor from Residual Variance:**
        - Converts information gain values to an array named `ig_array`.
        - Aligns global mu values with the shape of `factor_matrix` using `align_vector(self.meta_learner.mu, factor_matrix.shape[1])`.
        - Predicts information gains using a linear model (`factor_matrix @ mu`) and calculates residuals.
        - If the variance of residuals is greater than 0.01:
        - Spawns a new curiosity factor named `residual_{int(time.time())}` with an initial weight of 0.2.
        - Defines a placeholder function `residual_compute` to return a small random value.
        - Adds this new factor using `self.meta_learner.add_factor`.
        - Prints a message indicating the creation of a new curiosity factor.

    **Variables Used and Initialized:**
    - `factor_vals`: List of collected factor vectors.
    - `ig_vals`: List of information gain values.
    - `factor_matrix`: NumPy array of valid factor vectors.
    - `stds`: Standard deviations for each column in `factor_matrix`.
    - `valid_cols`: Boolean array indicating columns with non-zero variance.
    - `corr`: Correlation matrix computed from `factor_matrix.T`.
    - `redundant`: List of highly correlated factor pairs.
    - `thresh`: Correlation threshold (0.9).
    - `valid_indices`: Indices of valid columns in `factor_matrix`.
    - `mu`: Global mu values aligned with the shape of `factor_matrix`.
    - `ig_array`: Information gain array converted from `ig_vals`.
    - `pred_ig`: Predicted information gains using linear model.
    - `residual`: Residuals calculated as `ig_array - pred_ig`.

    **Mathematical Formulas:**
    The function uses a weighted average to merge factors and computes residuals for predicting new curiosity factors:
    $$ \
    \text{new\_w} = \frac{\text{w}_i + \text{w}_j}{2.0}
    \ $$
    Residuals are calculated as:
    $$ \
    \text{residual} = \text{ig\_array} - (\text{factor\_matrix} @ \mu)
    \ $$

    **Summary:**
    The function collects recent factor vectors and information gains, removes constant columns, computes correlation matrices, identifies redundant factors based on high correlations, merges weak factors, and spawns new curiosity factors from unexplained variance in residuals. The process is throttled to avoid resource strain.

---

45) ### concept_graph_compression
    The function `concept_graph_compression` performs the following actions:

    1. **Initial Check:**
    - If `self.config_manager.get('idle_reasoner.enable_concept_graph_compression', True)` is `False`, it returns immediately.

    2. **Resource Status Check:**
    - Calls `_get_resource_status()` to get CPU and memory status, returning specific variables:
        - `cpu_ok`: Boolean indicating if CPU usage is below 80%.
        - `mem_ok`: Boolean indicating if memory usage is below 90%.
        - `throttle`: Float in the range [0.3, 1.0] used to adjust workload based on resource strain (0.3 if resources are strained, 1.0 otherwise).
        - `cpu_p`: CPU usage percentage.
        - `mem_p`: Memory usage percentage.
    - If either `cpu_ok` or `mem_ok` is `False`, it prints a message and returns.

    3. **Print Message:**
    - Prints "🗂️ [Concept Graph Compression] Creating higher-order concepts..."

    4. **Cluster Concepts Based on Word Membership:**
    - Initializes an empty dictionary `concept_words` to store words associated with each concept.
    - Iterates through the word-concept counts in `self.concept_model.word_concept_counts`.
        - For each word and its corresponding count distribution, assigns the word to its most probable concept using `max(cnts, key=cnts.get)`.
        - Appends the word to the list of words associated with that concept in `concept_words`.

    5. **Create Meta-Concepts:**
    - Iterates through each concept and its associated words in `concept_words`.
    - For concepts with three or more associated words:
        - Creates a meta-concept label as `f"meta_{'_'.join(words[:2])}"`, where `words` are the first two words of the list.
        - Stores the meta-concept as a relation for each word in the reasoning engine using `self.reasoner.relations[w].append(("category", meta_label, 0.8))`.
        - Creates a definition string for the meta-concept: `f"a group of words: {', '.join(words)}"`.
        - Computes an average embedding for the member embeddings:
        - Retrieves word embeddings from `self.semantic_learner.word_embeddings` and stores them in `emb_list`.
        - If `emb_list` is not empty, computes the average embedding using `np.mean(emb_list, axis=0)`, storing it in `avg_emb`.
        - Stores the definition and dummy average embedding for the meta-concept using `self.reasoner.store_definition(meta_label, meta_def, avg_emb)`.
        - Prints a message indicating the creation of the meta-concept.

    **Variables Used and Initialized:**
    - `concept_words`: Dictionary mapping concepts to lists of associated words.
    - `words`: List of words associated with each concept.
    - `meta_label`: String representing the label for the meta-concept.
    - `emb_list`: List of word embeddings from `self.semantic_learner.word_embeddings`.
    - `avg_emb`: Average embedding vector computed from member embeddings.

    Mathematical Formulas:
    The function computes the average embedding as follows:
    \[
    \text{avg\_emb} = \frac{\sum_{i=1}^{n} \text{emb}_i}{n}
    \]
    where \( n \) is the number of word embeddings in `emb_list`.

    **Summary:**
    The function clusters words based on their most probable concept and creates higher-order meta-concepts for concepts with three or more associated words. It stores these meta-concepts as relations and definitions in the reasoning engine, using an average embedding of member words' embeddings. The process is resource-aware and skips execution if CPU or memory usage exceeds specified thresholds.

---

46) ### autonomous_hypothesis_planning
    The function `autonomous_hypothesis_planning` performs the following actions:

    1. **Initialization and Configuration Check**:
        - The function starts by checking if the autonomous hypothesis planning is enabled through the configuration managed by `self.config_manager.get('idle_reasoner.enable_autonomous_hypothesis_planning', True)`. If it is not enabled, the function returns immediately.
    
    2. **Resource Status Assessment**:
        - It calls `_get_resource_status()` to check CPU and memory status, returning a tuple `(cpu_ok, mem_ok, throttle, cpu_p, mem_p)` where `cpu_ok` and `mem_ok` are boolean values indicating if the system's CPU and memory resources are adequate (`True`) or inadequate (`False`). The variables `throttle`, `cpu_p`, and `mem_p` represent the resource utilization thresholds: `throttle` (a float between 0 and 1), `cpu_p` (CPU usage as a percentage, e.g., 50.0%), and `mem_p` (memory usage as a percentage, e.g., 75.0%).
        - If either `cpu_ok` or `mem_ok` is `False`, the function prints a message indicating that it is skipping autonomous hypothesis planning due to resource constraints and returns.

    3. **Printing Start Message**:
        - It prints "🔬 [Autonomous Hypothesis Planning] Generating word and sentence experiments..."

    4. **Generating Word-Level Candidates**:
        - The function initializes an empty list `candidates` for storing candidate words.
        - For each word in `self.concept_model.word_concept_counts`, it calculates the entropy of that word using `self.concept_model.entropy_word(word)`. If the entropy is greater than 0.5 and if the word has not been defined by `self.reasoner.definitions`, an information gain estimate (`ig_estimate`) is calculated as `entropy - 0.1` and appended to `candidates` with type "definition".
        - For each hypothesis in the first 5,000 hypotheses of `self.reasoner.hypotheses` (if `_max_hypotheses` attribute is set), if it's a synonym hypothesis that has not been asked yet (`hyp['asked'] == False`), it appends the word with an information gain estimate of `0.8` to `candidates` with type "synonym".

    5. **Generating Sentence-Level Candidates**:
        - It initializes an empty list `sentence_candidates`.
        - For each episode in the last 100 episodes of `self.episodic_memory.episodes`, it extracts user input (`sent`) and tokenizes it into `tokens`. If there are fewer than three tokens, or if any token is not known (checked by `_all_tokens_known(tokens)`), it continues to the next episode.
        - For each token in `tokens` that has a history stored in `self.word_history`, it adds its novelty factor to `avg_novelty` and increments `count`.
        - If `count > 0`, it normalizes `avg_novelty` by dividing by `count`. If the normalized average novelty is greater than 0.5, an information gain estimate (`ig_estimate`) is calculated as `avg_novelty * 0.5` and appended to `sentence_candidates` with type "sentence_meaning".

    6. **Combining and Sorting Candidates**:
        - It combines both word-level and sentence-level candidates into a single list `all_candidates`.
        - The list is sorted in descending order by the information gain estimate (`ig_estimate`) using a lambda function.
        
    7. **Generating Research Queue**:
        - An empty list `self.research_queue` is initialized to store research queries.
        - It sets `max_queue` as the maximum number of items to add, calculated as `int(10 * throttle)` but with a minimum value of 1.
        - For each item in the top `max_queue` elements of `all_candidates`, it generates an appropriate query based on the type ("definition", "synonym", or "sentence_meaning").
            - If the type is "definition", the query is generated as: `"What does '{item}' mean?"`.
            - If the type is "synonym", a hypothesis matching the word and type in `self.reasoner.hypotheses` is found, and if present, the query is generated as: `"Is '{item}' similar to '{hyp['candidate']}'?"`.
            - If the type is "sentence_meaning", the query is generated as: `"What does the sentence '{item}' mean?"`.
        - Each valid query is added to `self.research_queue` with a dictionary containing `'query'`, `'target'`, `'ig'`, `'type'`, and `'timestamp'`.

    8. **Printing Final Message**:
        - Finally, it prints the number of high-priority queries in the research queue along with the throttle value.

    In summary, this function identifies words and sentences that are potentially informative based on their entropy/novelty scores and generates corresponding research queries to be processed later by a reasoner or planner.

---

47) ### identity_stabilization
    The function `identity_stabilization` performs the following steps:

    1. **Configuration Check**:
    - It checks if the configuration parameter `'idle_reasoner.enable_identity_stabilization'` is set to `True`. If not, the function returns immediately.

    2. **Resource Status Check**:
    - It retrieves CPU and memory status using `_get_resource_status()`, which returns five values: `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
        - If either `cpu_ok` or `mem_ok` is `False`, it prints a message indicating that the identity stabilization will be skipped due to resource constraints.
        - The CPU and memory usage percentages (`cpu_p` and `mem_p`) are printed with one decimal place precision.

    3. **Consolidation of Self-Model**:
    - If both `cpu_ok` and `mem_ok` are `True`, it prints a message indicating that identity stabilization is being consolidated.
    - It then collects recent identity vectors from the last 50 episodes in `self.episodic_memory.episodes`.
        - For each episode, if the intrinsic reward (`ep.get('intrinsic_reward', 0)`) is greater than `0.1` and contains a `'factor_vector'`, it extracts the factor vector.
        - The extracted vector is aligned to the embedding dimension using `_align_factor_vector()` and `_align_vector_to_emb_dim()`.
    - If fewer than 5 identity vectors are collected, the function returns early.

    4. **Resource Management**:
    - It keeps only the last 100 identity vectors in `self.identity_history`.

    5. **Clustering (Optional)**:
    - If the system is not heavily throttled (`throttle < 0.5`), it skips clustering and computes a simple moving average of the last 10 identity vectors as the attractor.
        \[
        attractor = \frac{1}{N} \sum_{i=0}^{N-1} identity\_history[-10 + i]
        \]
    - Otherwise, if the system is heavily throttled, it performs k-means clustering on the `identity_history` with \( k = \min(\text{config_manager.get('idle_reasoner.identity_cluster_k', 3)}, \text{len(identity_matrix)}) \).
        - The largest cluster centroid (based on frequency of occurrence) is selected as the attractor.
    - If an exception occurs during clustering, it returns early.

    6. **Identity Adjustment**:
    - It adjusts `self.identity_manager.identity` towards the attractor using a weighted average with a smoothing factor (\( \alpha = 0.1 \)).
        \[
        new\_identity = (1 - \alpha) \cdot current\_identity + \alpha \cdot attractor
        \]
    - The identity vector is normalized if its norm is greater than zero.
        \[
        self.identity_manager.identity = \frac{new\_identity}{\|new\_identity\|}
        \]
    - The confidence in the self-model is incremented by `0.01`, capped at `1.0`.
    - The updated identity and confidence are saved.

    7. **Phase Transition Detection**:
    - It detects a phase transition by comparing the current identity to the attractor using `self.identity_manager.detect_drift(attractor)`.
        \[
        drift = \|current\_identity - attractor\|
        \]
    - If the drift is greater than `0.4`, it prints a message indicating that an identity phase transition has been detected with the specific drift value.

    The variables used and initialized are:
    - `cpu_ok`: A boolean representing if CPU resources are sufficient.
    - `mem_ok`: A boolean representing if memory resources are sufficient.
    - `throttle`: A float between 0.0 and 1.0 indicating system throttling level.
    - `cpu_p`: A float representing the current CPU usage percentage.
    - `mem_p`: A float representing the current memory usage percentage.
    - `identity_history`: A list of identity vectors collected from episodes.
    - `alpha`: A constant smoothing factor (\( \alpha = 0.1 \)).
    - `k`: The number of clusters for k-means, capped at 3 or the length of `identity_matrix`.

    Mathematical formulas used:
    - **Moving Average**:
        \[
        attractor = \frac{1}{N} \sum_{i=0}^{N-1} identity\_history[-10 + i]
        \]

    - **Identity Adjustment**:
        \[
        new\_identity = (1 - \alpha) \cdot current\_identity + \alpha \cdot attractor
        \]

    - **Normalization**:
        \[
        self.identity_manager.identity = \frac{new\_identity}{\|new\_identity\|}
        \]

---

48) ### cross_modal_alignment
    The function `cross_modal_alignment` performs cross-modal representation alignment using contrastive learning. Here is a precise description of its functionality:

    Variables Initialized and Used:
    - **cpu_ok**: Boolean indicating if CPU usage is within acceptable limits.
    - **mem_ok**: Boolean indicating if memory usage is within acceptable limits.
    - **throttle**: Float representing the throttle level, which indicates resource availability.
    - **cpu_p**: Float representing the percentage of CPU utilization.
    - **mem_p**: Float representing the percentage of memory utilization.

    **Functionality:**
    1. The function checks if cross-modal alignment is enabled via `self.config_manager.get('idle_reasoner.enable_cross_modal_alignment', True)`. If not, it returns immediately without performing any further operations.
    2. It then calls `_get_resource_status()` to obtain CPU usage (`cpu_p`), memory usage (`mem_p`), and the throttle level (`throttle`). The function checks if both `cpu_ok` and `mem_ok` are true (indicating that resources are sufficient).
    3. If either `cpu_ok` or `mem_ok` is false, it prints a message indicating insufficient resources and returns without performing cross-modal alignment.
    4. Next, the function checks if `throttle < 0.5`, which would indicate low resource availability. If true, it skips cross-modal alignment by printing an appropriate message and returning.
    5. The function then proceeds to align semantic, state, and episodic embeddings by gathering positive pairs `(state_vector, text_embedding)` from the last 100 episodes in `self.episodic_memory.episodes` if certain conditions are met (specifically, if `'factor_vector'` exists and is not empty).
    6. For each episode meeting these conditions:
    - It converts the factor vector to a state vector using `_align_factor_vector()` and `_align_vector_to_emb_dim()`.
    - Extracts the user input text from the episode.
    - Converts the text into an embedding representation using `self.semantic_learner.text_to_embedding(text)`, appending valid pairs `(state_vec, text_emb)` to `pos_pairs`.
    7. If fewer than 5 positive pairs are found, the function returns without further processing.
    8. For each pair in `pos_pairs`:
    - Computes the cosine similarity between `state_vec` and `text_emb` as follows:
        \[
        \text{sim_pos} = \frac{\mathbf{v}_\text{state} \cdot \mathbf{v}_\text{text}}{\|\mathbf{v}_\text{state}\|_2 \|\mathbf{v}_\text{text}\|_2 + 10^{-12}}
        \]
    - Samples a negative text embedding from other positive pairs, ensuring it is not the same as `text_emb`.
    9. For each pair:
    - If a valid negative pair `(state_vec, neg_emb)` is found, computes the cosine similarity between `state_vec` and `neg_emb` similarly.
    - Computes an InfoNCE-like loss to maximize the probability of positive pairs over negative ones:
        \[
        \text{prob_pos} = \frac{\exp(\text{sim_pos})}{\exp(\text{sim_pos}) + \exp(\text{sim_neg})}
        \]
    - Updates the text embedding towards `state_vec` using gradient ascent on log probability:
        \[
        \mathbf{v}_\text{text}^\prime = \mathbf{v}_\text{text} + lr \cdot (1 - \text{prob_pos}) \cdot (\mathbf{v}_\text{state} - \mathbf{v}_\text{text})
        \]
    - Adjusts the word embeddings in `self.semantic_learner.word_embeddings` based on the updated text embedding, normalizing updates by the number of tokens.
    10. Prints a message indicating that contrastive alignment is completed with the current throttle level.

    **Mathematical Formulas:**
    - **Cosine Similarity**:
    \[
    \text{similarity} = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2 + 10^{-12}}
    \]
    - **InfoNCE Loss Probability**:
    \[
    P(\text{positive}) = \frac{\exp(\text{sim_pos})}{\exp(\text{sim_pos}) + \exp(\text{sim_neg})}
    \]

    **Summary of Operations:**
    - The function performs cross-modal alignment by pulling positive pairs closer and pushing negative pairs apart using a contrastive loss.
    - It adjusts embeddings based on the similarity scores computed between state vectors and text embeddings, updating word embeddings accordingly.

---

49) ### exploration_policy_distillation
    The function `exploration_policy_distillation` performs exploration policy distillation by training a logistic regression classifier to determine when to ask questions based on system responses. Here is a precise description of its functionality:

    **Variables Initialized and Used:**
    - **cpu_ok**: Boolean indicating if CPU usage is within acceptable limits.
    - **mem_ok**: Boolean indicating if memory usage is within acceptable limits.
    - **throttle**: Float representing the throttle level, which indicates resource availability.
    - **cpu_p**: Float representing the percentage of CPU utilization.
    - **mem_p**: Float representing the percentage of memory utilization.

    **Functionality:**
    1. The function checks if exploration policy distillation is enabled via `self.config_manager.get('idle_reasoner.enable_exploration_policy_distillation', True)`. If not, it returns immediately without performing any further operations.
    2. It then calls `_get_resource_status()` to obtain CPU usage (`cpu_p`), memory usage (`mem_p`), and the throttle level (`throttle`). The function checks if both `cpu_ok` and `mem_ok` are true (indicating that resources are sufficient).
    3. If either `cpu_ok` or `mem_ok` is false, it prints a message indicating insufficient resources and returns without performing exploration policy distillation.
    4. Next, the function checks if `throttle < 0.4`, which would indicate low resource availability. If true, it skips policy training by printing an appropriate message and returning.
    5. The function proceeds to collect decision points from the last 200 episodes in `self.episodic_memory.episodes`:
    - For each episode containing a `'factor_vector'` that is not empty and a `'system_response'`, convert the factor vector into a state vector using `_align_factor_vector()` and normalize it with `_normalize_vector()`.
    - Append the normalized state vector to `X` and classify whether the system response contains a question mark (`?`) by appending 1 if true, otherwise 0 to `y`.
    6. If fewer than 20 samples are collected or if there are fewer than two unique labels in `y`, the function returns without further processing.
    7. Convert `X` and `y` into numpy arrays.
    8. Train a logistic regression classifier using scipy's minimize function:
    - Define a sigmoid function to compute probabilities:
        \[
        \sigma(z) = \frac{1}{1 + e^{-z}}
        \]
    - Define the loss function for logistic regression (negative log-likelihood):
        \[
        L(w, X, y) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
        \]
    - Initialize the weight vector `w0` as an array of zeros with length equal to the number of features plus one for the intercept.
    - Use a simpler optimizer (`L-BFGS-B`) if `throttle < 0.7`, otherwise use `BFGS`.
    9. Store the trained model weights in `self.policy_model` and print a message indicating successful policy distillation, including the intercept value and optimization method used.

    **Mathematical Formulas:**
    - **Sigmoid Function**:
    \[
    \sigma(z) = \frac{1}{1 + e^{-z}}
    \]
    - **Logistic Loss (Negative Log-Likelihood)**:
    \[
    L(w, X, y) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]
    \]

    **Summary of Operations:**
    - The function collects decision points from the last 200 episodes, classifying system responses as questions or not.
    - It trains a logistic regression classifier using scipy's minimize function to determine when to ask questions based on state vectors.
    - The trained model weights are stored in `self.policy_model`, and a message is printed indicating successful policy distillation.

---

50) ### cognitive_drive_evolution

    The function `cognitive_drive_evolution` performs cognitive drive evolution by removing weak dynamic factors and spawning new drives based on certain criteria. Here is a precise description of its functionality:

    Variables Initialized and Used:
    - **cpu_ok**: Boolean indicating if CPU usage is within acceptable limits.
    - **mem_ok**: Boolean indicating if memory usage is within acceptable limits.
    - **throttle**: Float representing the throttle level, which indicates resource availability.
    - **cpu_p**: Float representing the percentage of CPU utilization.
    - **mem_p**: Float representing the percentage of memory utilization.

    **Functionality:**
    1. The function checks if cognitive drive evolution is enabled via `self.config_manager.get('idle_reasoner.enable_cognitive_drive_evolution', True)`. If not, it returns immediately without performing any further operations.
    2. It then calls `_get_resource_status()` to obtain CPU usage (`cpu_p`), memory usage (`mem_p`), and the throttle level (`throttle`). The function checks if both `cpu_ok` and `mem_ok` are true (indicating that resources are sufficient).
    3. If either `cpu_ok` or `mem_ok` is false, it prints a message indicating insufficient resources and returns without performing cognitive drive evolution.
    4. The function proceeds to remove weak dynamic factors where the absolute value of their mean (`mu`) is less than a threshold (`weak_thresh = 0.001`). Specifically:
    - It iterates over each dynamic factor in `self.meta_learner.dynamic_factor_names`.
    - For each factor, it finds its index in `self.meta_learner.factor_names` and checks if the absolute value of the corresponding mean (`mu`) is less than `weak_thresh`.
    - If true, the factor name is added to a list `to_remove`.
    5. It then removes each factor in `to_remove` one by one:
    - For each factor name in `to_remove`, it checks if the name still exists in `self.meta_learner.dynamic_factor_names`. If not, it continues to the next iteration.
    - Otherwise, it finds the index of the factor in `self.meta_learner.factor_names`.
    - It removes the factor from `self.meta_learner.dynamic_factor_names` and updates arrays (`mu`, `lambda_`, `alpha`, `beta`) by keeping only elements at indices different from the current index.
    - If the factor name is present in `self.meta_learner.dynamic_compute_functions` or `self.meta_learner.factor_metadata`, it removes those entries as well.
    6. For each removed drive, a message indicating successful removal is printed.
    7. The function then spawns a new cognitive drive with a probability proportional to the throttle level (`spawn_prob = 0.2 * throttle`). Specifically:
    - A random number `np.random.rand()` between 0 and 1 is generated.
    - If this number is less than `spawn_prob`, a new dynamic factor name is created as `new_name` using the current time in milliseconds.
    - It defines a new drive computation function `new_drive_compute` which returns a random value between 0 and 0.5.
    - The new drive is added to the meta-learner with an initial mean (`initial_mu = 0.1`) and the defined computation function.
    8. If a new drive is spawned, it prints a message indicating successful spawning along with the throttle level.

    Mathematical Formulas:
    No specific mathematical formulas are used in this function beyond basic arithmetic operations such as absolute value and random number generation.

    Summary of Operations:
    - The function removes weak dynamic factors whose mean values (`mu`) have an absolute value less than `0.001`.
    - It spawns a new cognitive drive with probability `0.2 * throttle` if a random number is less than this probability.
    - Each removed and spawned drive is appropriately logged with print statements.

    **Detailed Steps:**
    1. **Remove Weak Dynamic Factors**:
    - Iterate over each dynamic factor name in `self.meta_learner.dynamic_factor_names`.
    - For each factor, find its index in `self.meta_learner.factor_names` if the absolute value of corresponding mean (`mu`) is less than `0.001`.
    - If true, add the factor name to `to_remove`.

    2. **Remove Each Factor**:
    - Iterate over each factor name in `to_remove`.
    - Remove the factor from `self.meta_learner.dynamic_factor_names` and update arrays (`mu`, `lambda_`, `alpha`, `beta`) by keeping only elements at indices different from the current index.
    - If present, remove the factor from `self.meta_learner.dynamic_compute_functions` and `self.meta_learner.factor_metadata`.

    3. **Spawn New Drive**:
    - Generate a random number between 0 and 1.
    - If less than `spawn_prob`, create a new dynamic factor name using current time in milliseconds.
    - Define the computation function for the new drive as returning a random value between 0 and 0.5.
    - Add the new drive to the meta-learner with an initial mean of `0.1`.
    - Print message indicating successful spawning.

    **Example Output:**
    ```plaintext
    Removed weak drive: 'drive_123456789'
    Spawned new cognitive drive: 'drive_123456789' (throttle=0.60)
    ```

    This function ensures that the cognitive drives are updated dynamically based on resource availability and current factor strengths, maintaining a balance between existing and newly created drives.

---

51) ### counterfactual_reasoning

    The function `counterfactual_reasoning` performs counterfactual reasoning by simulating alternative actions in recent episodes and updating the weights of relevant factors based on information gain estimates. Here is a precise description of its functionality:

    **Variables Initialized and Used:**
    - **cpu_ok**: Boolean indicating if CPU usage is within acceptable limits.
    - **mem_ok**: Boolean indicating if memory usage is within acceptable limits.
    - **throttle**: Float representing the throttle level, which indicates resource availability.
    - **cpu_p**: Float representing the percentage of CPU utilization.
    - **mem_p**: Float representing the percentage of memory utilization.

    **Functionality:**
    1. The function checks if counterfactual reasoning is enabled via `self.config_manager.get('idle_reasoner.enable_counterfactual_reasoning', True)`. If not, it returns immediately without performing any further operations.
    2. It then calls `_get_resource_status()` to obtain CPU usage (`cpu_p`), memory usage (`mem_p`), and the throttle level (`throttle`). The function checks if both `cpu_ok` and `mem_ok` are true (indicating that resources are sufficient).
    3. If either `cpu_ok` or `mem_ok` is false, it prints a message indicating insufficient resources and returns without performing counterfactual reasoning.
    4. The function proceeds to simulate "what if I asked X instead?" for each recent episode:
    - It selects the last 20 episodes from `self.episodic_memory.episodes`.
    - A maximum of `max_eps` episodes are processed, where `max_eps = max(1, int(len(recent) * throttle))`.
    5. For each selected episode in the range `[0, max_eps)`:
    - If the episode does not contain a `'user_input'`, it continues to the next iteration.
    - Tokens of the user input are obtained using `_tokenize()`. If no tokens are found, it continues to the next iteration.
    - A random token from the user input is chosen as an alternative word (`alt_word`).
    6. The function estimates the information gain (IG) if the alternative word had been asked about:
    - `entropy = self.concept_model.entropy_word(alt_word)` computes the entropy of the selected word using the concept model.
    - A crude IG estimate is computed as: \(\text{alt\_ig} = \text{entropy} - 0.1\), assuming the answer reduces entropy to 0.1.
    7. The function compares the estimated IG with the actual IG from the episode (if available):
    - If `actual_ig` is not present in the episode, it defaults to a crude proxy of `0`.
    - If \(\text{alt\_ig} > \text{actual\_ig} + 0.1\), indicating that asking about the alternative word would have been more informative:
        - It checks if the episode contains a `'factor_vector'` and if it is not empty.
        - Converts the factor vector into an aligned state vector using `_align_factor_vector()`.
        - Normalizes the state vector if its norm is greater than 0: \(\text{vec} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}\).
        - Boosts the meta-learner weights for factors with high activation:
        - For each factor, it increases its weight by `0.01 * vec[i]`.
    8. After processing the episodes, a message is printed indicating that counterfactual updates have been applied, specifying the number of processed episodes and throttle level.

    **Mathematical Formulas:**
    - **Entropy Reduction**:
    \[
    \text{alt\_ig} = H(\text{entropy}) - 0.1
    \]
    where \(H\) is the entropy function.

    Summary of Operations:
    - The function processes up to `max_eps` episodes from the last 20, simulating alternative user inputs.
    - For each episode, it computes an estimated information gain for a randomly chosen word and compares it with the actual information gain.
    - If the estimated IG is higher by more than 0.1, it updates the weights of factors in the meta-learner based on their activation levels.

    **Detailed Steps:**
    1. **Check Configuration**:
    - Ensure counterfactual reasoning is enabled via configuration settings.

    2. **Resource Check**:
    - Obtain CPU and memory usage status.
    - If resources are insufficient, print a message and return.

    3. **Recent Episodes Selection**:
    - Select the last 20 episodes from `self.episodic_memory.episodes`.
    - Process up to `max_eps` episodes where \( \text{max\_eps} = \max(1, \lfloor \text{len(recent)} \times \text{throttle} \rfloor) \).

    4. **Process Each Episode**:
    - Ensure the episode contains a `'user_input'`.
    - Tokenize the user input.
    - Choose a random token as an alternative word.

    5. **Estimate Information Gain**:
    - Compute entropy of the selected word using `self.concept_model.entropy_word(alt_word)`.
    - Estimate IG: \(\text{alt\_ig} = \text{entropy} - 0.1\).

    6. **Compare Actual and Estimated IG**:
    - Retrieve actual IG from episode or use a crude proxy of `0`.
    - If estimated IG is higher by more than 0.1, update factor weights.

    7. **Update Meta-Learner Weights**:
    - Align and normalize the factor vector.
    - Increase weights for factors with high activation: \(\text{meta\_learner.mu}[i] += 0.01 \times \text{vec}[i]\).

    8. **Print Summary**:
    - Print a message indicating successful counterfactual updates, specifying processed episodes and throttle level.

    **Example Output:**
    ```plaintext
    [Resource Aware] Skipping Counterfactual Reasoning (CPU:75.2%  MEM:40.3%).
    ```

    This function ensures that the system considers alternative actions in recent episodes to improve its reasoning capabilities by updating factor weights based on information gain estimates.

---

52) ### semantic_self_talk

    The `semantic_self_talk` function performs advanced self-talk by generating new sentences based on a trigram model derived from the user's previous inputs stored in episodic memory. The function has the following behavior:

    1. **Resource Check**: It first checks if semantic self-talk is enabled via configuration (`idle_reasoner.enable_semantic_self_talk`). If not, it returns immediately without further execution.
    2. **CPU and Memory Status**:
    - Calls `_get_resource_status()` to retrieve CPU usage (`cpu_p`), memory usage (`mem_p`), and a throttle value used for adjusting the number of generated sentences.
    - Skips self-talk if either CPU or memory is not okay (usage below 100%).
    3. **Trigram Model Construction**:
    - Constructs a trigram model from the last 200 episodes in episodic memory, ensuring only known words are included.
    - For each episode, it tokenizes the user input and builds the trigram by counting transitions between word pairs.

    4. **Sentence Generation**:
    - Determines the number of sentences to generate based on `throttle` (at least 1 sentence; maximum determined as \( \text{num\_sentences} = \max(1, \lfloor 5 \times \text{throttle}\rfloor) \)).
    - For each generated sentence:
        - With a probability of 70%, it performs a trigram walk by randomly selecting starting contexts and extending the sequence based on transition probabilities until the length is between 5 to 12 words.
        - Otherwise, with a probability of 30%, it creates a crossover sentence by combining prefixes from one episode and suffixes from another.

    5. **Learning**:
    - Adds generated sentences to `syntax_learner`.
    - Observes each token in the context of other tokens for semantic learning via `semantic_learner`.

    6. **Output**:
    - Prints information about skipped self-talk due to resource constraints.
    - Reports details on trigram data if available and proceeds with sentence generation.
    - Logs generated sentences (up to 70 characters) and total count.

    **Variables Initialized/Used**
    - `cpu_ok`: Boolean indicating CPU status.
    - `mem_ok`: Boolean indicating memory status.
    - `throttle`: Float value representing the resource constraint factor for adjusting sentence count.
    - `trigram`: Dictionary of Counters, where keys are word pairs (contexts) and values are dictionaries mapping next words to their frequencies.
    - `episodes`: List of the last 200 episodes from episodic memory.
    - `num_sentences`: Integer determining how many sentences should be generated based on throttle value.

    **Mathematical Formulas**
    1. Sentence count determination:
    $$ \
    \text{num\_sentences} = \max(1, \lfloor 5 \times \text{throttle}\rfloor)
    \ $$
    2. Trigram walk length (randomly chosen between 5 and 12):
    $$\
    \text{walk\_length} \in [5, 12)
    \ $$

    **Precise Steps:**
    - The function initializes `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p` by calling `_get_resource_status()`.
    - If resources are not sufficient (either CPU or memory below threshold), it prints a message indicating the skip.
    - Constructs trigram model from recent episodes, ensuring only known tokens are considered.
    - Generates sentences based on trigram walk (70% probability) or crossover (30% probability).
    - Adds generated sentences to `syntax_learner` and observes context for semantic learning.
    - Prints details of generated sentences up to 70 characters and total count.

--- 

53) ### state_space_topology_learning

    The function `state_space_topology_learning` performs state-space topology learning (cognitive manifold discovery) under specific conditions. Here are the details:

    **Function Overview:**
    - **Objective**: To discover attractors and macro-states by clustering similar state vectors.
    - **Conditions for Execution**:
    - The configuration option `idle_reasoner.enable_state_space_topology_learning` must be set to `True`.
    - CPU usage (`cpu_p`) and memory usage (`mem_p`) should each be below a threshold (default thresholds not explicitly defined in the provided code).
    - Throttle value (`throttle`) must be greater than or equal to 0.6.

    **Initialization and Variables Used:**
    - **Inputs**:
    - `self.config_manager`: Manages configuration settings.
    - `self.state_manager.states.items()`: A dictionary where keys are labels and values are state vectors (numpy arrays).

    - **Internal Variables Initialized**:
    - `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, `mem_p`: Resource status variables obtained from `_get_resource_status`.
    - `all_vectors`: List of normalized state vectors.
    - `labels`: Labels corresponding to each vector in `all_vectors`.
    - `n_clusters`: Number of clusters, determined by the configuration option `idle_reasoner.topology_n_clusters` or the number of available vectors, whichever is smaller.

    **Steps Performed:**
    1. **Resource Check**:
    - Checks if state-space topology learning is enabled.
    - Verifies CPU and memory usage against thresholds (values not specified).
        ```python
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        ```
    - Prints a message skipping the process if either CPU or memory is above threshold.

    2. **Throttle Check**:
    - Skips learning if `throttle` is less than 0.6.
    ```python
    if throttle < 0.6:
        print(f"   [Resource Aware] Skipping topology learning (throttle={throttle:.2f}).")
        return
    ```

    3. **Data Collection**:
    - Collects state vectors from `self.state_manager.states` and normalizes them.
        ```python
        all_vectors = []
        labels = []
        for label, vec in self.state_manager.states.items():
            all_vectors.append(self._normalize_vector(vec))
            labels.append(label)
        ```
    - Converts the list of vectors into a numpy array if there are at least four vectors.

    4. **Clustering**:
    - Performs k-means clustering using `kmeans2` with `n_clusters` number of clusters.
        ```python
        n_clusters = min(self.config_manager.get('idle_reasoner.topology_n_clusters', 5), len(all_vectors))
        try:
            centroids, assignments = kmeans2(all_vectors, n_clusters, minit='points')
        except:
            return
        ```

    5. **Macro-State Creation**:
    - For each centroid (cluster center):
        - Identifies the states belonging to that cluster.
        - If at least two states belong to a cluster, it creates a macro-state from those vectors and saves it as a shadow state in `self.state_manager`.
        ```python
        for i, cent in enumerate(centroids):
            members = [labels[j] for j, a in enumerate(assignments) if a == i]
            if len(members) >= 2:
                macro_name = f"macro_{i}_{int(time.time())}"
                self.state_manager.save_state(macro_name, cent, metadata={'members': members}, is_shadow=True)
                print(f"   Created macro-state '{macro_name}' from {len(members)} states.")
        ```
        - Assigns the centroids to `self.state_manifold`.

    **Summary:**
    - The function clusters state vectors into a specified number of groups.
    - Creates and saves macro-states (shadow states) for each cluster with at least two member states.
    - Updates `self.state_manifold` with the centroid vectors.

    This process ensures that resources are efficiently utilized by skipping learning when resource constraints or throttling conditions are not met.

54) ### knowledge_gap_frontier_detection
    The function `knowledge_gap_frontier_detection` performs the following steps:

    1. It checks a configuration setting (`idle_reasoner.enable_knowledge_gap_frontier`) using `self.config_manager.get('idle_reasoner.enable_knowledge_gap_frontier', True)`. If this setting is `False`, the function immediately returns without performing any further operations.

    2. The function initializes and uses the following variables:
    - `cpu_ok`: A boolean indicating whether CPU status is acceptable.
    - `mem_ok`: A boolean indicating whether memory status is acceptable.
    - `throttle`: A floating-point number representing a resource utilization adjustment factor, typically between 0 and 1.
    - `cpu_p`: A floating-point percentage value representing the current CPU usage.
    - `mem_p`: A floating-point percentage value representing the current memory usage.

    3. If either `cpu_ok` or `mem_ok` is `False`, it prints a message indicating that Knowledge Gap Frontier processing is skipped due to insufficient resources and returns without further execution.

    4. It initializes an empty list called `frontiers`.

    5. For each word in `self.concept_model.word_concept_counts`:
    - If the word does not have a definition in `self.reasoner.definitions`, it calculates the entropy of the word using `self.concept_model.entropy_word(word)`.
    - If this entropy is greater than 0.3, it computes an information gain (IG) value as \( \text{IG} = \text{entropy} - 0.1 \) and appends a tuple `(word, ig, "definition")` to the `frontiers` list.

    6. For each unresolved hypothesis in `self.reasoner.hypotheses`:
    - If the hypothesis has not been asked about (`hyp.get('asked', False)` is `False`) and contains a word, it appends a tuple `(word, 0.8, "hypothesis")` to the `frontiers` list.

    7. It sorts the `frontiers` list in descending order by IG value using the key function `lambda x: -x[1]`.

    8. It initializes an empty list called `self.research_queue`.

    9. It calculates the maximum number of items (`max_items`) to keep in the research queue based on the formula:
    $$ \
    \text{max\_items} = \max(1, 20 \times \text{throttle})
    \ $$
    where `throttle` is a value between 0 and 1.

    10. For each of the top `max_items` elements in the sorted `frontiers` list:
        - If the type of the element is "definition", it constructs a query string as `"What does '{word}' mean?"`.
        - Otherwise, if the type is "hypothesis", it constructs a query string as `"Could you clarify '{word}'?"`.
        - It appends a dictionary to `self.research_queue` containing:
        - `'query'`: The constructed query string.
        - `'target'`: The word related to the query.
        - `'ig'`: The information gain value associated with the word.
        - `'type'`: The type of the entry ("definition" or "hypothesis").
        - `'timestamp'`: The current Unix timestamp.

    11. It prints a message indicating that the research queue has been updated, including the number of items in `self.research_queue` and the throttle value.

    The function does not modify any global variables other than those initialized within it (`cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, `mem_p`, `frontiers`, `self.research_queue`).

---

55) ### social_cognition_simulation

    The function `social_cognition_simulation` performs the following steps:

    1. It checks a configuration setting (`idle_reasoner.enable_social_cognition_simulation`) using `self.config_manager.get('idle_reasoner.enable_social_cognition_simulation', True)`. If this setting is `False`, the function immediately returns without performing any further operations.

    2. The function initializes and uses the following variables:
    - `cpu_ok`: A boolean indicating whether CPU status is acceptable.
    - `mem_ok`: A boolean indicating whether memory status is acceptable.
    - `throttle`: A floating-point number representing a resource utilization adjustment factor, typically between 0 and 1.
    - `cpu_p`: A floating-point percentage value representing the current CPU usage.
    - `mem_p`: A floating-point percentage value representing the current memory usage.

    3. If either `cpu_ok` or `mem_ok` is `False`, it prints a message indicating that Social Cognition Simulation processing is skipped due to insufficient resources and returns without further execution.

    4. It initializes two empty lists called `pos_features` and `neg_features`.

    5. For each hypothesis in `self.reasoner.hypotheses`:
    - If the hypothesis has a type `"shadow_trust"` and contains an attribute `'utterance'`, it would theoretically store state vectors associated with these hypotheses.
    - However, due to simplicity, no actual storage or processing of state vectors is performed.

    6. The function then checks if there are any positive (`pos_features`) and negative (`neg_features`) examples available for training a naive mean classifier:
    - If such examples were present, it would update the `user_model` by recording the mean state for positive feedback.
    - Since no explicit storage of feedback is implemented, detailed training is skipped.

    7. It prints a message indicating that user model updates are placeholders and would require feedback storage

---

56) ### long_horizon_planning_rehearsal

    The `long_horizon_planning_rehearsal` function performs long-horizon cognitive planning rehearsal by simulating multi-turn dialogues. It initializes and uses the following variables:

    - `cpu_ok`: A boolean indicating whether CPU resources are sufficient.
    - `mem_ok`: A boolean indicating whether memory resources are sufficient.
    - `throttle`: A float value between 0 and 1 representing resource throttling level.
    - `cpu_p`: A float representing the current percentage of CPU usage.
    - `mem_p`: A float representing the current percentage of memory usage.

    The function works as follows:

    1. It checks if long-horizon planning is enabled via a configuration setting (`idle_reasoner.enable_long_horizon_planning`). If not, it returns immediately without performing any further operations.
    2. It retrieves resource status using `_get_resource_status`, which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
    3. If either `cpu_ok` or `mem_ok` is false, the function prints a message indicating that long-horizon planning is skipped due to insufficient CPU or memory resources, and then returns.
    4. The function prints a message indicating that it is simulating multi-turn dialogues.
    5. It checks if there are any states stored in the `state_manager.states`. If not, it returns without further operations.
    6. A random start state label is selected from the keys of `self.state_manager.states` using `np.random.choice`.
    7. The state vector corresponding to the chosen start label is copied into `state_vec`.
    8. An integer value `num_turns` is calculated as `max(1, int(3 * throttle))`, representing the number of simulated dialogue turns.
    9. For each turn, it performs the following steps:
        - Finds the word with the highest entropy from `self.concept_model.word_concept_counts`.
        - Calculates an information gain (IG) estimate using the entropy difference before and after asking about the selected word: `ig_estimate = self.concept_model.entropy_word(word) - 0.1`.
        - Accumulates the IG estimate into `total_reward`.
        - Updates the state vector by aligning it to embedding dimension, then blending with identity vector as follows:
            $$ \
            \text{aligned\_state} = \_align\_vector\_to\_emb\_dim(\text{state\_vec})
            \ $$
            $$ \
            \text{aligned\_identity} = \_align\_vector\_to\_emb\_dim(self.identity\_manager.identity)
            \ $$
            $$ \
            \text{state\_vec} = 0.9 \times \text{aligned\_state} + 0.1 \times \text{aligned\_identity}
            \ $$
        - Normalizes the updated state vector by dividing it with its Euclidean norm if the norm is greater than zero.
    10. After all dialogue turns, it prints a message summarizing the simulated conversation and the total IG reward achieved.

    The final print statement outputs:
    $$ \
    \text{Simulated } num\_turns \text{-turn conversation, total IG = total\_reward:.3f} \left(\text{throttle=throttle:.2f}\right)
    \ $$

    In summary, this function simulates a specified number of dialogue turns, calculates an estimate for the information gain (IG) at each turn, and accumulates these estimates to provide a measure of the effectiveness of long-horizon cognitive planning under given resource constraints.

---

57) ### behavioral_rule_discovery

    The `behavioral_rule_discovery` function performs the task of discovering behavioral rules by simulating a process akin to dreaming. It initializes and uses the following variables:

    - `cpu_ok`: A boolean indicating whether CPU resources are sufficient.
    - `mem_ok`: A boolean indicating whether memory resources are sufficient.
    - `throttle`: A float value between 0 and 1 representing resource throttling level.
    - `cpu_p`: A float representing the current percentage of CPU usage.
    - `mem_p`: A float representing the current percentage of memory usage.

    The function works as follows:

    1. It checks if behavioral rule discovery is enabled via a configuration setting (`idle_reasoner.enable_behavioral_rule_discovery`). If not, it returns immediately without performing any further operations.
    2. It retrieves resource status using `_get_resource_status`, which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
    3. If either `cpu_ok` or `mem_ok` is false, the function prints a message indicating that behavioral rule discovery is skipped due to insufficient CPU or memory resources, and then returns.
    4. The function prints a message indicating that it is dreaming about new rules.
    5. It calls `self.behavioral_learner.propose_new_rules(self)` to generate candidates for new rules. This method returns a list of candidate rules (`candidates`).
    6. If the list of candidates is not empty, the function iterates over each rule in the list and prints a message indicating the proposed new rule, its type, condition, and confidence.
    7. If no new rule candidates are found, it prints "No new rule candidates found."
    8. After proposing any new rules, the function calls `self.behavioral_learner.decay_confidences()` to decay the confidences of existing rules.

    In summary, this function performs behavioral rule discovery by generating and proposing new candidate rules based on current conditions, printing details about each proposed rule if any are generated, and decaying the confidence values of existing rules. The specific steps are as follows:

    1. Check if behavioral rule discovery is enabled.
    2. Retrieve resource status to ensure sufficient CPU and memory resources.
    3. Print a message indicating that new rules are being dreamed about.
    4. Generate candidate rules using `self.behavioral_learner.propose_new_rules(self)`.
    5. For each generated rule, print its type, condition, and confidence if any candidates are found.
    6. Decay the confidences of existing rules.


---

58) ### generate_logic_from_experience

    The `generate_logic_from_experience` function creates new logic rules based on highly confident behavioral rules that are not yet conscience beliefs. It initializes and uses the following variables:

    - `candidates`: A list to store candidate behavioral rules.
    - `filepath`: A string representing the file path where the generated logic rule is written.

    The function works as follows:

    1. If the configuration setting `idle_reasoner.enable_logic_inheritance` is false, the function returns immediately without performing any further operations.
    2. Initializes an empty list `candidates`.
    3. Iterates over each behavioral rule in `self.behavioral_learner.rules.values()`:
        - Checks if the rule is confirmed and has a confidence greater than 0.9.
        - Ensures that the rule's ID is not already present in the beliefs managed by `self.conscience_manager.beliefs`.
        - If all conditions are met, appends the rule to the list of candidates.
    4. If there are any candidates:
        - Selects the candidate with the highest confidence using `max(candidates, key=lambda r: r.confidence)`.
        - Constructs a dictionary `logic_rule` containing the description, condition, action, ethical status, and source of the selected rule.
    5. Calls `self.logic_inheritance.generate_logic_file(logic_rule)` to generate a logic file based on the constructed `logic_rule`, storing the resulting filepath in `filepath`.
    6. If a filepath is generated, calls `self.logic_inheritance.generate_doubt(logic_rule['logic_id'])` to queue doubt for the new logic rule.
    7. Prints a message indicating whether a new logic rule was written to a file or if no candidates were found.

    In summary, this function performs the following specific steps:

    1. Checks if logic inheritance is enabled via configuration (`idle_reasoner.enable_logic_inheritance`). If not, returns immediately.
    2. Initializes an empty list `candidates`.
    3. Iterates over each behavioral rule in `self.behavioral_learner.rules.values()` and checks if it meets the criteria (confirmed, confidence > 0.9, not already a belief).
    4. Appends qualifying rules to the `candidates` list.
    5. If there are candidates, selects the one with the highest confidence.
    6. Constructs a dictionary `logic_rule` based on the selected rule's details.
    7. Generates a logic file using `self.logic_inheritance.generate_logic_file(logic_rule)`, storing the filepath in `filepath`.
    8. Queues doubt for the new logic rule if a filepath is generated.
    9. Prints a message indicating whether a new logic rule was written to a file or if no candidates were found.

    The final print statement outputs:
    $$ \
    \text{📜 [Logic Generation] New logic rule written to } \text{filepath}, \text{ doubt queued.}
    \ $$
    or
    $$ \
    \text{📜 [Logic Generation] No new rule candidates found.}
    \ $$

---

59) ### contextual_sentiment_analysis

    The `contextual_sentiment_analysis` function performs deep analysis of a feedback corpus to learn emergent sentiment patterns. It initializes and uses the following variables:

    - `cpu_ok`: A boolean indicating whether CPU resources are sufficient.
    - `mem_ok`: A boolean indicating whether memory resources are sufficient.
    - `throttle`: A float value between 0 and 1 representing resource throttling level.
    - `cpu_p`: A float representing the current percentage of CPU usage.
    - `mem_p`: A float representing the current percentage of memory usage.
    - `confident_contexts`: An integer count of contexts with confidence above a threshold.
    - `total_entries`: The total number of entries in the feedback corpus.

    The function works as follows:

    1. It checks if contextual sentiment analysis is enabled via a configuration setting (`idle_reasoner.enable_contextual_sentiment_analysis`). If not, it returns immediately without performing any further operations.
    2. Retrieves resource status using `_get_resource_status`, which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
    3. If either `cpu_ok` or `mem_ok` is false, the function prints a message indicating that contextual sentiment analysis is skipped due to insufficient CPU or memory resources, and then returns.
    4. The function prints a message indicating that it is mining the feedback corpus for emergent patterns.
    5. Initializes an integer variable `confident_contexts` to zero.
    6. Iterates over each context in `self.reasoner.feedback_corpus.context_models.items()`:
        - Retrieves the confidence of the current context using `self.reasoner.feedback_corpus.get_confidence(ctx)`.
        - If the confidence is greater than or equal to `ContextualSentimentCorpus.CONFIDENCE_THRESHOLD`, increments `confident_contexts` by 1.
    7. Calculates the total number of entries in the feedback corpus as `len(self.reasoner.feedback_corpus.entries)`.
    8. Prints a summary message indicating the total number of feedback entries and the count of confident contexts.

    In summary, this function performs the following specific steps:

    1. Checks if contextual sentiment analysis is enabled via configuration (`idle_reasoner.enable_contextual_sentiment_analysis`). If not, returns immediately.
    2. Retrieves resource status to ensure sufficient CPU and memory resources.
    3. Prints a message indicating that feedback corpus mining for emergent patterns is being performed.
    4. Initializes an integer variable `confident_contexts` to zero.
    5. Iterates over each context in the feedback corpus and checks if its confidence exceeds the threshold (`ContextualSentimentCorpus.CONFIDENCE_THRESHOLD`).
    6. Increments `confident_contexts` for each qualifying context.
    7. Calculates the total number of entries in the feedback corpus.
    8. Prints a summary message indicating the total number of feedback entries and the count of confident contexts.

    The final print statement outputs:
    $$ \
    \text{📊 [Contextual Sentiment Analysis] Feedback entries: } \text{total\_entries}, \text{ confident contexts: } \text{confident\_contexts}.
    \ $$

---

60) ### autonomous_code_generation_and_testing

    The `autonomous_code_generation_and_testing` function generates and tests code for promising logic candidates based on high-confidence behavioral rules, unresolved hypotheses, and new state definitions. It initializes and uses the following variables:

    - `cpu_ok`: A boolean indicating whether CPU resources are sufficient.
    - `mem_ok`: A boolean indicating whether memory resources are sufficient.
    - `throttle`: A float value between 0 and 1 representing resource throttling level.
    - `cpu_p`: A float representing the current percentage of CPU usage.
    - `mem_p`: A float representing the current percentage of memory usage.
    - `candidates`: A list to store potential candidates for code generation.
    - `logic_id`: A string identifier for each candidate rule or hypothesis.

    The function works as follows:

    1. It checks if logic inheritance is enabled via a configuration setting (`idle_reasoner.enable_logic_inheritance`). If not, it returns immediately without performing any further operations.
    2. Retrieves resource status using `_get_resource_status`, which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
    3. If either `cpu_ok` or `mem_ok` is false, the function prints a message indicating that autonomous code generation is skipped due to insufficient CPU or memory resources, and then returns.
    4. The function prints a message indicating that it is selecting candidates for code generation.
    5. Initializes an empty list `candidates`.
    6. Iterates over each behavioral rule in `self.behavioral_learner.rules.values()`:
        - Checks if the rule is confirmed and has a confidence greater than 0.85.
        - Ensures that the rule's ID is not already present in `self.logic_inheritance.generated_code_paths`.
        - If all conditions are met, appends the tuple (`"behavioral"`, rule) to the list of candidates.
    7. Iterates over each hypothesis in `self.reasoner.hypotheses`:
        - Checks if the hypothesis type is "synonym" and has a similarity greater than 0.9.
        - Ensures that the hypothesis is not already asked.
        - If all conditions are met, generates a unique `logic_id` based on the word and candidate, and appends the tuple (`"semantic"`, hyp, logic_id) to the list of candidates if `logic_id` is not in `self.logic_inheritance.generated_code_paths`.
    8. Checks if there are any current state labels and a stored embedding:
        - Constructs a unique `state_id` based on the first label.
        - Ensures that `state_id` is not already present in `self.logic_inheritance.generated_code_paths`.
        - If all conditions are met, constructs a dictionary for state definition and appends the tuple (`"state"`, state_rule, state_id) to the list of candidates.
    9. If no candidates are found, prints a message indicating that there are no candidates for code generation and returns immediately.
    10. Processes up to 2 candidates per idle cycle:
        - For each candidate, determines its type and corresponding data (`behavioral`, `semantic`, or `state`).
        - Constructs a dictionary based on the candidate's type.
        - Creates a logic JSON file using `self.logic_inheritance.generate_logic_file(rule_dict)`.
        - Generates a doubt for the rule by calling `self.logic_inheritance.generate_doubt(logic_id)`.
        - Prints a message indicating that a doubt has been created.
        - Generates code files based on the generated doubts using `self.logic_inheritance.generate_code_for_logic(logic_id)`.
        - Performs shadow testing of the generated code using `self.logic_inheritance.shadow_test_generated_code(logic_id)`.
        - Depending on the success of shadow testing, prints a message indicating whether the generation and testing were successful or if there was an error.

    In summary, this function performs the following specific steps:

    1. Checks if logic inheritance is enabled via configuration (`idle_reasoner.enable_logic_inheritance`). If not, returns immediately.
    2. Retrieves resource status to ensure sufficient CPU and memory resources.
    3. Prints a message indicating that it is selecting candidates for code generation.
    4. Initializes an empty list `candidates`.
    5. Iterates over each behavioral rule in `self.behavioral_learner.rules.values()`:
        - Checks if the rule meets criteria (confirmed, confidence > 0.85).
        - Ensures the rule's ID is not already present in generated code paths.
        - Appends qualifying rules to `candidates`.
    6. Iterates over each hypothesis in `self.reasoner.hypotheses`:
        - Checks if the hypothesis type is "synonym" and has similarity > 0.9.
        - Ensures the hypothesis has not been asked before.
        - Generates a unique `logic_id` based on word and candidate, appending to `candidates`.
    7. Constructs state definitions based on current labels and embeddings:
        - Checks if there are any current state labels and stored embedding.
        - Constructs unique `state_id`, ensuring it is not already present in generated code paths.
        - Appends qualifying states to `candidates`.
    8. If no candidates are found, prints a message indicating that there are no candidates for code generation and returns immediately.
    9. Processes up to 2 candidates per idle cycle:
        - For each candidate, constructs a dictionary based on type (`behavioral`, `semantic`, or `state`).
        - Creates a logic JSON file using `self.logic_inheritance.generate_logic_file(rule_dict)`.
        - Generates doubt for the rule by calling `self.logic_inheritance.generate_doubt(logic_id)`.
        - Prints message indicating doubt creation.
        - Generates code files based on doubts using `self.logic_inheritance.generate_code_for_logic(logic_id)`.
        - Performs shadow testing of generated code using `self.logic_inheritance.shadow_test_generated_code(logic_id)`.
        - Depending on success, prints messages indicating generation and testing results.

---

61) ### deep_behavioral_pattern_mining

    The `deep_behavioral_pattern_mining` function performs deep offline mining of conversational patterns to discover behavioral rules. It initializes and uses the following variables:

    - `cpu_ok`: A boolean indicating whether CPU resources are sufficient.
    - `mem_ok`: A boolean indicating whether memory resources are sufficient.
    - `throttle`: A float value between 0 and 1 representing resource throttling level.
    - `cpu_p`: A float representing the current percentage of CPU usage.
    - `mem_p`: A float representing the current percentage of memory usage.
    - `candidates`: A list to store candidate behavioral rules.

    The function works as follows:

    1. It checks if deep pattern mining is enabled via a configuration setting (`idle_reasoner.enable_behavioral_rule_discovery`). If not, it returns immediately without performing any further operations.
    2. Retrieves resource status using `_get_resource_status`, which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
    3. If either `cpu_ok` or `mem_ok` is false, the function prints a message indicating that deep pattern mining is skipped due to insufficient CPU or memory resources, and then returns.
    4. The function prints a message indicating that it is mining long-range conversational patterns.
    5. Calls `self.behavioral_learner.deep_pattern_mining(self)` to generate candidate behavioral rules and stores the result in `candidates`.
    6. If there are any candidates:
        - For each candidate rule, calls `self.behavioral_learner.add_candidate_rule(rule)` to add the candidate rule to the list of candidate rules.
        - Prints a message indicating that a new pattern has been discovered: \(\text{{condition}} \rightarrow \text{{action}}\).
    7. If no candidates are found, prints "No new deep patterns found."

    In summary, this function performs the following specific steps:

    1. Checks if deep pattern mining is enabled via configuration (`idle_reasoner.enable_behavioral_rule_discovery`). If not, returns immediately.
    2. Retrieves resource status to ensure sufficient CPU and memory resources.
    3. Prints a message indicating that it is mining long-range conversational patterns.
    4. Calls `self.behavioral_learner.deep_pattern_mining(self)` to generate candidate behavioral rules.
    5. Stores the result in `candidates`.
    6. If there are any candidates:
        - For each candidate rule, adds the rule to the list of candidate rules using `self.behavioral_learner.add_candidate_rule(rule)`.
        - Prints a message indicating that a new pattern has been discovered: \(\text{{condition}} \rightarrow \text{{action}}\).
    7. If no candidates are found, prints "No new deep patterns found."

---

62) ### propagate_word_state_associations
    
    The function `propagate_word_state_associations` performs semantic similarity-based propagation of word-state labels during idle periods, as described below:

    **Initialization and Variables Used**

    1. **Configuration Check**:
    - `cpu_ok`, `mem_ok`: Boolean variables indicating whether CPU and memory resources are sufficient.
    - `throttle`: Not used in the function body.
    - `cpu_p`, `mem_p`: Float values representing current CPU and memory usage percentages.

    2. **Resource Status Retrieval**:
    ```python
    cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
    ```

    3. **Word Embeddings List**:
    - `all_words`: A list containing all words that have embeddings stored in the `self.semantic_learner.word_embeddings` dictionary.

    4. **Sampled Words**:
    - `sample`: A sample of 50 (or fewer if there are less than 50) unique words from `all_words`.

    5. **Similarity Dictionary**:
    - `similarities`: A dictionary storing pairs of similar words and their similarity scores, where the score is above a threshold.

    6. **Propagated Count**:
    - `propagated`: An integer initialized to 0, used to count how many label associations were propagated between similar words.

    **Function Execution**

    1. **Configuration Check**:
    The function first checks if semantic self-talk (word-state propagation via similarity) is enabled in the configuration settings (`self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True)`). If not, it returns immediately without performing any operations.

    2. **Resource Status and Print Statement**:
    - Retrieves CPU and memory status.
    - Skips execution if either `cpu_ok` or `mem_ok` is false.
    - Prints a message indicating the skipping of word-state propagation due to insufficient resources: 
        ```python
        print(f"   [Resource Aware] Skipping word-state propagation (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
        ```

    3. **Print Propagation Start**:
    - Prints a message indicating the start of the word-state propagation process.
        ```python
        print("🔄 [Word-State Propagation] Spreading labels via semantic similarity...")
        ```

    4. **Sampled Word Embeddings**:
    - Generates a sample of 50 (or fewer) words from `all_words` using `random.sample(all_words, min(50, len(all_words)))`.
    
    5. **Compute Similarities**:
    - Computes pairwise similarities for the sampled words.
    - For each pair `(w1, w2)` in the sample where `similarity(w1, w2) > 0.8` (high similarity), adds the pair and its similarity score to `similarities`.

    6. **Propagate Labels**:
    - Iterates over pairs of similar words.
    - Retrieves labels for each word with a confidence threshold of 0.3 using `assoc.get_labels(w1, threshold=0.3)` and `assoc.get_labels(w2, threshold=0.3)`.
    - For each label in `labels1` not already present in `labels2`, adds the label to `w2` with a scaled confidence (`sim * 0.8`) using `assoc.add_association(w2, lab, confidence=sim * 0.8, source="similarity")` and increments `propagated`.
    - Similarly, for each label in `labels2` not already present in `labels1`, adds the label to `w1` with a scaled confidence (`sim * 0.8`) using `assoc.add_association(w1, lab, confidence=sim * 0.8, source="similarity")` and increments `propagated`.

    7. **Print Propagation Summary**:
    - Prints the number of label associations propagated.
        ```python
        print(f"   Propagated {propagated} label associations.")
        ```

    8. **Detect Conflicts**:
    - Iterates over all words to detect any ambiguous labels using `assoc.detect_conflicts(word)`.
    - If conflicts are found, prints a warning message indicating the word and its conflicting labels.
        ```python
        print(f"   ⚠️ Word '{word}' has ambiguous labels: {conflicts}. (Will ask user later)")
        ```

    **Formulas Used**

    - **Confidence Scaling**:
    $$ \
    \text{confidence} = \text{similarity} \times 0.8
    \ $$

    The function ensures that word-state labels are propagated based on semantic similarity, provided sufficient resources and configuration settings allow it.

---

63) ### mine_phrase_relations

    The function `mine_phrase_relations` performs the following actions:

    1. **Initializes Variables and Checks Configuration**:
        - It initializes a boolean value `cpu_ok`, `mem_ok`, `throttle`, and two float values `cpu_p` and `mem_p`.
        - These variables are initialized by calling `_get_resource_status()`. The function checks if `idle_reasoner.enable_semantic_self_talk` is set to `True` in the configuration. If not, it returns immediately.

    2. **Resource Status Check**:
        - It retrieves resource status using `_get_resource_status()` which returns `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`.
        - If either `cpu_ok` or `mem_ok` is `False`, it prints a message indicating that the phrase relation mining is skipped due to insufficient resources:
        ```python
        print(f"   [Resource Aware] Skipping phrase relation mining (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
        ```

    3. **Collect Recent User Inputs**:
        - If resource status is acceptable, it prints a message indicating the start of phrase relation mining.
        - It collects recent user inputs from episodic memory by accessing `self.episodic_memory.episodes[-200:]`.
        - For each episode in these episodes, if there is a non-empty 'user_input' and its length (in words) does not exceed 10, it appends the input to the list `phrases`.

    4. **Filter Short List of Phrases**:
        - If the number of collected phrases (`len(phrases)`) is less than 5, the function returns without further processing.

    5. **Compute Embeddings for Collected Phrases**:
        - It initializes an empty dictionary `phrase_embs`.
        - For each phrase in `phrases`, it attempts to compute its embedding using `self.semantic_learner.text_to_embedding(p, fallback_to_char=True)`. If the embedding is not `None`, it stores the phrase and its embedding in `phrase_embs`.

    6. **Pairwise Comparison of Embeddings**:
        - It converts `phrase_embs` into a list of tuples.
        - If this list has more than 30 items, it randomly samples 30 items from `phrase_embs`.
        - For each pair of embeddings in the sampled list (excluding self-pair comparisons), it computes cosine similarity using:
        $$\
        \text{similarity} = \frac{\mathbf{emb1} \cdot \mathbf{emb2}}{\|\mathbf{emb1}\| \|\mathbf{emb2}\| + 10^{-12}}
        \ $$
        - If the computed similarity is greater than 0.85, it checks if there already exists a hypothesis with the same key.
        - If no such hypothesis exists, it creates and appends a new hypothesis to `self.reasoner.hypotheses`:
        ```python
        hyp = {
            'key': f"phrase_similar:{p1}:{p2}",
            'type': 'phrase_similarity',
            'phrase1': p1,
            'phrase2': p2,
            'similarity': sim,
            'text': f"I notice that '{p1}' and '{p2}' seem similar. Are they related?",
            'asked': False,
        }
        ```
        - It prints a message indicating the proposed hypothesis.

    The function initializes and uses the following variables:
    - `cpu_ok`: A boolean variable representing CPU status.
    - `mem_ok`: A boolean variable representing memory status.
    - `throttle`: An unknown type variable (could be boolean or float) from resource status.
    - `cpu_p`: A float value representing CPU percentage usage.
    - `mem_p`: A float value representing memory percentage usage.
    - `episodes`: A list of recent user inputs retrieved from episodic memory.
    - `phrases`: A list of selected phrases from the episodes, each containing up to 10 words.
    - `phrase_embs`: A dictionary storing phrases and their corresponding embeddings.
    - `items`: A list of tuples representing phrase-emission pairs.
    - `hyp`: A hypothesis dictionary created if a new relation is found.

    The function makes use of external objects:
    - `self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True)`: Retrieves configuration for enabling semantic self-talk.
    - `self.episodic_memory.episodes[-200:]`: Accesses the last 200 episodes from episodic memory.
    - `self.semantic_learner.text_to_embedding(p, fallback_to_char=True)`: Computes an embedding for a given text with a character-level fallback if necessary.
    - `self.reasoner.hypotheses.append(hyp)`: Appends a new hypothesis to the list of hypotheses.

    The cosine similarity formula used is:
    $$ \
    \text{similarity} = \frac{\mathbf{emb1} \cdot \mathbf{emb2}}{\|\mathbf{emb1}\| \|\mathbf{emb2}\| + 10^{-12}}
    \ $$
    where $\mathbf{emb1}$ and $\mathbf{emb2}$ are the embeddings of two phrases, and $+$ $10^{-12}$ is added to avoid division by zero.

---

64) ### converge_knowledge_graph

    The function `converge_knowledge_graph` performs idle-time consolidation of a knowledge graph by executing the following steps:

    1. **Initial Checks and Resource Status**:
        - The function first checks if an attribute `knowledge_graph` exists within the `reasoner` object. If not, it returns immediately.
        - It initializes the variable `kg` to reference the `knowledge_graph` attribute of the `self.reasoner`.
        - It calls `_get_resource_status()` which returns five values: `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, and `mem_p`. If either `cpu_ok` or `mem_ok` is `False`, it prints a message indicating that knowledge graph convergence is skipped due to resource constraints, specifically CPU usage exceeding 50% or memory usage exceeding 50%. The exact thresholds are not specified in the code but can be inferred from the context.

    2. **Infer Transitive Relations**:
        - For each relation type 'is_a' and 'means', it infers transitive relations with a confidence threshold of 0.6.
        - It iterates through each inferred edge `(n1, n2)` along with its relationship type `rt` and confidence value `conf`, adds the edge to the knowledge graph using `kg.add_edge(n1, n2, rt, conf)`, and prints a message indicating the inference.

    3. **Merge Synonym Candidates**:
        - This step is currently a placeholder where merging synonym candidates would be performed if required. The code comments indicate that this functionality has been skipped due to the need for careful handling.

    4. **Prune Weak Edges**:
        - It prunes edges with confidence values less than 0.1.
        - Prints a message indicating weak edge pruning and runs `run_constraint_propagation()` on the reasoner object.

    5. **Identify Central Nodes**:
        - Identifies top central nodes (top 10) in terms of degree centrality using `kg.get_central_nodes(top_k=10)`.
        - Stores these central nodes in a separate structure, `self.central_graph_nodes`.

    6. **Save Knowledge Graph**:
        - The knowledge graph is saved after the convergence process.

    The function initializes and uses the following variables:

    - `cpu_ok`: Boolean indicating if CPU usage is within acceptable limits.
    - `mem_ok`: Boolean indicating if memory usage is within acceptable limits.
    - `throttle`: Placeholder variable, not used in this context.
    - `cpu_p`: Percentage of CPU usage as a float value.
    - `mem_p`: Percentage of memory usage as a float value.
    - `kg`: Reference to the knowledge graph stored in `self.reasoner.knowledge_graph`.
    - `inferred`: List of inferred transitive relations.
    - `central`: Top 10 central nodes from the knowledge graph.

---

65) ### form_meta_concepts_from_concepts

    The function `form_meta_concepts_from_concepts` is a method of an object that calls the `form_meta_concepts` method on its `concept_model` attribute. This function does not initialize any variables; it solely relies on the state of the `self.concept_model` attribute.

    description:

    1. **Function Name**: `form_meta_concepts_from_concepts`
    2. **Parameters**: None
    3. **Return Value**: None

    **Description**:
    - The function checks if the `concept_model` object has an attribute named `form_meta_concepts`.
    - If the attribute exists, it calls this method on the `concept_model`.

    No variables are initialized by this function. It uses one variable: `self.concept_model`, which is expected to be an object with a method `form_meta_concepts`.

    In summary:
    - The function does not initialize any new variables.
    - The only attribute used is `self.concept_model`.
    - No return value or output is generated by the function.

---

66) ### perform_all_idle_tasks

    The function `perform_all_idle_tasks` orchestrates all idle-time tasks while respecting the configuration settings. Here are the details:

    1. **Initialization and Variables**:
        - `now`: Current timestamp obtained from `time.time()`.
        - `self._idle_tasks_last_run`: Tracks the last time idle tasks were run.
        - `cpu_ok`, `mem_ok`, `throttle`, `cpu_p`, `mem_p`: Resource status variables obtained from `_get_resource_status()` method.

    2. **Functionality**:
        - If less than 300 seconds (5 minutes) have passed since the last execution of idle tasks (`now - self._idle_tasks_last_run < 300`), the function returns immediately.
        - Updates `self._idle_tasks_last_run` to the current time if no tasks were skipped due to timing constraints.
        
        - Calls `_get_resource_status()` method to get CPU and memory status:
            - If either CPU or memory is not okay (`not cpu_ok or not mem_ok`), it prints a message indicating resource shortage (CPU percentage: `cpu_p`, Memory percentage: `mem_p`) and returns without running any tasks.
        
        - Prints a header for the idle session task execution.
        - Defines a list of tasks to be executed:
            ```python
            tasks = [
                self.world_model_dreaming,
                self.active_memory_rehearsal,
                self.curiosity_landscape_optimization,
                self.concept_graph_compression,
                self.autonomous_hypothesis_planning,
                self.identity_stabilization,
                self.cross_modal_alignment,
                self.exploration_policy_distillation,
                self.cognitive_drive_evolution,
                self.counterfactual_reasoning,
                self.semantic_self_talk,
                self.state_space_topology_learning,
                self.knowledge_gap_frontier_detection,
                self.social_cognition_simulation,
                self.long_horizon_planning_rehearsal,
                self.behavioral_rule_discovery,
                self.generate_logic_from_experience,
                self.autonomous_code_generation_and_testing,
                self.contextual_sentiment_analysis,
                self.deep_behavioral_pattern_mining,
                self.propagate_word_state_associations,
                self.mine_phrase_relations,
                self._metabolic_maintenance,
                self.converge_knowledge_graph,
                self.train_language_model,
                self.propose_lm_improvement,
                self.reasoner.run_symbolic_inference,
                self.form_meta_concepts_from_concepts,
                self.build_world_model_transitions
            ]
            ```
        
        - Iterates through each task in the `tasks` list, executing them one by one:
            - Calls each task using `task()`.
            - Catches any exceptions raised during execution and prints an error message along with a traceback.
            - Delays between tasks based on throttling factor (`throttle`). The delay is calculated as `max(0.05, 0.2 * throttle)` seconds.

    3. **Output**:
        - Prints a header for the idle session task execution.
        - Executes each task in sequence with appropriate delays and exception handling.
        - Prints a footer after all tasks are completed or skipped due to resource constraints.


---

67) ### train_language_model
    The function `train_language_model` trains a neural language model using textual data. It initializes and uses the following variables:

    - `cpu_ok`, `mem_ok`: Boolean values indicating whether CPU and memory resources are sufficient.
    - `throttle`: A float value representing resource throttling factor.
    - `cpu_p`, `mem_p`: Float values representing current CPU and memory usage percentages.
    - `sentences`: A list of sentences collected from episodic memory, definitions, and potentially synthetic data.
    - `batch`: A subset of `sentences` used for training the language model.
    - `total_loss`: A float value initialized to 0.0, tracking the cumulative loss during training.
    - `lr`: Learning rate adjusted by throttling factor.

    The function performs the following steps:

    1. Checks if the language model is enabled via configuration and returns early if not.
    2. Retrieves resource status (CPU and memory usage) and checks for sufficient resources before proceeding with training.
    3. Collects sentences from episodic memory, ensuring each sentence has a maximum of 30 words and that system responses do not contain '?'.
    4. Optionally adds synthetic sentences from dreaming to the list if available (though this part is currently unimplemented).
    5. Adds definitions from `self.reasoner.definitions` as additional training data.
    6. Returns early if fewer than 10 sentences are collected for training.
    7. Shuffles and selects a batch of sentences for training, ensuring the batch size does not exceed `train_batch_size`.
    8. Updates the vocabulary with any new words found in the selected batch.
    9. Trains the language model on each sentence in the batch using a learning rate adjusted by throttling factor, accumulating total loss.
    10. Calculates and prints average loss, current vocabulary size, and total steps taken during training.

    The formula for calculating average loss is:
    $$ \
    \text{avg\_loss} = \frac{\text{total\_loss}}{\text{len(batch)}}
    \ $$

    Mathematically, the process of accumulating losses can be represented as:
    $$ \
    \text{total\_loss} = \sum_{i=0}^{\text{len(batch)-1}} \text{loss}_i
    \ $$
    where `loss_i` is the loss from training on the i-th sentence in the batch.

---

68) ### _build_lm_validation_set
    The function `_build_lm_validation_set` creates a validation set from the most recent episodes stored in `self.episodic_memory`, excluding training data. The function initializes and uses the following variables:

    - **sentences**: A list that stores user inputs and system responses that meet certain criteria.
    - **size**: An integer parameter with a default value of 50, specifying the number of sentences to be used for validation.

    The function performs the following steps:

    1. Initializes an empty list `sentences`.
    2. Iterates over the last 500 episodes from `self.episodic_memory.episodes`:
    - For each episode `ep`, it retrieves and processes `user_input` (if present) by stripping any leading or trailing whitespace.
    - If `user_input` is not empty and its length in terms of words does not exceed 30, the processed user input is appended to `sentences`.
    - Similarly, if `system_response` (if present) is not empty, does not contain a question mark (`?`), and its word count does not exceed 30, it is also stripped and added to `sentences`.

    3. Shuffles the list of sentences using NumPy's `np.random.shuffle`.
    4. Returns the last `size` elements from the shuffled `sentences` if the total number of sentences in `sentences` is at least equal to `size`. Otherwise, it returns all available sentences.

    Here is a summary of the function’s operations:

    - The episodes are selected from the most recent 500 entries in `self.episodic_memory.episodes`.
    - User inputs and system responses that are not empty, contain fewer than or exactly 30 words, and for which the system response does not include a question mark (if applicable) are added to `sentences`.
    - The sentences list is shuffled using NumPy’s random shuffle function.
    - If there are at least `size` entries in `sentences`, the last `size` entries are returned as the validation set. Otherwise, all available sentences are returned.

 
--- 

69) ### evaluate_language_model

---


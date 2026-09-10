## Doc for learner
---

This Python file contains classes and functions related to a knowledge-based reasoning system. Key components include the UnificationEngine for expression unification with variables, Procedure and ProceduralLearner for learning step-by-step problem-solving procedures from user input, SyntaxLearner for Bayesian grammar induction, WordStateAssociator for mapping words to semantic state labels, KnowledgeGraph for storing a heterogeneous graph of knowledge relationships, and StructuralPatternLearner for extracting structural patterns from sentences. The design is modular with each class handling specific aspects like unification, procedural learning, syntax learning, word-state association, knowledge graph management, and pattern extraction respectively.

---

### UnificationEngine
The `UnificationEngine` class is responsible for unifying two expressions containing variables. It performs this by recursively comparing and binding variable names to expressions where possible, ensuring that no contradictions arise (e.g., a variable does not unify with an expression in which it already appears). The key attributes include the ability to handle both leaf nodes (constants or single variables) and non-leaf nodes (expressions with sub-expressions), applying substitutions consistently throughout.

### Functions

#### `unify`

The `unify` function in the `UnificationEngine` class attempts to unify two expressions (`expr1`, `expr2`) and returns a dictionary mapping variable names to expressions that represent the unification. If no valid unification can be found, it returns `None`.

- **Parameters:**
  - `expr1`: An instance of the `Expression` class representing one expression.
  - `expr2`: Another instance of the `Expression` class representing a second expression.
  - `subs`: Optional dictionary (default is `None`) that maps variable names to expressions, used for tracking substitutions.

- **Return Value:**
  A dictionary mapping variable names to expressions if unification succeeds; otherwise, returns `None`.

The function handles various cases:
1. If both expressions are leaves and one of them is a variable, it attempts to bind the variable to the other expression.
2. If one expression is a leaf (constant or single variable) and the other is not, checks whether the non-leaf can be unified with the leaf by binding variables appropriately.
3. For two non-leaves, ensures that their roots are equal before recursively unifying their children.

If any step fails to unify due to contradictions or mismatches, it returns `None`.

---

#### `_bind_variable`

The `_bind_variable` function in the `UnificationEngine` class binds a variable name to an expression within a substitution dictionary, ensuring consistency.

- **Parameters:**
  - `var_name`: A string representing the name of the variable.
  - `expr`: An instance of the `Expression` class representing the expression to which the variable should be bound.
  - `subs`: A dictionary mapping variable names to expressions that represents current substitutions.

- **Return Value:**
  Returns a new substitution dictionary with the binding applied if consistent, otherwise returns `None`.

The function checks for consistency in the following ways:
1. If `var_name` is already present in `subs`, it attempts to unify the existing expression bound to `var_name` with `expr`. This ensures that any previously established bindings are not contradicted.
2. If `var_name` is not yet bound, a new copy of `subs` is created and updated by adding the binding `(var_name: expr)`.

If unification fails at any point due to contradictions or mismatches, it returns `None`. Otherwise, it returns the updated substitution dictionary.

---

#### `substitute`

The `substitute` function in the `UnificationEngine` class applies a given substitution dictionary to an expression.

- **Parameters:**
  - `expr`: An instance of the `Expression` class representing the expression to which substitutions should be applied.
  - `subs`: A dictionary mapping variable names to expressions that represents current substitutions.

- **Return Value:**
  Returns a new expression with all variables in `expr` substituted according to `subs`.

The function handles substitution as follows:
1. If `expr` is a leaf (i.e., it does not have children):
   - Checks if the root of `expr` is a variable and whether that variable name exists in `subs`.
   - If so, returns the expression corresponding to the value from `subs`. Otherwise, returns the original expression.
2. If `expr` has non-leaf structure (i.e., it has children):
   - Recursively applies substitution to its left and right child expressions (`left`, `right`) if they exist.
   - Returns a new instance of `Expression` with the root unchanged but with substituted children.

This ensures that all variables in the expression are consistently replaced according to the provided substitutions.

---


### Procedure
The `Procedure` class represents a sequence of steps designed to solve a specific class of problems by transforming expressions. Each step is defined as an action with optional arguments. Key attributes include the procedure's name, list of steps (each a dictionary containing an action and its arguments), associated variables, confidence level, and usage count. The purpose of this class is to encapsulate problem-solving logic in a structured format that can be applied repeatedly to expressions using a reasoning engine.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `Procedure` class and initializes its attributes.

- **Parameters:**
  - `name`: A string representing the name of the procedure.
  - `steps`: A list of dictionaries, where each dictionary represents a step in the procedure. Each step contains an action (a string) and optional arguments (`args`, a list).
  - `variables` (optional): A list of instances of the `Variable` class representing variables associated with this procedure.

- **Return Value:**
  The constructor does not return any value; it initializes instance attributes instead.

The function sets up several key attributes:
1. `self.name`: Stores the name provided.
2. `self.steps`: Stores the steps list, where each step is a dictionary containing an action and optional arguments.
3. `self.variables`: Initializes with the given variables or an empty list if no variables are specified.
4. `self.confidence`: Sets to 0.5 by default; this represents the confidence level of the procedure.
5. `self.uses`: Initialized to 0, representing how many times the procedure has been used.

This constructor ensures that a new instance of `Procedure` is properly initialized with all necessary attributes for solving problems according to defined steps and variables.

---

#### `apply`

The `apply` function in the `Procedure` class applies a sequence of steps to an expression using a reasoning engine.

- **Parameters:**
  - `expr`: An instance of the `Expression` class representing the initial expression.
  - `engine`: An instance of the `ReasoningEngine` class used for applying transformation rules or actions.

- **Return Value:**
  Returns the final transformed expression as an instance of `Expression`, or `None` if any step fails to apply successfully.

The function iterates through each step in the procedure's list of steps:
1. For each step, it extracts the action and optional arguments.
2. Depending on the action (e.g., 'isolate_variable', 'simplify'), it applies the corresponding transformation using the provided reasoning engine or placeholder logic.
3. If any step fails to apply successfully, the function will return `None`.

This ensures that the expression is transformed according to each defined step in the procedure until completion or failure.

### Example Actions:
- **isolate_variable**: Placeholder for isolating a variable by moving terms around (e.g., transforming \( ax + b = c \) into \( x = \frac{c - b}{a} \)).
- **simplify**: Evaluates constant parts of the expression to simplify it further.

These actions are placeholders and would be implemented with specific logic or rule schemas in a real application.

---


### ProceduralLearner
The `ProceduralLearner` class is responsible for learning and applying step-by-step solution procedures from user-provided explanations. Key attributes include an associated reasoning engine (`engine`) and a dictionary mapping problem patterns to learned procedures (`procedures`). The purpose of this class is to record new procedures based on user input and apply existing procedures to solve problems, ensuring that each procedure is stored and can be retrieved for future use.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `ProceduralLearner` class and initializes its attributes.

- **Parameters:**
  - `engine`: An instance of the `ReasoningEngine` class used for reasoning and applying transformations.

- **Return Value:**
  The constructor does not return any value; it initializes instance attributes instead.

The function sets up several key attributes:
1. `self.engine`: Stores the provided reasoning engine.
2. `self.procedures`: Initializes a dictionary to store problem patterns as keys and corresponding procedures as values.

This ensures that a new instance of `ProceduralLearner` is properly initialized with an associated reasoning engine and an empty set of learned procedures.

---

#### `learn_procedure`

The `learn_procedure` function in the `ProceduralLearner` class records a new step-by-step solution procedure from user-provided information.

- **Parameters:**
  - `problem`: An instance of the `Expression` class representing the initial problem expression.
  - `steps`: A list of strings containing natural language descriptions of each step in the solution process.
  - `solution`: An instance of the `Expression` class representing the final solved expression.

- **Return Value:**
  Returns an instance of the `Procedure` class that was created and stored for this problem pattern.

The function performs the following actions:
1. Converts the list of steps into a formal procedure object using the `Procedure` constructor.
2. Stores the new procedure in the dictionary `self.procedures`, with the string representation of the problem as the key.
3. Returns the newly created procedure object.

This ensures that each learned procedure is associated with its corresponding problem pattern and can be retrieved for future use.

---

#### `find_procedure`

The `find_procedure` function in the `ProceduralLearner` class searches for an existing solution procedure based on the given problem.

- **Parameters:**
  - `problem`: An instance of the `Expression` class representing the initial problem expression.

- **Return Value:**
  Returns a stored `Procedure` object if one matches the provided problem, otherwise returns `None`.

The function performs the following actions:
1. Converts the input problem to its string representation.
2. Looks up this string in the dictionary `self.procedures`.
3. If a matching key is found, it returns the associated procedure; otherwise, it returns `None`.

This ensures that if a previously learned solution exists for the given problem pattern, it can be retrieved and applied.

---

#### `apply_procedure`

The `apply_procedure` function in the `ProceduralLearner` class applies a previously learned procedure to solve a given problem.

- **Parameters:**
  - `problem`: An instance of the `Expression` class representing the initial problem expression.

- **Return Value:**
  Returns an instance of the `Expression` class if the solution is successfully applied, otherwise returns `None`.

The function performs the following actions:
1. Calls `find_procedure` to locate a stored procedure that matches the given problem.
2. If a matching procedure (`proc`) is found, it applies this procedure to the input problem using the associated reasoning engine and returns the resulting expression.
3. If no matching procedure is found, it returns `None`.

This ensures that if a previously learned solution exists for the given problem pattern, it can be applied to generate or modify the initial problem expression accordingly.

---


### SyntaxLearner
The `SyntaxLearner` class is responsible for incrementally learning a probabilistic context-free grammar (PCFG) from observed sentences using a Dirichlet process prior over rules. It maintains a lexicon of words and their part-of-speech tags, counts for rule occurrences (`rule_counts`), and total counts for nonterminals (`rule_total`). The class also includes methods to parse input sentences into syntax trees, update parameters based on configuration changes, add new sentences to the grammar, generate grammatical responses about a target, and save/load model state.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `SyntaxLearner` class and initializes its attributes.

- **Parameters:**
  - `config_manager`: An optional instance of the `ConfigManager` class (default is `None`).

- **Return Value:**
  The constructor does not return any value; it initializes instance attributes instead.

The function sets up several key attributes:
1. `self.config_manager`: Stores the provided configuration manager or defaults to `None`.
2. `self.nonterminals`: A set of nonterminal symbols in the grammar, such as 'S', 'NP', etc.
3. `self.lexicon`: An empty dictionary that will map words to sets of part-of-speech (POS) tags.
4. `self.rule_counts`: A nested defaultdict and Counter structure where keys are left-hand side (LHS) nonterminals and values are counts for right-hand side (RHS) rules or word occurrences.
5. `self.rule_total`: A Counter that stores the total count of each LHS symbol across all rules.
6. `self.dp_alpha`: The Dirichlet process concentration parameter, set to 1.0 by default.

This ensures that a new instance of `SyntaxLearner` is properly initialized with necessary attributes for learning and maintaining probabilistic context-free grammar (PCFG) rules.

---

#### `parse`

The `parse` function in the `SyntaxLearner` class performs a simple CYK ( Cocke–Younger–Kasami) parser to generate the highest-probability parse tree for an input sentence.

- **Parameters:**
  - `tokens`: A list of strings representing the tokens or words in the sentence.

- **Return Value:**
  Returns an instance of the `SyntaxTree` class if a valid parse is found, otherwise returns `None`.

The function performs the following actions:
1. Initializes a chart (table) to store possible nonterminals and corresponding syntax trees for each span of tokens.
2. Fills lexical rules (terminal symbols) by looking up part-of-speech tags in the lexicon and creating initial tree nodes for each word.
3. Uses dynamic programming over lengths to fill the chart with higher-level parse options, combining smaller spans based on grammar rules stored in `rule_counts`.
4. Returns the best parse tree covering the entire sentence (rooted at 'S') if available; otherwise returns a fallback first valid tree or `None`.

This ensures that input sentences are parsed into syntax trees according to learned grammatical rules and part-of-speech tags, with support for incremental learning of new rules based on observed data.

---

#### `update_parameters_from_config`

The `update_parameters_from_config` function in the `SyntaxLearner` class serves as a placeholder for future functionality to dynamically update parameters based on configuration changes.

- **Parameters:**
  None

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

- **Behavior and Side Effects:**
  Currently, this function does nothing (it passes). It is intended as a placeholder for future implementation where the class might need to update its parameters based on configuration changes dynamically. This could include updating attributes like `dp_alpha`, `lexicon`, or other internal state variables from a provided configuration manager.

This ensures that the method can be expanded in the future without breaking existing code, allowing for dynamic parameter updates if needed.

---

#### `add_sentence`

The `add_sentence` function in the `SyntaxLearner` class processes an input sentence to update the grammar rules or lexicon.

- **Parameters:**
  - `sentence`: A string representing the input sentence to be parsed and added to the model's knowledge base.

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

The function performs the following actions:
1. Splits the input sentence into tokens (words).
2. Iterates through each token in the sentence.
3. If a word (`t`) is not already present in the lexicon, adds it with a default part-of-speech tag ('NN' for noun) and an initial count of 1.

This ensures that new words encountered during parsing are added to the lexicon with basic POS tagging information, facilitating incremental learning of grammar rules based on observed data.

---

#### `generate_response`

The `generate_response` function in the `SyntaxLearner` class generates a grammatically structured response about a given target.

- **Parameters:**
  - `target`: A string representing the target for which to generate a sentence.

- **Return Value:**
  Returns a formatted string that represents a simple template sentence about the target. For now, it uses a placeholder format.

The function performs the following action:
1. Constructs and returns a basic template response in the form of `TPL.HYP.REL word='target' candidate='unknown'`.

This ensures that a grammatically structured output is generated for the specified target, even though the current implementation is very simple and will be expanded later.

---

#### `save`

The `save` function in the `SyntaxLearner` class saves the current state of the model to a specified file.

- **Parameters:**
  - `path`: A string representing the filepath where the model's state should be saved.

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

The function performs the following actions:
1. Creates a temporary filename by appending ".tmp" to the specified path.
2. Opens and writes the serialized (pickled) model state, including `lexicon`, `rule_counts`, and `rule_total`, to the temporary file in binary write mode.
3. Renames or moves the temporary file to the final destination using `os.replace` to ensure atomicity.

This ensures that the current internal state of the `SyntaxLearner` object is safely saved to disk, preventing data loss during the save process by writing to a temporary location first and then replacing the original file.

---

#### `load`

The `load` function in the `SyntaxLearner` class loads the model's state from a specified file.

- **Parameters:**
  - `path`: A string representing the filepath from which to load the model's state.

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

The function performs the following actions:
1. Opens and reads the serialized (pickled) data from the specified file in binary read mode.
2. Unpickles the loaded data, which includes `lexicon`, `rule_counts`, and `rule_total`.
3. Updates the internal attributes of the `SyntaxLearner` object with the deserialized data.

This ensures that the model's previous state is restored correctly from disk, allowing for continued operation where it left off previously.

---


### WordStateAssociator
The `WordStateAssociator` class tracks associations between words/phrases and user-defined state labels (e.g., "greeting"), supporting multiple labels per word with confidence scores. It includes features for conflict detection, resolution through user feedback, and integration of interaction history.

Key attributes include:
- `word_to_labels`: A dictionary mapping each word to a set of associated labels with their respective confidence scores.
- `label_to_words`: A dictionary mapping each label to the set of words associated with it.
- `word_label_stats`: Detailed statistics for each word-label pair, used for probabilistic tracking and normalization.
- `conflict_resolution_threshold`: The minimum difference in confidence required to automatically resolve conflicts.
- `conflict_history`: A list recording resolved conflicts.

The class is responsible for adding associations between words and labels, normalizing confidence scores based on user feedback, detecting potential label ambiguities (conflicts), resolving these conflicts through user input, and saving/loading the state of its internal data structures.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `WordStateAssociator` class and initializes its attributes.

- **Parameters:**
  None

- **Return Value:**
  The constructor does not return any value; it initializes instance attributes instead.

The function sets up several key attributes:
1. `self.word_to_labels`: A dictionary where each word maps to another dictionary of labels with their respective confidence scores.
2. `self.label_to_words`: A dictionary mapping each label to a set of words associated with that label.
3. `self.word_label_stats`: Detailed statistics for each word-label pair, used for probabilistic tracking and normalization.
4. `self.conflict_resolution_threshold`: The minimum difference in confidence required to automatically resolve conflicts (set to 0.3 by default).
5. `self.conflict_history`: An empty list that will store resolved conflict records.

This ensures that a new instance of `WordStateAssociator` is properly initialized with necessary attributes for tracking word-label associations, managing statistics, and handling conflicts.

---

#### `_ensure_word_stats`

The `_ensure_word_stats` function in the `WordStateAssociator` class ensures that detailed statistics are available for each word-label pair.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `interaction_index`: An integer (default is 0) indicating an interaction index used to track different stages of interactions or sessions.

- **Return Value:**
  Returns the dictionary containing statistics (`stats`) for each label associated with the given word.

The function performs the following actions:
1. Retrieves existing statistics and labels for the specified word.
2. Calculates the total confidence score across all labels associated with the word, ensuring it is non-negative.
3. For each label-confidence pair, calculates an initial probability based on its relative contribution to the total confidence.
4. Ensures that detailed probabilistic statistics are available for each label-word combination using `ensure_probabilistic_stats`.

This ensures that comprehensive and normalized statistical data is maintained for tracking word-label associations over multiple interactions or sessions.

---

#### `_normalize_word_labels`

The `_normalize_word_labels` function in the `WordStateAssociator` class normalizes and updates confidence scores for labels associated with a given word.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `interaction_index`: An integer (default is 0) indicating an interaction index used to track different stages of interactions or sessions.
  - `reward`: A float (default is 0.0) that represents the reward value for updating statistics.
  - `touched_label`: A string representing a specific label that was interacted with, if applicable.
  - `weight`: A float (default is 0.0) indicating the weight of the interaction.

- **Return Value:**
  Returns the dictionary containing updated statistics (`stats`) for each label associated with the given word.

The function performs the following actions:
1. Ensures that detailed statistics are available for each label-word pair using `_ensure_word_stats`.
2. Normalizes and updates probability fields in these statistics based on interaction data.
3. Updates `self.word_to_labels` to reflect only active labels (non-archived) with their probabilities.
4. Synchronizes the set of words associated with each label (`label_to_words`) by removing any outdated associations.

This ensures that confidence scores are consistently updated and maintained for tracking word-label associations over multiple interactions or sessions, while also handling archival and synchronization of related data structures.

---

#### `get_label_probabilities`

The `get_label_probabilities` function in the `WordStateAssociator` class retrieves and normalizes confidence scores (probabilities) for labels associated with a given word.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `interaction_index`: An integer (default is 0) indicating an interaction index used to track different stages of interactions or sessions.
  - `include_archived`: A boolean (default is False) that determines whether archived labels should be included in the output.

- **Return Value:**
  Returns a dictionary mapping each label associated with the word to its probability score as a float.

The function performs the following actions:
1. Normalizes and updates statistics for the given word using `_normalize_word_labels`.
2. Constructs a dictionary of probabilities for active labels (non-archived) unless `include_archived` is True, in which case archived labels are included as well.

This ensures that confidence scores are consistently updated and maintained for tracking word-label associations over multiple interactions or sessions, while also providing an option to include historical data if needed.

---

#### `add_association`

The `add_association` function in the `WordStateAssociator` class adds a new association between a word and a label, updating confidence scores as necessary.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `label`: A string representing a specific label to associate with the word.
  - `confidence`: A float (default is 1.0) indicating the initial or updated confidence score for this association.
  - `source`: A string (default is "explicit") specifying the source of the association, e.g., user input or automated process.
  - `interaction_index`: An integer (default is 0) indicating an interaction index used to track different stages of interactions or sessions.
  - `reward`: A float (default is 0.0) that represents a reward value for updating statistics.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Ensures detailed statistics are available for each label-word pair using `_ensure_word_stats`.
2. Adds or updates the confidence score for the specified word-label association.
3. Normalizes and updates probability fields in these statistics based on interaction data.
4. Updates the set of words associated with the given label (`label_to_words`).

This ensures that associations are consistently added, updated, and maintained while handling potential conflicts between labels and normalizing confidence scores over multiple interactions or sessions.

---

#### `get_labels`

The `get_labels` function in the `WordStateAssociator` class retrieves labels associated with a given word that have confidence scores above a specified threshold.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `threshold`: A float (default is 0.5) indicating the minimum required probability for inclusion of a label in the output list.

- **Return Value:**
  Returns a sorted list of labels associated with the given word, where each label has a confidence score greater than the specified threshold.

The function performs the following actions:
1. Retrieves and normalizes the probabilities (confidence scores) for all labels associated with the input word using `get_label_probabilities`.
2. Filters out labels whose probability is less than or equal to the threshold.
3. Sorts the remaining labels by their confidence score in descending order.

This ensures that only high-confidence labels are returned, providing a filtered and sorted list of relevant associations for the given word based on user-defined thresholds.

---

#### `get_words`

The `get_words` function in the `WordStateAssociator` class retrieves a list of words associated with a given label that have confidence scores above a specified threshold.

- **Parameters:**
  - `label`: A string representing a specific label.
  - `threshold`: A float (default is 0.5) indicating the minimum required probability for inclusion of a word in the output list.

- **Return Value:**
  Returns a list of words associated with the given label, where each word has an association confidence score greater than the specified threshold.

The function performs the following actions:
1. Iterates through all words associated with the input label.
2. Retrieves and normalizes the probabilities (confidence scores) for labels associated with each word using `get_label_probabilities`.
3. Filters out words whose probability of being associated with the given label is less than or equal to the threshold.
4. Collects the remaining words into a list.

This ensures that only high-confidence associations are returned, providing a filtered list of relevant words based on user-defined thresholds for the specified label.

---

#### `find_similar_words_by_label`

The `find_similar_words_by_label` function in the `WordStateAssociator` class finds other words that share at least one label with a given word.

- **Parameters:**
  - `word`: A string representing a specific word.

- **Return Value:**
  Returns a list of strings, where each element is another word that shares at least one common label with the input word and has confidence scores above a specified threshold (0.3).

The function performs the following actions:
1. Retrieves labels associated with the given word using `get_labels` with a threshold of 0.3.
2. For each retrieved label, finds all words associated with that label using `get_words`.
3. Collects these similar words into a set to remove duplicates and ensures the input word itself is not included in the result by discarding it from the set.
4. Converts the resulting set of similar words back into a list.

This function provides a way to identify related or semantically similar words based on shared labels, ensuring that only high-confidence associations are considered.

---

#### `detect_conflicts`

The `detect_conflicts` function in the `WordStateAssociator` class identifies pairs of labels for a given word that have similar confidence scores, indicating potential ambiguity.

- **Parameters:**
  - `word`: A string representing a specific word.

- **Return Value:**
  Returns a list of tuples, where each tuple contains two strings representing label pairs with similar confidence scores.

The function performs the following actions:
1. Checks if the given word has any associated labels.
2. Retrieves and normalizes the probabilities (confidence scores) for all labels associated with the input word using `get_label_probabilities`.
3. Iterates through each pair of labels to compare their confidence scores.
4. Identifies pairs where the absolute difference in confidence is below a predefined threshold (`conflict_resolution_threshold`).
5. Collects these conflicting label pairs into a list.

This function helps identify potential ambiguities or conflicts between different labels associated with the same word, ensuring that only significant differences are considered for resolution.

---

#### `break_bridge`

The `break_bridge` function in the `WordStateAssociator` class resolves conflicts by reducing the confidence of a rejected label and updating statistics accordingly.

- **Parameters:**
  - `word`: A string representing a specific word.
  - `label1`: A string representing one conflicting label associated with the word.
  - `label2`: A string representing another conflicting label associated with the word.
  - `keep_label`: A string indicating which of the two labels should be kept (the other will be reduced).
  - `interaction_index`: An integer (default is 0) indicating an interaction index used to track different stages of interactions or sessions.
  - `reward`: A float (default is -1.0) that represents a reward value for updating statistics.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Determines which label to reject based on the input labels and keep_label.
2. Adds an association with `keep_label` at full confidence (1.0) using `add_association`.
3. Ensures detailed statistics are available for each label-word pair using `_ensure_word_stats`.
4. Reduces the usage count of the rejected label by half, updates its reward trace, and normalizes word-label probabilities.
5. Records the resolved conflict in a history list (`conflict_history`).

This function ensures that conflicts between labels associated with the same word are properly resolved based on user input or feedback, updating internal statistics to reflect this resolution while maintaining historical records for future reference.

---

#### `save`

The `save` function in the `WordStateAssociator` class saves the current state of its internal data structures to a specified file.

- **Parameters:**
  - `path`: A string representing the filepath where the model's state should be saved.

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

The function performs the following actions:
1. Converts internal data structures (sets) into lists for JSON serialization.
2. Constructs a dictionary `data` containing all relevant attributes (`word_to_labels`, `label_to_words`, `word_label_stats`, and `conflict_history`) in a format suitable for JSON dumping.
3. Opens the specified file path in write mode and writes the serialized data to it using `json.dump`.

This ensures that the current state of the model is saved correctly, allowing for future loading and continued operation from where it left off previously.

### Example Conversion:
- Converts sets like `{word: {'label1': 0.8}}` into dictionaries.
- Converts sets in `label_to_words` to lists before saving.

This function ensures that all internal data structures are properly serialized and saved to disk for persistent storage.

---

#### `load`

The `load` function in the `WordStateAssociator` class loads previously saved state from a specified file into its internal data structures.

- **Parameters:**
  - `path`: A string representing the filepath from which to load the model's state.

- **Return Value:**
  The function does not return any value; it is designed to perform side effects only.

The function performs the following actions:
1. Checks if the specified file exists.
2. Opens and reads the serialized data from the file using `json.load`.
3. Reconstructs internal dictionaries (`word_to_labels`, `label_to_words`) and other attributes (`word_label_stats` and `conflict_history`) based on the loaded JSON data.
4. Ensures that detailed statistics are available for each word-label pair by calling `_ensure_word_stats` with an interaction index of 0.

This function ensures that previously saved state is correctly restored, allowing continued operation from where it left off previously.

### Example Reconstruction:
- Converts dictionaries in `data['word_to_labels']` back into the appropriate defaultdict structure.
- Converts lists in `data['label_to_words']` back into sets for proper internal representation.

The side effects include updating all relevant attributes to match the loaded state, ensuring consistency and continuity of operations.

---


### KnowledgeGraph
The `KnowledgeGraph` class stores a heterogeneous graph of knowledge with nodes representing words, phrases, states (each node has a type and unique identifier) and edges representing relationships between nodes with confidence scores and metadata. It supports incremental updates, transitive inference, merging nodes, pruning weak edges, and saving/loading the graph state.

Key attributes include:
- `inference_log`: A list to store logs of inferences.
- `nodes`: A dictionary mapping node IDs to their type, label, embedding (optional), and metadata.
- `edges`: A nested defaultdict representing relationships between nodes with confidence scores.
- `edge_list`: A list for fast iteration over edges.
- `node_counter`: An integer used to generate unique node IDs.

The class is responsible for adding and updating nodes and edges, applying symbolic rules to infer new relations, merging similar nodes, pruning weak edges, retrieving central (high-degree) nodes, saving the graph state to a file, and loading it back from a file.

### Functions

#### `__init__`

The `__init__` function is the constructor for the `KnowledgeGraph` class and initializes its attributes.

- **Parameters:**
  None

- **Return Value:**
  The constructor does not return any value; it initializes instance attributes instead.

The function sets up several key attributes:
1. Initializes an empty list to store inference logs (`inference_log`).
2. Creates a dictionary `nodes` where each node ID maps to its type, label, embedding (optional), and metadata.
3. Sets up nested defaultdicts for edges (`edges`) that support multiple relation types between the same nodes.
4. Initializes a list `edge_list` for fast iteration over edges with tuples containing `(node1, node2, rel_type, confidence, timestamp)`.
5. Creates a dictionary `edge_metadata` to store metadata associated with each edge using a unique key format (`"node1|node2|rel_type"`).
6. Initializes an integer counter `node_counter` for generating unique node IDs.

Additionally, the function calls `_load()` to load any existing state from disk upon initialization.

This ensures that a new instance of `KnowledgeGraph` is properly initialized with necessary attributes and loaded data if available.

---

#### `_next_node_id`

The `_next_node_id` function generates a unique node ID for the `KnowledgeGraph` class.

- **Parameters:**
  None

- **Return Value:**
  Returns a string representing the next available unique node ID in the format `"n<counter>"`, where `<counter>` is an integer incremented each time this method is called.

The function performs the following actions:
1. Increments `self.node_counter` by 1.
2. Formats and returns the new node ID as a string using the pattern `"n{node_counter}"`.

This ensures that each call to `_next_node_id` generates a unique identifier for nodes in the knowledge graph, facilitating proper management of nodes with distinct IDs.

---

#### `add_node`

The `add_node` function in the `KnowledgeGraph` class adds a new node to the graph and returns its unique ID.

- **Parameters:**
  - `label`: A string representing the label of the node.
  - `node_type`: A string (default is "word") specifying the type of the node.
  - `embedding`: An optional numpy array (`np.ndarray`) representing an embedding for the node.
  - `metadata`: An optional dictionary providing additional metadata about the node.

- **Return Value:**
  Returns a string representing the unique ID assigned to the newly added or existing node.

The function performs the following actions:
1. Checks if a node with the same label and type already exists in the graph by iterating through the nodes.
2. If an identical node is found, returns its existing ID.
3. Otherwise, generates a new unique node ID using `_next_node_id`.
4. Adds the new node to `self.nodes` dictionary with details including type, label, embedding (if provided), and metadata (defaulting to an empty dict if not specified).
5. Returns the newly generated or found node ID.

This ensures that nodes are uniquely identified and properly added to the graph while avoiding duplicates based on their label and type.

---

#### `get_node_id`

The `get_node_id` function in the `KnowledgeGraph` class retrieves the unique ID of a node based on its label and type.

- **Parameters:**
  - `label`: A string representing the label of the node.
  - `node_type`: A string (default is "word") specifying the type of the node.

- **Return Value:**
  Returns an optional string representing the unique ID of the node if found, otherwise returns `None`.

The function performs the following actions:
1. Iterates through all nodes in the graph.
2. Checks if a node with the specified label and type exists by comparing their attributes.
3. If such a node is found, its unique ID (`nid`) is returned.
4. If no matching node is found after iterating through all nodes, `None` is returned.

This ensures that the function can efficiently retrieve the unique identifier of a specific node based on its label and type, or indicate if no such node exists in the graph.

---

#### `apply_rule`

The `apply_rule` function in the `KnowledgeGraph` class applies a symbolic rule to update relationships within the graph.

- **Parameters:**
  - `rule`: A dictionary representing the symbolic rule. The format is:
    ```python
    {'if': (subject_predicate, relation, object_predicate),
     'then': (new_subject, new_relation, new_object)}
    ```
    where each predicate can be a variable or constant.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Checks if the rule is of type `'transitive'` and has a specified relation.
2. If so, extracts the relevant relationship (`rel`) from the rule.
3. Builds an adjacency list for this specific relation by iterating through all edges in the graph.
4. Applies transitive closure using breadth-first search (BFS) with a depth limit:
   - For each node `start` that has outgoing edges of type `rel`, perform BFS to find nodes connected via multiple steps.
   - If there is no direct edge between two nodes or if the cumulative confidence score exceeds an existing one, add the new edge and log the inference.

This ensures that transitive relationships are inferred based on specified rules, updating the graph with higher-confidence edges while logging each inference for later review.

---

#### `_log_inference`

The `_log_inference` function in the `KnowledgeGraph` class logs an inference for later review.

- **Parameters:**
  - `n1`: A string representing the ID of the first node involved in the inference.
  - `n2`: A string representing the ID of the second node involved in the inference.
  - `rel`: A string representing the relationship type between the nodes.
  - `conf`: A float representing the confidence score for the inferred edge.
  - `rule_name`: A string specifying the name or identifier of the rule that generated this inference.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Checks if an attribute `inference_log` exists in the instance.
2. If not, initializes `self.inference_log` as an empty list.
3. Appends a dictionary entry to `self.inference_log`, containing details of the inference including timestamp, node IDs (`n1`, `n2`), relationship type (`rel`), confidence score (`conf`), and rule name (`rule_name`).

This ensures that each inferred edge is logged with relevant information for later review or analysis.

---

#### `add_edge`

The `add_edge` function in the `KnowledgeGraph` class adds or updates an edge between two nodes.

- **Parameters:**
  - `node1`: A string representing the ID of the first node.
  - `node2`: A string representing the ID of the second node.
  - `rel_type`: A string specifying the type of relationship between the nodes.
  - `confidence`: A float (default is 1.0) indicating the confidence score for this edge.
  - `metadata`: An optional dictionary providing additional metadata about the edge.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Checks if both nodes exist in the graph by verifying their presence in `self.nodes`.
2. If either node is missing, returns without further action.
3. Updates or adds the edge between `node1` and `node2` with the specified relationship type (`rel_type`) and confidence score (`confidence`).
4. Stores metadata for this edge if provided using a unique key format `"node1|node2|rel_type"`.
5. Adds an entry to `self.edge_list`, which is used for fast iteration over edges.
6. Optionally, adds the reverse edge (from `node2` to `node1`) and corresponding metadata for symmetric relationships like 'synonym', 'similar', or 'co_occur'.

This ensures that edges are properly added or updated in a directed manner unless specified otherwise, while also maintaining separate lists for easy iteration over all edges.

---

#### `get_neighbors`

The `get_neighbors` function in the `KnowledgeGraph` class retrieves a list of neighboring nodes connected to a specified node with optional filtering by relationship type.

- **Parameters:**
  - `node_id`: A string representing the ID of the node for which neighbors are requested.
  - `rel_type`: An optional string specifying the type of relationship. If provided, only edges matching this type will be included in the result.

- **Return Value:**
  Returns a list of tuples where each tuple contains `(neighbor_id, rel_type, confidence)`, representing neighboring nodes connected to the specified node along with their relationship types and confidence scores.

The function performs the following actions:
1. Checks if the given `node_id` exists in the graph's edges.
2. If not found, returns an empty list.
3. Iterates through all relationships (`rels`) of the specified node.
4. For each edge, checks if it matches the provided relationship type or if no specific type is required.
5. Appends matching tuples to a neighbors list.

This ensures that neighboring nodes and their connecting edges are retrieved based on optional filtering by relationship type.

---

#### `get_all_edges`

The `get_all_edges` function in the `KnowledgeGraph` class retrieves a list of all edges in the graph.

- **Parameters:**
  None

- **Return Value:**
  Returns a list of tuples where each tuple contains `(node1, node2, rel_type, confidence)`, representing an edge between two nodes along with their relationship type and confidence score.

The function performs the following actions:
1. Initializes an empty list `edges` to store all edges.
2. Iterates through all nodes (`n1`) in the graph's edges dictionary.
3. For each node, iterates through its connected nodes (`n2dict`).
4. For each edge between a pair of nodes, iterates through their relationship types and confidence scores.
5. Appends tuples representing each edge to the `edges` list.

This ensures that all edges in the graph are retrieved for further processing or analysis.

---

#### `infer_transitive_relations`

The `infer_transitive_relations` function in the `KnowledgeGraph` class infers new edges for transitive relations using a breadth-first search (BFS) approach.

- **Parameters:**
  - `rel_type`: A string representing the type of relationship to infer, typically 'is_a' or 'means'.
  - `threshold`: A float (default is 0.5) indicating the minimum confidence score required for an edge to be considered in inference.

- **Return Value:**
  Returns a list of tuples where each tuple contains `(node1, node2, rel_type, confidence)` representing inferred edges based on transitive relationships.

The function performs the following actions:
1. Checks if `rel_type` is one of 'is_a' or 'means'.
2. If not, returns an empty list.
3. Builds an adjacency list for nodes connected by the specified relationship type and above a certain confidence threshold.
4. Uses BFS to traverse from each node up to depth 3 (to find paths of length 2+):
   - Tracks visited nodes to avoid cycles.
   - For each level, checks if there is not already a direct edge with higher or equal confidence between the start and current node.
5. If no such direct edge exists, adds an inferred tuple for the new edge.

This ensures that transitive relationships are properly inferred by finding indirect paths of length 2+ while avoiding redundant edges based on existing high-confidence connections.

---

#### `merge_nodes`

The `merge_nodes` function in the `KnowledgeGraph` class merges one node into another, transferring all edges and deleting the merged node.

- **Parameters:**
  - `node_id1`: A string representing the ID of the first node.
  - `node_id2`: A string representing the ID of the second node to be merged into the first.
  - `keep_id`: An optional string specifying which node should remain after merging. If not provided, defaults to `node_id1`.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Determines whether `node_id1` or `node_id2` will be kept based on the optional parameter `keep_id`.
2. Checks if the node to remove (`remove_id`) exists in the graph.
3. Transfers all edges connected to `remove_id` (both incoming and outgoing) to `keep_id`, updating edge lists accordingly.
4. Deletes any references to `remove_id` from the adjacency dictionary (`self.edges`).
5. Removes the node with ID `remove_id` from the nodes list.

This ensures that one node is merged into another, transferring all connected edges while maintaining graph integrity and consistency.

---

#### `prune_weak_edges`

The `prune_weak_edges` function in the `KnowledgeGraph` class removes edges with confidence scores below a specified threshold.

- **Parameters:**
  - `threshold`: A float (default is 0.1) indicating the minimum confidence score required for an edge to be retained.

- **Return Value:**
  The function does not return any value; it performs side effects only.

The function performs the following actions:
1. Initializes a list `to_remove` to store edges that will be deleted.
2. Iterates through all nodes (`n1`) in the graph's adjacency dictionary (`self.edges`).
3. For each node, iterates through its connected nodes (`n2dict`).
4. For each edge between two nodes, checks if the confidence score is below the specified threshold and adds it to `to_remove`.
5. Deletes edges with low confidence scores from the adjacency dictionary.
6. Removes empty dictionaries for nodes that no longer have any connections after pruning.

This ensures that weak or less confident edges are removed while maintaining graph integrity by cleaning up unused entries in the adjacency dictionary.

---

#### `get_central_nodes`

The `get_central_nodes` function in the `KnowledgeGraph` class returns the node IDs with the highest degree (number of edges).

- **Parameters:**
  - `top_k`: An integer (default is 10) specifying how many top nodes to return based on their degree.

- **Return Value:**
  Returns a list of tuples where each tuple contains `(node_id, degree)` representing the node ID and its corresponding degree.

The function performs the following actions:
1. Initializes a `degree` dictionary to count the number of edges for each node.
2. Iterates through all nodes (`n1`) in the graph's adjacency dictionary (`self.edges`).
3. For each edge connected to a node, increments the degree count for both endpoints (`n1` and `n2`).
4. Sorts the nodes by their degree in descending order.
5. Returns the top `top_k` nodes with the highest degrees.

This ensures that central or highly interconnected nodes are identified based on their number of edges, providing a measure of node importance within the graph.

---


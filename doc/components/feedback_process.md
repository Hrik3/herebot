## Doc for feedback_process
---

This Python file defines a `ContextualSentimentCorpus` class that maintains and learns from user feedback to improve sentiment prediction in conversational AI systems. The corpus stores past interactions as `FeedbackEntry` instances, which include timestamps, input/output text, emotional factors, ratings, and tokens. It uses Beta distributions to model the probability of positive/negative/neutral sentiments given different contexts (tokens, factor states, response templates). Confidence thresholds and minimum sample requirements are used to determine when sufficient feedback has been gathered for each context, allowing the system to stop asking users for more sentiment evaluations once it becomes highly confident in its predictions.

---

### FeedbackEntry
The `FeedbackEntry` class stores a single instance of user feedback for an AI interaction. It includes the timestamp, user input and system response with their tokenized versions, emotional/control state factors at the time of response, a rating indicating sentiment or context-specific reaction, and optional metadata such as the template ID if used.


### ContextualSentimentCorpus
The `ContextualSentimentCorpus` class maintains a corpus of user feedback with full context, learning sentiment probabilities conditioned on various contexts such as tokens, emotional factors, and response templates. It automatically stops asking for more feedback when confidence in the learned models exceeds a specified threshold. Key attributes include the list of feedback entries (`entries`) and dictionaries mapping different contextual keys to Beta distribution parameters representing positive, negative, and neutral sentiments (`context_models`).

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `ContextualSentimentCorpus` class. It sets up the list of feedback entries (`entries`) and a dictionary for context models (`context_models`). The function also calls `_load()` to load existing data from a file into these attributes.

Parameters:
- None

Return value:
- None

Important behavior:
- Initializes `self.entries` as an empty list.
- Initializes `self.context_models` as an empty dictionary.
- Calls the private method `_load()` to potentially populate `entries` and `context_models` by loading data from a file.

---

#### `_load`

The `_load` function attempts to load a serialized dictionary from a file into the instance attributes `self.entries` and `self.context_models`.

Parameters:
- None

Return value:
- None

Important behavior:
- Checks if the specified corpus file (`CORPUS_FILE`) exists.
- Opens the file in binary read mode and loads its contents using `pickle.load`.
- Updates `self.entries` with the 'entries' key from the loaded data, defaulting to an empty list if not present.
- Updates `self.context_models` with the 'context_models' key from the loaded data, defaulting to an empty dictionary if not present.
- Catches and prints any exceptions that occur during file loading or deserialization.

Side-effects:
- Populates `self.entries` and `self.context_models` based on the contents of the serialized file.
- Prints an error message if there is a problem loading the file.

---

#### `save`

The `save` function serializes and saves the instance attributes `self.entries` and `self.context_models` to a file.

Parameters:
- None

Return value:
- None

Important behavior:
- Creates a temporary filename by appending ".tmp" to `CORPUS_FILE`.
- Opens the temporary file in binary write mode.
- Dumps a dictionary containing `entries` and `context_models` using `pickle.dump`.
- Renames the temporary file to replace the original corpus file.

Side-effects:
- Serializes and writes the instance attributes to disk, overwriting any existing data at `CORPUS_FILE`.

Purpose:
- Ensures that the current state of feedback entries and context models is saved persistently.

---

#### `add_entry`

The `add_entry` function adds a new feedback entry to the corpus and updates the context models accordingly.

Parameters:
- `entry`: A `FeedbackEntry` object representing user feedback.

Return value:
- None

Important behavior:
- Appends the provided `entry` to the list of entries (`self.entries`).
- Calls `_update_model_for_entry(entry)` to update the context models based on the new entry.
- Calls `save()` to persist the updated corpus and context models to disk.

Side-effects:
- Modifies `self.entries` by adding a new feedback entry.
- Updates `self.context_models`.
- Persists changes to disk by saving the updated data.

Purpose:
- Adds user feedback to the corpus while ensuring that the associated context models are also updated and saved persistently.

---

#### `_update_model_for_entry`

The `_update_model_for_entry` function updates the Beta distribution parameters for all relevant contexts based on a given feedback entry.

Parameters:
- `entry`: A `FeedbackEntry` object representing user feedback.

Return value:
- None

Important behavior:
- Converts the rating to lowercase and checks if it is one of 'positive', 'negative', or 'neutral' (or their context-specific variants). If not, returns early.
- Extracts the base sentiment from the normalized rating ('positive', 'negative', or 'neutral').
- Iterates over each token in `entry.context_tokens` and updates the corresponding Beta distribution using `_update_context_model`.
- Iterates over emotional factors in `['threat', 'trust', 'curiosity', 'energy', 'conscience_active']`, checks their values, categorizes them into high, low, or mid ranges, and updates the relevant context models.
- If a response template ID is available, it updates the corresponding Beta distribution.

Side-effects:
- Updates `self.context_models` with new counts for each relevant context-sentiment pair.

Purpose:
- Incrementally update sentiment probability estimates based on user feedback while considering different contextual factors.

---


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

#### `_update_context_model`

The `_update_context_model` function updates the Beta distribution parameters for a specific context-sentiment pair.

Parameters:
- `context_key`: A string representing the key of the contextual information.
- `sentiment`: A string indicating the sentiment ('positive', 'negative', or 'neutral').

Return value:
- None

Important behavior:
- Checks if `context_key` is already in `self.context_models`. If not, initializes it with default Beta distributions for positive, negative, and neutral sentiments.
- Retrieves the relevant Beta distribution parameters (`a`, `b`) from `self.context_models[context_key][sentiment]`.
- Updates the count of either 'a' or 'b' based on the sentiment:
  - Increments 'a' if the sentiment is 'positive'.
  - Increments 'b' otherwise.

Side-effects:
- Modifies `self.context_models` by updating the Beta distribution parameters for the specified context-sentiment pair.

Purpose:
- Incrementally update the counts of positive, negative, and neutral sentiments in a Bayesian manner using Beta distributions.

---

#### `get_sentiment_probability`

The `get_sentiment_probability` function returns the posterior mean probability of a given sentiment for a specified context.

Parameters:
- `context_key`: A string representing the key of the contextual information.
- `sentiment`: A string indicating the sentiment ('positive', 'negative', or 'neutral').

Return value:
- A float representing the posterior mean probability of the specified sentiment in the given context.

Important behavior:
- Checks if `context_key` is present in `self.context_models`. If not, returns a uniform prior probability of 0.33.
- Retrieves the relevant Beta distribution parameters (`a`, `b`) from `self.context_models[context_key][sentiment]`.
- Computes and returns the posterior mean probability using the formula:
  \[
  P(\text{sentiment} | \text{context}) = \frac{a}{a + b}
  \]

Side-effects:
- None

Purpose:
- Provides a probabilistic estimate of sentiment given a specific context, based on previously observed feedback.

---

#### `get_confidence`

The `get_confidence` function calculates and returns the confidence level for the most probable sentiment in a given context.

Parameters:
- `context_key`: A string representing the key of the contextual information.

Return value:
- A float representing the maximum confidence (1 - posterior variance) among all sentiments for the specified context.

Important behavior:
- Checks if `context_key` is present in `self.context_models`. If not, returns 0.0.
- Iterates over each sentiment (`sent`) and its corresponding Beta distribution parameters (`a`, `b`) from `self.context_models[context_key]`.
- Computes the probability of the sentiment using the formula:
  \[
  P(\text{sentiment} | \text{context}) = \frac{a}{a + b}
  \]
- Calculates the posterior variance for each sentiment using the formula:
  \[
  \text{variance} = \frac{ab}{(a+b)^2 (a+b+1)}
  \]
- Computes confidence as \(1 - \text{variance}\).
- Tracks and returns the maximum confidence among all sentiments.
- Ensures that there are at least `self.MIN_SAMPLES` samples before returning a non-zero confidence.

Side-effects:
- None

Purpose:
- Provides an estimate of how confident the model is in predicting the most probable sentiment for a given context, based on previously observed feedback.

---

#### `should_ask_feedback`

The `should_ask_feedback` function determines whether more feedback is needed for the current context based on confidence levels.

Parameters:
- `tokens`: A list of strings representing tokens from user input.
- `factor_dict`: A dictionary mapping emotional factors to their values (floats).
- `template_id`: An optional string representing a response template ID, defaulting to None if not provided.

Return value:
- A boolean indicating whether more feedback is needed (`True`) or not (`False`).

Important behavior:
- Checks the confidence level for each token context. If any token's confidence is below `self.CONFIDENCE_THRESHOLD`, returns `True`.
- Iterates over emotional factors in `['threat', 'trust', 'curiosity', 'energy', 'conscience_active']`. For each factor, categorizes its value into high, low, or mid ranges and checks the corresponding context's confidence. If any factor context has a confidence below `self.CONFIDENCE_THRESHOLD`, returns `True`.
- Checks the response template ID if provided. If the confidence for this context is below `self.CONFIDENCE_THRESHOLD`, returns `True`.

Side-effects:
- None

Purpose:
- Determines whether additional feedback should be requested based on the current level of confidence in sentiment predictions for different contextual factors and tokens, ensuring that more data is collected when necessary to improve model accuracy.

---



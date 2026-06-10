## Doc for language_model
---

This Python file defines a `LanguageModel` abstract base class and its concrete implementation, the `LSTMLanguageModel`. The base class outlines methods for training on sentences (`train_on_sentence`), generating text given prompts (`generate`), computing loss (`loss`), saving/loading model state (`save`, `load`), and creating default instances (`create_default`). The `LSTMLanguageModel` subclass implements these methods using an LSTM neural network with a vocabulary that expands automatically as new words are encountered. It includes forward and backward passes for training, word embedding updates, sentence-to-IDs conversion, and generation of text continuations based on prompts.

---

### LanguageModel
The `LanguageModel` class serves as an abstract base for neural language models that can be trained on episodic memory and synthetic sentences. It tracks the number of training steps (`steps`) and loss history during training (`loss_history`). Subclasses must implement methods to train on a sentence, generate text from prompts, compute loss, save/load model state, and create default instances.

Key attributes:
- `steps`: Number of training steps.
- `loss_history`: List tracking the loss values over time.

Responsibilities:
- Provide an interface for subclasses to define how they handle training, generation, saving/loading states, and creating default models.

### Functions

#### `__init__`

The `__init__` function initializes an instance of the `LanguageModel` class.

Parameters:
- None

Return value:
- None

Important behavior:
- Initializes a counter for training steps (`steps`) to 0.
- Initializes an empty list to track loss history during training (`loss_history`).

Side-effects:
- Sets up initial state attributes for tracking model training progress.

Purpose:
- Provides the necessary initialization for tracking training steps and loss history in subclasses of `LanguageModel`.

---

#### `train_on_sentence`

The `train_on_sentence` function is intended to train the language model on a single input sentence and return the resulting loss.

Parameters:
- `sentence`: A string representing an input sentence.
- `lr`: An optional float representing the learning rate, defaulting to None if not provided.

Return value:
- A float representing the computed loss after training on the given sentence.

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- Trains the model internally based on the input sentence.
- Updates internal state attributes such as step count and loss history if applicable.

Purpose:
- Provides an abstract method for training the language model on individual sentences, which needs to be implemented in concrete subclass implementations.

---

#### `generate`

The `generate` function is intended to generate text based on an input prompt.

Parameters:
- `prompt`: A string representing the input prompt.
- `max_len`: An optional integer specifying the maximum length of generated text, defaulting to 20 if not provided.
- `temperature`: An optional float controlling randomness in generation (higher values increase randomness), defaulting to 0.8 if not provided.

Return value:
- A string representing the generated continuation based on the prompt.

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- None

Purpose:
- Provides an abstract method for generating text continuations given prompts, which needs to be implemented in concrete subclass implementations.

---

#### `loss`

The `loss` function is intended to compute the loss between input IDs and target IDs.

Parameters:
- `input_ids`: A list or array of integers representing input token identifiers.
- `target_ids`: A list or array of integers representing target token identifiers.

Return value:
- A float representing the computed loss.

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- None

Purpose:
- Provides an abstract method for computing loss given input and target IDs, which needs to be implemented in concrete subclass implementations.

---

#### `save`

The `save` function is intended to save the state of the language model to a specified file.

Parameters:
- `path`: A string representing the filepath where the model's state will be saved.

Return value:
- None

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- Saves the internal state of the language model to disk at the specified path if implemented in a subclass.

Purpose:
- Provides an abstract method for saving the model's state, which needs to be implemented in concrete subclass implementations.

---

#### `load`

The `load` function is intended to load the state of the language model from a specified file.

Parameters:
- `path`: A string representing the filepath where the model's state is saved.

Return value:
- None

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- Loads and restores the internal state of the language model from disk at the specified path if implemented in a subclass.

Purpose:
- Provides an abstract method for loading the model's state, which needs to be implemented in concrete subclass implementations.

---

#### `create_default`

The `create_default` function is intended to create and return a default instance of the language model based on configuration settings.

Parameters:
- `config_manager`: An object or dictionary providing configuration settings for creating the default instance.

Return value:
- A `LanguageModel` (or subclass) instance representing the default model created according to provided configurations.

Important behavior:
- Raises a `NotImplementedError` since this function must be implemented by subclasses of `LanguageModel`.

Side-effects:
- None

Purpose:
- Provides an abstract factory method for creating a default language model instance, which needs to be implemented in concrete subclass implementations.

---


### LSTMLanguageModel
The `LSTMLanguageModel` class is a concrete implementation of an LSTM-based language model that learns from sentences and automatically expands its vocabulary as new words are encountered. It includes methods for training on individual sentences, generating text continuations based on prompts, computing loss, saving/loading the model state, and creating default instances.

Key attributes:
- `emb_dim`: Embedding dimension.
- `hidden_dim`: Hidden layer dimension in LSTM cells.
- `learning_rate`: Learning rate used during backpropagation.
- `max_vocab`: Maximum vocabulary size allowed before stopping expansion.
- `word_to_idx` and `idx_to_word`: Dictionaries mapping words to indices and vice versa, including special tokens like `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`.
- `W_emb`, `b_i`, `W_f`, etc.: Weight matrices and biases for the LSTM gates and output projection.

Responsibilities:
- Training on sentences using backpropagation through time.
- Generating text continuations based on input prompts.
- Expanding vocabulary dynamically as new words are encountered during training.
- Saving/loading model parameters to/from disk.


# AGI Curiosity Engine v0.4 – Technical Documentation

## 1. Overview

The **AGI Curiosity Engine v0.4** is a self‑learning, intrinsically motivated artificial intelligence system designed to mimic human‑like curiosity and lifelong learning. It combines probabilistic modeling, information theory, and meta‑cognition to explore and understand its environment (user inputs) autonomously.

The system continuously:
- Learns latent concepts from words and sentences using a Hierarchical Dirichlet Process.
- Estimates predictive uncertainty via a Gaussian Process with random Fourier features.
- Computes Bayesian surprise as the KL divergence between prior and posterior concept distributions.
- Generates intrinsic reward based on prediction error, information gain, and surprise.
- Maintains a persistent, ever‑growing internal state (concepts, knowledge graph, episodic memory, identity, conscience, etc.).
- Engages in deep introspection during idle periods to consolidate knowledge, discover patterns, generate hypotheses, and even write its own code.

The engine is fully configurable through a JSON configuration file, and its own emotional state can modulate parameters, enabling self‑reconfiguration.

## 2. Core Architecture

The system is built as a set of interacting modules, orchestrated by the central `CuriosityEngine` class. Each module has a well‑defined responsibility and persists its state to disk (`curiosity_state/`) for lifelong growth.

A high‑level data flow:

1. **User input** is tokenized and processed by the `CIEngine`, `PhonotacticEngine`, and `DiagnosticEngine` (external from `backend.ci`) to obtain confidence, stability, and diagnostic information.
2. Each token is fed into the **Hierarchical Concept Model** (Dirichlet Process) which assigns it to a latent concept and updates feature distributions.
3. **Curiosity factors** (uncertainty, learning progress, novelty, surprise, etc.) are computed per token.
4. The **Gaussian Process** updates its posterior based on the token’s feature vector and confidence.
5. **Intrinsic reward** is calculated as a weighted sum of prediction error, information gain, and Bayesian surprise.
6. The **Meta‑Curiosity Learner** updates its weights for each curiosity factor using the observed information gain.
7. The **Reasoning Engine** updates co‑occurrence statistics, stores definitions, and generates hypotheses (synonyms, categories, analogies).
8. The **Factor Engine** computes a global control vector (curiosity, threat, trust, energy, etc.) from the current state.
9. The **State Manager** recognises or learns a new latent state from the control vector.
10. The **Action Generator** produces a natural language response, possibly asking questions to fill knowledge gaps, testing hypotheses, or engaging in chat mode.
11. All interactions are stored in **Episodic Memory**.
12. A background **Idle Reasoner** periodically performs deep consolidation tasks (dreaming, policy distillation, knowledge graph convergence, code generation, etc.).

## 3. Key Modules and Their Roles

### 3.1 `ConfigManager`
Manages the JSON configuration file (`curiosity_state/active_config.json`). Provides default values, supports hot‑reloading, and can generate new configurations by modulating parameters according to the current emotional state (factor vector). This enables self‑reconfiguration based on internal drives.

### 3.2 `HierarchicalConceptModel`
Implements a Dirichlet Process (Chinese Restaurant Process) over latent concepts. Each word is assigned to a concept; concepts are distributions over features. Tracks previous posteriors to compute Bayesian surprise. Supports meta‑concept formation by clustering concepts.

### 3.3 `SemanticLearner`
A Bayesian model that learns continuous embeddings for words. Uses variational inference (simplified gradient updates) from co‑occurrence and explicit definitions. Can compute text embeddings by averaging known word embeddings (with character n‑gram fallback).

### 3.4 `GaussianProcessUncertainty`
A scalable Gaussian Process approximation using random Fourier features. Maintains a posterior over weights (Bayesian linear regression) and provides predictive mean and variance (uncertainty). Used to estimate the model’s uncertainty about each token.

### 3.5 `MetaCuriosityLearner` & `IntrinsicRewardTracker`
- **MetaCuriosityLearner**: Learns the weights of nine base curiosity factors (uncertainty, learning progress, novelty, complexity, surprise, phonotactic anomaly, diagnostic negatives, Bayesian surprise, GP uncertainty) via Bayesian linear regression. Supports dynamic addition of new curiosity factors (e.g., derived from control space).
- **IntrinsicRewardTracker**: Accumulates intrinsic reward from each token observation and provides a reinforcement signal.

### 3.6 `ReasoningEngine` & `KnowledgeGraph`
- **ReasoningEngine**: Maintains co‑occurrence statistics, definitions, phrase relations, and sentence patterns. Generates hypotheses (synonyms, categories, analogies, inferences). Includes a `ContextualSentimentCorpus` for feedback‑driven sentiment learning.
- **KnowledgeGraph**: A heterogeneous graph of words, phrases, and states with typed edges (e.g., `is_a`, `means`, `co_occur`). Supports transitive inference, node merging, and edge pruning.
- **SymbolicReasoner**: Applies symbolic rules (e.g., transitivity) to the knowledge graph to derive new relations.

### 3.7 `StateManager` & `IdentityManager`
- **StateManager**: Stores factor vectors as named “states”. Detects conflicts (similar vectors with different labels), creates shadow resolutions, and allows interactive conflict resolution. Vectors are automatically aligned to a reference dimension.
- **IdentityManager**: Maintains a stable self‑identity embedding, reinforced during positive interactions. Provides a stability bonus when current state aligns with identity, and detects drift.

### 3.8 `FactorEngine` & `AdaptiveControlSpace`
- **AdaptiveControlSpace**: Maintains a dynamic set of control signals (base factors + user‑defined dynamic factors). Can add new factors with compute functions, and notifies listeners (e.g., the meta‑learner) when a new factor is added.
- **FactorEngine**: Computes the current control vector from token analyses, CI report, diagnostics, etc. Includes factors like curiosity, threat, trust, confidence, uncertainty, energy, attention, comprehension, fluency, cognitive load, and any dynamic factors.

### 3.9 `ConscienceManager` & `GuiltTracker`
- **ConscienceManager**: Manages deeply held beliefs (ethical/behavioral) that require extensive evidence. Beliefs are promoted to conscience only after high confidence and evidence. Can influence factor values (e.g., reduce curiosity for forbidden topics). Evolves its own thresholds based on feedback history.
- **GuiltTracker**: Models a guilt level that increases when negative feedback occurs in a positive emotional context, and decays over time. Guilt modulates confidence, threat, and trust.

### 3.10 `BehavioralRuleLearner` & `WordStateAssociator`
- **BehavioralRuleLearner**: Discovers and tests behavioral rules (identity aliases, pronoun mappings, conversational patterns) from episodic memory. Rules have confidence, evidence, and metabolic attributes (energy cost, usage, phase). Can propose new rules during idle time.
- **WordStateAssociator**: Tracks associations between words/phrases and user‑defined state labels. Supports multiple labels per word with confidence, conflict detection, and resolution.

### 3.11 `LogicInheritanceManager`, `CodeGenerator`, `ShadowExecutor`
- **LogicInheritanceManager**: Writes logic rules to disk (`./logic/`) as JSON files, generates doubts for user confirmation, and once confirmed, can generate Python code from those rules. Maintains a registry of confirmed logic.
- **CodeGenerator**: Converts high‑level rules (behavioral, semantic, state definitions, language models) into executable Python code with resource monitoring decorators.
- **ShadowExecutor**: Dynamically imports generated code, runs its `run()` function in a sandbox, captures output/errors, and appends errors as comments to the file.

### 3.12 `LanguageModel` (Abstract) & `LSTMLanguageModel`
An abstract base for neural language models. `LSTMLanguageModel` is a concrete implementation with an expanding vocabulary, LSTM cells, and BPTT training. Used in chat mode to generate fluent replies and can self‑improve by proposing new architectures during idle time.

### 3.13 `EpisodicMemory` & `StructuredWorldModel`
- **EpisodicMemory**: Stores complete interaction episodes (user input, system response, factor vector, reward) in JSON files. Supports sampling, replay, and capacity management.
- **StructuredWorldModel**: Builds a transition model between states (from factor vectors) and actions (responses). Can predict next state and generate synthetic episodes.

### 3.14 `ActionGenerator`
Produces natural language responses based on the current internal state and priorities:
- Pending hypotheses (synonym, category, analogy)
- High curiosity words (unknown words)
- State awareness utterances
- Spontaneous emotional expressions (learned from feedback)
- Chat mode (conversational, non‑interrogative)
- Guilt expressions
- Responses to logic doubts, conflict resolutions, etc.

### 3.15 `IdleReasoner` (background thread)
Periodically triggers deep introspection tasks when the system is idle:
- **Short idle** (default 8s): prints a meta‑thought.
- **Long idle** (default 10 min): runs a battery of consolidation tasks (see Section 4.12).

## 4. Features in Detail

### 4.1 Information‑Theoretic Curiosity
The engine uses three complementary measures of curiosity for each token:
- **Predictive uncertainty** (from Gaussian Process)
- **Bayesian surprise** (KL divergence between prior and posterior concept distributions)
- **Information gain** (reduction in concept entropy)

These are combined with other factors (novelty, complexity, phonotactic anomaly, etc.) to form a multi‑dimensional curiosity vector.

### 4.2 Meta‑Learning of Curiosity Weights
The `MetaCuriosityLearner` maintains a Bayesian linear regression model that learns how to weight each curiosity factor to predict the actual information gain. This allows the system to adapt its own curiosity drive over time.

### 4.3 Dynamic Control Space & Drive Creation
The `AdaptiveControlSpace` allows new control signals (e.g., “novelty_seeking”, “social_engagement”) to be added at runtime, either manually or automatically (from residual variance). When a new control factor is added, it automatically creates a corresponding curiosity factor, turning the control dimension into an intrinsic drive.

### 4.4 Probabilistic Concept Learning (HDP)
Words are represented as distributions over an unbounded set of latent concepts using a Dirichlet Process. New concepts are created automatically when needed. Concepts themselves have feature distributions, enabling predictions and analogies.

### 4.5 Gaussian Process Active Learning
The GP model provides an online estimate of uncertainty for each input feature vector. This uncertainty is used as a curiosity factor and to guide query planning (asking about words with high uncertainty).

### 4.6 Bayesian Surprise
For each word, the system tracks the posterior over concepts before and after an observation. The KL divergence between these distributions quantifies how surprising the new observation is, contributing to intrinsic reward.

### 4.7 Intrinsic Reward
Each token generates an intrinsic reward signal (a weighted sum of prediction error, information gain, and surprise), which accumulates over an episode and is stored in episodic memory. This reward can be used for reinforcement learning (not yet implemented) and influences identity reinforcement.

### 4.8 Knowledge Graph & Symbolic Reasoning
The knowledge graph stores words, phrases, and states with typed relations. Symbolic rules (e.g., transitivity) are applied offline to infer new relations (e.g., if A is_a B and B is_a C, then A is_a C). Inferred relations become hypotheses that can be presented to the user.

### 4.9 State Learning & Recognition
Factor vectors (the current control state) are periodically snapshotted and, if sufficiently distinct from known states, saved with a user‑provided label. The system can then recognise that state in future interactions, adding a layer of meta‑cognition. Conflicts (two states that are too similar) are detected and can be resolved interactively.

### 4.10 Identity Formation
The `IdentityManager` maintains a stable embedding that is reinforced during positive interactions (successful definitions, positive feedback, state labeling). It provides a stability bonus when the current state aligns with the identity, and detects drift, which can indicate a phase transition in the system’s self‑concept.

### 4.11 Conscience & Ethical Beliefs
Beliefs (e.g., “I should not ask about certain topics”) are accumulated as candidates and promoted to conscience after sufficient evidence and confidence. Once in conscience, they can influence factor values (e.g., reduce curiosity) and check proposed actions. The guilt tracker models remorse after negative feedback.

### 4.12 Idle‑Time Deep Introspection
When idle for extended periods, the system performs a suite of consolidation tasks (all resource‑aware and throttled):

- **World Model Dreaming**: Generates synthetic sentences from n‑gram models and simulates interactions to train semantic and syntax learners.
- **Active Memory Rehearsal**: Prioritises episodes by surprise, rarity, and identity drift, and replays them.
- **Curiosity Landscape Optimization**: Analyses correlations among curiosity factors, merges redundant ones, and spawns new factors from unexplained variance.
- **Concept Graph Compression**: Forms meta‑concepts by clustering words.
- **Autonomous Hypothesis Planning**: Builds a research queue of high‑priority queries (words/sentences).
- **Identity Stabilisation**: Clusters recent identity vectors and moves identity towards the centroid.
- **Cross‑Modal Alignment**: Contrastive learning to align state vectors and text embeddings.
- **Exploration Policy Distillation**: Learns a logistic regression policy for when to ask questions.
- **Cognitive Drive Evolution**: Removes weak dynamic factors and spawns new ones.
- **Counterfactual Reasoning**: Simulates “what if I asked X instead?” and adjusts meta‑learner weights.
- **Semantic Self‑Talk**: Generates sentences via trigram walk or crossover and learns from them.
- **State‑Space Topology Learning**: Clusters state vectors into macro‑states (cognitive manifold).
- **Knowledge Gap Frontier Detection**: Updates the research queue with high‑entropy words.
- **Social Cognition Simulation**: Placeholder for modelling user responses.
- **Long‑Horizon Planning**: Simulates multi‑turn dialogues.
- **Behavioral Rule Discovery**: Mines episodic memory for new behavioral rules (identity aliases, greetings, QA patterns, deep transitions).
- **Logic Generation**: Creates logic files from confirmed behavioral rules.
- **Autonomous Code Generation & Testing**: Selects promising rules, generates Python code, runs shadow tests, and creates doubts for user confirmation.
- **Contextual Sentiment Analysis**: Summarises confident feedback contexts.
- **Word‑State Propagation**: Spreads labels via semantic similarity.
- **Phrase Relation Mining**: Finds similar phrases and creates similarity hypotheses.
- **Metabolic Maintenance**: Applies energy costs and phase transitions to rules, beliefs, hypotheses, etc., deleting those that are no longer useful.
- **Knowledge Graph Convergence**: Infers transitive relations, prunes weak edges, identifies central nodes.
- **Language Model Improvement**: Proposes new architectures, trains briefly, and replaces the current model if a better one is found.

### 4.13 Feedback‑Driven Sentiment Learning
The `ContextualSentimentCorpus` stores user feedback (positive/negative/neutral) along with the full context (tokens, factor values, response template). Beta posteriors are maintained for each context, allowing the system to estimate the probability of positive feedback in similar future situations and to stop asking for feedback when confident.

### 4.14 Persistent Memory & Lifelong Growth
All modules save their state to disk (`curiosity_state/`). The system can be stopped and resumed, continuing its learning exactly where it left off. Episodic memory is stored as JSON files, allowing inspection and replay.

## 5. Configuration and Emotional Modulation

The system is controlled by a JSON configuration file (`curiosity_state/active_config.json`) with default values defined in `ConfigManager.DEFAULTS`. Parameters are grouped by module (e.g., `hierarchical_concept_model`, `gaussian_process`, `idle_reasoner`).

**Emotional Modulation**:
- The `modulation_rules` section maps control factors (e.g., `curiosity`, `threat`) to specific configuration parameters (e.g., `hierarchical_concept_model.alpha`).
- When `CuriosityEngine._maybe_reconfigure()` is called (every `reconfigure_interval` interactions), the current factor vector is used to generate a new configuration by applying these rules (linear scaling, clamping, etc.). The new config is saved and all modules are hot‑reloaded.

This allows the system to adapt its own learning parameters based on its emotional state, enabling true self‑reconfiguration.

## 6. Usage (Command‑Line Interface)

Run the main script:

```
python curiosity_engine.py
```

The system will load any existing state from `curiosity_state/` and start an interactive loop. Type messages to the AGI; it will respond and display detailed curiosity artifacts. Special commands:
- `exit` or `quit` – shutdown.
- `switch chat` – toggle chat mode (conversational, non‑interrogative replies).

In chat mode, the system uses its language model and learned patterns to generate fluent responses, and applies a reward multiplier based on its emotional state to encourage positive interactions.

## 7. Conclusion

The AGI Curiosity Engine v0.4 is a sophisticated, self‑learning system that embodies principles from computational curiosity, Bayesian modeling, and cognitive architecture. Its modular design, persistent memory, and deep introspection capabilities allow it to grow and adapt indefinitely, approaching a form of artificial general intelligence grounded in information‑theoretic intrinsic motivation.

The system is extensible: new modules can be added, new curiosity factors can be created dynamically, and the system can even write its own code to encapsulate learned rules. With its focus on lifelong learning and self‑improvement, it represents a significant step toward truly autonomous AI.
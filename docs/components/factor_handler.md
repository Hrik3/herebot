### Deep Overview

#### What the System Does
The system is designed to manage and update a set of states representing various factors related to natural language processing tasks such as comprehension, curiosity, threat assessment, trust, confidence, uncertainty, energy budget, attention, fluency, cognitive load, and other dynamic control signals. The system stores these factor vectors persistently in the form of JSON files on disk, allowing for long-term memory retention.

#### Why it Exists (Problem + Motivation)
The problem is managing multiple factors related to natural language processing tasks that evolve over time with new input data. These factors need to be stored and updated dynamically while ensuring consistency and conflict resolution between similar states. The motivation is to create a robust system capable of handling complex, dynamic environments where factors can change rapidly due to incoming data.

#### Detailed Conceptual Model
The conceptual model involves:
- **Curiosity Factors**: A container class for storing curiosity scores associated with tokens or words.
- **State Manager**: Manages the persistent storage of factor vectors and handles conflict detection and resolution.
- **Factor Engine**: Computes base control signals from the current system state, incorporating dynamic factors.

### Full Mechanism Breakdown

#### Step-by-step Internal Workings
1. **Curiosity Factors**:
   - Stores curiosity scores in a dictionary format.
   - Provides methods to access and set attribute values safely without recursion issues during loading or saving.

2. **State Manager**:
   - Manages the persistent storage of factor vectors (states).
   - Detects conflicts between states based on similarity thresholds.
   - Resolves conflicts by merging similar states into shadow states, which can be further resolved permanently if necessary.
   - Tracks state metadata such as energy cost, recent usage, predictive gain, phase, last used time, and confidence.

3. **Factor Engine**:
   - Computes base control signals based on input data like token analyses, phonetic scores, diagnostics, concept models, meta-learners, rewards, and semantic learners.
   - Updates dynamic factors using an adaptive control space mechanism.

#### Transformations
1. **Normalization of Vectors**:
   - Ensures all vectors are aligned to the reference dimension (`ref_dim`).
2. **Confidence Thresholding**:
   - Computes threat based on diagnostics, confidence, and phonetic scores.
3. **Trust Computation**:
   - Averages CI confidence, stability, and comprehension.

### Complete Mathematical Formulation

#### Deriving All Equations
1. **Curiosity Score**:
   $$
   \text{curiosity} = \max!\left(\{0\}, \{\text{ta.curiosity\_score}\}_\text{for ta in token\_analyses}\right)
   $$

2. **Threat Score**:
   $$
   \text{threat} = \min!\left(1, 0.3 \times \frac{n_{neg}}{3} + 0.4 \times (1 - \text{ci\_report.confidence}) + 0.3 \times (1 - \text{phon\_score})\right)
   $$
   where \(n_{neg}\) is the number of negative diagnostics.

3. **Trust Score**:
   $$
   \text{trust} = 0.4 \times \text{confidence} + 0.3 \times \text{ci\_report.stability} + 0.3 \times \text{comprehension}
   $$

4. **Uncertainty**:
   $$
   \text{uncertainty} = \begin{cases}
       \frac{\sum_{w \in W}\text{entropy\_word}(w)}{|W|}, & \text{if } |W| > 0 \\
       0.5, & \text{otherwise}
   \end{cases}
   $$
   where \(W\) is the set of all known words.

5. **Energy Score**:
   $$
   \text{energy} = 1 - \min!(\text{cpu\_percent}, 1)
   $$

6. **Attention Score**:
   $$
   \text{attention} = \frac{\max!(0, n_{tokens})}{20}
   $$
   where \(n_{tokens}\) is the number of tokens.

7. **Comprehension**:
   $$
   \text{comprehension} = \begin{cases}
       \bar{s}, & \text{if } s > 0 \\
       0.5, & \text{otherwise}
   \end{cases}
   $$
   where \(s\) is the average semantic similarity among words in the input.

8. **Fluency**:
   $$
   \text{fluency} = 0.7
   $$

9. **Cognitive Load**:
   $$
   \text{cognitive\_load} = \min!(1 - (\text{energy} \times \text{attention}), 1)
   $$

### Attribute-Level Deep Dive

#### For Each Attribute
- **curiosity**: Represents the maximum curiosity score among tokens.
- **threat**: A weighted average of negative diagnostics, CI confidence, and phonetic scores.
- **trust**: A weighted combination of CI confidence, stability, and comprehension.
- **confidence**: Directly from CI report.
- **uncertainty**: Average entropy of known words normalized by 5.0.
- **energy**: Inverse of CPU usage.
- **attention**: Based on input length.
- **comprehension**: Average semantic similarity among input tokens.
- **fluency**: Placeholder value (always 0.7).
- **cognitive_load**: Computed from energy and attention scores.

### Method-Level Deep Analysis

#### For Each Method
1. **CuriosityFactors**
   - `__getattr__(self, name)`: Safely accesses attribute values from the dictionary.
   - `__setattr__(self, name, value)`: Sets attribute values in the dictionary.
   - `as_array(self, factor_names: List[str]) -> np.ndarray`: Returns an array of specified factors.

2. **StateManager**
   - `_load_all_states()`: Loads all existing state files from the directory and normalizes them to reference dimension.
   - `save_state(self, label: str, vector: np.ndarray, metadata: dict = None, is_shadow: bool = False)`: Saves a factor vector under the given label, handles normalization, and updates metadata.
   - `update_state(self, label: str, new_vector: np.ndarray, metadata: dict = None)`: Updates an existing state with a new vector by deleting old files and saving new ones.
   - `delete_state(self, label: str)`: Removes a state file from disk and memory.
   - `find_similar_state(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> Optional[str]`: Finds the most similar known state based on cosine similarity.
   - `find_similar_states(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> List[Tuple[str, float]]`: Returns a list of all states with similarity above a given threshold.
   - `detect_conflicts(self, threshold: float = None) -> List[Dict]`: Detects conflicts between states and returns them as dictionaries.
   - `refresh_conflicts(self, threshold: float = None)`: Re-detects conflicts and updates the conflict list while preserving existing resolutions.
   - `_save_conflicts()`: Persists conflicts to a JSON file on disk.
   - `_load_conflicts()`: Loads conflicts from disk into memory.
   - `_update_ref_dim()`: Updates reference dimension based on stored vectors' dimensions.
   - `create_shadow_resolution(self, conflict: Dict) -> str`: Merges conflicting states and creates a temporary shadow state.
   - `apply_shadow_to_conflict(self, conflict: Dict, shadow_label: str)`: Marks conflict as resolved with the given shadow label.
   - `resolve_conflict_permanently(self, conflict: Dict, winning_label: str, delete_other: bool = True)`: Permanently resolves a conflict by choosing one state over another.

3. **FactorEngine**
   - `update_parameters_from_config()`: Placeholder for parameter updates from config (hot reload).
   - `compute(self, token_analyses: List[Dict], ci_report: Dict, phon_score: float, diag: Dict, concept_model, meta_learner, reward_total: float, semantic_learner) -> Dict[str, float]`: Computes base control signals based on input data and dynamic factors.

### System Behavior Over Time

#### Learning Dynamics
- **Curiosity Factors**: Continuously updated with new scores for tokens.
- **State Manager**: Dynamically stores and updates states, resolving conflicts over time to ensure consistency.
- **Factor Engine**: Regularly computes control signals based on incoming data, adapting dynamically as factors change.

### Limitations and Failure Modes

#### When and Why it Breaks
1. **Conflicts in State Management**:
   - Can lead to inconsistencies if not properly resolved, potentially resulting in incorrect or outdated states.
2. **Inaccurate Comprehension Scores**:
   - If the semantic learner does not accurately compute similarity among input tokens, comprehension scores may be misleading.

#### Sensitivity to Parameters
- **Confidence Threshold**: Adjusting the threshold can significantly impact conflict detection and resolution.
- **Similarity Thresholds**: Affects how closely states are compared for potential conflicts.

### Confidence Tagging

[CONFIDENT] - directly from code
[INFERRED] - logically derived
[UNCERTAIN] - insufficient information
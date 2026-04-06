### Deep Overview

#### What the System Does
The system is a behavioral rule learner that identifies, tests, and integrates rules governing conversational behavior from past interactions. These rules can be of different types: identity aliases (mapping words/phrases to self-reference), pronoun mappings, response templates for specific inputs, or action rules based on certain states.

#### Why It Exists
The primary motivation is to improve the agent's understanding of its own role and how it should respond in various conversational contexts. This system helps the agent learn from user interactions and refine its responses over time, leading to more coherent and contextually appropriate behavior.

### Detailed Conceptual Model

- **Behavioral Rules**: These are learned rules that govern specific behaviors based on past experiences.
  - Identity aliases: Maps a word/phrase to self-reference (e.g., "herebot" = "I").
  - Pronoun mapping: Links pronouns like "you" or "he/she" to either the agent itself or another entity.
  - Response templates: Specific patterns for responding to certain inputs.
  - Action rules: Tendency to perform specific actions in given states.

- **Rule Engine**: Executes deterministic rules when conditions are met, modifying the knowledge base accordingly.
- **Constraint Propagator**: Simple constraint propagation mechanism that maintains a set of constraints and applies them iteratively until no new facts can be added (fixpoint).

### Full Mechanism Breakdown

#### Step-by-step Internal Workings

1. **Rule Class**:
    - `__init__(self, name: str, condition: Expression, action: Callable, reasoning_engine: 'ReasoningEngine')` initializes a rule with a name, condition expression, and an action function.
    - `matches(self, context: Dict) -> bool`: Evaluates the given condition in the specified context using the provided reasoning engine. Returns `True` if the condition is met, otherwise `False`.
    - `execute(self, context: Dict)`: Executes the rule's action on the given context.

2. **Rule Engine Class**:
    - `__init__(self)` initializes an empty list of rules.
    - `add_rule(self, rule: Rule)`: Adds a new rule to the engine's list.
    - `run(self, context: Dict)`: Iterates through all rules and executes any that match the current context.

3. **Constraint Propagator Class**:
    - `__init__(self, knowledge_graph)` initializes with a reference to the knowledge graph and an empty list of constraints.
    - `add_constraint(self, constraint_func: Callable)`: Adds a new constraint function to the list.
    - `propagate(self, max_iter=10)`: Iteratively applies all constraints until no new facts can be added.

4. **Behavioral Rule Class**:
    - Represents a behavioral rule with attributes like type, condition, action, confidence, prior evidence, energy cost, recent usage, and phase (active, weak, dormant, archived, deleted).

5. **BehavioralRuleLearner Class**:
   - `__init__(self, config_manager: Optional['ConfigManager'] = None)`: Initializes with a configuration manager.
   - `_load(self)`: Loads previously saved behavioral rules from disk if they exist.
   - `record_usage(self, rule_id: str)`: Tracks the recent usage of a specific rule.
   - `save(self)`: Saves current state to disk.
   - `add_candidate_rule(self, rule: BehavioralRule)`: Adds a new candidate rule, removing an older one if maximum limit is reached.
   - `update_confidence(self, rule_id: str, delta: float, feedback: bool = False)`: Updates the confidence of a specific rule based on incremental evidence.
   - `decay_confidences(self)`: Decays the confidence of unconfirmed rules over time.
   - `get_testable_rules(self, context: Dict) -> List[BehavioralRule]`: Returns a list of testable rules that meet certain criteria (confidence threshold and not already asked).
   - `_matches_condition(self, rule: BehavioralRule, context: Dict) -> bool`: Checks if the given rule's condition matches the provided context.
   - `propose_new_rules(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`: Proposes new rules based on patterns detected in episodic memory and recent interactions.
     - `detect_greeting_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`: Detects common opening utterances (greetings) and their typical responses to generate conversational pattern rules.
     - `detect_qa_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`: Identifies question-answer pairs from episodic memory to create response templates for specific questions.
   - `deep_pattern_mining(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`: Offline mining of longer sequences (Markov chain) to identify patterns in user-agent interactions and generate conversational rules.

### Complete Mathematical Formulation

#### Feature Transformations
- **Confidence Update**:
$$ \text{new\_confidence} = \max(0.0, \min(1.0, \text{old\_confidence} + \Delta)) $$
Where $\Delta$ is the incremental evidence (positive or negative).

- **Energy Cost and Usage Smoothing**:
$$ \text{recent\_usage} = (\text{current\_usage} \times 0.99^{\Delta t}) + 0.1 $$
Where $\Delta t$ is the time since last usage.

#### Probability Equations
- **Rule Matching Condition**:
The condition for a rule to match in context can be expressed as:
$$ P(\text{match}|C) = \begin{cases} 
1 & \text{if } E(C; \text{condition}) \\
0 & \text{otherwise}
\end{cases} $$
Where $E(C; \text{condition})$ is the evaluation of condition in context $C$.

- **Rule Confidence Decay**:
The confidence decays over time if not confirmed, given by:
$$ \text{new\_confidence} = \max(0.0, \min(1.0, \text{old\_confidence} \times d^{t/\Delta t})) $$
Where $d$ is the decay rate, and $\Delta t$ is the time elapsed since last tested.

#### Update Rules
- **Confidence Update**:
$$ \text{new\_confidence} = \max(0.0, \min(1.0, \text{old\_confidence} + \delta)) $$
Where $\delta$ is the incremental evidence (positive or negative).

- **Recent Usage Smoothing**:
The smoothed usage count updates as follows:
$$ \text{recent\_usage} = (\text{current\_usage} \times 0.99^{\Delta t}) + 0.1 $$
Where $\Delta t$ is the time since last used.

#### Prediction Functions
- **Prediction of Matching Rules**:
The probability a rule matches given context can be modeled as:
$$ P(\text{match}|C) = \begin{cases} 
1 & \text{if } E(C; \text{condition}) \\
0 & \text{otherwise}
\end{cases} $$
Where $E(C; \text{condition})$ is the evaluation of condition in context $C$.

### Attribute-Level Deep Dive

#### Attributes
- **rule_id**: Unique identifier for each rule.
- **rule_type**: Type of behavioral rule (identity_alias, pronoun_mapping, conversational_pattern).
- **condition**: Condition under which the rule applies.
- **action**: Action taken when condition is met.
- **confidence**: Degree to which the system believes in this rule.
- **prior**: Initial belief before any evidence.
- **evidence**: Number of times confidence has been updated.
- **created**: Timestamp of creation.
- **last_tested**: Timestamp of last test.
- **asked**: Whether the rule was queried for confirmation.
- **confirmed**: Whether the rule's accuracy was confirmed by feedback.
- **energy_cost**: Cost per time unit to keep alive.
- **recent_usage**: Smoothed usage count over time.
- **predictive_gain**: Estimated contribution to prediction.
- **phase**: Active, weak, dormant, archived, or deleted state of the rule.

### Method-Level Deep Analysis

#### BehavioralRule Class
- `__init__(self, rule_id: str, rule_type: str, condition: Dict[str, Any], action: Dict[str, Any], confidence: float = 0.5, prior: float = 0.1, evidence: int = 0, created: float = time.time(), last_tested: Optional[float] = None, asked: bool = False, confirmed: bool = False, energy_cost: float = 1.0, recent_usage: float = 0.0, predictive_gain: float = 0.0, phase: str = "active", last_used: Optional[float] = None)`
- `matches(self, context: Dict) -> bool`: Evaluates the rule's condition in a given context.
- `execute(self, context: Dict)`: Executes the rule’s action on the provided context.

#### RuleEngine Class
- `__init__(self)`: Initializes an empty list of rules.
- `add_rule(self, rule: Rule)`: Adds a new rule to the engine's list.
- `run(self, context: Dict)`: Iterates through all rules and executes any that match the current context.

#### ConstraintPropagator Class
- `__init__(self, knowledge_graph)`
- `add_constraint(self, constraint_func: Callable)`
- `propagate(self, max_iter=10)`

#### BehavioralRuleLearner Class
- `__init__(self, config_manager: Optional['ConfigManager'] = None)`
- `_load(self)`
- `record_usage(self, rule_id: str)`
- `save(self)`
- `add_candidate_rule(self, rule: BehavioralRule)`
- `update_confidence(self, rule_id: str, delta: float, feedback: bool = False)`
- `decay_confidences(self)`
- `get_testable_rules(self, context: Dict) -> List[BehavioralRule]`
- `_matches_condition(self, rule: BehavioralRule, context: Dict) -> bool`
- `propose_new_rules(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`
- `detect_greeting_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`
- `detect_qa_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`
- `deep_pattern_mining(self, engine: 'CuriosityEngine') -> List[BehavioralRule]`

### System Behavior Over Time

#### Learning Dynamics
The system continuously learns and integrates new behavioral rules based on past interactions. These rules are refined over time through usage and feedback, with confidence levels adjusted accordingly.

#### Evolution with Data
As more data is added (new episodes of user-agent interaction), the learner identifies patterns, proposes new rules, tests them, and updates their confidence based on evidence from actual use.

### Limitations and Failure Modes

- **Inconsistent User Input**: If users provide inconsistent input or feedback, it may lead to incorrect rule generation.
- **Overfitting**: Rules might become too specific to past interactions and fail in novel scenarios.
- **Confidence Decay**: Unconfirmed rules decay over time if no new evidence is provided.
- **Resource Constraints**: The system can only maintain a fixed number of rules due to resource limitations, leading to potential loss of important rules.

### Confidence Tagging

[CONFIDENT] Directly from code
[INFERRED] Logically derived
[UNCERTAIN] Insufficient information
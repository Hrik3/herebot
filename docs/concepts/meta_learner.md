### Deep Overview

#### What the system does
The `MetaCuriosityLearner` class in `meta_learner.py` learns the weights of curiosity factors via Bayesian linear regression. The target is to model actual information gain (reduction in uncertainty) for each factor.

#### WHY it exists (problem + motivation)
This system addresses the need to understand and quantify different sources of curiosity-driven behavior by learning their relative importance through data. Traditional approaches often rely on predefined weights or fixed models, which may not capture dynamic changes over time. The Bayesian approach allows continuous adaptation based on new observations, improving accuracy and relevance.

#### Conceptual Model Explained in Detail
The model consists of a set of base curiosity factors (e.g., uncertainty, novelty) and dynamically learned weights for these factors. Each factor's importance is quantified by its weight in predicting information gain. The Bayesian linear regression framework enables the continuous updating of these weights as new data becomes available.

### Full Mechanism Breakdown

#### Initialization
Upon initialization, the class sets up base curiosity factors and initializes structures to store their weights:

```python
def __init__(self, config_manager: Optional['ConfigManager'] = None,
             n_factors: int = None):
    self.config_manager = config_manager
    
    # Base curiosity factors (always present)
    self.base_factor_names = [
        "uncertainty", "learning_progress", "novelty",
        "complexity", "surprise", "phonotactic_anomaly",
        "diagnostic_negatives", "bayesian_surprise", "gp_uncertainty"
    ]
    
    # Dynamic factor names (can be added later)
    self.dynamic_factor_names = []
    
    if config_manager:
        pass
    
    # Initialize weights for base factors
    self.n_factors = len(self.base_factor_names)
    self.mu = np.zeros(self.n_factors)
    self.lambda_ = np.ones(self.n_factors) * 0.1
    self.alpha = np.ones(self.n_factors) * 1.0
    self.beta = np.ones(self.n_factors) * 1.0
    self.factor_metadata: Dict[str, dict] = {}   # factor_name -> {energy_cost, recent_usage, predictive_gain, phase, last_used}
    
    self._load()
```

#### Internal Working Step-by-Step

##### Base Factors Initialization
Base curiosity factors are defined as predefined signals that always exist:

```python
self.base_factor_names = [
        "uncertainty", "learning_progress", "novelty",
        "complexity", "surprise", "phonotactic_anomaly",
        "diagnostic_negatives", "bayesian_surprise", "gp_uncertainty"
    ]
```

##### Weight Initialization
Weights for these base factors are initialized with zero means and small variances:

```python
self.mu = np.zeros(self.n_factors)
self.lambda_ = np.ones(self.n_factors) * 0.1
```

Alpha and beta parameters control the precision of prior distributions:
```python
self.alpha = np.ones(self.n_factors) * 1.0
self.beta = np.ones(self.n_factors) * 1.0
```

##### Factor Metadata Initialization
Metadata for each factor, including energy cost, usage history, predictive gain, phase, and last used time, is initialized:

```python
self.factor_metadata: Dict[str, dict] = {}   # factor_name -> {energy_cost, recent_usage, predictive_gain, phase, last_used}
```

### Complete Mathematical Formulation

#### Feature Transformations and Probability Equations

##### Bayesian Linear Regression Model
The model assumes a linear relationship between curiosity factors \( \mathbf{x} \) (input features) and the target variable \( y \) (information gain):

\[ y = \mathbf{w}^\top \mathbf{x} + \epsilon, \quad \epsilon \sim N(0, \sigma^2) \]

Where:
- \( \mathbf{x} \): Curiosity factors
- \( \mathbf{w} \): Weights to be learned
- \( \epsilon \): Noise term

The prior over weights is assumed to be Gaussian:

\[ p(\mathbf{w}) = \mathcal{N}(0, \lambda^{-1} I) \]

Where:
- \( \lambda \): Precision of the prior (inverse variance)

Given data points \( (\mathbf{x}_i, y_i) \), the posterior distribution over weights is:

\[ p(\mathbf{w} | \mathcal{D}) = \mathcal{N}(\mu_N, \Lambda_N^{-1}) \]

Where:
- \( \mu_N \): Posterior mean
- \( \Lambda_N \): Precision matrix

The update equations for the posterior are:

\[ \Lambda_N = \lambda I + X^\top X \]
\[ \mu_N = \Lambda_N^{-1} (X^\top y) \]

Where:
- \( X \): Matrix of input features
- \( y \): Vector of target variables

#### Update Rules
The system updates the posterior mean and precision matrix based on new observations:

```python
def update(self, x: np.ndarray, y: float):
    self.lambda_ += (x ** 2).sum(axis=0)
    self.mu = (self.beta * self.lambda_) / (self.alpha + self.beta) * x @ y
```

##### Prior and Likelihood Integration
The prior and likelihood are combined to update the posterior:

\[ \mu_N = \frac{\lambda}{\alpha} X^\top y \]
\[ \Lambda_N = \lambda I + X^\top X \]

Where:
- \( \alpha \): Precision of the prior
- \( \beta \): Inverse variance of the likelihood

#### Prediction Equations
Predictions are made by taking the expected value under the posterior distribution:

\[ \hat{y} = \mathbf{x}^\top \mu_N \]

### Attribute-Level Deep Dive

#### What it represents
Each attribute in the class represents a specific aspect of the model's state or functionality:

1. **self.base_factor_names (list)**
   - Represents predefined curiosity factors that are always present.
2. **self.dynamic_factor_names (list)**
   - Tracks dynamically discovered curiosity factors and their names.
3. **self.n_factors (int)**
   - Number of base factors in the system.
4. **self.mu (np.ndarray)**
   - Posterior mean weights for each factor.
5. **self.lambda_ (np.ndarray)**
   - Precision matrix for the prior distribution over weights.
6. **self.alpha (np.ndarray)**
   - Precision parameter for the prior distribution over weights.
7. **self.beta (np.ndarray)**
   - Inverse variance parameter for the likelihood of observations.
8. **self.factor_metadata (Dict[str, dict])** 
   - Metadata for each factor including energy cost and usage history.

#### Why it exists
Each attribute serves a specific purpose in maintaining the state and functionality of the model:

1. **base_factor_names** ensures that essential curiosity factors are always present.
2. **dynamic_factor_names** allows for adaptive behavior by adding or removing factors based on new observations.
3. **n_factors** tracks the number of base factors to ensure consistent updates.
4. **mu** stores the current posterior mean weights for each factor.
5. **lambda_** controls the precision of the prior distribution over weights.
6. **alpha** and **beta** control the precision of prior and likelihood distributions, respectively.
7. **factor_metadata** provides metadata about each factor to track usage and energy costs.

#### How it influences behavior over time
The attributes collectively influence how the model evolves as new data is processed:

1. **base_factor_names**: Ensures that essential curiosity factors are always present, providing a stable baseline.
2. **dynamic_factor_names**: Allows for adaptive behavior by adding or removing factors based on new observations.
3. **n_factors**: Tracks the number of base factors to ensure consistent updates.
4. **mu** stores the current posterior mean weights for each factor, which are updated over time.
5. **lambda_**, **alpha**, and **beta** control the precision of prior distributions, influencing how quickly the model adapts to new data.
6. **factor_metadata**: Tracks metadata about each factor, enabling informed decisions on usage and energy costs.

### Method-Level Deep Analysis

#### Purpose
Each method serves a specific purpose within the model:

1. **__init__(self, config_manager=None, n_factors=None)**
   - Initializes the class with configuration parameters or default settings.
2. **update(self, x: np.ndarray, y: float)**
   - Updates the posterior weights based on new observations.

#### FULL Internal Logic (Line-by-Line Explanation)

##### __init__(self, config_manager=None, n_factors=None)
1. Initialize `config_manager` for loading configuration parameters.
2. Define base curiosity factors as predefined signals.
3. Initialize structures to store dynamic factor names and metadata.
4. Set up weights and precision matrices for Bayesian linear regression.

##### update(self, x: np.ndarray, y: float)
1. Update the prior precision matrix based on new feature vector \( \mathbf{x} \).
2. Compute the posterior mean weights using the updated precision and observed target value \( y \).

#### Mathematical Reasoning Behind Operations
- The initialization sets up predefined base factors and structures to store dynamic factors.
- Updating the model involves integrating the prior and likelihood distributions based on new observations.

#### Edge Cases and Hidden Behaviors
1. If `config_manager` is not provided, default settings are used instead.
2. When updating weights, precision matrices ensure numerical stability by controlling how quickly the prior adapts to new data.

### System Behavior Over Time

#### How the Model Evolves with Data
The model evolves by continuously updating the posterior distribution over weights based on new observations:

\[ \mu_N = \frac{\lambda}{\alpha} X^\top y \]
\[ \Lambda_N = \lambda I + X^\top X \]

Where:
- \( X \): Matrix of input features
- \( y \): Vector of target variables

#### Learning Dynamics
The system's learning dynamics involve periodic updates to the posterior mean and precision matrix:

1. **Prior Integration**: The prior distribution over weights is integrated with new observations.
2. **Posterior Update**: The posterior mean and precision are updated based on observed data.

#### Stability / Convergence Insights
Stability is maintained through careful control of prior precision and likelihood variance:

- **Precision Control**: Higher values of \( \lambda \) make the prior more confident, reducing overfitting.
- **Variance Control**: Lower values of \( \sigma^2 \) (controlled by \( \beta \)) reduce noise in observations.

### Limitations and Failure Modes

#### When This Approach Breaks
- **Overfitting**: If precision parameters are too low, the model may adapt too quickly to noise.
- **Underfitting**: Conversely, if precision is too high, the model may not learn from new data effectively.

#### Sensitivity to Parameters
The behavior of the model is highly sensitive to the choice of parameters:

1. **lambda_** controls the prior confidence:
   - Higher values make the prior more confident.
   - Lower values allow for more flexibility in adapting to new data.
2. **alpha** and **beta** control the precision of prior and likelihood distributions, respectively.

#### Trade-offs (Accuracy vs Compute, Approximation Limits)
- **Flexibility**: Lower precision parameters allow faster adaptation but may lead to overfitting.
- **Stability**: Higher precision ensures robustness against noise but might miss important signals.
- **Complexity**: Adding more dynamic factors increases computational complexity but improves adaptability.

### Confidence Tagging

[CONFIDENT] directly from code
[INFERRED] logically derived
[UNCERTAIN] insufficient info
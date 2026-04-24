from importglobal import Optional, Dict,Callable,time,np,os,traceback
from components.config_manager import ConfigManager
from components.directory_handler  import STATE_DIR
from functionsglobal import align_factor_vector
class MetaCuriosityLearner:
    """
    Learns the weights of curiosity factors via Bayesian linear regression.
    Target = actual information gain (reduction in uncertainty).

    Now config-aware and supports dynamic expansion of factor set.
    """

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
            # We'll keep the config parameter for backward compatibility, but it's no longer fixed.
            # Actually, we ignore the config n_factors because we manage our own list.
            pass
        
        # Initialize weights for base factors
        self.n_factors = len(self.base_factor_names)
        self.mu = np.zeros(self.n_factors)
        self.lambda_ = np.ones(self.n_factors) * 0.1
        self.alpha = np.ones(self.n_factors) * 1.0
        self.beta = np.ones(self.n_factors) * 1.0
        self.factor_metadata: Dict[str, dict] = {}   # factor_name -> {energy_cost, recent_usage, predictive_gain, phase, last_used}
        # Dictionary to store compute functions for dynamic curiosity factors
        self.dynamic_compute_functions = {}  # factor_name -> callable
        self._load()

    @property
    def factor_names(self):
        """Complete list of all curiosity factors (base + dynamic)."""
        return self.base_factor_names + self.dynamic_factor_names

    def update_parameters_from_config(self):
        """Placeholder for hot reload (no fixed n_factors anymore)."""
        pass

    def add_factor(self, factor_name: str, initial_mu: float = 0.0,
                compute_function: Optional[Callable] = None,
                metadata: dict = None):

        if factor_name in self.factor_names:
            return

        # append ONLY to dynamic list
        self.dynamic_factor_names.append(factor_name)

        if compute_function is not None:
            self.dynamic_compute_functions[factor_name] = compute_function

        # resize
        self.mu = np.append(self.mu, initial_mu)
        self.lambda_ = np.append(self.lambda_, 1.0)
        self.alpha = np.append(self.alpha, 1.0)
        self.beta = np.append(self.beta, 1.0)

        self.n_factors = len(self.mu)

        # ---- metabolic metadata ----
        self.factor_metadata[factor_name] = {
            'energy_cost': 1.0,
            'recent_usage': 0.0,
            'predictive_gain': abs(initial_mu),
            'phase': 'active',
            'last_used': None
        }

        self.save(os.path.join(STATE_DIR, 'meta_learner.npz'))

    def compute_dynamic_factor(self, factor_name: str, token: str,
                               token_ci: Dict, token_phon: float,
                               diag: Dict, features_array: np.ndarray,
                               global_factor_dict: Optional[Dict] = None) -> float:
        """
        Compute the value of a dynamic curiosity factor using its registered compute function.
        Returns 0.0 if no function is registered.
        """
        if factor_name in self.dynamic_compute_functions:
            func = self.dynamic_compute_functions[factor_name]
            try:
                val = func(token, token_ci, token_phon, diag, features_array, global_factor_dict)
                # ---- update metabolic stats ----
                meta = self.factor_metadata.get(factor_name)
                if meta:
                    now = time.time()
                    if meta['last_used'] is not None:
                        # decay recent_usage over time
                        dt = now - meta['last_used']
                        meta['recent_usage'] *= 0.99 ** dt
                    meta['recent_usage'] += 0.1   # each usage adds 0.1 (smooth)
                    meta['last_used'] = now
                    # predictive_gain could be updated with the absolute value of the factor (or something)
                    meta['predictive_gain'] = 0.9 * meta['predictive_gain'] + 0.1 * abs(val)
                # ---------------------------------
                return val
            except Exception as e:
                print(f"Error computing dynamic factor '{factor_name}': {e}")
                print(traceback.format_exc())
                return 0.0
        return 0.0
    
    def update(self, factors: np.ndarray, information_gain: float):

        target_dim = len(factors)

        # ---- ensure all parameter arrays match the same size ----
        def _expand(arr, fill):
            if len(arr) < target_dim:
                return np.append(arr, np.full(target_dim - len(arr), fill))
            return arr

        self.mu = _expand(self.mu, 0.0)
        self.lambda_ = _expand(self.lambda_, 0.1)
        self.alpha = _expand(self.alpha, 1.0)
        self.beta = _expand(self.beta, 1.0)

        # if learner larger than factors → pad factors
        if len(factors) < len(self.mu):
            factors = np.pad(factors, (0, len(self.mu) - len(factors)))

        # ---- safe update ----
        for i in range(len(self.mu)):
            x = factors[i]
            self.lambda_[i] += x ** 2
            self.mu[i] = (
                (self.lambda_[i] * self.mu[i] + x * information_gain)
                / (self.lambda_[i] + x ** 2)
            )

    def predict_curiosity(self, factors: np.ndarray) -> float:
        """Weighted sum of factors using posterior mean weights."""
        # ---- Auto align dimensionality (dynamic factor safety) ----
        if len(self.mu) < len(factors):
            # expand mu
            pad = np.zeros(len(factors) - len(self.mu))
            self.mu = np.concatenate([self.mu, pad])

        elif len(self.mu) > len(factors):
            # pad factors
            factors = np.pad(factors, (0, len(self.mu) - len(factors)))

        factors = align_factor_vector(factors, len(self.mu))
        return np.dot(self.mu, factors)

    def get_weights(self) -> Dict[str, float]:
        """Return dictionary mapping factor names to learned weights."""
        return dict(zip(self.factor_names, self.mu))

    def apply_guilt_to_weights(self, guilt_factor: float):
        """
        guilt_factor ∈ [0.5,1.0] (higher guilt → lower factor).
        Reduce weights for factors that may have caused negative feedback.
        For simplicity, we reduce all weights slightly and increase the weight of a
        hypothetical "caution" factor if it exists.
        """
        # Scale down all weights by guilt_factor
        self.mu *= guilt_factor
        # Boost a "caution" factor if present
        if "caution" in self.factor_names:
            idx = self.factor_names.index("caution")
            self.mu[idx] = min(1.0, self.mu[idx] + (1.0 - guilt_factor) * 0.2)

    def save(self, path: str):
        """Save weights and factor names."""
        np.savez(path,
                 mu=self.mu,
                 lambda_=self.lambda_,
                 alpha=self.alpha,
                 beta=self.beta,
                 base_factor_names=self.base_factor_names,
                 dynamic_factor_names=self.dynamic_factor_names,
                factor_metadata=self.factor_metadata) 

    def load(self, path: str):
        """Load weights and factor names."""
        data = np.load(path, allow_pickle=True)
        self.mu = data['mu']
        self.lambda_ = data['lambda_']
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.base_factor_names = list(data['base_factor_names'])
        self.dynamic_factor_names = list(data['dynamic_factor_names'])
        self.n_factors = len(self.mu)
        if 'factor_metadata' in data:
            self.factor_metadata = data['factor_metadata'].item()
        else:
            self.factor_metadata = {}
    
    def _load(self):
        """Load from default path if exists."""
        path = os.path.join(STATE_DIR, 'meta_learner.npz')
        if os.path.exists(path):
            self.load(path)

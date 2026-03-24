from importglobal import Optional,Dict,defaultdict,json,Callable,List,os,time,traceback,np
from config_manager import ConfigManager
from directory_handler import STATE_DIR
# ------------------------------------------------------------
# Adaptive Control Space – dynamic expansion of control factors
# ------------------------------------------------------------
class AdaptiveControlSpace:
    """
    Manages the dynamic set of control signals (factors).
    Base factors are fixed; new factors can be discovered and added
    based on statistical variance, state clustering, or utility.
    """
    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        
        # Base factor names (always present)
        self.base_factors = [
            "curiosity", "threat", "trust", "confidence",
            "uncertainty", "energy", "attention",
            "comprehension", "fluency", "cognitive_load"
        ]
        
        # Dynamic factor names (start empty)
        self.dynamic_factors = []   # e.g., "novelty_seeking", "social_engagement", ...
        
        # For each dynamic factor, we store its definition: how it is computed,
        # its current value, and its historical statistics.
        self.factor_definitions = {}  # name -> dict of metadata
        
        # History for variance/utility tracking
        self.history = defaultdict(list)  # name -> list of past values
        self.max_history = 100
        
        # Expansion parameters (can be overridden from config)
        self.variance_threshold = 0.05
        self.utility_threshold = 0.01
        self.max_factors = 50
        self.expansion_cooldown = 100
        self._load_config()
        # Listeners to be notified when a new factor is added
        self._listeners = []
        self._load()

    def add_listener(self, callback: Callable[[str, Callable, dict], None]):
        """
        Register a listener that will be called whenever a new dynamic factor is added.
        The callback receives (factor_name, compute_function, metadata).
        """
        self._listeners.append(callback)

    def _load_config(self):
        """Load parameters from config if available."""
        if self.config_manager:
            self.variance_threshold = self.config_manager.get('adaptive_control_space.variance_threshold', 0.05)
            self.utility_threshold = self.config_manager.get('adaptive_control_space.utility_threshold', 0.01)
            self.max_factors = self.config_manager.get('adaptive_control_space.max_factors', 50)
            self.expansion_cooldown = self.config_manager.get('adaptive_control_space.expansion_cooldown', 100)
    
    def _load(self):
        """Load dynamic factor definitions from disk (if any)."""
        path = os.path.join(STATE_DIR, "control_space.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.dynamic_factors = data.get('dynamic_factors', [])
                # Deserialize compute functions? We store only metadata; compute functions must be registered at runtime.
                self.factor_definitions = data.get('factor_definitions', {})
            except Exception as e:
                print(f"Could not load control space: {e}")
    
    def save(self):
        """Persist dynamic factor definitions (compute functions are not saved)."""
        # Remove callable objects before saving
        serializable_defs = {}
        for name, defn in self.factor_definitions.items():
            serializable_defs[name] = {k: v for k, v in defn.items() if k != 'compute'}
        path = os.path.join(STATE_DIR, "control_space.json")
        with open(path, 'w') as f:
            json.dump({
                'dynamic_factors': self.dynamic_factors,
                'factor_definitions': serializable_defs
            }, f, indent=2)
    
    @property
    def all_factors(self) -> List[str]:
        """Return the complete list of control factors (base + dynamic)."""
        return self.base_factors + self.dynamic_factors
    
    def add_factor(self, name: str, compute_function: Callable, metadata: dict = None):
        """
        Register a new dynamic factor.
        compute_function will be called with the current system state (dict)
        and should return a float in [0,1].
        """
        if name in self.all_factors:
            return
        if len(self.all_factors) >= self.max_factors:
            print(f"⚠️ Cannot add factor '{name}': max factors ({self.max_factors}) reached.")
            return
        self.dynamic_factors.append(name)
        self.factor_definitions[name] = {
            'compute': compute_function,
            'metadata': metadata or {},
            'created': time.time()
        }
        self.save()
        # Notify all listeners
        for listener in self._listeners:
            try:
                listener(name, compute_function, metadata)
            except Exception:
                print(f"[Listener ERROR] factor '{name}'")
                print(traceback.format_exc())

    
    def compute_factor(self, name: str, state: Dict) -> float:
        """Compute the current value of a factor (base or dynamic)."""
        if name in self.base_factors:
            # Base factors are computed by FactorEngine; this method is not used for them.
            return state.get(name, 0.5)
        elif name in self.dynamic_factors:
            func = self.factor_definitions[name].get('compute')
            if callable(func):
                return np.clip(func(state), 0, 1)
        return 0.5
    
    def record_value(self, name: str, value: float):
        """Store the value in history for later analysis."""
        self.history[name].append(value)
        if len(self.history[name]) > self.max_history:
            self.history[name].pop(0)
    
    def propose_new_factor(self, all_state_vectors: List[np.ndarray], 
                           factor_values: Dict[str, List[float]]) -> Optional[str]:
        """
        Analyze recent state vectors and factor histories to propose a new factor.
        Returns a suggested factor name if one is warranted, else None.
        
        This is a placeholder – you can replace this with sophisticated discovery logic.
        """
        # Example heuristic: if a base factor's variance drops below threshold,
        # but there is still unexplained variance in the state space, propose a new factor.
        # For now, we return None (no auto‑expansion) – you can enable it later.
        return None

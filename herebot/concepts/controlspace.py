from importglobal import Optional,Dict,defaultdict,json,Callable,List,os,time,traceback,np,Counter
from components.config_manager import ConfigManager
from components.directory_handler import STATE_DIR
from components.learner import KnowledgeGraph
from components.factor_handler import StateManager
from components.factor_space import FactorSpace, FactorSpaceInvariantError
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
        if not hasattr(self, 'factor_space'):
            self.factor_space = FactorSpace('control', self.all_factors)

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
                self.base_factors = data.get('base_factors', self.base_factors)
                self.dynamic_factors = data.get('dynamic_factors', [])
                # Deserialize compute functions? We store only metadata; compute functions must be registered at runtime.
                self.factor_definitions = data.get('factor_definitions', {})
                persisted = data.get('factor_space')
                self.factor_space = FactorSpace(
                    (persisted or {}).get('space_id', 'control'), self.all_factors,
                    version=(persisted or {}).get('version', 0),
                    history=(persisted or {}).get('history'),
                    factor_metadata=(persisted or {}).get('factor_metadata', {}))
            except Exception as e:
                print(f"Could not load control space: {e}")
    
    def save(self):
        """Persist dynamic factor definitions (compute functions are not saved)."""
        # Remove callable objects before saving
        serializable_defs = {}
        for name, defn in self.factor_definitions.items():
            serializable_defs[name] = {k: v for k, v in defn.items() if k != 'compute'}
        path = os.path.join(STATE_DIR, "control_space.json")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'base_factors': self.base_factors,
                'dynamic_factors': self.dynamic_factors,
                'factor_definitions': serializable_defs,
                'factor_space': self.factor_space.as_dict(),
            }, f, indent=2)
    
    @property
    def all_factors(self) -> List[str]:
        """Return the complete list of control factors (base + dynamic)."""
        return self.base_factors + self.dynamic_factors

    def add_base_factor(self, name: str, metadata: dict = None):
        """Register a non-dynamic semantic control signal without mutating a schema in place."""
        if name in self.all_factors:
            return
        self.base_factors.append(name)
        self.factor_space.add(name, metadata=metadata or {'kind': 'base'})
        self.save()
    
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
        self.factor_space.add(name, metadata=metadata)
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

    def validate_control_vector(self, vector: np.ndarray, *, factor_names: List[str] = None):
        names = self.factor_space.factor_names if factor_names is None else list(factor_names)
        if names != self.factor_space.factor_names:
            raise FactorSpaceInvariantError(
                f"control factor ordering mismatch at v{self.factor_space.version}: "
                f"provided={names}, expected={self.factor_space.factor_names}")
        self.factor_space.validate_vector(vector)
    
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
        # For now, we return None (no auto-expansion) – you can enable it later.
        return None
    
# ------------------------------------------------------------
# StructuredWorldModel – uses knowledge graph and episodic memory to simulate state transitions
# ------------------------------------------------------------
class StructuredWorldModel:
    """
    Models transitions between states and actions, using the knowledge graph
    to represent possible outcomes. Can generate synthetic future episodes.
    """
    def __init__(self, kg: KnowledgeGraph, state_manager: StateManager):
        self.kg = kg
        self.state_manager = state_manager
        self.transition_counts = defaultdict(lambda: defaultdict(Counter))  # state_label -> action -> next_state_label -> count

    def observe_episode(self, state_label, action, next_state_label):
        """Record a transition from episodic memory."""
        self.transition_counts[state_label][action][next_state_label] += 1

    def predict_next_state(self, state_label, action):
        """Return the most probable next state given current state and action."""
        if state_label not in self.transition_counts or action not in self.transition_counts[state_label]:
            return None
        next_states = self.transition_counts[state_label][action]
        return max(next_states, key=next_states.get) if next_states else None

    def generate_synthetic_episode(self, start_state_label, max_steps=5):
        """Walk through the transition graph to create a synthetic sequence."""
        if not self.transition_counts:
            return []
        current = start_state_label
        sequence = [current]
        for _ in range(max_steps):
            # Choose a random action from known actions for this state
            if current not in self.transition_counts:
                break
            actions = list(self.transition_counts[current].keys())
            if not actions:
                break
            action = np.random.choice(actions)
            next_state = self.predict_next_state(current, action)
            if next_state is None or next_state == current:
                break
            sequence.append((action, next_state))
            current = next_state
        return sequence

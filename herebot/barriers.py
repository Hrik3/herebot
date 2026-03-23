from importglobal import Optional,copy,Dict, Any,List,json,os,Tuple,dataclass,field,time,pickle,np
from .config_manager import ConfigManager
from directory_handler import STATE_DIR, LOGIC_DIR, SHADOW_LOG_DIR
from traits.code_generator import CodeGenerator,ShadowExecutor,GeneratedCodeFeedback

# ------------------------------------------------------------
# GuiltTracker – models unease after negative feedback in positive contexts
# ------------------------------------------------------------
class GuiltTracker:
    """
    Guilt level ∈ [0,1]. Increases when negative feedback is received while
    the system is in a "positive" emotional state (high trust, low threat, high energy).
    Decays over time.
    """

    def __init__(self, decay_rate: float = 0.99, guilt_boost: float = 0.2):
        self.guilt_level = 0.0          # 0 = no guilt, 1 = intense guilt
        self.decay_rate = decay_rate    # per hour
        self.guilt_boost = guilt_boost  # how much guilt increases per negative feedback
        self.history = []              # list of (timestamp, guilt_before, guilt_after, context)
        self.last_update = time.time()

    def update_guilt(self, feedback_rating: str, emotional_state: dict = None):
        """
        Called after receiving user feedback.
        If rating is negative and emotional_state indicates positive environment,
        guilt increases. Otherwise, guilt decays.
        """
        # Compute positivity score (0..1)
        if emotional_state is None:
            emotional_state = {}
        trust = emotional_state.get("trust", 0.5)
        threat = emotional_state.get("threat", 0.0)
        energy = emotional_state.get("energy", 0.5)
        comprehension = emotional_state.get("comprehension", 0.5)
        positivity = (trust + (1 - threat) + energy + comprehension) / 4.0

        # Decay first
        self.decay()

        before = self.guilt_level

        if feedback_rating.lower().startswith("negative"):
            self.guilt_level = min(1.0, self.guilt_level + self.guilt_boost * positivity)
        else:
            self.guilt_level *= 0.95

        self.history.append((time.time(), before, self.guilt_level, emotional_state))

    def decay(self):
        """Explicit decay call (called periodically)."""
        now = time.time()
        hours_passed = (now - self.last_update) / 3600.0
        self.guilt_level *= (self.decay_rate ** hours_passed)
        self.last_update = now

    def get_guilt_factor(self) -> float:
        """Return a multiplier ∈ [0.5, 1.0] for confidence/curiosity."""
        # Higher guilt → lower confidence, lower curiosity
        return 1.0 - self.guilt_level * 0.5

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "guilt_level": self.guilt_level,
                "decay_rate": self.decay_rate,
                "guilt_boost": self.guilt_boost,
                "history": self.history,
                "last_update": self.last_update
            }, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.guilt_level = data.get("guilt_level", 0.0)
                self.decay_rate = data.get("decay_rate", 0.99)
                self.guilt_boost = data.get("guilt_boost", 0.2)
                self.history = data.get("history", [])
                self.last_update = data.get("last_update", time.time())

@dataclass
class ConscienceBelief:
    """A deeply held behavioral/ethical belief."""
    belief_id: str
    description: str      # natural language description
    rule_type: str        # e.g., 'ethical', 'behavioral', 'meta'
    condition: Dict[str, Any]  # conditions under which it applies
    action: Dict[str, Any]     # prescribed action or constraint
    confidence: float = 0.0
    evidence: int = 0
    added_to_conscience: bool = False   # True when formally integrated
    created: float = field(default_factory=time.time)
    last_updated: Optional[float] = None
    # ---- metabolic attributes ----
    energy_cost: float = 1.0
    recent_usage: float = 0.0
    predictive_gain: float = 0.0
    phase: str = "active"
    last_used: Optional[float] = None

    def __hash__(self):
        return hash(self.belief_id)

class ConscienceManager:
    """
    Manages deep behavioral beliefs (conscience). Beliefs are only added to
    conscience after extensive evidence and high trust. The system can
    also adjust its own parameters (thresholds) after sufficient maturity.
    """
    STATE_FILE = os.path.join(STATE_DIR, "conscience_manager.pkl")

    DEFAULT_PARAMS = {
        "trust_threshold_for_addition": 0.85,
        "min_evidence": 5,
        "decay_rate": 0.99,
        "max_beliefs": 50,
        "evolution_trigger": 10,
    }

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.params = copy.deepcopy(self.DEFAULT_PARAMS)
        if config_manager:
            cm_config = config_manager.config.get('conscience_manager', {})
            for key in self.params:
                if key in cm_config:
                    self.params[key] = cm_config[key]

        self.beliefs: Dict[str, ConscienceBelief] = {}
        self.conscience_beliefs: Dict[str, ConscienceBelief] = {}
        self.belief_history: List[Dict] = []
        self.guilt_tracker = GuiltTracker()
        self._load()

    def _load(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.params = data.get('params', self.DEFAULT_PARAMS)
                    self.beliefs = data.get('beliefs', {})
                    self.conscience_beliefs = data.get('conscience_beliefs', {})
                    self.belief_history = data.get('history', [])
            except Exception as e:
                print(f"Could not load conscience manager: {e}")

    def save(self):
        tmp = self.STATE_FILE + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'beliefs': self.beliefs,
                'conscience_beliefs': self.conscience_beliefs,
                'history': self.belief_history,
            }, f)
        os.replace(tmp, self.STATE_FILE)

    def add_belief_candidate(self, belief: ConscienceBelief):
        if belief.belief_id in self.beliefs:
            return
        if len(self.beliefs) >= self.params['max_beliefs']:
            non_conscience = [b for b in self.beliefs.values() if not b.added_to_conscience]
            if non_conscience:
                oldest = min(non_conscience, key=lambda b: b.created)
                del self.beliefs[oldest.belief_id]
        self.beliefs[belief.belief_id] = belief
        self.belief_history.append({
            'event': 'add',
            'belief_id': belief.belief_id,
            'timestamp': time.time()
        })
        self.save()

    def update_confidence(self, belief_id: str, delta: float, feedback: bool = False, emotional_state: dict = None):
        if belief_id not in self.beliefs:
            return
        belief = self.beliefs[belief_id]
        belief.confidence = np.clip(belief.confidence + delta, 0.0, 1.0)
        if feedback:
            belief.evidence += 1 if delta > 0 else -1
        if feedback:
            if delta < 0 and emotional_state:   # negative feedback
                self.guilt_tracker.update_guilt("negative", emotional_state)
            elif delta > 0 and emotional_state:
                self.guilt_tracker.update_guilt("positive", emotional_state)
        belief.last_updated = time.time()
        self.belief_history.append({
            'event': 'update',
            'belief_id': belief_id,
            'delta': delta,
            'confidence': belief.confidence,
            'timestamp': time.time()
        })
        self.save()

    def apply_guilt_to_control_signals(self, factor_dict: Dict) -> Dict:
        """
        Modify emotional factors based on current guilt level.
        Guilt reduces confidence, increases threat, lowers trust.
        """
        guilt = self.guilt_tracker.guilt_level
        if guilt > 0:
            factor_dict['confidence'] = factor_dict.get('confidence', 0.5) * (1 - guilt * 0.3)
            factor_dict['threat'] = min(1.0, factor_dict.get('threat', 0.0) + guilt * 0.2)
            factor_dict['trust'] = max(0.0, factor_dict.get('trust', 0.5) - guilt * 0.2)
            factor_dict['guilt'] = guilt   # expose guilt as a control signal
        return factor_dict
    
    def decay_confidences(self):
        for belief in self.beliefs.values():
            if not belief.added_to_conscience and belief.last_updated:
                days_since = (time.time() - belief.last_updated) / 86400
                decay = self.params['decay_rate'] ** days_since
                belief.confidence *= decay
        self.save()

    def maybe_promote_to_conscience(self, belief_id: str) -> bool:
        if belief_id not in self.beliefs:
            return False
        belief = self.beliefs[belief_id]
        if belief.added_to_conscience:
            return False
        if belief.confidence >= self.params['trust_threshold_for_addition'] and \
           belief.evidence >= self.params['min_evidence']:
            belief.added_to_conscience = True
            self.conscience_beliefs[belief_id] = belief
            self.belief_history.append({
                'event': 'promote',
                'belief_id': belief_id,
                'timestamp': time.time()
            })
            self.save()
            print(f"🧠 [Conscience] New belief integrated: {belief.description[:60]}...")
            return True
        return False

    def get_active_conscience(self) -> List[ConscienceBelief]:
        return list(self.conscience_beliefs.values())

    def check_action(self, proposed_action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        for belief in self.conscience_beliefs.values():
            if self._matches_condition(belief.condition, proposed_action):
                if 'forbid' in belief.action or belief.action.get('effect') == 'forbid':
                    return False, belief.description
        return True, None

    def _matches_condition(self, condition: Dict, action: Dict) -> bool:
        for k, v in condition.items():
            if k in action and action[k] != v:
                return False
        return True

    def evolve_parameters(self):
        if len(self.conscience_beliefs) < self.params['evolution_trigger']:
            return
        recent_history = [h for h in self.belief_history if h['timestamp'] > time.time() - 86400*7]
        if not recent_history:
            return
        confirm_ratio = sum(1 for h in recent_history if h['event'] == 'update' and h.get('delta',0) > 0) / len(recent_history)
        if confirm_ratio > 0.8:
            self.params['trust_threshold_for_addition'] = max(0.6, self.params['trust_threshold_for_addition'] - 0.05)
            self.params['min_evidence'] = max(3, self.params['min_evidence'] - 1)
            print(f"⚙️ [Conscience] Evolving parameters: trust_threshold={self.params['trust_threshold_for_addition']:.2f}, min_evidence={self.params['min_evidence']}")
        elif confirm_ratio < 0.3:
            self.params['trust_threshold_for_addition'] = min(0.95, self.params['trust_threshold_for_addition'] + 0.05)
            self.params['min_evidence'] = min(10, self.params['min_evidence'] + 1)
            print(f"⚙️ [Conscience] Evolving parameters (cautious): trust_threshold={self.params['trust_threshold_for_addition']:.2f}, min_evidence={self.params['min_evidence']}")
        if self.config_manager:
            cm_config = self.config_manager.config.get('conscience_manager', {})
            for key in self.params:
                cm_config[key] = self.params[key]
            self.config_manager.config['conscience_manager'] = cm_config
            self.config_manager.save()
        self.save()

    def apply_ethical_influence(self, factor_dict: Dict) -> Dict:
        if self.conscience_beliefs:
            factor_dict['conscience_active'] = min(1.0, len(self.conscience_beliefs) / self.params.get('max_beliefs', 50))
        else:
            factor_dict['conscience_active'] = 0.0
        return factor_dict

class LogicInheritanceManager:
    """
    Allows the AGI to write its own logical rules to disk (./logic)
    and later ask the user for confirmation. Once confirmed, the logic
    can be integrated into the reasoning engine or become a conscience belief.
    """
    LOGIC_DIR = LOGIC_DIR

    def __init__(self, config_manager: Optional['ConfigManager'] = None, conscience_manager: Optional[ConscienceManager] = None):
        self.config_manager = config_manager
        self.conscience_manager = conscience_manager
        self.pending_doubts: List[Dict] = []
        self.confirmed_logic: Dict[str, Dict] = {}
        self.code_generator = CodeGenerator()
        self.shadow_executor = ShadowExecutor()
        self.generated_code_feedback = GeneratedCodeFeedback()
        self.generated_code_paths = {}   # logic_id -> filepath
        self._load()

    def _load(self):
        path = os.path.join(self.LOGIC_DIR, 'confirmed_logic.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.confirmed_logic = data.get('confirmed_logic', {})
                    self.pending_doubts = data.get('pending_doubts', [])
                    self.generated_code_paths = data.get('generated_code_paths', {})
            except Exception as e:
                print(f"Could not load logic inheritance: {e}")

    def save(self):
        path = os.path.join(self.LOGIC_DIR, 'confirmed_logic.pkl')
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'confirmed_logic': self.confirmed_logic,
                'pending_doubts': self.pending_doubts,
                'generated_code_paths': self.generated_code_paths,
            }, f)
        os.replace(tmp, path)

    def generate_logic_file(self, rule: Dict) -> str:
        logic_id = f"logic_{int(time.time())}_{np.random.randint(1000)}"
        safe_id = "".join(c for c in logic_id if c.isalnum() or c == "_")
        filename = f"{safe_id}.json"
        logic_id = safe_id
        filepath = os.path.join(self.LOGIC_DIR, filename)
        rule['logic_id'] = logic_id
        rule['created'] = time.time()
        rule['confirmed'] = False
        rule['energy_cost'] = 0.5
        rule['recent_usage'] = 0.0
        rule['predictive_gain'] = 0.5
        rule['phase'] = 'active'
        rule['last_used'] = None
        with open(filepath, 'w') as f:
            json.dump(rule, f, indent=2)
        return filepath

    def generate_doubt(self, logic_id: str) -> Optional[Dict]:
        filepath = os.path.join(self.LOGIC_DIR, f"{logic_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            rule = json.load(f)
        doubt = {
            'doubt_id': f"doubt_{logic_id}_{int(time.time())}",
            'logic_id': logic_id,
            'rule': rule,
            'timestamp': time.time(),
            'asked': False,
            'response': None
        }
        self.pending_doubts.append(doubt)
        self.save()
        return doubt

    def generate_code_for_logic(self, logic_id: str) -> Optional[str]:
        """
        Generate Python code from a confirmed logic rule and save it.
        Returns filepath if successful, else None.
        """
        # Look for the rule in confirmed_logic or pending_doubts
        rule = self.confirmed_logic.get(logic_id)
        if not rule:
            for d in self.pending_doubts:
                if d['logic_id'] == logic_id:
                    rule = d['rule']
                    break
        if not rule:
            return None

        code = self.code_generator.generate(rule)
        filepath = self.code_generator.save_code_file(logic_id, code)
        self.generated_code_paths[logic_id] = filepath
        return filepath

    def shadow_test_generated_code(self, logic_id: str) -> dict:
        """
        Run the generated code in shadow mode, record errors, and return test result.
        """
        if logic_id not in self.generated_code_paths:
            return {"success": False, "error": "No generated file"}

        filepath = self.generated_code_paths[logic_id]
        success, error, output = self.shadow_executor.test_code_file(filepath)

        # Log the test run
        log_entry = {
            "timestamp": time.time(),
            "success": success,
            "error": error,
            "output": output[:500]   # truncate
        }
        log_file = os.path.join(SHADOW_LOG_DIR, f"{logic_id}_test.json")
        try:
            with open(log_file, "w") as f:
                json.dump(log_entry, f)
        except:
            pass

        return {"success": success, "error": error, "output": output}
        
    def confirm_logic(self, logic_id: str, is_correct: bool):
        doubt = None
        for d in self.pending_doubts:
            if d['logic_id'] == logic_id:
                doubt = d
                break
        if not doubt:
            return
        if is_correct:
            self.confirmed_logic[logic_id] = doubt['rule']
            rule = doubt['rule']
            rule['last_used'] = time.time()
            rule['recent_usage'] = rule.get('recent_usage', 0.0) * 0.99 + 0.1
            # ---- NEW: generate code and shadow test ----
            filepath = self.generate_code_for_logic(logic_id)
            if filepath:
                test_result = self.shadow_test_generated_code(logic_id)
                if not test_result["success"]:
                    # Error already appended to file as comment
                    print(f"⚠️ [Shadow Test] Generated code for {logic_id} has errors.")
            # ---------------------------------------------
            if self.conscience_manager:
                belief = ConscienceBelief(
                    belief_id=f"conscience_{logic_id}",
                    description=doubt['rule'].get('description', f"Logical rule: {doubt['rule']}"),
                    rule_type='ethical' if doubt['rule'].get('ethical', False) else 'behavioral',
                    condition=doubt['rule'].get('condition', {}),
                    action=doubt['rule'].get('action', {}),
                    confidence=0.7,
                    evidence=1,
                    added_to_conscience=False
                )
                self.conscience_manager.add_belief_candidate(belief)
                self.conscience_manager.update_confidence(belief.belief_id, 0.2, feedback=True)
        else:
            # ---- Negative feedback: store and possibly trigger guilt ----
            if self.conscience_manager and hasattr(self.conscience_manager, 'guilt_tracker'):
                # We'll need to pass the emotional state – will be handled in CuriosityEngine
                pass
        self.pending_doubts = [d for d in self.pending_doubts if d['logic_id'] != logic_id]
        self.save()

    def get_next_doubt(self) -> Optional[Dict]:
        for d in self.pending_doubts:
            if not d.get('asked', False):
                d['asked'] = True
                self.save()
                return d
        return None

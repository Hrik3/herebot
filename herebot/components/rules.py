from importglobal import Optional, Dict, Any, Set, Callable, List,json,os,Tuple,dataclass,field,time
from components.config_manager import ConfigManager
from traits.traits_global import Expression
from concepts.concepts_global import *
from components.resoning import ReasoningEngine
from core.heart import CuriosityEngine
# =============================================================================
# RULE ENGINE & CONSTRAINT PROPAGATOR
# =============================================================================
class Rule:
    """Deterministic rule with conditions and actions."""
    def __init__(self, name: str, condition: Expression, action: Callable, reasoning_engine: 'ReasoningEngine'):
        self.name = name
        self.condition = condition   # an Expression that must evaluate to True
        self.action = action         # function that modifies the knowledge base
        self.reasoning_engine = reasoning_engine

    def matches(self, context: Dict) -> bool:
        """Evaluate the condition in the given context."""
        try:
            result = self.reasoning_engine.eval_expression(self.condition, context)
            return bool(result)
        except:
            return False

    def execute(self, context: Dict):
        self.action(context)

class RuleEngine:
    """Executes deterministic rules when conditions are met."""
    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def run(self, context: Dict):
        for rule in self.rules:
            if rule.matches(context):
                rule.execute(context)

class ConstraintPropagator:
    """
    Simple constraint propagation over typed symbols.
    Maintains a set of constraints and repeatedly applies them until fixpoint.
    """
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.constraints = []   # list of callables that return new facts

    def add_constraint(self, constraint_func: Callable):
        """constraint_func receives knowledge_graph and returns list of new edges."""
        self.constraints.append(constraint_func)

    def propagate(self, max_iter=10):
        changed = True
        for _ in range(max_iter):
            if not changed:
                break
            changed = False
            for constraint in self.constraints:
                new_edges = constraint(self.kg)
                for (n1, n2, rel, conf) in new_edges:
                    # check if already exists with higher or equal confidence
                    existing = self.kg.edges.get(n1, {}).get(n2, {}).get(rel, 0)
                    if conf > existing:
                        self.kg.add_edge(n1, n2, rel, conf)
                        changed = True
# =============================================================================
# ------------------------------------------------------------
# Behavioral Rule Learner – discovers, tests, and adopts behavioral rules
# ------------------------------------------------------------
@dataclass
class BehavioralRule:
    """
    Represents a behavioral rule learned by the system.
    Types:
        - 'identity_alias': maps a word/phrase to self-reference (e.g., "herebot" = "I")
        - 'pronoun_mapping': maps pronouns to self/other (e.g., "you" -> self when referring to the bot)
        - 'response_template': specific pattern for responding to certain inputs
        - 'action_rule': tendency to perform certain actions in given states
    """
    rule_id: str
    rule_type: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float = 0.5
    prior: float = 0.1
    evidence: int = 0
    created: float = field(default_factory=time.time)
    last_tested: Optional[float] = None
    asked: bool = False
    confirmed: bool = False
    # ---- metabolic attributes ----
    energy_cost: float = 1.0               # cost per time unit to keep alive
    recent_usage: float = 0.0               # smoothed usage count
    predictive_gain: float = 0.0            # estimated contribution to prediction
    phase: str = "active"                   # active, weak, dormant, archived, deleted
    last_used: Optional[float] = None       # last time the rule was applied

class BehavioralRuleLearner:
    """
    Learns behavioral rules from:
    - Patterns in episodic memory (e.g., user says "you are herebot", then uses "herebot" as self)
    - Self-reflection during dreaming (analogy, co-occurrence)
    - User feedback on hypotheses
    """
    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.rules: Dict[str, BehavioralRule] = {}
        self.rule_history: List[Dict] = []
        self.max_rules = 100
        self.confirmation_threshold = 0.8
        self.test_threshold = 0.6
        self.decay_rate = 0.99
        self._load()

    def _load(self):
        path = os.path.join(STATE_DIR, 'behavioral_rules.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.rules = data.get('rules', {})
                    self.rule_history = data.get('history', [])
            except Exception as e:
                print(f"Could not load behavioral rules: {e}")

    def record_usage(self, rule_id: str):
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            now = time.time()
            if rule.last_used:
                dt = now - rule.last_used
                rule.recent_usage *= 0.99 ** dt
            rule.recent_usage += 0.1
            rule.last_used = now

    def save(self):
        path = os.path.join(STATE_DIR, 'behavioral_rules.pkl')
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'history': self.rule_history
            }, f)
        os.replace(tmp, path)

    def add_candidate_rule(self, rule: BehavioralRule):
        if rule.rule_id in self.rules:
            return
        if len(self.rules) >= self.max_rules:
            oldest = min(self.rules.values(), key=lambda r: r.created)
            del self.rules[oldest.rule_id]
        self.rules[rule.rule_id] = rule
        self.rule_history.append({
            'event': 'add',
            'rule_id': rule.rule_id,
            'timestamp': time.time(),
            'rule': rule
        })
        self.save()

    def update_confidence(self, rule_id: str, delta: float, feedback: bool = False):
        if rule_id not in self.rules:
            return
        rule = self.rules[rule_id]
        rule.confidence = np.clip(rule.confidence + delta, 0.0, 1.0)
        rule.evidence += 1 if delta > 0 else -1
        if feedback:
            rule.confirmed = delta > 0
            rule.asked = True
        rule.last_tested = time.time()
        self.rule_history.append({
            'event': 'update',
            'rule_id': rule_id,
            'delta': delta,
            'confidence': rule.confidence,
            'timestamp': time.time()
        })
        self.save()

    def decay_confidences(self):
        for rule in self.rules.values():
            if not rule.confirmed and rule.last_tested:
                time_since = time.time() - rule.last_tested
                decay = self.decay_rate ** (time_since / 3600)
                rule.confidence *= decay
        self.save()

    def get_testable_rules(self, context: Dict) -> List[BehavioralRule]:
        testable = []
        for rule in self.rules.values():
            if rule.confidence >= self.test_threshold and not rule.asked:
                if self._matches_condition(rule, context):
                    testable.append(rule)
        for rule in testable:
            self.record_usage(rule.rule_id)
        return testable

    def _matches_condition(self, rule: BehavioralRule, context: Dict) -> bool:
        cond = rule.condition
        if rule.rule_type == 'identity_alias':
            alias = cond.get('alias')
            return alias in context.get('tokens', [])
        elif rule.rule_type == 'pronoun_mapping':
            pronoun = cond.get('pronoun')
            return pronoun in context.get('tokens', [])
        return False

    def propose_new_rules(self, engine: 'CuriosityEngine') -> List[BehavioralRule]:
        candidates = []
        # Existing identity alias detection
        episodes = engine.episodic_memory.episodes[-200:]
        alias_candidates = defaultdict(int)
        for ep in episodes:
            user_input = ep.get('user_input', '').lower()
            if 'you are' in user_input or "you're" in user_input or "your name is" in user_input:
                idx = -1
                if 'you are' in user_input:
                    idx = user_input.find('you are') + len('you are')
                elif "you're" in user_input:
                    idx = user_input.find("you're") + len("you're")
                elif "your name is" in user_input:
                    idx = user_input.find("your name is") + len("your name is")
                if idx != -1:
                    rest = user_input[idx:].strip()
                    if rest:
                        name_token = rest.split()[0].strip('.,?!')
                        if name_token and len(name_token) > 1:
                            alias_candidates[name_token] += 1
        for alias, count in alias_candidates.items():
            if count >= 2:
                rule_id = f"identity_alias:{alias}"
                if rule_id not in self.rules:
                    rule = BehavioralRule(
                        rule_id=rule_id,
                        rule_type='identity_alias',
                        condition={'alias': alias},
                        action={'self_reference': alias},
                        confidence=0.3 + 0.1 * count,
                        prior=0.2
                    )
                    candidates.append(rule)

        # New pattern detections
        candidates.extend(self.detect_greeting_patterns(engine))
        candidates.extend(self.detect_qa_patterns(engine))
        # deep_pattern_mining will be called separately during idle

        return candidates

    def detect_greeting_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]:
        """
        Scan episodic memory for common opening utterances and their typical responses.
        Creates rules of type 'conversational_pattern'.
        """
        candidates = []
        episodes = engine.episodic_memory.episodes[-500:]  # recent history

        # Count user utterance frequencies (first turn of a session or after long gaps)
        user_utterance_counter = Counter()
        response_counter = defaultdict(Counter)  # user_input -> Counter of system_responses

        for ep in episodes:
            user_input = ep.get('user_input', '').lower().strip()
            system_response = ep.get('system_response', '').strip()
            if not user_input or not system_response:
                continue
            user_utterance_counter[user_input] += 1
            response_counter[user_input][system_response] += 1

        # Find frequent user utterances (e.g., greetings)
        threshold = 3  # at least 3 occurrences
        for user_input, freq in user_utterance_counter.items():
            if freq >= threshold and len(user_input.split()) <= 3:  # short phrases
                # Get the most common system response
                if user_input in response_counter:
                    most_common_resp = response_counter[user_input].most_common(1)[0][0]
                    rule_id = f"pattern_greeting_{hash(user_input) % 10000:04x}"
                    if rule_id not in self.rules:
                        rule = BehavioralRule(
                            rule_id=rule_id,
                            rule_type='conversational_pattern',
                            condition={'user_input_pattern': user_input},
                            action={'response_template': most_common_resp},
                            confidence=0.4 + 0.1 * freq,
                            prior=0.3
                        )
                        candidates.append(rule)
        return candidates

    def detect_qa_patterns(self, engine: 'CuriosityEngine') -> List[BehavioralRule]:
        """
        Find question–answer pairs: user inputs ending with '?' and system responses.
        """
        candidates = []
        episodes = engine.episodic_memory.episodes[-300:]
        question_counter = Counter()
        answer_counter = defaultdict(Counter)

        for ep in episodes:
            user_input = ep.get('user_input', '').strip()
            system_response = ep.get('system_response', '').strip()
            if not user_input or not system_response:
                continue
            if user_input.endswith('?'):
                # Normalize question (remove punctuation, lower)
                norm_q = user_input.lower().rstrip('?').strip()
                question_counter[norm_q] += 1
                # Store a simplified version: if question contains "what is X", we might generalize
                words = norm_q.split()
                if len(words) >= 3 and words[0] in ['what', 'who', 'where', 'when', 'why', 'how']:
                    # Create a template: e.g., "what is X" -> response pattern?
                    # For now, just store exact match
                    answer_counter[norm_q][system_response] += 1
                else:
                    answer_counter[norm_q][system_response] += 1

        for q, freq in question_counter.items():
            if freq >= 2 and q in answer_counter:
                most_common_ans = answer_counter[q].most_common(1)[0][0]
                rule_id = f"pattern_qa_{hash(q) % 10000:04x}"
                if rule_id not in self.rules:
                    rule = BehavioralRule(
                        rule_id=rule_id,
                        rule_type='conversational_pattern',
                        condition={'question_pattern': q},
                        action={'answer_template': most_common_ans},
                        confidence=0.3 + 0.1 * freq,
                        prior=0.2
                    )
                    candidates.append(rule)
        return candidates

    def deep_pattern_mining(self, engine: 'CuriosityEngine') -> List[BehavioralRule]:
        """
        Offline mining of longer sequences (e.g., 2-turn exchanges) using simple Markov chain.
        Called only during idle.
        """
        candidates = []
        episodes = engine.episodic_memory.episodes[-1000:]  # larger history

        # Build a bigram model over (user_input, system_response) pairs
        transition = defaultdict(Counter)  # prev_system -> next_user
        # We need to consider sequences: user1 -> system1 -> user2 -> system2 ...
        for i in range(len(episodes) - 1):
            ep1 = episodes[i]
            ep2 = episodes[i+1]
            if 'system_response' in ep1 and 'user_input' in ep2:
                prev_resp = ep1['system_response'].strip()
                next_user = ep2['user_input'].strip().lower()
                if prev_resp and next_user:
                    transition[prev_resp][next_user] += 1

        # Find common transitions
        for prev_resp, next_users in transition.items():
            total = sum(next_users.values())
            for next_user, count in next_users.items():
                if count >= 3 and count / total > 0.3:
                    rule_id = f"pattern_transition_{hash(prev_resp+next_user) % 10000:04x}"
                    if rule_id not in self.rules:
                        rule = BehavioralRule(
                            rule_id=rule_id,
                            rule_type='conversational_pattern',
                            condition={'previous_response_pattern': prev_resp},
                            action={'expected_user_input': next_user},
                            confidence=0.5 + 0.1 * (count / total),
                            prior=0.2
                        )
                        candidates.append(rule)
        return candidates
    
    def integrate_rule(self, rule_id: str):
        """Mark rule as fully integrated (confidence = 1.0)."""
        if rule_id in self.rules:
            self.rules[rule_id].confidence = 1.0
            self.save()


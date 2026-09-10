from concepts.meta_learner import MetaCuriosityLearner
from importglobal import Set,Dict,List,Any,Tuple, Optional, Counter, defaultdict, pickle, np, os, time, math
from traits.traits_global import Expression, Variable
from components.learner import KnowledgeGraph, ConfigManager, STATE_DIR, StructuralPatternLearner, UnificationEngine, ProceduralLearner, WordStateAssociator
from traits.traits_global import TypeHierarchy, Operator, Symbol, SymbolRegistry, RuleSchema,SchemaLearner
from components.rules import RuleEngine, ConstraintPropagator
from components.feedback_process import ContextualSentimentCorpus, FeedbackEntry
from components.pattern_observer import PatternObserver
# ------------------------------------------------------------
# SymbolicReasoner – performs rule-based inference on the knowledge graph
# ------------------------------------------------------------
class SymbolicReasoner:
    """
    Applies symbolic rules to the knowledge graph to derive new relations,
    and generates hypotheses for user confirmation.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, reasoner: 'ReasoningEngine'):
        self.kg = knowledge_graph
        self.reasoner = reasoner
        self.rules = []          # list of rule dicts
        self._load_default_rules()

    def _load_default_rules(self):
        """Add some default transitive rules."""
        self.rules.append({
            'type': 'transitive',
            'relation': 'is_a',
            'max_depth': 3,
            'threshold': 0.6,
            'description': 'If X is_a Y and Y is_a Z, then X is_a Z'
        })
        self.rules.append({
            'type': 'transitive',
            'relation': 'means',
            'max_depth': 3,
            'threshold': 0.6,
            'description': 'If X means Y and Y means Z, then X means Z'
        })
        self.rules.append({
            'type': 'transitive',
            'relation': 'similar',
            'max_depth': 2,
            'threshold': 0.8,
            'description': 'If X similar Y and Y similar Z, then X similar Z'
        })

    def run_inference(self):
        """Apply all rules to the knowledge graph."""
        for rule in self.rules:
            self.kg.apply_rule(rule)

    def generate_hypotheses_from_inferences(self):
        """Convert recent inferences into user-facing hypotheses."""
        if not hasattr(self.kg, 'inference_log'):
            return
        for entry in self.kg.inference_log[-10:]:  # recent only
            # Avoid duplicate hypotheses
            key = f"inference:{entry['node1']}:{entry['relation']}:{entry['node2']}"
            if any(h.get('key') == key for h in self.reasoner.hypotheses):
                continue
            # Get labels from kg
            node1_label = self.kg.nodes.get(entry['node1'], {}).get('label', 'unknown')
            node2_label = self.kg.nodes.get(entry['node2'], {}).get('label', 'unknown')
            hyp = {
                'key': key,
                'type': 'inference',
                'subject': node1_label,
                'predicate': entry['relation'],
                'object': node2_label,
                'confidence': entry['confidence'],
                'text': f"I inferred that {node1_label} {entry['relation']} {node2_label}. Is that correct?",
                'asked': False,
                'energy_cost': 0.5,
                'recent_usage': 0.0,
                'predictive_gain': entry['confidence'],
                'phase': 'active',
                'last_used': None,
            }
            self.reasoner.hypotheses.append(hyp)
# ------------------------------------------------------------
# Reasoning Engine (enhanced with active learning via GP)
# ------------------------------------------------------------
class ReasoningEngine:
    """
    Stores world knowledge, co-occurrence graphs, definitions, and semantic relations.
    Generates hypotheses about word meanings and plans queries.

    Now config-aware and supports definition learning & synonym detection.
    Extended with typed symbols, formal operators, expression trees,
    deterministic rules, and constraint propagation.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.co_occurrence = defaultdict(Counter)
        self.relations = defaultdict(list)          # word -> list of (rel_type, target_word, confidence)
        self.hypotheses = []                       # pending hypotheses to ask user
        # GP for predicting curiosity from word features (optional)
        self.gp_predictor = None
        self.knowledge_graph = KnowledgeGraph()
        self.symbolic_reasoner = SymbolicReasoner(self.knowledge_graph, self)

        # --- NEW: Definition memory ---
        self.definitions: Dict[str, str] = {}           # word -> definition text
        self.definition_embeddings: Dict[str, np.ndarray] = {}

        # --- NEW: Typed symbols & operators ---
        self.type_hierarchy = TypeHierarchy()
        self.operators: Dict[str, Operator] = {}
        self.symbol_registry = SymbolRegistry()
        self._register_builtin_operators()

        # --- NEW: Rule engine & constraint propagation ---
        self.rule_engine = RuleEngine()
        self.constraint_propagator = ConstraintPropagator(self.knowledge_graph)
        self._add_builtin_constraints()


    def _register_builtin_operators(self):
        """Define basic arithmetic and comparison operators and register them as symbols."""
        self.operators['+'] = Operator('+', 2, 10, lambda a,b: a+b)
        self.operators['-'] = Operator('-', 2, 10, lambda a,b: a-b)
        self.operators['*'] = Operator('*', 2, 20, lambda a,b: a*b)
        self.operators['/'] = Operator('/', 2, 20, lambda a,b: a/b if b!=0 else float('inf'))
        self.operators['='] = Operator('=', 2, 5, lambda a,b: a==b)
        self.operators['>'] = Operator('>', 2, 5, lambda a,b: a>b)
        self.operators['<'] = Operator('<', 2, 5, lambda a,b: a<b)

        # Register them in the symbol registry
        for op_name, op in self.operators.items():
            sym = Symbol(op_name, 'operator', arity=op.arity, semantics=op.evaluator)
            self.symbol_registry.register(sym)

    def _add_builtin_constraints(self):
        """Example constraints: if X is_a Y and Y is_a Z, then X is_a Z."""
        def transitive_is_a(kg):
            new_edges = []
            for n1, n2dict in kg.edges.items():
                for n2, rels in n2dict.items():
                    if 'is_a' in rels:
                        for n3, rels2 in kg.edges.get(n2, {}).items():
                            if 'is_a' in rels2 and n3 != n1:
                                conf = rels['is_a'] * rels2['is_a']
                                new_edges.append((n1, n3, 'is_a', conf))
            return new_edges
        self.constraint_propagator.add_constraint(transitive_is_a)
        # --- NEW: conversational pattern memory ---
        self.context_patterns = defaultdict(lambda: {
            "sentence_initial": 0,
            "short_utterance": 0,
            "total": 0
        })
        self.pattern_asked = set()
        # word -> embedding of definition
        # ========== SENTENCE-LEVEL REASONING (cutting-edge) ==========
        self.sentence_patterns = defaultdict(int)      # n-gram -> count
        self.sentence_ngram_order = 3                     # default trigram
        self.sentence_embeddings: Dict[str, np.ndarray] = {}   # hash -> embedding
        self.sentence_metadata: Dict[str, dict] = {}           # hash -> metadata
        self.sentence_asked = set()                           # queries already asked
        self.sentence_similarity_threshold = 0.85            # for analogies
        # ---- Feedback corpus for emergent sentiment learning ----
        self.feedback_corpus = ContextualSentimentCorpus()
        self.response_template_registry = {}   # template_id -> template text (for future learning)
        self.word_state = WordStateAssociator()
        # --- NEW: Phrase-level reasoning ---
        self.phrase_definitions: Dict[str, str] = {}           # phrase -> definition text
        self.phrase_embeddings: Dict[str, np.ndarray] = {}     # phrase -> embedding
        self.phrase_metadata: Dict[str, dict] = {}
        self.phrase_relations: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)  # phrase -> (rel_type, target, confidence)
        self.inferences: List[Dict] = []                        # pending inferred relations to ask user
        self.is_a_relations: Dict[str, Set[str]] = defaultdict(set)   # subject -> set of objects (for "is a")
        self.means_relations: Dict[str, Set[str]] = defaultdict(set)  # subject -> set of objects (for "means")
        # ---- Knowledge graph for connected reasoning ----
        self.structural_patterns = StructuralPatternLearner()
        # ---- Expression store for ASTs ----
        self.expressions: Dict[str, Expression] = {}  # sentence hash -> expression
        # ---- Rule schema learning ----
        self.schemas: List['RuleSchema'] = []          # learned rule schemas
        self.variable_counter: int = 0                  # for generating variable names
        self.schema_learner = SchemaLearner(self)       # the learner engine
        # ---- Unification engine for constraint solving ----
        self.unifier = UnificationEngine()
        self.constraints = []  # list of (left, right) equations
        # ---- Procedural learning ----
        self.procedural_learner = ProceduralLearner(self)
        # ---- Pattern observer: learns structural words + triplet extraction ----
        self.pattern_observer = PatternObserver()
        self.last_observed_triplets = []
    # ------------------------------------------------------------------
    # Unification and constraint solving
    # ------------------------------------------------------------------
    def solve_equation(self, left: Expression, right: Expression) -> Optional[Dict]:
        """
        Solve an equation left = right by unification or arithmetic solving.
        Returns a substitution dict if solvable, else None.
        """
        # Try unification first (for logical equations)
        subs = self.unifier.unify(left, right)
        if subs is not None:
            return subs
        # Try arithmetic solving
        return self.solve_arithmetic(left, right)

    def add_constraint(self, left: Expression, right: Expression):
        """Add an equation to the constraint store and attempt to simplify."""
        self.constraints.append((left, right))

    def substitute_in_expression(self, expr: Expression, subs: Dict) -> Expression:
        """Apply a substitution to an expression."""
        return self.unifier.substitute(expr, subs)

    def is_constant(self, expr: Expression) -> bool:
        """Check if expression contains no variables."""
        if expr.is_leaf():
            if isinstance(expr.root, Variable):
                return False
            try:
                float(expr.root)
                return True
            except:
                return False
        else:
            left_const = self.is_constant(expr.left) if expr.left else True
            right_const = self.is_constant(expr.right) if expr.right else True
            return left_const and right_const

    def find_single_variable(self, expr: Expression) -> Optional[str]:
        """If expression contains exactly one variable, return its name, else None."""
        if expr.is_leaf():
            if isinstance(expr.root, Variable):
                return expr.root.name
            else:
                return None
        else:
            left_var = self.find_single_variable(expr.left) if expr.left else None
            right_var = self.find_single_variable(expr.right) if expr.right else None
            if left_var and right_var:
                return None
            if left_var:
                return left_var
            if right_var:
                return right_var
            return None

    def eval_constant(self, expr: Expression) -> Optional[float]:
        """Evaluate a constant expression to a number. Return None if not constant."""
        if not self.is_constant(expr):
            return None
        return self.eval_expression(expr, {})

    def solve_arithmetic(self, left: Expression, right: Expression) -> Optional[Dict[str, Expression]]:
        """
        Solve a linear arithmetic equation with one variable.
        Returns a dict mapping variable name to an Expression (constant), or None if unsolvable.
        """
        # If both sides constant
        if self.is_constant(left) and self.is_constant(right):
            left_val = self.eval_constant(left)
            right_val = self.eval_constant(right)
            if left_val is not None and right_val is not None and abs(left_val - right_val) < 1e-9:
                return {}  # no variables, equation holds
            else:
                return None

        # If left is variable
        if left.is_leaf() and isinstance(left.root, Variable):
            if self.is_constant(right):
                val = self.eval_constant(right)
                if val is not None:
                    return {left.root.name: Expression(str(val))}
            return None

        # If right is variable
        if right.is_leaf() and isinstance(right.root, Variable):
            if self.is_constant(left):
                val = self.eval_constant(left)
                if val is not None:
                    return {right.root.name: Expression(str(val))}
            return None

        # If left is an operation and right is constant, and left has exactly one variable
        if not left.is_leaf() and self.is_constant(right) and self.find_single_variable(left) is not None:
            return self._solve_recursive(left, right)

        return None

    def _solve_recursive(self, left: Expression, right: Expression) -> Optional[Dict[str, Expression]]:
        """
        Helper for solve_arithmetic.
        Assumes right is constant, left contains exactly one variable.
        Returns substitution.
        """
        # If left is variable
        if left.is_leaf() and isinstance(left.root, Variable):
            val = self.eval_constant(right)
            if val is not None:
                return {left.root.name: Expression(str(val))}
            return None

        # left must be an operation
        if left.is_leaf():
            return None

        op = left.root
        if left.left is None or left.right is None:
            return None

        left_has_var = self.find_single_variable(left.left) is not None
        right_has_var = self.find_single_variable(left.right) is not None

        if left_has_var and right_has_var:
            return None

        if left_has_var:
            var_child = left.left
            const_child = left.right
            if op == '+':
                new_right_expr = Expression('-', right, const_child)
            elif op == '-':
                new_right_expr = Expression('+', right, const_child)
            elif op == '*':
                if self.eval_constant(const_child) == 0:
                    return None
                new_right_expr = Expression('/', right, const_child)
            elif op == '/':
                new_right_expr = Expression('*', right, const_child)
            else:
                return None
            return self._solve_recursive(var_child, new_right_expr)

        elif right_has_var:
            var_child = left.right
            const_child = left.left
            if op == '+':
                new_right_expr = Expression('-', right, const_child)
            elif op == '-':
                new_right_expr = Expression('-', const_child, right)
            elif op == '*':
                if self.eval_constant(const_child) == 0:
                    return None
                new_right_expr = Expression('/', right, const_child)
            elif op == '/':
                new_right_expr = Expression('/', const_child, right)
            else:
                return None
            return self._solve_recursive(var_child, new_right_expr)

        return None

    def eval_expression(self, expr: Expression, context: Dict) -> Any:
        """Evaluate an expression tree using the operator registry and context."""
        if expr.is_leaf():
            # leaf can be a constant or a variable from context
            if isinstance(expr.root, str):
                # Try to convert to number
                try:
                    return float(expr.root)
                except ValueError:
                    return context.get(expr.root, expr.root)  # fallback to string
            return expr.root
        else:
            op = expr.root
            if not isinstance(op, Operator):
                # maybe it's a string that should be an operator
                op = self.operators.get(expr.root)
                if op is None:
                    raise ValueError(f"Unknown operator: {expr.root}")
            left_val = self.eval_expression(expr.left, context)
            right_val = self.eval_expression(expr.right, context)
            return op.evaluate(left_val, right_val)
        
    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass
    # ------------------------------------------------------------------
    # Co-occurrence and basic hypothesis
    # ------------------------------------------------------------------
    def observe_sentence(self, sentence: str, tokens: List[str]):
        for i, w1 in enumerate(tokens):
            # pattern tracking
            pat = self.context_patterns[w1]
            pat["total"] += 1
            if i == 0:
                pat["sentence_initial"] += 1
            if len(tokens) <= 3:
                pat["short_utterance"] += 1

            # update co-occurrence counts (original functionality)
            for w2 in tokens[i+1:]:
                self.co_occurrence[w1][w2] += 1
                self.co_occurrence[w2][w1] += 1

        # ---- Add edges to knowledge graph for co-occurring words ----
        if len(tokens) >= 2:
            # Get node IDs for each token (add if not present)
            node_ids = {}
            for t in tokens:
                nid = self.knowledge_graph.get_node_id(t, "word")
                if nid is None:
                    emb = None
                    if hasattr(self, 'config_manager') and hasattr(self.config_manager, 'semantic_learner'):
                        # Attempt to get embedding
                        emb = self.config_manager.semantic_learner.word_embeddings.get(t)
                    nid = self.knowledge_graph.add_node(t, "word", embedding=emb)
                node_ids[t] = nid

            # For each pair, add co-occurrence edge with confidence based on frequency
            for i, w1 in enumerate(tokens):
                for w2 in tokens[i+1:]:
                    count = self.co_occurrence[w1][w2]   # now includes current sentence
                    # Confidence: 1 - exp(-count/5)  (saturates at ~1 after many co-occurrences)
                    conf = 1.0 - math.exp(-count / 5.0)
                    self.knowledge_graph.add_edge(node_ids[w1], node_ids[w2], "co_occur", confidence=conf)

        # --- store sentence (without embedding now; embedding added later) ---
        self.store_sentence(sentence, tokens, embedding=None,
                            metadata={'source': 'user', 'timestamp': time.time()})

        # --- Pattern observer: learn structural words from positional statistics ---
        self.last_observed_triplets = self.pattern_observer.observe(tokens)

        # --- NEW: Structural pattern learning ---
        pattern = self.structural_patterns.extract_pattern(sentence, tokens)
        if pattern:
            self.structural_patterns.add_pattern(sentence, tokens, pattern)

    def store_phrase_relation(self, subject: str, object: str, rel_type: str, confidence: float = 1.0):
        """
        Store a relation between two phrases (e.g., 'cat' 'is_a' 'animal').
        Also checks for transitive inferences immediately.
        """
        if rel_type == 'is_a':
            self.is_a_relations[subject].add(object)
        elif rel_type == 'means':
            self.means_relations[subject].add(object)
        # Also store in generic phrase_relations for completeness
        self.phrase_relations[subject].append((rel_type, object, confidence))
        # ---- Add to knowledge graph ----
        # Get or create nodes for subject and object
        subj_nid = self.knowledge_graph.get_node_id(subject, "phrase")
        if subj_nid is None:
            subj_nid = self.knowledge_graph.add_node(subject, "phrase")
        obj_nid = self.knowledge_graph.get_node_id(object, "phrase")
        if obj_nid is None:
            obj_nid = self.knowledge_graph.add_node(object, "phrase")
        # Add edge
        self.knowledge_graph.add_edge(subj_nid, obj_nid, rel_type, confidence=confidence)
        self._check_transitive_inferences(subject, object, rel_type)

    def run_constraint_propagation(self):
        """Apply constraint propagation to derive new edges."""
        self.constraint_propagator.propagate()

    def _check_transitive_inferences(self, subject: str, object: str, rel_type: str):
        """
        When a new relation (subject rel_type object) is added, look for
        chains that yield new inferred relations.
        """
        if rel_type == 'is_a':
            # If object has outgoing is_a relations, subject inherits them
            for obj2 in self.is_a_relations.get(object, []):
                if obj2 not in self.is_a_relations[subject]:
                    self.add_inference_hypothesis(subject, 'is_a', obj2, confidence=0.8)
            # Also check if subject appears as object of another relation
            for subj2, objs in self.is_a_relations.items():
                if subject in objs and object not in self.is_a_relations[subj2]:
                    self.add_inference_hypothesis(subj2, 'is_a', object, confidence=0.8)
        elif rel_type == 'means':
            # Similar transitive closure for "means"
            for obj2 in self.means_relations.get(object, []):
                if obj2 not in self.means_relations[subject]:
                    self.add_inference_hypothesis(subject, 'means', obj2, confidence=0.8)
            for subj2, objs in self.means_relations.items():
                if subject in objs and object not in self.means_relations[subj2]:
                    self.add_inference_hypothesis(subj2, 'means', object, confidence=0.8)

    def add_inference_hypothesis(self, subject: str, rel_type: str, object: str, confidence: float):
        """
        Create a natural language hypothesis for a transitive inference and
        add it to the hypotheses list to be asked later.
        """
        key = f"inference:{subject}:{rel_type}:{object}"
        if any(h.get('key') == key for h in self.hypotheses):
            return
        if rel_type == 'is_a':
            text = f"TPL.ASK.ISA subject='{subject}' object='{object}'?"
        elif rel_type == 'means':
            text = f"TPL.ASK.MEANS subject='{subject}' object='{object}'?"
        else:
            text = f"TPL.ASK.REL subject='{subject}' object='{object}'?"
        hyp = {
            'key': key,
            'type': 'inference',
            'subject': subject,
            'predicate': rel_type,
            'object': object,
            'confidence': confidence,
            'text': text,
            'asked': False,
            # ---- metabolic ----
            'energy_cost': 0.5,
            'recent_usage': 0.0,
            'predictive_gain': confidence,   # initial guess
            'phase': 'active',
            'last_used': None,
        }
        self.hypotheses.append(hyp)

    def _shadow_link_content_tokens(self, tokens: List[str]) -> List[str]:
        """Return compact content tokens for shadow differentiation proposals."""
        if not tokens:
            return []
        clean = []
        for token in tokens:
            token = str(token).strip().lower()
            if not token or token in {'tpl', 'ask', 'def', 'mean', 'fb'}:
                continue
            clean.append(token)
        try:
            content = self.pattern_observer.filter_content(clean)
        except Exception:
            content = clean
        return content or clean

    def _one_hop_neighbor_labels(self, tokens: List[str]) -> List[str]:
        labels = []
        seen = set(tokens)
        for token in tokens:
            nid = (
                self.knowledge_graph.get_node_id(token, "word")
                or self.knowledge_graph.get_node_id(token, "phrase")
            )
            if not nid:
                continue
            for neighbor_id, rel_type, confidence in self.knowledge_graph.get_neighbors(nid):
                label = self.knowledge_graph.nodes.get(neighbor_id, {}).get('label')
                if label and label not in seen and label not in labels:
                    labels.append(label)
                if len(labels) >= 4:
                    return labels
        return labels

    def propose_shadow_differentiation_link(
        self,
        context_tokens: List[str] = None,
        response_tokens: List[str] = None,
        condition: str = None,
        source: str = "feedback",
        rejected_response: str = None,
        metadata: Dict = None,
    ) -> Optional[Dict]:
        """
        Create a pending hypothesis for a differentiating relation without
        committing the edge to the knowledge graph.
        """
        context_tokens = self._shadow_link_content_tokens(context_tokens or [])
        response_tokens = self._shadow_link_content_tokens(response_tokens or [])
        triplets = list((metadata or {}).get('triplets') or self.last_observed_triplets or [])

        subject = None
        object_ = None
        if len(triplets) >= 2:
            h1, _c1, t1 = triplets[-2]
            h2, _c2, t2 = triplets[-1]
            subject = f"{h1} {t1}".strip()
            object_ = f"{h2} {t2}".strip()
        elif len(context_tokens) >= 4:
            subject = " ".join(context_tokens[:2])
            object_ = " ".join(context_tokens[2:4])
        elif len(context_tokens) >= 2:
            subject = context_tokens[0]
            neighbors = self._one_hop_neighbor_labels(context_tokens)
            object_ = " ".join(neighbors[:2]) if neighbors else context_tokens[1]
        elif context_tokens and response_tokens:
            subject = context_tokens[0]
            object_ = response_tokens[0]

        if not subject or not object_ or subject == object_:
            return None

        if condition is None:
            condition = " ".join(context_tokens[:3]) if context_tokens else "feedback"

        key = f"shadow_diff:{subject}:{object_}:{condition}"
        for hyp in self.hypotheses:
            if hyp.get('key') == key:
                return hyp if hyp.get('status', 'pending') == 'pending' else None

        hyp = {
            'key': key,
            'type': 'shadow_differentiation_link',
            'subject': subject,
            'object': object_,
            'predicate': 'different_context',
            'condition': condition,
            'confidence': 0.25,
            'status': 'pending',
            'source': source,
            'rejected_response': rejected_response,
            'metadata': metadata or {},
            'text': f"TPL.ASK.DIFF.LINK subject='{subject}' object='{object_}' condition='{condition}'?",
            'asked': False,
            'energy_cost': 0.4,
            'recent_usage': 0.0,
            'predictive_gain': 0.5,
            'phase': 'active',
            'last_used': None,
        }
        self.hypotheses.append(hyp)
        return hyp

    def _get_shadow_differentiation_hypothesis(self, proposal) -> Optional[Dict]:
        if isinstance(proposal, dict):
            return proposal
        for hyp in self.hypotheses:
            if hyp.get('key') == proposal and hyp.get('type') == 'shadow_differentiation_link':
                return hyp
        return None

    def commit_shadow_differentiation_link(self, proposal) -> bool:
        """Commit a pending differentiation proposal as low-confidence graph edges."""
        hyp = self._get_shadow_differentiation_hypothesis(proposal)
        if not hyp or hyp.get('status') in ('committed', 'rejected'):
            return False
        subject = hyp.get('subject')
        object_ = hyp.get('object')
        if not subject or not object_:
            return False
        n1 = self.knowledge_graph.get_node_id(subject, "phrase")
        if n1 is None:
            n1 = self.knowledge_graph.add_node(subject, "phrase")
        n2 = self.knowledge_graph.get_node_id(object_, "phrase")
        if n2 is None:
            n2 = self.knowledge_graph.add_node(object_, "phrase")
        metadata = {
            'condition': hyp.get('condition'),
            'source': hyp.get('source'),
            'rejected_response': hyp.get('rejected_response'),
            'proposal_key': hyp.get('key'),
        }
        confidence = hyp.get('confidence', 0.25)
        self.knowledge_graph.add_edge(n1, n2, 'different_context', confidence=confidence, metadata=metadata)
        self.knowledge_graph.add_edge(n2, n1, 'different_context', confidence=confidence, metadata=metadata)
        hyp['status'] = 'committed'
        hyp['asked'] = True
        hyp['committed_at'] = time.time()
        return True

    def reject_shadow_differentiation_link(self, proposal) -> bool:
        """Reject a pending differentiation proposal without touching the graph."""
        hyp = self._get_shadow_differentiation_hypothesis(proposal)
        if not hyp:
            return False
        hyp['status'] = 'rejected'
        hyp['asked'] = True
        hyp['rejected_at'] = time.time()
        return True

    def infer_possible_meanings(self, word: str) -> List[str]:
        if word not in self.co_occurrence:
            return []
        candidates = sorted(self.co_occurrence[word].items(), key=lambda x: -x[1])
        return [c for c, _ in candidates[:5]]

    def generate_hypothesis(self, word: str, context: Dict) -> str:
        candidates = self.infer_possible_meanings(word)
        if candidates:
            return f"TPL.HYP.REL word='{word}' candidate='{candidates[0]}'"
        else:
            return f"TPL.NO.CTX word='{word}'"

    def plan_query(self, target: str, uncertainty: float = None) -> str:
        """Generate a question; if uncertainty provided, mention it."""
        base = f"TPL.ASK.DEF target='{target}'?"
        if uncertainty is not None:
            return f"{base} uc={uncertainty:.2f}"
        return base
    
    def run_symbolic_inference(self):
        """Call during idle to apply rules and generate hypotheses."""
        self.symbolic_reasoner.run_inference()
        self.symbolic_reasoner.generate_hypotheses_from_inferences()
    # ------------------------------------------------------------------
    # Definition storage and similarity
    # ------------------------------------------------------------------
    def store_definition(self, word: str, definition_text: str, embedding: np.ndarray):
        """Save a definition and its embedding."""
        self.definitions[word] = definition_text
        self.definition_embeddings[word] = embedding

    def find_similar_definitions(self, word: str, threshold: float = 0.85) -> List[Tuple[str, float]]:
        """
        Return list of (other_word, similarity) for words whose definition embedding
        is similar to the given word's definition embedding.
        """
        if word not in self.definition_embeddings:
            return []
        emb = self.definition_embeddings[word]
        results = []
        for other, other_emb in self.definition_embeddings.items():
            if other == word:
                continue
            sim = np.dot(emb, other_emb) / (np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-12)
            if sim >= threshold:
                results.append((other, sim))
        results.sort(key=lambda x: -x[1])
        return results

    def find_similar_phrases(self, phrase_emb: np.ndarray, threshold: float = 0.85) -> List[Tuple[str, float]]:
        """
        Return list of (phrase, similarity) for stored phrases with embedding similar to query.
        """
        results = []
        for key, emb in self.phrase_embeddings.items():
            sim = np.dot(phrase_emb, emb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(emb) + 1e-12)
            if sim >= threshold:
                # retrieve original phrase from metadata; skip entries whose
                # text was never stored (legacy hash-only keys) instead of
                # leaking an internal hash into user-visible output.
                phrase = self.phrase_metadata.get(key, {}).get('phrase')
                if not phrase:
                    continue
                results.append((phrase, sim))
        results.sort(key=lambda x: -x[1])
        return results

    def store_phrase(self, phrase: str, embedding: np.ndarray, metadata: dict = None):
        """Store a phrase and its embedding for future similarity."""
        canon = phrase.strip().lower()
        if not canon:
            return
        key = str(hash(canon))
        self.phrase_embeddings[key] = embedding
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        meta.setdefault('phrase', canon)
        meta.setdefault('timestamp', time.time())
        self.phrase_metadata[key] = meta
    # ------------------------------------------------------------------
    # Synonym/category hypothesis generation
    # ------------------------------------------------------------------
    def generate_synonym_hypothesis(self, word: str) -> Optional[str]:
        """
        If the word has a definition and another word has a very similar definition,
        produce a hypothesis string (or store it in self.hypotheses).
        Returns the hypothesis text if one is newly created, else None.
        """
        if word not in self.definitions:
            return None
        similar = self.find_similar_definitions(word, threshold=0.85)
        if not similar:
            return None
        best_word, sim = similar[0]
        # Avoid repeating the same hypothesis
        hypothesis_key = f"synonym:{word}:{best_word}"
        if any(h.get('key') == hypothesis_key for h in self.hypotheses):
            return None
        hypothesis = {
            'key': hypothesis_key,
            'type': 'synonym',
            'word': word,
            'candidate': best_word,
            'similarity': sim,
            'text': f"TPL.ASK.SYN word='{word}' candidate='{best_word}'?",
            'asked': False,
            # ---- metabolic ----
            'energy_cost': 0.5,
            'recent_usage': 0.0,
            'predictive_gain': sim,
            'phase': 'active',
            'last_used': None,
        }
        self.hypotheses.append(hypothesis)
        return hypothesis['text']

    def plan_synonym_query(self, word: str, candidate: str) -> str:
        """Generate a direct question about synonymy."""
        return f"TPL.ASK.SYN word='{word}' candidate='{candidate}'?"

    # ------------------------------------------------------------------
    # Persistence (updated to include definitions and relations)
    # ------------------------------------------------------------------
    def save(self, path: str):
        # Convert numpy arrays to lists
        def_emb_serializable = {w: emb.tolist() for w, emb in self.definition_embeddings.items()}
        sent_emb_serializable = {h: emb.tolist() for h, emb in self.sentence_embeddings.items()}
        phrase_embed_serializable = {p: emb.tolist() for p, emb in self.phrase_embeddings.items()} 
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'co_occurrence': self.co_occurrence,
                'relations': self.relations,
                'hypotheses': self.hypotheses,
                'definitions': self.definitions,
                'definition_embeddings': def_emb_serializable,
                'sentence_patterns': self.sentence_patterns,
                'sentence_embeddings': sent_emb_serializable,
                'sentence_metadata': self.sentence_metadata,
                'sentence_asked': self.sentence_asked,
                'phrase_definitions': self.phrase_definitions,
                'phrase_embeddings': phrase_embed_serializable,
                'phrase_relations': self.phrase_relations,
                'phrase_metadata': self.phrase_metadata, 
                'inferences': self.inferences,
                'is_a_relations': dict(self.is_a_relations),
                'means_relations': dict(self.means_relations),
                'structural_patterns': self.structural_patterns,
                'type_hierarchy': self.type_hierarchy,
                'schemas': self.schemas,
                'procedures': self.procedural_learner.procedures,
                'po_retrofix_done': True,
            }, f)
        os.replace(tmp, path)
        # Save symbol registry
        self.symbol_registry.save(os.path.join(STATE_DIR, 'symbol_registry.json'))
        # Save word-state associations separately
        self.word_state.save(os.path.join(STATE_DIR, 'word_state.json'))
        self.knowledge_graph._save()

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.co_occurrence = data['co_occurrence']
            self.relations = data['relations']
            self.hypotheses = data.get('hypotheses', [])
            self.definitions = data.get('definitions', {})
            emb_dict = data.get('definition_embeddings', {})
            self.definition_embeddings = {w: np.array(emb) for w, emb in emb_dict.items()}
            raw = data.get('sentence_patterns', {})
            self.sentence_patterns = defaultdict(int)
            for k, v in raw.items():
                if isinstance(v, int):
                    self.sentence_patterns[k] = v
                elif isinstance(v, Counter):
                    self.sentence_patterns[k] = sum(v.values())
                else:
                    self.sentence_patterns[k] = int(v)
            sent_emb_dict = data.get('sentence_embeddings', {})
            self.sentence_embeddings = {h: np.array(emb) for h, emb in sent_emb_dict.items()}
            self.sentence_metadata = data.get('sentence_metadata', {})
            self.sentence_asked = data.get('sentence_asked', set())
            # NEW: load phrase data
            self.phrase_definitions = data.get('phrase_definitions', {})
            phrase_emb_dict = data.get('phrase_embeddings', {})
            self.phrase_embeddings = {p: np.array(emb) for p, emb in phrase_emb_dict.items()}
            self.phrase_relations = data.get('phrase_relations', defaultdict(list))
            self.inferences = data.get('inferences', [])
            self.is_a_relations = defaultdict(set, data.get('is_a_relations', {}))
            self.means_relations = defaultdict(set, data.get('means_relations', {}))
            self.phrase_metadata = data.get('phrase_metadata', {})
            self.type_hierarchy = data.get('type_hierarchy', TypeHierarchy())
            self.schemas = data.get('schemas', [])
            self.procedural_learner.procedures = data.get('procedures', {})
        # Load word-state associations
        self.word_state.load(os.path.join(STATE_DIR, 'word_state.json'))
        self.knowledge_graph._load()
        # Load structural patterns
        if 'structural_patterns' in data:
            self.structural_patterns = data['structural_patterns']
        else:
            self.structural_patterns = StructuralPatternLearner()
        # Load symbol registry
        self.symbol_registry.load(os.path.join(STATE_DIR, 'symbol_registry.json'))
        # Fix structural scores polluted by PC-off gaps — runs once per old save file.
        # After the first fix, 'po_retrofix_done' is written by save() so it never re-runs.
        if not data.get('po_retrofix_done', False):
            self.pattern_observer.recompute_active_counts()

    def detect_pattern_and_query(self, word: str) -> Optional[str]:
        pat = self.context_patterns[word]
        if pat["total"] < 6:
            return None

        if word in self.pattern_asked:
            return None

        init_ratio = pat["sentence_initial"] / pat["total"]
        short_ratio = pat["short_utterance"] / pat["total"]

        if init_ratio > 0.7 and short_ratio > 0.6:
            self.pattern_asked.add(word)
            return f"TPL.ASK.PATTERN.LABEL word='{word}'?"
        return None
    # ------------------------------------------------------------------
    # Sentence-level reasoning (cutting-edge)
    # ------------------------------------------------------------------
    def store_sentence(self, sentence: str, tokens: List[str],
                       embedding: Optional[np.ndarray] = None,
                       metadata: Optional[dict] = None):
        """
        Store a sentence for pattern learning, dreaming and analogy.
        Extracts n-grams and stores sentence embedding if provided.
        """
        canon = sentence.strip().lower()
        if not canon:
            return
        sent_hash = str(hash(canon))   # simple key

        # Store n-grams
        n = self.sentence_ngram_order
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                self.sentence_patterns[ngram] += 1

        # Store embedding and metadata
        if embedding is not None:
            self.sentence_embeddings[sent_hash] = embedding
            self.sentence_metadata[sent_hash] = metadata or {
                'sentence': canon,
                'timestamp': time.time(),
                'tokens': tokens
            }

    def find_similar_sentences_by_emb(self, query_emb: np.ndarray,
                                      threshold: float = None) -> List[Tuple[str, float, dict]]:
        """Return (hash, similarity, metadata) for sentences with cos sim > threshold."""
        if threshold is None:
            threshold = self.sentence_similarity_threshold
        results = []
        for h, emb in self.sentence_embeddings.items():
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-12)
            if sim >= threshold:
                results.append((h, sim, self.sentence_metadata.get(h, {})))
        results.sort(key=lambda x: -x[1])
        return results

    def generate_sentence_hypothesis(self, sentence: str, tokens: List[str],
                                     sentence_emb: np.ndarray) -> Optional[str]:
        """
        Generate a hypothesis about the sentence meaning or relations.
        Only called when all tokens are known.
        """
        # --- Definition pattern ---
        if "means" in sentence.lower() or "is" in sentence.lower() or "refers to" in sentence.lower():
            return None   # handled elsewhere

        # --- Category statement (X is Y) ---
        if len(tokens) >= 3 and tokens[1] in ['is', 'are', 'was', 'were', 'means']:
            subj, verb, obj = tokens[0], tokens[1], tokens[2]
            hyp_key = f"category:{subj}:{obj}"
            if not any(h.get('key') == hyp_key for h in self.hypotheses):
                hypothesis = {
                    'key': hyp_key,
                    'type': 'category',
                    'subject': subj,
                    'object': obj,
                    'text': f"TPL.ASK.CAT subject='{subj}' object='{obj}'?",
                    'asked': False,
                }
                self.hypotheses.append(hypothesis)
                return hypothesis['text']

        # --- Analogy from similar sentence ---
        similar = self.find_similar_sentences_by_emb(sentence_emb, threshold=0.9)
        if similar:
            best_hash, sim, meta = similar[0]
            best_sent = meta.get('sentence', '')
            if best_sent and best_sent != sentence:
                hyp_key = f"analogy:{hash(sentence)}:{best_hash}"
                if not any(h.get('key') == hyp_key for h in self.hypotheses):
                    hypothesis = {
                        'key': hyp_key,
                        'type': 'analogy',
                        'sentence': sentence,
                        'similar_sentence': best_sent,
                        'similarity': sim,
                        'text': f"TPL.ASK.ANALOGY sentence='{sentence}' similar='{best_sent}'?",
                        'asked': False,
                    }
                    self.hypotheses.append(hypothesis)
                    return hypothesis['text']
        return None

    def plan_sentence_query(self, sentence: str, uncertainty: float = None) -> Optional[str]:
        """Generate a question about a sentence when its meaning is unclear."""
        if sentence in self.sentence_asked:
            return None
        self.sentence_asked.add(sentence)
        base = f"TPL.ASK.SENT sentence='{sentence}'?"
        if uncertainty is not None:
            return f"{base} comp={1-uncertainty:.2f}"
        return base

    def detect_recurring_patterns(self, min_freq: int = 3) -> List[Tuple[tuple, int]]:
        """Return frequent n-grams from stored sentences."""
        patterns = [(ng, cnt) for ng, cnt in self.sentence_patterns.items() if cnt >= min_freq]
        patterns.sort(key=lambda x: -x[1])
        return patterns

    def add_feedback(self, entry: FeedbackEntry):
        """Store user feedback and update sentiment models."""
        self.feedback_corpus.add_entry(entry)

    def get_sentiment_probability(self, context_key: str, sentiment: str) -> float:
        return self.feedback_corpus.get_sentiment_probability(context_key, sentiment)

    def get_context_confidence(self, context_key: str) -> float:
        return self.feedback_corpus.get_confidence(context_key)

    def should_ask_feedback(self, tokens: List[str], factor_dict: Dict, template_id: Optional[str] = None) -> bool:
        return self.feedback_corpus.should_ask_feedback(tokens, factor_dict, template_id)

    def register_response_template(self, template_id: str, template_text: str):
        """Store a response template for future learning."""
        self.response_template_registry[template_id] = template_text

    def store_expression(self, sentence: str, expr: Expression):
        """Store an expression for a sentence."""
        # Use sentence hash as key to avoid duplicates
        key = str(hash(sentence))
        self.expressions[key] = expr
# ------------------------------------------------------------
# ParallelReasoner – generates multiple reasoning paths
# ------------------------------------------------------------
class ParallelReasoner:
    """
    Spawns multiple reasoning threads (hypotheses) by applying different
    curiosity factor weightings or perspectives.
    """
    def __init__(self, base_reasoner: ReasoningEngine, meta_learner: 'MetaCuriosityLearner'):
        self.base = base_reasoner
        self.meta = meta_learner
        self.threads = []   # list of dicts with 'name', 'weight_bias', 'hypotheses'

    def generate_thoughts(self, token: str, context: Dict) -> List[Dict]:
        """
        Produce several different hypotheses about `token` by perturbing
        the curiosity factor weights or using different reasoning strategies.
        """
        thoughts = []
        # Thread 1: default reasoning (original)
        thoughts.append({
            'name': 'default',
            'hypothesis': self.base.generate_hypothesis(token, context),
            'confidence': 0.7
        })
        # Thread 2: focus on uncertainty (high weight on uncertainty factor)
        high_uncertainty = self.meta.get_weights().copy()
        high_uncertainty['uncertainty'] = high_uncertainty.get('uncertainty', 0.5) * 2
        hyp2 = self.base.generate_hypothesis(token, context)  # simplified; could be more sophisticated
        thoughts.append({
            'name': 'high_uncertainty',
            'hypothesis': hyp2,
            'confidence': 0.6
        })
        # Thread 3: use knowledge graph to find related words
        nid = self.base.knowledge_graph.get_node_id(token, "word")
        related = []
        if nid:
            neighbors = self.base.knowledge_graph.get_neighbors(nid, 'co_occur')
            for nb, rt, conf in neighbors[:3]:
                nb_label = self.base.knowledge_graph.nodes.get(nb, {}).get('label', 'unknown')
                related.append(nb_label)
        if related:
            hyp3 = f"TPL.HYP.GRAPH token='{token}' related='{', '.join(related)}'"
        else:
            hyp3 = self.base.generate_hypothesis(token, context)
        thoughts.append({
            'name': 'graph',
            'hypothesis': hyp3,
            'confidence': 0.5
        })
        return thoughts

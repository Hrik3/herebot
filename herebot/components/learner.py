
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from components.resoning import ReasoningEngine
from importglobal import Optional,Dict,time,Set,np,List,json,os,Tuple,defaultdict,Counter,pickle
from traits.traits_global import Expression, Variable, SyntaxTree
from components.config_manager import ConfigManager
from components.directory_handler import STATE_DIR
from components.probabilistic_labeling import (
    ensure_probabilistic_stats,
    normalize_probability_field,
)

# =============================================================================
# UNIFICATION ENGINE – Bidirectional constraint solving
# =============================================================================
class UnificationEngine:
    """
    Performs unification of expressions with variables.
    """
    def unify(self, expr1: Expression, expr2: Expression, subs: Optional[Dict] = None) -> Optional[Dict]:
        """
        Unify two expressions and return a substitution dict mapping variable names to expressions.
        Returns None if unification fails.
        """
        if subs is None:
            subs = {}

        # If both are leaves
        if expr1.is_leaf() and expr2.is_leaf():
            # If expr1 is a variable
            if isinstance(expr1.root, Variable):
                return self._bind_variable(expr1.root.name, expr2, subs)
            # If expr2 is a variable
            if isinstance(expr2.root, Variable):
                return self._bind_variable(expr2.root.name, expr1, subs)
            # Both constants: must be equal
            if expr1.root == expr2.root:
                return subs
            else:
                return None

        # If one is leaf and other is not
        if expr1.is_leaf() and not expr2.is_leaf():
            if isinstance(expr1.root, Variable):
                return self._bind_variable(expr1.root.name, expr2, subs)
            else:
                return None
        if expr2.is_leaf() and not expr1.is_leaf():
            if isinstance(expr2.root, Variable):
                return self._bind_variable(expr2.root.name, expr1, subs)
            else:
                return None

        # Both are non-leaves
        if expr1.root != expr2.root:
            return None
        # Check same arity (both have left and right or both missing)
        if (expr1.left is None) != (expr2.left is None) or (expr1.right is None) != (expr2.right is None):
            return None

        # Recursively unify children
        if expr1.left is not None:
            subs = self.unify(expr1.left, expr2.left, subs)
            if subs is None:
                return None
        if expr1.right is not None:
            subs = self.unify(expr1.right, expr2.right, subs)
            if subs is None:
                return None
        return subs

    def _bind_variable(self, var_name: str, expr: Expression, subs: Dict) -> Optional[Dict]:
        """Bind var_name to expr in subs, checking for consistency."""
        if var_name in subs:
            # Already bound, unify the existing binding with expr
            return self.unify(subs[var_name], expr, subs)
        else:
            # Check for occurs check (optional) – skip for now
            subs = subs.copy()
            subs[var_name] = expr
            return subs

    def substitute(self, expr: Expression, subs: Dict) -> Expression:
        """Apply substitution to expression."""
        if expr.is_leaf():
            if isinstance(expr.root, Variable) and expr.root.name in subs:
                return subs[expr.root.name]
            else:
                return expr
        else:
            left = self.substitute(expr.left, subs) if expr.left else None
            right = self.substitute(expr.right, subs) if expr.right else None
            return Expression(expr.root, left, right)
# =============================================================================
# PROCEDURAL LEARNING – Step-by-step problem solving
# =============================================================================
class Procedure:
    """
    A sequence of steps to solve a class of problems.
    Each step is a transformation of an expression.
    """
    def __init__(self, name: str, steps: List[Dict], variables: List['Variable'] = None):
        self.name = name
        self.steps = steps   # each step: {'action': str, 'args': list}
        self.variables = variables or []
        self.confidence = 0.5
        self.uses = 0

    def apply(self, expr: Expression, engine: 'ReasoningEngine') -> Optional[Expression]:
        """Apply procedure steps to expression. Returns final expression or None if fails."""
        current = expr
        for step in self.steps:
            action = step['action']
            args = step.get('args', [])
            if action == 'isolate_variable':
                # e.g., move terms to isolate variable
                # This is a placeholder; real implementation would use rule schemas.
                pass
            elif action == 'simplify':
                # evaluate constant parts
                pass
            # ... more actions
        return current

class ProceduralLearner:
    """
    Learns step-by-step solution procedures from user explanations.
    """
    def __init__(self, engine: 'ReasoningEngine'):
        self.engine = engine
        self.procedures: Dict[str, Procedure] = {}  # problem pattern -> procedure

    def learn_procedure(self, problem: Expression, steps: List[str], solution: Expression):
        """
        Record a procedure from user-provided steps.
        steps: list of natural language descriptions (for now).
        """
        # Convert steps to internal representation (simplified)
        proc = Procedure(f"proc_{len(self.procedures)}", steps)
        self.procedures[str(problem)] = proc
        return proc

    def find_procedure(self, problem: Expression) -> Optional[Procedure]:
        """Find a procedure that matches the problem structure."""
        # For now, exact match on string representation.
        return self.procedures.get(str(problem))

    def apply_procedure(self, problem: Expression) -> Optional[Expression]:
        proc = self.find_procedure(problem)
        if proc:
            return proc.apply(problem, self.engine)
        return None
# ------------------------------------------------------------
# Syntax Learner – Bayesian grammar induction
# ------------------------------------------------------------
class SyntaxLearner:
    """
    Incrementally learns a probabilistic context-free grammar (PCFG)
    from observed sentences. Uses a Dirichlet process prior over rules
    to allow unbounded rule creation.

    Now config-aware (placeholder for future parameters).
    Extended with simple CYK parser to produce syntax trees.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        # rule probabilities: (A -> BC) or (A -> word)
        self.nonterminals = {'S', 'NP', 'VP', 'PP', 'N', 'V', 'Adj', 'Det'}
        self.lexicon = {}        # word -> set of POS tags
        self.rule_counts = defaultdict(Counter)  # LHS -> RHS -> count
        self.rule_total = Counter()              # LHS -> total count
        self.dp_alpha = 1.0     # Dirichlet process concentration

    def parse(self, tokens: List[str]) -> Optional[SyntaxTree]:
        """
        Simple CYK parser returning the highest-probability parse tree.
        Returns None if no parse found.
        """
        n = len(tokens)
        if n == 0:
            return None

        # Chart: table[i][j] = list of (nonterminal, tree) for span i..j (inclusive)
        chart = [[[] for _ in range(n)] for _ in range(n)]

        # Fill lexical rules (terminals)
        for i, word in enumerate(tokens):
            # find POS tags for word from lexicon
            tags = self.lexicon.get(word, {'NN'})  # default noun
            for tag in tags:
                tree = SyntaxTree(root=tag, children=[SyntaxTree(word)])
                chart[i][i].append((tag, tree))

        # Dynamic programming over lengths
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    for A_rule in self.rule_counts:
                        for (B, C), count in self.rule_counts[A_rule].items():
                            # look for B in chart[i][k] and C in chart[k+1][j]
                            for (B_tag, left_tree) in chart[i][k]:
                                if B_tag != B:
                                    continue
                                for (C_tag, right_tree) in chart[k+1][j]:
                                    if C_tag != C:
                                        continue
                                    tree = SyntaxTree(root=A_rule, children=[left_tree, right_tree])
                                    chart[i][j].append((A_rule, tree))
        # Return best S covering entire sentence
        if chart[0][n-1]:
            # choose one with highest probability? For simplicity, return first S
            for nonterm, tree in chart[0][n-1]:
                if nonterm == 'S':
                    return tree
            # fallback: first tree
            return chart[0][n-1][0][1]
        return None

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def add_sentence(self, sentence: str):
        """
        Use a simple CYK parser with current grammar to parse the sentence.
        If parsing fails, hypothesise new rules based on POS tags from CI engine?
        Simplified: we'll just record word–tag co-occurrences from existing lexicon.
        """
        tokens = sentence.split()
        # for now, just add to lexicon if unknown
        for t in tokens:
            if t not in self.lexicon:
                # assign a default tag (NN) and count
                self.lexicon[t] = {'NN': 1}
                # also assign a basic type in the type hierarchy (if accessible)

    def generate_response(self, target: str) -> str:
        """
        Generate a grammatical sentence about `target`.
        Very simple template for now – expand later.
        """
        return f"TPL.HYP.REL word='{target}' candidate='unknown'"

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'lexicon': self.lexicon,
                'rule_counts': self.rule_counts,
                'rule_total': self.rule_total,
            }, f)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.lexicon = data['lexicon']
            self.rule_counts = data['rule_counts']
            self.rule_total = data['rule_total']
# ------------------------------------------------------------
# Word-State Associator – maps words to semantic state labels
# ------------------------------------------------------------
class WordStateAssociator:
    """
    Tracks associations between words/phrases and user-defined state labels (e.g., "greeting").
    Supports multiple labels per word with confidence scores.
    Enhanced with conflict detection and user feedback integration.
    """
    def __init__(self):
        self.word_to_labels: Dict[str, Dict[str, float]] = defaultdict(dict)  # word -> {label: confidence}
        self.label_to_words: Dict[str, Set[str]] = defaultdict(set)           # label -> set of words
        self.word_label_stats: Dict[str, Dict[str, dict]] = defaultdict(dict)
        self.conflict_resolution_threshold = 0.3   # minimum confidence difference to auto-resolve
        self.conflict_history = []                  # list of resolved conflicts
        self.conflict_cooldown: Dict[str, int] = {}   # word -> interaction_index until which conflicts are suppressed

    def _ensure_word_stats(self, word: str, interaction_index: int = 0):
        stats = self.word_label_stats[word]
        labels = self.word_to_labels.get(word, {})
        total = float(sum(max(0.0, float(v)) for v in labels.values()) or 0.0)
        for label, confidence in labels.items():
            initial_probability = (float(confidence) / total) if total > 0 else 0.0
            stats[label] = ensure_probabilistic_stats(
                stats.get(label, {}),
                interaction_index=interaction_index,
                initial_usage=float(confidence),
                initial_probability=initial_probability,
            )
        return stats

    def _normalize_word_labels(self, word: str, interaction_index: int = 0, reward: float = 0.0, touched_label: str = None, weight: float = 0.0):
        stats = self._ensure_word_stats(word, interaction_index)
        normalize_probability_field(
            stats,
            interaction_index=interaction_index,
            reward=reward,
            touched_key=touched_label,
            touched_weight=weight,
            archive=True,
        )
        self.word_to_labels[word] = {
            label: float(meta.get('probability', 0.0))
            for label, meta in stats.items()
            if meta.get('phase', 'active') != 'archived'
        }
        for label in list(self.label_to_words.keys()):
            if word in self.label_to_words[label] and label not in self.word_to_labels[word]:
                self.label_to_words[label].discard(word)
        for label in self.word_to_labels[word]:
            self.label_to_words[label].add(word)
        return stats

    def get_label_probabilities(self, word: str, interaction_index: int = 0, include_archived: bool = False) -> Dict[str, float]:
        stats = self._normalize_word_labels(word, interaction_index)
        return {
            label: float(meta.get('probability', 0.0))
            for label, meta in stats.items()
            if include_archived or meta.get('phase', 'active') != 'archived'
        }

    def add_association(
        self,
        word: str,
        label: str,
        confidence: float = 1.0,
        source: str = "explicit",
        interaction_index: int = 0,
        reward: float = 0.0,
    ):
        """
        Add a word-label association. If the word already has the label, update confidence.
        If the word has other labels with similar confidence, we may need to ask later.
        """
        self._ensure_word_stats(word, interaction_index)
        if label not in self.word_label_stats[word]:
            self.word_to_labels[word][label] = max(0.0, float(confidence or 0.0))
            self._ensure_word_stats(word, interaction_index)
        weight = max(0.0, float(confidence or 0.0))
        self._normalize_word_labels(
            word,
            interaction_index=interaction_index,
            reward=reward,
            touched_label=label,
            weight=weight,
        )
        self.label_to_words[label].add(word)

    def get_labels(self, word: str, threshold: float = 0.5) -> List[str]:
        """Return labels for word with confidence > threshold."""
        probs = self.get_label_probabilities(word)
        return [lab for lab, conf in sorted(probs.items(), key=lambda item: item[1], reverse=True) if conf > threshold]

    def get_words(self, label: str, threshold: float = 0.5) -> List[str]:
        """Return words associated with a label (with any confidence)."""
        words = []
        for word in list(self.label_to_words.get(label, set())):
            probs = self.get_label_probabilities(word)
            if probs.get(label, 0.0) > threshold:
                words.append(word)
        return words

    def find_similar_words_by_label(self, word: str) -> List[str]:
        """Find other words that share at least one label with the given word."""
        labels = self.get_labels(word, threshold=0.3)
        similar = set()
        for lab in labels:
            similar.update(self.get_words(lab))
        similar.discard(word)
        return list(similar)

    def detect_conflicts(self, word: str, interaction_index: int = 0) -> List[Tuple[str, str]]:
        """
        Return pairs of labels for the same word that have similar confidence,
        indicating possible ambiguity that should be clarified.
        Respects conflict_cooldown to avoid prompting on every turn for dismissed conflicts.
        """
        if word not in self.word_to_labels:
            return []
        if self.conflict_cooldown.get(word, 0) > int(interaction_index or 0):
            return []
        labels = list(self.get_label_probabilities(word, interaction_index=interaction_index).items())
        conflicts = []
        for i, (l1, c1) in enumerate(labels):
            for l2, c2 in labels[i+1:]:
                if abs(c1 - c2) < self.conflict_resolution_threshold:
                    conflicts.append((l1, l2))
        return conflicts

    def mark_unresolved(self, word: str, labels: List[str], interaction_index: int = 0, cooldown_turns: int = 20):
        """
        Handle a conflict where neither label was confirmed (NEITHER / UNRESOLVED).
        Applies a small symmetric penalty to both labels, normalizes,
        sets a cooldown backoff, and records the event in conflict_history.
        """
        for lab in labels:
            stats = self.word_label_stats.get(word, {}).get(lab)
            if stats:
                stats['reward_trace'] = float(stats.get('reward_trace', 0.0)) - 0.5  # small symmetric penalty
        self._normalize_word_labels(word, interaction_index=interaction_index)
        self.conflict_cooldown[word] = int(interaction_index or 0) + cooldown_turns
        self.conflict_history.append({
            'word': word,
            'labels': list(labels),
            'kept': None,
            'outcome': 'NEITHER',
            'cooldown_until_interaction': int(interaction_index or 0) + cooldown_turns,
            'interaction_index': int(interaction_index or 0),
            'timestamp': time.time(),
        })

    def break_bridge(self, word: str, label1: str, label2: str, keep_label: str, interaction_index: int = 0, reward: float = -1.0):
        """
        User clarified that `word` is associated with `keep_label`, not the other.
        Reduce confidence of the other label.
        """
        if word in self.word_to_labels:
            reject_label = label2 if keep_label == label1 else label1
            self.add_association(word, keep_label, confidence=1.0, source="conflict_keep", interaction_index=interaction_index, reward=abs(reward))
            self._ensure_word_stats(word, interaction_index)
            rejected = self.word_label_stats[word].get(reject_label)
            if rejected:
                rejected['usage'] = max(0.0, float(rejected.get('usage', 0.0)) * 0.5)
                rejected['reward_trace'] = float(rejected.get('reward_trace', 0.0)) - abs(float(reward or 0.0))
                self._normalize_word_labels(
                    word,
                    interaction_index=interaction_index,
                    reward=-abs(float(reward or 0.0)),
                    touched_label=reject_label,
                    weight=0.0,
                )
        self.conflict_history.append({
            'word': word,
            'labels': [label1, label2],
            'kept': keep_label,
            'interaction_index': int(interaction_index or 0),
            'timestamp': time.time()
        })

    def save(self, path: str):
        # Convert sets to lists for JSON serialization
        data = {
            'word_to_labels': {w: dict(labs) for w, labs in self.word_to_labels.items()},
            'label_to_words': {lab: list(words) for lab, words in self.label_to_words.items()},
            'word_label_stats': {w: labs for w, labs in self.word_label_stats.items()},
            'conflict_history': self.conflict_history,
            'conflict_cooldown': self.conflict_cooldown,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.word_to_labels = defaultdict(dict, {w: defaultdict(float, labs) for w, labs in data['word_to_labels'].items()})
            self.label_to_words = defaultdict(set, {lab: set(words) for lab, words in data['label_to_words'].items()})
            self.word_label_stats = defaultdict(dict, data.get('word_label_stats', {}))
            for word in list(self.word_to_labels.keys()):
                self._ensure_word_stats(word, interaction_index=0)
            self.conflict_history = data.get('conflict_history', [])
            self.conflict_cooldown = data.get('conflict_cooldown', {})
# ------------------------------------------------------------
# Knowledge Graph – maps relationships between words, phrases, states
# ------------------------------------------------------------
class KnowledgeGraph:
    """
    Stores a heterogeneous graph of knowledge:
    - Nodes: words, phrases, states (each node has a type and a unique identifier)
    - Edges: relationships with confidence and metadata.
    Supports incremental updates, transitive inference, and merging.
    """
    def __init__(self):
        self.inference_log = [] 
        self.nodes = {}          # node_id -> {'type': str, 'label': str, 'embedding': np.ndarray (optional), 'metadata': dict}
        self.edges = defaultdict(lambda: defaultdict(dict))  # node1 -> node2 -> {rel_type: confidence, ...} (supports multiple relation types between same nodes)
        self.edge_list = []       # list of (node1, node2, rel_type, confidence, timestamp) for fast iteration
        self.edge_metadata = {}   # "node1|node2|rel_type" -> metadata
        self.node_counter = 0
        self._load()

    def _next_node_id(self):
        self.node_counter += 1
        return f"n{self.node_counter}"

    def add_node(self, label: str, node_type: str = "word", embedding: np.ndarray = None, metadata: dict = None) -> str:
        """Add a node, return its ID. If node with same label and type exists, return existing ID."""
        # Check if already exists
        for nid, ndata in self.nodes.items():
            if ndata['label'] == label and ndata['type'] == node_type:
                return nid
        nid = self._next_node_id()
        self.nodes[nid] = {
            'type': node_type,
            'label': label,
            'embedding': embedding.copy() if embedding is not None else None,
            'metadata': metadata or {}
        }
        return nid

    def get_node_id(self, label: str, node_type: str = "word") -> Optional[str]:
        for nid, ndata in self.nodes.items():
            if ndata['label'] == label and ndata['type'] == node_type:
                return nid
        return None

    def apply_rule(self, rule: dict):
        """
        Apply a symbolic rule to the graph.
        Rule format: {'if': (subject_predicate, relation, object_predicate),
                     'then': (new_subject, new_relation, new_object)}
        where each predicate can be a variable (e.g., '?x') or a constant.
        This is a simplified forward chaining.
        """
        # For now, only handle simple transitive rules: if A rel B and B rel C then A rel C
        if rule.get('type') == 'transitive' and rule.get('relation'):
            rel = rule['relation']
            # Build adjacency for this relation
            adj = defaultdict(list)
            for n1, n2, rt, conf in self.get_all_edges():
                if rt == rel and conf >= rule.get('threshold', 0.5):
                    adj[n1].append((n2, conf))
            # Apply transitive closure
            for start in adj:
                # BFS with depth limit
                visited = set()
                queue = [(start, 1.0, 0)]
                while queue:
                    cur, cum_conf, depth = queue.pop(0)
                    if depth > 0 and cur != start and depth <= rule.get('max_depth', 3):
                        existing = self.edges[start].get(cur, {}).get(rel, 0)
                        if existing < cum_conf:
                            self.add_edge(start, cur, rel, cum_conf)
                            # Log inference
                            self._log_inference(start, cur, rel, cum_conf, 'transitive')
                    if depth < rule.get('max_depth', 3):
                        for nxt, conf in adj.get(cur, []):
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append((nxt, cum_conf * conf, depth+1))

    def _log_inference(self, n1, n2, rel, conf, rule_name):
        """Store inference in a log for later review."""
        if not hasattr(self, 'inference_log'):
            self.inference_log = []
        self.inference_log.append({
            'timestamp': time.time(),
            'node1': n1,
            'node2': n2,
            'relation': rel,
            'confidence': conf,
            'rule': rule_name
        })

    def add_edge(self, node1: str, node2: str, rel_type: str, confidence: float = 1.0, metadata: dict = None):
        """Add or update an edge between two nodes."""
        if node1 not in self.nodes or node2 not in self.nodes:
            return
        # Update the adjacency dict
        self.edges[node1][node2][rel_type] = confidence
        if metadata:
            self.edge_metadata[f"{node1}|{node2}|{rel_type}"] = metadata
        # Also store in reverse direction for undirected? We'll treat as directed unless specified.
        # For many relations, we want symmetric (synonym, similar) but we'll store both ways if needed.
        # We'll store a separate list for easy iteration
        self.edge_list.append((node1, node2, rel_type, confidence, time.time()))
        # Optional: add reverse edge for symmetric relations
        if rel_type in ['synonym', 'similar', 'co_occur']:
            self.edges[node2][node1][rel_type] = confidence
            if metadata:
                self.edge_metadata[f"{node2}|{node1}|{rel_type}"] = metadata

    def get_neighbors(self, node_id: str, rel_type: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Return list of (neighbor_id, rel_type, confidence) for given node."""
        if node_id not in self.edges:
            return []
        neighbors = []
        for n2, rels in self.edges[node_id].items():
            for rt, conf in rels.items():
                if rel_type is None or rt == rel_type:
                    neighbors.append((n2, rt, conf))
        return neighbors

    def get_all_edges(self) -> List[Tuple[str, str, str, float]]:
        """Return list of (node1, node2, rel_type, confidence)."""
        edges = []
        for n1, n2dict in self.edges.items():
            for n2, rels in n2dict.items():
                for rt, conf in rels.items():
                    edges.append((n1, n2, rt, conf))
        return edges

    def infer_transitive_relations(self, rel_type: str, threshold: float = 0.5) -> List[Tuple[str, str, str, float]]:
        """
        For transitive relations (e.g., is_a, means), infer new edges via path.
        Returns list of inferred edges.
        """
        if rel_type not in ['is_a', 'means']:
            return []
        # Build adjacency list for this relation
        adj = defaultdict(list)
        for n1, n2, rt, conf in self.get_all_edges():
            if rt == rel_type and conf >= threshold:
                adj[n1].append((n2, conf))
        # Run Floyd-Warshall? Too heavy. Use BFS from each node to find paths of length 2+.
        inferred = []
        for start in adj:
            # BFS up to depth 3
            visited = set()
            queue = [(start, 1.0, 0)]  # (node, cumulative_confidence, depth)
            while queue:
                cur, cum_conf, depth = queue.pop(0)
                if depth > 0 and cur != start and depth <= 3:
                    # If there's not already a direct edge with confidence >= cum_conf, add inference
                    existing = self.edges[start].get(cur, {}).get(rel_type, 0)
                    if existing < cum_conf:
                        inferred.append((start, cur, rel_type, cum_conf))
                if depth < 3:
                    for nxt, conf in adj.get(cur, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, cum_conf * conf, depth+1))
        return inferred

    def merge_nodes(self, node_id1: str, node_id2: str, keep_id: Optional[str] = None):
        """Merge node2 into node1 (or keep specified). Transfer all edges, delete node2."""
        if keep_id is None:
            keep_id = node_id1
            remove_id = node_id2 if node_id2 != keep_id else node_id1
        else:
            if keep_id == node_id1:
                remove_id = node_id2
            else:
                remove_id = node_id1
        if remove_id not in self.nodes:
            return
        # Transfer edges from remove_id to keep_id
        # Incoming edges to remove_id: we need to update edges where target is remove_id
        # We'll collect all edges where remove_id is either end.
        edges_to_add = []
        # Outgoing
        for n2, rels in self.edges.get(remove_id, {}).items():
            for rt, conf in rels.items():
                edges_to_add.append((keep_id, n2, rt, conf))
        # Incoming
        for n1, n2dict in self.edges.items():
            if n1 != remove_id:
                if remove_id in n2dict:
                    for rt, conf in n2dict[remove_id].items():
                        edges_to_add.append((n1, keep_id, rt, conf))
        # Add transferred edges
        for n1, n2, rt, conf in edges_to_add:
            self.add_edge(n1, n2, rt, conf)
        # Remove old edges involving remove_id
        # Remove from edges dict
        if remove_id in self.edges:
            del self.edges[remove_id]
        for n1 in list(self.edges.keys()):
            if remove_id in self.edges[n1]:
                del self.edges[n1][remove_id]
        # Remove from node list
        del self.nodes[remove_id]
        # Optionally, update embeddings (e.g., average) for keep_id

    def prune_weak_edges(self, threshold: float = 0.1):
        """Remove edges with confidence below threshold."""
        to_remove = []
        for n1, n2dict in self.edges.items():
            for n2, rels in n2dict.items():
                for rt, conf in list(rels.items()):
                    if conf < threshold:
                        to_remove.append((n1, n2, rt))
        for n1, n2, rt in to_remove:
            del self.edges[n1][n2][rt]
            if not self.edges[n1][n2]:
                del self.edges[n1][n2]
        # Also clean edge_list? We'll regenerate on save.

    def get_central_nodes(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Return node IDs with highest degree (number of edges)."""
        degree = defaultdict(int)
        for n1, n2dict in self.edges.items():
            for n2, rels in n2dict.items():
                degree[n1] += len(rels)
                degree[n2] += len(rels)
        sorted_nodes = sorted(degree.items(), key=lambda x: -x[1])
        return sorted_nodes[:top_k]

    def save(self, path: str):
        """Serialize graph to file."""
        # Convert embeddings to lists for JSON
        nodes_serial = {}
        for nid, ndata in self.nodes.items():
            nd = ndata.copy()
            if nd['embedding'] is not None:
                nd['embedding'] = nd['embedding'].tolist()
            nodes_serial[nid] = nd
        edges_serial = []
        for n1, n2dict in self.edges.items():
            for n2, rels in n2dict.items():
                for rt, conf in rels.items():
                    edges_serial.append((n1, n2, rt, conf))
        data = {
            'nodes': nodes_serial,
            'edges': edges_serial,
            'edge_metadata': self.edge_metadata,
            'node_counter': self.node_counter
        }
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        nodes_serial = data['nodes']
        edges_serial = data['edges']
        self.node_counter = data.get('node_counter', 0)
        self.edge_metadata = data.get('edge_metadata', {})
        self.nodes = {}
        for nid, ndata in nodes_serial.items():
            nd = ndata.copy()
            if nd['embedding'] is not None:
                nd['embedding'] = np.array(nd['embedding'])
            self.nodes[nid] = nd
        self.edges = defaultdict(lambda: defaultdict(dict))
        for n1, n2, rt, conf in edges_serial:
            self.edges[n1][n2][rt] = conf
        # Rebuild edge_list? Not needed, can be regenerated.
        self.edge_list = edges_serial  # for compatibility

    def _load(self):
        path = os.path.join(STATE_DIR, 'knowledge_graph.json')
        if os.path.exists(path):
            self.load(path)

    def _save(self):
        path = os.path.join(STATE_DIR, 'knowledge_graph.json')
        self.save(path)
# ------------------------------------------------------------
# Structural Pattern Learner – learns sentence templates with slots
# ------------------------------------------------------------
class StructuralPatternLearner:
    """
    Extracts structural patterns from sentences (e.g., "X told Y 'Z'")
    and stores them as templates. Can generate new sentences by recombining
    templates with known words that fit the slots.
    """
    def __init__(self):
        self.templates = defaultdict(list)   # template_string -> list of slot_values
        self.slot_types = ['PERSON', 'QUOTE', 'THING', 'ACTION', 'PLACE']
        self.pattern_counts = Counter()
        self.generated = set()

    def extract_pattern(self, sentence: str, tokens: List[str]) -> Optional[str]:
        """
        Convert a sentence into a template by replacing content words with slot markers.
        Uses simple heuristics: nouns -> PERSON, quoted text -> QUOTE, etc.
        Returns a template string like "[PERSON] told [PERSON] '[QUOTE]'"
        """
        # crude POS tagging? For now, use regex for common patterns
        pattern = sentence
        # Replace quoted text
        import re
        quote_matches = re.findall(r'"([^"]*)"|' + "'([^']*)'", sentence)
        for q in quote_matches:
            q = q[0] or q[1]
            pattern = pattern.replace(f'"{q}"', '[QUOTE]').replace(f"'{q}'", '[QUOTE]')
        # Replace proper names (capitalized words) – heuristic
        words = sentence.split()
        for i, w in enumerate(words):
            if w.istitle() and w not in ['I', 'A'] and len(w) > 1:
                pattern = pattern.replace(w, '[PERSON]', 1)
        # If pattern changed, store it
        if pattern != sentence:
            # clean up multiple spaces
            pattern = ' '.join(pattern.split())
            return pattern
        return None

    def add_pattern(self, sentence: str, tokens: List[str], pattern: str):
        """Store the pattern and the original tokens for slot filling."""
        self.templates[pattern].append(tokens)
        self.pattern_counts[pattern] += 1

    def generate_from_pattern(self, pattern: str) -> Optional[str]:
        """
        Generate a new sentence by filling slots in the pattern with random words
        that have appeared in those slot positions in other occurrences.
        """
        if pattern not in self.templates or not self.templates[pattern]:
            return None
        # Choose a random example to borrow slot values
        example = np.random.choice(self.templates[pattern])
        # For each slot in pattern, replace with a random value from the same slot position across examples
        slot_map = {}
        words_in_pattern = pattern.split()
        new_words = []
        for w in words_in_pattern:
            if w.startswith('[') and w.endswith(']'):
                slot = w
                if slot not in slot_map:
                    # collect all values for this slot from stored examples
                    candidates = []
                    for ex in self.templates[pattern]:
                        # align positions – assumes same number of words
                        if len(ex) == len(words_in_pattern):
                            idx = words_in_pattern.index(w)
                            if idx < len(ex):
                                candidates.append(ex[idx])
                    if candidates:
                        slot_map[slot] = np.random.choice(candidates)
                    else:
                        slot_map[slot] = 'something'
                new_words.append(slot_map[slot])
            else:
                new_words.append(w)
        return ' '.join(new_words)

    def get_frequent_patterns(self, min_freq=2) -> List[str]:
        """Return patterns that appear at least min_freq times."""
        return [p for p, cnt in self.pattern_counts.items() if cnt >= min_freq]

"""
PatternObserver – learns structural vs content words purely from usage patterns.

No hardcoded word lists. Words that consistently appear in "bridge" positions
(between content-word phrases, across many different sentence pairs) are
identified as structural/grammatical connectors.

Algorithm:
  structural_score(w) = 0.4 * sentence_coverage(w)
                      + 0.3 * bridge_ratio(w)
                      + 0.3 * neighbor_entropy(w)

  High score → structural (function word, copula, connector).
  Low score  → content (noun, verb with real meaning, adjective).

Also provides:
  SentenceTripletExtractor  – (head_phrase, connector, tail_phrase) per sentence
  MathSentenceParser        – NL math → symbolic expression (all learned, no hardcoding)
  PatternObserver           – unified facade
"""
from importglobal import (
    np, Counter, defaultdict,
    Dict, List, Optional, Set, Tuple, math
)


# ---------------------------------------------------------------------------
# 1. Word Role Tracker
# ---------------------------------------------------------------------------
class WordRoleTracker:
    """
    Per-word positional statistics across observed sentences.
    Tracks bridge frequency, neighbour diversity, and sentence coverage.
    """
    def __init__(self):
        self.bridge_count: Counter = Counter()
        self.left_neighbors:  Dict[str, Counter] = defaultdict(Counter)
        self.right_neighbors: Dict[str, Counter] = defaultdict(Counter)
        self.sentence_count: Counter = Counter()
        self.total_sentences: int = 0
        self.total_count: Counter = Counter()

    def observe(self, tokens: List[str]):
        """Record statistics for one tokenised sentence."""
        self.total_sentences += 1
        seen: Set[str] = set()
        for i, tok in enumerate(tokens):
            self.total_count[tok] += 1
            if tok not in seen:
                self.sentence_count[tok] += 1
                seen.add(tok)
            # bridge position: has both a left and a right neighbour
            if 0 < i < len(tokens) - 1:
                self.bridge_count[tok] += 1
                self.left_neighbors[tok][tokens[i - 1]] += 1
                self.right_neighbors[tok][tokens[i + 1]] += 1

    def neighbor_entropy(self, word: str) -> float:
        """Shannon entropy of the word's left+right neighbour distribution."""
        counts = (list(self.left_neighbors[word].values()) +
                  list(self.right_neighbors[word].values()))
        total = sum(counts)
        if total == 0:
            return 0.0
        ent = -sum((c / total) * math.log(c / total + 1e-12)
                   for c in counts if c > 0)
        max_ent = math.log(len(counts) + 1)
        return ent / (max_ent + 1e-12)

    def structural_score(self, word: str) -> float:
        """
        Returns a score in [0, 1] reflecting how 'structural' this word is.
        Requires at least 3 total occurrences; returns 0 otherwise.
        """
        n = self.total_count.get(word, 0)
        if n < 3:
            return 0.0
        coverage     = self.sentence_count.get(word, 0) / max(self.total_sentences, 1)
        bridge_ratio = self.bridge_count.get(word, 0) / max(n, 1)
        entropy      = self.neighbor_entropy(word)
        return 0.4 * coverage + 0.3 * bridge_ratio + 0.3 * entropy


# ---------------------------------------------------------------------------
# 2. Structural Word Detector
# ---------------------------------------------------------------------------
class StructuralWordDetector:
    """
    Identifies structural/grammatical words purely from co-occurrence patterns.
    Uses a cached set that is rebuilt lazily after each new observation.
    """
    def __init__(self, threshold: float = 0.25, min_count: int = 3):
        self.tracker   = WordRoleTracker()
        self.threshold = threshold
        self.min_count = min_count
        self._cache: Set[str] = set()
        self._dirty:  bool    = True

    def observe(self, tokens: List[str]):
        """Feed a sentence's tokens to update statistics."""
        self.tracker.observe(tokens)
        self._dirty = True

    def _rebuild_cache(self):
        self._cache = {
            w
            for w, c in self.tracker.total_count.items()
            if c >= self.min_count
            and self.tracker.structural_score(w) >= self.threshold
        }
        self._dirty = False

    def is_structural(self, word: str) -> bool:
        if self._dirty:
            self._rebuild_cache()
        return word in self._cache

    def filter_content(self, tokens: List[str]) -> List[str]:
        """Return only the content (non-structural) tokens."""
        if self._dirty:
            self._rebuild_cache()
        return [t for t in tokens if t not in self._cache]

    def get_structural_words(self) -> Set[str]:
        if self._dirty:
            self._rebuild_cache()
        return set(self._cache)

    def scores(self) -> Dict[str, float]:
        """Return structural score for every seen word (useful for inspection)."""
        return {w: self.tracker.structural_score(w)
                for w in self.tracker.total_count}


# ---------------------------------------------------------------------------
# 3. Sentence Triplet Extractor
# ---------------------------------------------------------------------------
class SentenceTripletExtractor:
    """
    Extracts (head_phrase, connector, tail_phrase) triplets.

    Algorithm:
      Walk tokens left-to-right, alternating between collecting content-word
      runs and structural-word runs.  When a structural run appears between
      two content runs, emit a triplet.

    Tracks connector frequencies and emits clarification questions when a
    connector has been seen ≥ question_threshold times without a known role.
    """
    def __init__(self, detector: StructuralWordDetector,
                 question_threshold: int = 3):
        self.detector            = detector
        self.question_threshold  = question_threshold
        self.connector_counts:   Counter                       = Counter()
        self.connector_examples: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
        self.connector_roles:    Dict[str, str]                = {}   # connector -> rel_type
        self.asked_connectors:   Set[str]                      = set()
        self.pending_questions:  List[Dict]                    = []

    def extract(self, tokens: List[str]) -> List[Tuple[str, str, str]]:
        """Return list of (head, connector, tail) from a tokenised sentence."""
        triplets: List[Tuple[str, str, str]] = []
        i = 0
        while i < len(tokens):
            # -- collect left content run --
            left_run: List[str] = []
            while i < len(tokens) and not self.detector.is_structural(tokens[i]):
                left_run.append(tokens[i])
                i += 1
            if not left_run:
                i += 1          # skip a structural word at the very start
                continue

            # -- collect structural bridge --
            bridge: List[str] = []
            while i < len(tokens) and self.detector.is_structural(tokens[i]):
                bridge.append(tokens[i])
                i += 1
            if not bridge:
                continue        # no bridge after this left run

            # -- collect right content run --
            right_run: List[str] = []
            while i < len(tokens) and not self.detector.is_structural(tokens[i]):
                right_run.append(tokens[i])
                i += 1

            if right_run:
                head      = ' '.join(left_run)
                connector = ' '.join(bridge)
                tail      = ' '.join(right_run)
                triplets.append((head, connector, tail))
                self.connector_counts[connector] += 1
                if len(self.connector_examples[connector]) < 10:
                    self.connector_examples[connector].append((head, tail))

        return triplets

    def connector_to_reltype(self, connector: str) -> Optional[str]:
        """Return learned relation type for this connector, or None."""
        return self.connector_roles.get(connector)

    def learn_connector_role(self, connector: str, rel_type: str):
        """Confirm what structural role a connector plays."""
        self.connector_roles[connector] = rel_type

    def get_pending_questions(self) -> List[Dict]:
        """
        Return new questions for connectors seen ≥ threshold times
        that have no confirmed role yet.  Each connector asked only once.
        """
        new_qs: List[Dict] = []
        for connector, count in self.connector_counts.items():
            if (count >= self.question_threshold
                    and connector not in self.connector_roles
                    and connector not in self.asked_connectors):
                examples = self.connector_examples[connector][:2]
                ex_str   = '; '.join(f"'{h}' {connector} '{t}'"
                                     for h, t in examples)
                q = {
                    'connector': connector,
                    'text': (
                        f"I've seen '{connector}' linking concepts {count} times "
                        f"(e.g. {ex_str}).  What does '{connector}' express — "
                        f"(1) equivalence/definition, (2) category membership, "
                        f"(3) similarity, or (4) something else?"
                    ),
                    'type': 'connector_role',
                }
                new_qs.append(q)
                self.pending_questions.append(q)
                self.asked_connectors.add(connector)
        return new_qs


# ---------------------------------------------------------------------------
# 4. Math Sentence Parser
# ---------------------------------------------------------------------------
class MathSentenceParser:
    """
    Converts sentence-level NL math to a symbolic result dict.
    All word→number and word→operator mappings are learned (teach_* methods),
    not hardcoded.  Numeric literals are always recognised directly.

    Usage:
        parser.teach_number('two', 2.0)
        parser.teach_operator('plus', '+')
        parser.teach_equality('equals')
        result = parser.try_parse(['two', 'plus', 'three', 'equals', 'five'], {})
        # → {'left': 2.0, 'operator': '+', 'right': 3.0, 'expected_result': 5.0, ...}
    """
    def __init__(self):
        self.number_words:   Dict[str, float] = {}
        self.operator_words: Dict[str, str]   = {}
        self.equality_words: Set[str]         = set()

    def teach_number(self, word: str, value: float):
        self.number_words[word.lower()] = value

    def teach_operator(self, word: str, symbol: str):
        self.operator_words[word.lower()] = symbol

    def teach_equality(self, word: str):
        self.equality_words.add(word.lower())

    def _classify(self, tok: str) -> Tuple[str, any]:
        """Return (category, value) for a token."""
        t = tok.lower()
        if t in self.number_words:
            return 'number', self.number_words[t]
        if t in self.operator_words:
            return 'operator', self.operator_words[t]
        if t in self.equality_words:
            return 'equality', '='
        if t in ('+', '-', '*', '/', '=', '>', '<'):
            return 'operator', t
        try:
            return 'number', float(t)
        except ValueError:
            return 'unknown', t

    def try_parse(self, tokens: List[str],
                  known_operators: Dict) -> Optional[Dict]:
        """
        Attempt to interpret the token list as a math sentence.
        Returns a result dict or None.
        """
        classified = [(tok, *self._classify(tok)) for tok in tokens]
        numbers = [(i, c[2]) for i, c in enumerate(classified) if c[1] == 'number']
        ops     = [(i, c[2]) for i, c in enumerate(classified)
                   if c[1] == 'operator'
                   or (c[1] == 'unknown' and c[0] in known_operators)]

        if len(numbers) < 2:
            return None

        n1_pos, n1 = numbers[0]
        n2_pos, n2 = numbers[1]

        # Operator must sit between the first two numbers
        mid_ops = [(i, sym) for i, sym in ops if n1_pos < i < n2_pos]
        if not mid_ops:
            return None

        op_sym = mid_ops[0][1]
        result: Dict = {
            'left':     n1,
            'operator': op_sym,
            'right':    n2,
            'sentence': ' '.join(tokens),
        }
        if len(numbers) >= 3:
            result['expected_result'] = numbers[2][1]
        return result

    def infer_operator_from_values(self, left: float, right: float,
                                   result: float) -> Optional[str]:
        """
        Given `left WORD right = result`, figure out which arithmetic
        operation WORD represents by checking all four candidates.
        """
        tol = 1e-9
        if abs(left + right - result) < tol:
            return '+'
        if abs(left - right - result) < tol:
            return '-'
        if abs(left * right - result) < tol:
            return '*'
        if right != 0 and abs(left / right - result) < tol:
            return '/'
        return None


# ---------------------------------------------------------------------------
# 5. Unified Facade
# ---------------------------------------------------------------------------
class PatternObserver:
    """
    Unified facade for StructuralWordDetector, SentenceTripletExtractor,
    and MathSentenceParser.

    Typical call sequence per sentence:
        triplets = observer.observe(tokens)
        math     = observer.try_parse_math(tokens, reasoner.operators)
        qs       = observer.get_pending_questions()
    """
    def __init__(self, structural_threshold: float = 0.25,
                 min_count: int = 3,
                 connector_question_threshold: int = 3):
        self.detector         = StructuralWordDetector(
                                    threshold=structural_threshold,
                                    min_count=min_count)
        self.triplet_extractor = SentenceTripletExtractor(
                                    self.detector,
                                    question_threshold=connector_question_threshold)
        self.math_parser      = MathSentenceParser()

    # --- core observe ---
    def observe(self, tokens: List[str]) -> List[Tuple[str, str, str]]:
        """
        Feed a tokenised sentence.  Updates structural statistics and
        returns extracted (head, connector, tail) triplets.
        """
        self.detector.observe(tokens)
        return self.triplet_extractor.extract(tokens)

    # --- structural word API ---
    def is_structural(self, word: str) -> bool:
        return self.detector.is_structural(word)

    def filter_content(self, tokens: List[str]) -> List[str]:
        return self.detector.filter_content(tokens)

    def get_structural_words(self) -> Set[str]:
        return self.detector.get_structural_words()

    def structural_scores(self) -> Dict[str, float]:
        return self.detector.scores()

    # --- connector role API ---
    def connector_to_reltype(self, connector: str) -> Optional[str]:
        return self.triplet_extractor.connector_to_reltype(connector)

    def learn_connector_role(self, connector: str, rel_type: str):
        self.triplet_extractor.learn_connector_role(connector, rel_type)

    def get_pending_questions(self) -> List[Dict]:
        return self.triplet_extractor.get_pending_questions()

    # --- math parsing API ---
    def try_parse_math(self, tokens: List[str],
                       known_operators: Dict) -> Optional[Dict]:
        return self.math_parser.try_parse(tokens, known_operators)

    def infer_operator(self, left: float, right: float,
                       result: float) -> Optional[str]:
        return self.math_parser.infer_operator_from_values(left, right, result)

    def teach_math_number(self, word: str, value: float):
        self.math_parser.teach_number(word, value)

    def teach_math_operator(self, word: str, symbol: str):
        self.math_parser.teach_operator(word, symbol)

    def teach_math_equality(self, word: str):
        self.math_parser.teach_equality(word)
from importglobal import Optional,float,Dict,Counter,defaultdict,np,math,pickle,os,time
from components.config_manager import ConfigManager
# ------------------------------------------------------------
# Hierarchical Probabilistic Concept Model (Dirichlet Process)
# ------------------------------------------------------------
class HierarchicalConceptModel:
    """
    Represents words as distributions over an unbounded set of latent concepts.
    Uses a Dirichlet Process prior (via Chinese Restaurant Process) to allow
    new concepts to be created as needed.

    Now config‑aware: reads alpha, beta from ConfigManager.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None,
                 alpha: Optional[float] = None, beta: Optional[float] = None):
        self.config_manager = config_manager
        if config_manager:
            self.alpha = config_manager.get('hierarchical_concept_model.alpha', 0.5)
            self.beta = config_manager.get('hierarchical_concept_model.beta', 1.0)
        else:
            self.alpha = alpha if alpha is not None else 0.5
            self.beta = beta if beta is not None else 1.0

        self.concept_counts = Counter()   # total assignments per concept
        self.word_concept_counts = defaultdict(Counter)  # word → concept counts
        self.concept_features = defaultdict(Counter)     # concept → feature counts
        self.feature_vocab = set()
        # Store previous posterior for each word (to compute Bayesian surprise)
        self.prev_posterior = {}     # word -> dict(concept: probability)
        # New: meta‑concept layer
        self.meta_concept_counts = Counter()           # meta‑concept -> count
        self.concept_to_meta = {}                      # concept -> meta‑concept
        self.meta_alpha = 0.1                           # concentration for meta‑DP

    def update_parameters_from_config(self):
        """Reload parameters from the config manager (for hot reload)."""
        if self.config_manager:
            self.alpha = self.config_manager.get('hierarchical_concept_model.alpha', 0.5)
            self.beta = self.config_manager.get('hierarchical_concept_model.beta', 1.0)

    def add_word_observation(self, word: str, features: Dict[str, float]) -> int:
        """
        Given a word and its feature vector, assign it to a concept
        via CRP, update counts, and return the assigned concept ID.
        """
        # Store previous posterior for Bayesian surprise calculation
        if word in self.word_concept_counts:
            total = sum(self.word_concept_counts[word].values())
            self.prev_posterior[word] = {
                c: cnt / total for c, cnt in self.word_concept_counts[word].items()
            }
        else:
            self.prev_posterior[word] = {}

        # Compute CRP probabilities
        probs = []
        concepts = []
        total = sum(self.concept_counts.values())
        for c, cnt in self.concept_counts.items():
            probs.append(cnt / (total + self.alpha))
            concepts.append(c)
        probs.append(self.alpha / (total + self.alpha))   # new concept
        concepts.append(max(self.concept_counts.keys(), default=-1) + 1)

        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
        concept = np.random.choice(concepts, p=probs)


        # Update counts
        self.concept_counts[concept] += 1
        self.word_concept_counts[word][concept] += 1

        # For new concept, initialize its feature distribution
        if concept not in self.concept_features:
            self._initialize_concept_features(concept, features)

        # Update concept feature counts (Dirichlet posterior)
        for k, v in features.items():
            self.concept_features[concept][k] += v
            self.feature_vocab.add(k)

        return concept

    def _initialize_concept_features(self, concept: int, features: Dict[str, float]):
        """Smooth initialization with prior beta."""
        for k, v in features.items():
            self.concept_features[concept][k] = self.beta * v

    def predict_features(self, word: str) -> Dict[str, float]:
        """
        Predictive distribution over features for a word, marginalizing over concepts.
        """
        if word not in self.word_concept_counts:
            return {}
        total = sum(self.word_concept_counts[word].values())
        probs = {c: cnt / total for c, cnt in self.word_concept_counts[word].items()}
        features = defaultdict(float)
        for c, p in probs.items():
            concept_total = sum(self.concept_features[c].values())
            if concept_total == 0:
                continue
            for f, cnt in self.concept_features[c].items():
                features[f] += p * (cnt / concept_total)
        return dict(features)

    def entropy_word(self, word: str) -> float:
        """Entropy of concept distribution for the word."""
        if word not in self.word_concept_counts:
            return 0.0
        counts = list(self.word_concept_counts[word].values())
        total = sum(counts)
        probs = [c / total for c in counts]
        return -sum(p * math.log(p + 1e-12) for p in probs)

    def bayesian_surprise(self, word: str) -> float:
        """
        KL divergence between prior and posterior over concepts for the word.
        If no prior, returns 0.
        """
        if word not in self.word_concept_counts or word not in self.prev_posterior:
            return 0.0
        prior = self.prev_posterior[word]
        total = sum(self.word_concept_counts[word].values())
        posterior = {c: cnt / total for c, cnt in self.word_concept_counts[word].items()}
        # KL divergence
        kl = 0.0
        for c, p_post in posterior.items():
            p_prior = prior.get(c, 1e-10)
            kl += p_post * math.log(p_post / p_prior)
        return np.clip(kl, 0, 10)  # prevent explosion

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'beta': self.beta,
                'concept_counts': self.concept_counts,
                'word_concept_counts': self.word_concept_counts,
                'concept_features': self.concept_features,
                'feature_vocab': self.feature_vocab,
                'prev_posterior': self.prev_posterior,
            }, f)
        os.replace(tmp, path)

    def register_definition_event(self, word: str):
        """
        Called when a word receives an explicit definition.
        Performs semantic stabilization:
        - Reinforces dominant concept
        - Reduces entropy
        - Marks word as stabilized
        """

        if word not in self.word_concept_counts:
            return

        counts = self.word_concept_counts[word]

        # Reinforce dominant concept
        if isinstance(counts, dict) and len(counts) > 0:
            dominant = max(counts, key=counts.get)
            counts[dominant] += 10

        # Mark stabilized
        if not hasattr(self, "stabilized_words"):
            self.stabilized_words = set()

        self.stabilized_words.add(word)

        # Track definition event
        if not hasattr(self, "definition_events"):
            self.definition_events = {}

        self.definition_events[word] = {
            "timestamp": time.time(),
            "stability": 1.0,
            "type": "explicit_definition"
        }

    def form_meta_concepts(self, min_concepts_per_meta=3):
        """
        Cluster concepts into meta‑concepts using their feature distributions.
        Uses a simple Chinese Restaurant Process on the concept features.
        """
        # Build feature vectors for concepts
        concept_ids = list(self.concept_features.keys())
        if len(concept_ids) < min_concepts_per_meta:
            return

        # For each concept, compute its feature vector (normalized)
        concept_vectors = {}
        for c in concept_ids:
            total = sum(self.concept_features[c].values())
            if total == 0:
                continue
            vec = np.array([self.concept_features[c].get(f, 0) for f in self.feature_vocab])
            concept_vectors[c] = vec / total

        if len(concept_vectors) < min_concepts_per_meta:
            return

        # Chinese Restaurant Process for meta‑concepts
        meta_counts = Counter()
        concept_to_meta = {}
        for c, vec in concept_vectors.items():
            # Compute probabilities for existing meta‑concepts based on average feature similarity
            probs = []
            metas = []
            for m, count in meta_counts.items():
                # Get average feature vector of concepts in this meta
                members = [c2 for c2, m2 in concept_to_meta.items() if m2 == m]
                if members:
                    avg_vec = np.mean([concept_vectors[m2] for m2 in members], axis=0)
                    sim = np.dot(vec, avg_vec) / (np.linalg.norm(vec)*np.linalg.norm(avg_vec)+1e-12)
                    probs.append(count * sim)   # popularity * similarity
                    metas.append(m)
            # New meta‑concept probability
            probs.append(self.meta_alpha)
            metas.append(max(meta_counts.keys(), default=-1) + 1)

            probs = np.array(probs)
            probs = probs / probs.sum()
            chosen = np.random.choice(metas, p=probs)

            if chosen in meta_counts:
                meta_counts[chosen] += 1
                concept_to_meta[c] = chosen
            else:
                meta_counts[chosen] = 1
                concept_to_meta[c] = chosen

        self.meta_concept_counts = meta_counts
        self.concept_to_meta = concept_to_meta
        print(f"   [Meta‑Concept] Formed {len(meta_counts)} meta‑concepts from {len(concept_vectors)} base concepts.")

    @classmethod
    def load(cls, path: str, config_manager: Optional['ConfigManager'] = None):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(alpha=data['alpha'], beta=data['beta'], config_manager=config_manager)
        model.concept_counts = data['concept_counts']
        model.word_concept_counts = data['word_concept_counts']
        model.concept_features = data['concept_features']
        model.feature_vocab = data['feature_vocab']
        model.prev_posterior = data['prev_posterior']
        return model

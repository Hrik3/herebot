from importglobal import os,pickle,np,Counter,List,Optional
from components.config_manager import ConfigManager
# ------------------------------------------------------------
# Semantic Learner – Bayesian model of word meanings
# ------------------------------------------------------------
class SemanticLearner:
    """
    Implements a hierarchical generative model of word meanings:
    - Each word has a latent vector in a continuous semantic space.
    - Observed co-occurrence and user definitions are generated from this vector.
    - Variational inference via stochastic gradient descent.

    Now config-aware: reads emb_dim, alpha.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None,
                 emb_dim: int = None, alpha: float = None):
        self.config_manager = config_manager
        if config_manager:
            self.emb_dim = config_manager.get('semantic_learner.emb_dim', 64)
            self.alpha = config_manager.get('semantic_learner.alpha', 0.1)
        else:
            self.emb_dim = emb_dim if emb_dim is not None else 64
            self.alpha = alpha if alpha is not None else 0.1

        # word -> (mu, log_sigma) for its embedding
        self.word_embeddings = {}   # word -> np.ndarray(mean)
        self.word_logsigma = {}     # word -> np.ndarray(log variance)
        self.word_counts = Counter()

    def update_parameters_from_config(self):
        """Reload parameters from the config manager (for hot reload)."""
        if self.config_manager:
            new_emb_dim = self.config_manager.get('semantic_learner.emb_dim', 64)
            new_alpha = self.config_manager.get('semantic_learner.alpha', 0.1)
            if new_emb_dim != self.emb_dim:
                # Changing embedding dimension is non-trivial; we ignore for hot reload
                print("Warning: semantic_learner.emb_dim changed but cannot be applied live.")
            self.alpha = new_alpha

    def initialize_word(self, word: str):
        """Initialise embedding with prior N(0, I/alpha)."""
        self.word_embeddings[word] = np.zeros(self.emb_dim)
        self.word_logsigma[word] = np.zeros(self.emb_dim)  # log(1) = 0
        self.word_counts[word] = 0

    def observe_context(self, word: str, context_words: List[str]):
        """
        Update embedding using co-occurrence information.
        Simplified: assume that if two words appear together,
        their embeddings should be close (cosine similarity).
        Performs one step of variational inference.
        """
        if word not in self.word_embeddings:
            self.initialize_word(word)
        # compute gradient of ELBO
        emb = self.word_embeddings[word]
        # for each context word, increase similarity
        grad = np.zeros_like(emb)
        for cw in context_words:
            if cw in self.word_embeddings:
                c_emb = self.word_embeddings[cw]
                # cosine similarity gradient
                norm_e = np.linalg.norm(emb)
                norm_c = np.linalg.norm(c_emb)
                if norm_e > 0 and norm_c > 0:
                    grad += (c_emb / (norm_e * norm_c)) - (emb * np.dot(emb, c_emb)) / (norm_e**3 * norm_c)
        # prior gradient: -alpha * emb
        grad -= self.alpha * emb
        # update (stochastic gradient ascent)
        lr = 1.0 / (self.word_counts[word] + 1.0)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            grad = np.zeros_like(grad)
        self.word_embeddings[word] += lr * grad
        self.word_counts[word] += 1

    def observe_definition(self, word: str, definition_text: str,
                           structural_detector=None):
        """
        When user provides a definition (e.g., "markus means a name"),
        treat it as supervised learning: the embedding should be close
        to the embedding of the definition words.

        If a structural_detector is supplied (from PatternObserver), function
        words like 'means', 'is', 'a' are stripped from the definition before
        building the target embedding, so only content words shape the vector.
        """
        # tokenise definition, optionally filter structural/function words
        tokens = definition_text.split()
        if structural_detector is not None:
            tokens = [t for t in tokens if not structural_detector.is_structural(t)]
        known_embs = [self.word_embeddings[t] for t in tokens if t in self.word_embeddings]
        if not known_embs:
            return
        target = np.mean(known_embs, axis=0)
        target = target / (np.linalg.norm(target) + 1e-12)

        if word not in self.word_embeddings:
            self.initialize_word(word)
        # pull embedding toward target
        emb = self.word_embeddings[word]
        grad = (target - emb) - self.alpha * emb
        self.word_embeddings[word] += 0.1 * grad
        self.word_counts[word] += 1

    def similarity(self, w1: str, w2: str) -> float:
        """Cosine similarity between embeddings."""
        if w1 not in self.word_embeddings or w2 not in self.word_embeddings:
            return 0.0
        e1 = self.word_embeddings[w1]
        e2 = self.word_embeddings[w2]
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(e1, e2) / (norm1 * norm2)

    def text_to_embedding(self, text: str, fallback_to_char=True) -> Optional[np.ndarray]:
        """
        Convert a text string to an embedding by averaging embeddings of known words.
        If fallback_to_char, use character n-gram embedding for unknown words.
        Returns None only if no words and no characters (empty).
        """
        tokens = text.split()
        known_embs = []
        for t in tokens:
            if t in self.word_embeddings:
                known_embs.append(self.word_embeddings[t])
            elif fallback_to_char:
                # use character embedding
                known_embs.append(self.char_ngram_embedding(t, emb_dim=self.emb_dim))
        if not known_embs:
            return None
        emb = np.mean(known_embs, axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def char_ngram_embedding(self, text: str, n=3, emb_dim=64) -> np.ndarray:
        """
        Compute a simple character n-gram embedding for unknown words.
        Hash each n-gram to a position, use bag-of-char-ngrams.
        """
        # Pad text
        text = ' ' + text.lower() + ' '
        ngrams = []
        for i in range(len(text)-n+1):
            ngram = text[i:i+n]
            ngrams.append(ngram)
        emb = np.zeros(emb_dim)
        for ng in set(ngrams):  # unique n-grams
            idx = hash(ng) % emb_dim
            emb[idx] += 1.0
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb
    
    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'embeddings': self.word_embeddings,
                'logsigma': self.word_logsigma,
                'counts': self.word_counts,
            }, f)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_embeddings = data['embeddings']
            self.word_logsigma = data['logsigma']
            self.word_counts = data['counts']

import hashlib
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

        # Schema version for semantic representation
        self.semantic_space_version = 1

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
        """
        Initialise word embedding with a deterministic morphological subword prior.
        Avoids the zero-vector absorbing barrier while providing an empirical prior
        where morphologically related words start proximal and unrelated words have
        expected cosine similarity of zero.
        """
        prior = self.char_ngram_embedding(word, emb_dim=self.emb_dim)
        self.word_embeddings[word] = prior
        self.word_logsigma[word] = np.zeros(self.emb_dim)  # log(1) = 0
        self.word_counts[word] = 0

    def observe_context(self, word: str, context_words: List[str]):
        """
        Update embedding using co-occurrence information.
        Simplified: assume that if two words appear together,
        their embeddings should be close (cosine similarity).
        Performs one step of variational inference on the unit semantic sphere.
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
        norm_post = np.linalg.norm(self.word_embeddings[word])
        if norm_post > 0:
            self.word_embeddings[word] /= norm_post
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
            content_tokens = [t for t in tokens if not structural_detector.is_structural(t)]
            if content_tokens:
                tokens = content_tokens
        known_embs = [self.word_embeddings[t] for t in tokens if t in self.word_embeddings]
        if not known_embs:
            # Fall back to character embeddings of definition tokens if none registered yet
            known_embs = [self.char_ngram_embedding(t, emb_dim=self.emb_dim) for t in tokens]
        if not known_embs:
            return
        target = np.mean(known_embs, axis=0)
        norm_t = np.linalg.norm(target)
        if norm_t > 0:
            target /= norm_t
        else:
            return

        if word not in self.word_embeddings:
            self.initialize_word(word)
        # pull embedding toward target
        emb = self.word_embeddings[word]
        grad = (target - emb) - self.alpha * emb
        self.word_embeddings[word] += 0.5 * grad
        norm_post = np.linalg.norm(self.word_embeddings[word])
        if norm_post > 0:
            self.word_embeddings[word] /= norm_post
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
        return float(np.dot(e1, e2) / (norm1 * norm2))

    def text_to_embedding(self, text: str, fallback_to_char=True, structural_detector=None) -> Optional[np.ndarray]:
        """
        Convert a text string to an embedding with content-aware weighting:
        tokens are weighted by inverse frequency and soft structural downweighting,
        preserving grammatical distinction without allowing structural frames to dominate.
        """
        tokens = text.split()
        if not tokens:
            return None
        weights = []
        vectors = []
        for t in tokens:
            if t in self.word_embeddings:
                v = self.word_embeddings[t]
            elif fallback_to_char:
                v = self.char_ngram_embedding(t, emb_dim=self.emb_dim)
            else:
                continue

            cnt = self.word_counts.get(t, 1)
            w = 1.0 / np.sqrt(1.0 + cnt * 0.1)

            if structural_detector is not None and structural_detector.is_structural(t):
                w *= 0.15

            weights.append(max(0.1, w))
            vectors.append(v)

        if not vectors:
            return None

        weighted_sum = np.zeros(self.emb_dim, dtype=float)
        for v, w in zip(vectors, weights):
            weighted_sum += v * w

        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum /= norm
            return weighted_sum
        return None

    def char_ngram_embedding(self, text: str, n=3, emb_dim=64) -> np.ndarray:
        """
        Compute a deterministic character n-gram feature representation for text/subwords.
        Uses cryptographic MD5 hash modulo emb_dim with alternating sign-hashing
        (feature hashing) to ensure:
        1. Strict process-invariance across Python runs and platforms.
        2. Unbiased inner products (expected inner product of unrelated strings is 0).
        3. Preservation of subword/morphological similarity.
        """
        text = ' ' + text.lower() + ' '
        ngrams = []
        for i in range(len(text)-n+1):
            ngrams.append(text[i:i+n])
        emb = np.zeros(emb_dim, dtype=float)
        for ng in set(ngrams):  # unique n-grams
            digest = hashlib.md5(ng.encode('utf-8')).digest()
            h = int.from_bytes(digest[:4], byteorder='little')
            idx = h % emb_dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            emb[idx] += sign
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'version': self.semantic_space_version,
                'embeddings': self.word_embeddings,
                'logsigma': self.word_logsigma,
                'counts': self.word_counts,
            }, f)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_embeddings = data.get('embeddings', {})
            self.word_logsigma = data.get('logsigma', {})
            self.word_counts = data.get('counts', Counter())
            version = data.get('version', 0)
            if version == 0:
                # Migration of legacy v0 representations:
                # 1. Clean up multi-word state labels and factor-contaminated vectors
                legacy_keys = list(self.word_embeddings.keys())
                for k in legacy_keys:
                    v = self.word_embeddings[k]
                    v_norm = np.linalg.norm(v)
                    is_factor_contaminated = (v_norm > 0 and len(v) >= 14 and np.all(np.abs(v[13:]) < 1e-9) and np.any(v[:11] > 0))
                    if ' ' in k and k not in ['new york']:
                        del self.word_embeddings[k]
                        self.word_counts.pop(k, None)
                        self.word_logsigma.pop(k, None)
                    elif is_factor_contaminated or v_norm == 0:
                        self.word_embeddings[k] = self.char_ngram_embedding(k, emb_dim=self.emb_dim)
                self.semantic_space_version = 1
            else:
                self.semantic_space_version = version

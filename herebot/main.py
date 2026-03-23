"""
AGI Curiosity Engine v0.4 - Information-Theoretic Curiosity & Intrinsic Motivation
Integrated with real CI/Phonotactic/Diagnostic engines from `backend.ci`.

This system implements cutting-edge, human-level curiosity:

1. **Hierarchical Dirichlet Process** - latent concept sharing.
2. **Gaussian Process Uncertainty** - active learning via predictive variance.
3. **Bayesian Surprise** - KL divergence between prior and posterior.
4. **Intrinsic Reward** - prediction error / knowledge gain as reinforcement.
5. **Multi-factor meta-curiosity** - adaptively weighted by learned utility.
6. **Reasoning engine** - co-occurrence hypotheses and query planning.
7. **Persistent state** - lifelong growth.

All maths is probabilistic; decisions are sampled from posteriors.
# Based on concepts from:
# - Horswill, I. (2015). "Computational Guilt". AIIDE Workshop.
# - Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences". NIPS.
# - Oudeyer, P-Y. & Kaplan, F. (2007). "What is intrinsic motivation? A typology of computational approaches".
# - Schmidhuber, J. (1991). "A possibility for implementing curiosity and boredom in model-building neural controllers".
"""
from .importglobal import *
from .directory_handler import *
from .functionsglobal import *
from .config_manager import *
from concepts.concepts_global import *
from traits.traits_global import *
from .factor_handler import *
from .rules import *
from .learner import *
from .barriers import *
# ------------------------------------------------------------
# Language Model – base class for all self‑expanding language models
# ------------------------------------------------------------
class LanguageModel:
    """
    Abstract base for a neural language model that can be trained on
    episodic memory and synthetic sentences. Subclasses must implement
    train_on_sentence, generate, save, load, and a static method
    create_default.
    """
    def __init__(self):
        self.steps = 0
        self.loss_history = []

    def train_on_sentence(self, sentence: str, lr: float = None) -> float:
        """Train on one sentence. Return loss."""
        raise NotImplementedError

    def generate(self, prompt: str, max_len: int = 20, temperature: float = 0.8) -> str:
        """Generate a continuation given a prompt."""
        raise NotImplementedError

    def loss(self, input_ids, target_ids) -> float:
        """Compute loss for given input and target ids."""
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    @staticmethod
    def create_default(config_manager) -> 'LanguageModel':
        """Factory method to create a default instance."""
        raise NotImplementedError
# ------------------------------------------------------------
# LSTM Language Model – concrete implementation
# ------------------------------------------------------------
class LSTMLanguageModel(LanguageModel):
    """
    A small LSTM‑based language model that learns from sentences.
    Vocabulary expands automatically as new words are encountered.
    """
    def __init__(self, emb_dim=64, hidden_dim=128, vocab_size=None,
                 learning_rate=0.01, max_vocab=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_vocab = max_vocab

        # Vocabulary
        self.word_to_idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4
        if vocab_size is not None:
            self.vocab_size = min(vocab_size, max_vocab)

        # Model parameters (LSTM)
        self.W_emb = np.random.randn(self.vocab_size, emb_dim) * 0.1
        self.num_layers = 1
        self.cell_type = 'lstm'

        # LSTM gates: input, forget, cell, output (each has weights for input and hidden)
        self.W_i = np.random.randn(emb_dim + hidden_dim, hidden_dim) * 0.1
        self.b_i = np.zeros(hidden_dim)
        self.W_f = np.random.randn(emb_dim + hidden_dim, hidden_dim) * 0.1
        self.b_f = np.zeros(hidden_dim)
        self.W_c = np.random.randn(emb_dim + hidden_dim, hidden_dim) * 0.1
        self.b_c = np.zeros(hidden_dim)
        self.W_o = np.random.randn(emb_dim + hidden_dim, hidden_dim) * 0.1
        self.b_o = np.zeros(hidden_dim)

        # Output projection to vocabulary
        self.W_out = np.random.randn(hidden_dim, self.vocab_size) * 0.1
        self.b_out = np.zeros(self.vocab_size)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _one_hot(self, idx, size):
        vec = np.zeros(size)
        vec[idx] = 1
        return vec

    def forward(self, input_ids, prev_h=None, prev_c=None):
        """
        input_ids: list of token ids (length T)
        Returns:
            logits: (T, vocab_size)
            (h, c): final hidden and cell states
        """
        T = len(input_ids)
        if prev_h is None:
            h = np.zeros(self.hidden_dim)
            c = np.zeros(self.hidden_dim)
        else:
            h, c = prev_h, prev_c

        logits = []
        for t in range(T):
            x = self.W_emb[input_ids[t]]
            xh = np.concatenate([x, h])
            i = self._sigmoid(xh @ self.W_i + self.b_i)
            f = self._sigmoid(xh @ self.W_f + self.b_f)
            o = self._sigmoid(xh @ self.W_o + self.b_o)
            g = self._tanh(xh @ self.W_c + self.b_c)
            c = f * c + i * g
            h = o * self._tanh(c)
            logit = h @ self.W_out + self.b_out
            logits.append(logit)
        return np.array(logits), (h, c)

    def loss(self, input_ids, target_ids):
        logits, _ = self.forward(input_ids)
        T = len(input_ids)
        loss = 0.0
        for t in range(T):
            probs = self._softmax(logits[t])
            loss -= np.log(probs[target_ids[t]] + 1e-12)
        return loss / T

    def backward(self, input_ids, target_ids, lr=None):
        if lr is None:
            lr = self.learning_rate

        T = len(input_ids)

        # Forward pass storing intermediates
        xs = []
        xhs = []
        i_vals = []
        f_vals = []
        o_vals = []
        g_vals = []
        c_vals = [np.zeros(self.hidden_dim)]
        h_vals = [np.zeros(self.hidden_dim)]
        logits = []

        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        for t in range(T):
            x = self.W_emb[input_ids[t]]
            xs.append(x)
            xh = np.concatenate([x, h])
            xhs.append(xh)

            i = self._sigmoid(xh @ self.W_i + self.b_i)
            f = self._sigmoid(xh @ self.W_f + self.b_f)
            o = self._sigmoid(xh @ self.W_o + self.b_o)
            g = self._tanh(xh @ self.W_c + self.b_c)

            i_vals.append(i)
            f_vals.append(f)
            o_vals.append(o)
            g_vals.append(g)

            c = f * c + i * g
            h = o * self._tanh(c)

            c_vals.append(c)
            h_vals.append(h)

            logit = h @ self.W_out + self.b_out
            logits.append(logit)

        # Compute loss gradients
        dlogits = []
        for t, logit in enumerate(logits):
            probs = self._softmax(logit)
            probs[target_ids[t]] -= 1.0
            dlogits.append(probs)

        # Initialize gradients
        dW_emb = np.zeros_like(self.W_emb)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros(self.hidden_dim)
        dc_next = np.zeros(self.hidden_dim)

        for t in reversed(range(T)):
            dout = dlogits[t]
            dW_out += np.outer(h_vals[t+1], dout)
            db_out += dout

            dh = dout @ self.W_out.T + dh_next

            do = dh * self._tanh(c_vals[t+1])
            dc = dh * o_vals[t] * (1 - self._tanh(c_vals[t+1])**2) + dc_next

            d_og = self._tanh(c_vals[t+1]) * dh * o_vals[t] * (1 - o_vals[t])
            d_i = dc * g_vals[t] * i_vals[t] * (1 - i_vals[t])
            d_f = dc * c_vals[t] * f_vals[t] * (1 - f_vals[t])
            d_g = dc * i_vals[t] * (1 - g_vals[t]**2)

            d_xh = np.zeros(self.emb_dim + self.hidden_dim)
            d_xh += d_i @ self.W_i.T
            d_xh += d_f @ self.W_f.T
            d_xh += d_g @ self.W_c.T
            d_xh += d_og @ self.W_o.T

            dx = d_xh[:self.emb_dim]
            dh_prev = d_xh[self.emb_dim:]

            dW_i += np.outer(xhs[t], d_i)
            db_i += d_i
            dW_f += np.outer(xhs[t], d_f)
            db_f += d_f
            dW_c += np.outer(xhs[t], d_g)
            db_c += d_g
            dW_o += np.outer(xhs[t], d_og)
            db_o += d_og

            dW_emb[input_ids[t]] += dx

            dh_next = dh_prev
            dc_next = f_vals[t] * dc

        # Clip gradients
        clip = 5.0
        for param in [dW_emb, dW_i, db_i, dW_f, db_f, dW_c, db_c, dW_o, db_o, dW_out, db_out]:
            np.clip(param, -clip, clip, out=param)

        # Update parameters (SGD)
        self.W_emb -= lr * dW_emb
        self.W_i -= lr * dW_i
        self.b_i -= lr * db_i
        self.W_f -= lr * dW_f
        self.b_f -= lr * db_f
        self.W_c -= lr * dW_c
        self.b_c -= lr * db_c
        self.W_o -= lr * dW_o
        self.b_o -= lr * db_o
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

        loss = np.mean([-np.log(self._softmax(logits[t])[target_ids[t]] + 1e-12) for t in range(T)])
        self.steps += 1
        self.loss_history.append(loss)
        return loss

    def add_word_to_vocab(self, word):
        """Add a new word to vocabulary if not already present and space left."""
        if word in self.word_to_idx or self.vocab_size >= self.max_vocab:
            return self.word_to_idx.get(word, self.word_to_idx["<UNK>"])
        idx = self.vocab_size
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.vocab_size += 1
        # Expand embedding matrix
        new_row = np.random.randn(1, self.emb_dim) * 0.1
        self.W_emb = np.vstack([self.W_emb, new_row])
        # Expand output matrix (add a column)
        new_col = np.random.randn(self.hidden_dim, 1) * 0.1
        self.W_out = np.hstack([self.W_out, new_col])
        # Expand bias
        self.b_out = np.append(self.b_out, 0.0)
        return idx

    def sentence_to_ids(self, sentence, add_bos=True, add_eos=True):
        tokens = sentence.split()
        ids = []
        if add_bos:
            ids.append(self.word_to_idx["<BOS>"])
        for t in tokens:
            if t in self.word_to_idx:
                ids.append(self.word_to_idx[t])
            else:
                ids.append(self.word_to_idx["<UNK>"])
        if add_eos:
            ids.append(self.word_to_idx["<EOS>"])
        return ids

    def ids_to_sentence(self, ids, skip_special=True):
        words = []
        for i in ids:
            if i == self.word_to_idx["<EOS>"]:
                break
            if skip_special and i < 4:
                continue
            words.append(self.idx_to_word.get(i, "<UNK>"))
        return " ".join(words)

    def generate(self, prompt, max_len=20, temperature=0.8):
        prompt_ids = self.sentence_to_ids(prompt, add_bos=True, add_eos=False)
        if not prompt_ids:
            prompt_ids = [self.word_to_idx["<BOS>"]]
        input_ids = prompt_ids[:]
        h, c = None, None
        if input_ids:
            _, (h, c) = self.forward(input_ids)

        generated = input_ids[:]
        for _ in range(max_len):
            last_id = generated[-1]
            x = self.W_emb[last_id]
            xh = np.concatenate([x, h if h is not None else np.zeros(self.hidden_dim)])
            i = self._sigmoid(xh @ self.W_i + self.b_i)
            f = self._sigmoid(xh @ self.W_f + self.b_f)
            o = self._sigmoid(xh @ self.W_o + self.b_o)
            g = self._tanh(xh @ self.W_c + self.b_c)
            if c is None:
                c = np.zeros(self.hidden_dim)
            c = f * c + i * g
            h = o * self._tanh(c)
            logit = h @ self.W_out + self.b_out
            probs = self._softmax(logit / temperature)
            next_id = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_id)
            if next_id == self.word_to_idx["<EOS>"]:
                break
        return self.ids_to_sentence(generated)

    def train_on_sentence(self, sentence, lr=None):
        ids = self.sentence_to_ids(sentence)
        if len(ids) < 2:
            return 0.0
        input_ids = ids[:-1]
        target_ids = ids[1:]
        return self.backward(input_ids, target_ids, lr)

    def save(self, path):
        data = {
            'W_emb': self.W_emb,
            'W_i': self.W_i,
            'b_i': self.b_i,
            'W_f': self.W_f,
            'b_f': self.b_f,
            'W_c': self.W_c,
            'b_c': self.b_c,
            'W_o': self.W_o,
            'b_o': self.b_o,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'steps': self.steps,
            'loss_history': self.loss_history,
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'max_vocab': self.max_vocab,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.W_emb = data['W_emb']
        self.W_i = data['W_i']
        self.b_i = data['b_i']
        self.W_f = data['W_f']
        self.b_f = data['b_f']
        self.W_c = data['W_c']
        self.b_c = data['b_c']
        self.W_o = data['W_o']
        self.b_o = data['b_o']
        self.W_out = data['W_out']
        self.b_out = data['b_out']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = data['vocab_size']
        self.steps = data['steps']
        self.loss_history = data['loss_history']
        self.emb_dim = data['emb_dim']
        self.hidden_dim = data['hidden_dim']
        self.learning_rate = data['learning_rate']
        self.max_vocab = data['max_vocab']

    @staticmethod
    def create_default(config_manager):
        emb_dim = config_manager.get('semantic_learner.emb_dim', 64)
        return LSTMLanguageModel(
            emb_dim=emb_dim,
            hidden_dim=128,
            learning_rate=0.01,
            max_vocab=10000
        )
# ------------------------------------------------------------
# GeneratedCodeFeedback – groups logics by user feedback
# ------------------------------------------------------------
class GeneratedCodeFeedback:
    """
    Stores all feedback received for each generated code file.
    """
    FEEDBACK_FILE = os.path.join(STATE_DIR, "generated_code_feedback.json")

    def __init__(self):
        self.feedback = {}   # logic_id -> list of {rating, timestamp, context}
        self._load()

    def _load(self):
        if os.path.exists(self.FEEDBACK_FILE):
            try:
                with open(self.FEEDBACK_FILE, "r") as f:
                    self.feedback = json.load(f)
            except:
                self.feedback = {}

    def save(self):
        with open(self.FEEDBACK_FILE, "w") as f:
            json.dump(self.feedback, f, indent=2)

    def add_feedback(self, logic_id: str, rating: str, context: dict = None):
        """rating: 'positive', 'negative', 'neutral', 'context_positive', etc."""
        if logic_id not in self.feedback:
            self.feedback[logic_id] = []
        entry = {
            "rating": rating,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.feedback[logic_id].append(entry)
        self.save()

    def get_feedback_summary(self, logic_id: str) -> dict:
        """Return counts of positive/negative/neutral feedback."""
        entries = self.feedback.get(logic_id, [])
        counts = {"positive": 0, "negative": 0, "neutral": 0, "other": 0}
        for e in entries:
            rating = e["rating"].lower()
            if "positive" in rating:
                counts["positive"] += 1
            elif "negative" in rating:
                counts["negative"] += 1
            elif "neutral" in rating:
                counts["neutral"] += 1
            else:
                counts["other"] += 1
        return counts
# ------------------------------------------------------------
# Contextual Sentiment Corpus – emergent learning from user feedback
# ------------------------------------------------------------
@dataclass
class FeedbackEntry:
    """Stores a single feedback instance from user."""
    timestamp: float
    user_input: str
    system_response: str
    factor_dict: Dict[str, float]          # emotional/control state at response time
    rating: str                           # 'positive', 'negative', 'neutral', 'context_positive', 'context_negative', 'context_neutral'
    context_tokens: List[str]             # tokens from user input
    response_tokens: List[str]            # tokens from system response
    response_template_id: Optional[str] = None   # if response came from a template
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextualSentimentCorpus:
    """
    Maintains a corpus of user feedback with full context.
    Learns sentiment probabilities conditioned on context (tokens, emotional factors, response templates).
    Automatically stops asking for feedback when confidence is high.
    """
    CORPUS_FILE = os.path.join(STATE_DIR, "feedback_corpus.pkl")
    CONFIDENCE_THRESHOLD = 0.85   # stop asking when confidence > threshold
    MIN_SAMPLES = 5               # minimum samples before considering confident

    def __init__(self):
        self.entries: List[FeedbackEntry] = []
        self.context_models = {}   # context_key -> {'pos': Beta(a,b), 'neg': Beta(a,b), 'neu': Beta(a,b)}
        self._load()

    def _load(self):
        if os.path.exists(self.CORPUS_FILE):
            try:
                with open(self.CORPUS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.entries = data.get('entries', [])
                    self.context_models = data.get('context_models', {})
            except Exception as e:
                print(f"Could not load feedback corpus: {e}")

    def save(self):
        tmp = self.CORPUS_FILE + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'entries': self.entries,
                'context_models': self.context_models
            }, f)
        os.replace(tmp, self.CORPUS_FILE)

    def add_entry(self, entry: FeedbackEntry):
        self.entries.append(entry)
        self._update_model_for_entry(entry)
        self.save()

    def _update_model_for_entry(self, entry: FeedbackEntry):
        """Update Beta posteriors for all relevant contexts."""
        rating = entry.rating.lower()
        if rating not in ['positive', 'negative', 'neutral', 'context_positive', 'context_negative', 'context_neutral']:
            return

        # Use normalized rating: extract base sentiment
        if 'positive' in rating:
            base_sent = 'positive'
        elif 'negative' in rating:
            base_sent = 'negative'
        else:
            base_sent = 'neutral'

        # 1. Token contexts
        for token in entry.context_tokens:
            key = f"token:{token}"
            self._update_context_model(key, base_sent)

        # 2. Emotional factor contexts
        for factor in ['threat', 'trust', 'curiosity', 'energy', 'conscience_active']:
            if factor in entry.factor_dict:
                val = entry.factor_dict[factor]
                if val > 0.7:
                    key = f"factor:{factor}_high"
                elif val < 0.3:
                    key = f"factor:{factor}_low"
                else:
                    key = f"factor:{factor}_mid"
                self._update_context_model(key, base_sent)

        # 3. Response template ID (if available)
        if entry.response_template_id:
            key = f"template:{entry.response_template_id}"
            self._update_context_model(key, base_sent)

    def _update_context_model(self, context_key: str, sentiment: str):
        """Update Beta posterior for a specific context-sentiment pair."""
        if context_key not in self.context_models:
            self.context_models[context_key] = {
                'positive': {'a': 1, 'b': 1},
                'negative': {'a': 1, 'b': 1},
                'neutral': {'a': 1, 'b': 1}
            }
        model = self.context_models[context_key][sentiment]
        if sentiment == 'positive':
            model['a'] += 1
        else:
            model['b'] += 1


    def get_sentiment_probability(self, context_key: str, sentiment: str) -> float:
        """Return posterior mean of P(sentiment | context)."""
        if context_key not in self.context_models:
            return 0.33   # uniform prior
        model = self.context_models[context_key].get(sentiment, {'a':1, 'b':1})
        a = model['a']
        b = model['b']
        return a / (a + b)

    def get_confidence(self, context_key: str) -> float:
        """Return confidence (1 - posterior variance) for the most probable sentiment."""
        if context_key not in self.context_models:
            return 0.0
        max_confidence = 0.0
        total_samples = 0
        for sent, model in self.context_models[context_key].items():
            a = model['a']
            b = model['b']
            total_samples += (a - 1) + (b - 1)   # counts added to prior
            p = a / (a + b)
            variance = (a * b) / ((a + b)**2 * (a + b + 1))
            confidence = 1 - variance
            if confidence > max_confidence:
                max_confidence = confidence
        # Require minimum samples
        if total_samples < self.MIN_SAMPLES:
            return 0.0
        return max_confidence

    def should_ask_feedback(self, tokens: List[str], factor_dict: Dict, template_id: Optional[str] = None) -> bool:
        """
        Determine whether we need more feedback for the current context.
        Returns False if all relevant contexts are confident.
        """
        # Check token contexts
        for token in tokens:
            key = f"token:{token}"
            if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                return True
        # Check factor contexts
        for factor in ['threat', 'trust', 'curiosity', 'energy', 'conscience_active']:
            if factor in factor_dict:
                val = factor_dict[factor]
                if val > 0.7:
                    key = f"factor:{factor}_high"
                elif val < 0.3:
                    key = f"factor:{factor}_low"
                else:
                    key = f"factor:{factor}_mid"
                if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                    return True
        # Check template context
        if template_id:
            key = f"template:{template_id}"
            if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                return True
        return False
# ------------------------------------------------------------
# Intrinsic Motivation: Reward based on Prediction Error / Information Gain
# ------------------------------------------------------------
class IntrinsicRewardTracker:
    """
    Tracks cumulative intrinsic reward and provides reinforcement signal.
    Reward = prediction error (1 - confidence) + information gain + Bayesian surprise.

    Now config‑aware (placeholder for future tunable parameters).
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.reward_history = []
        # No tunable parameters currently, but we keep the config manager for consistency

    def update_parameters_from_config(self):
        """Placeholder for hot reload (no parameters currently)."""
        pass

    def compute_reward(self, token_ci: Dict, igain: float, bayes_surprise: float) -> float:
        """Compute intrinsic reward for a single token observation."""
        # Prediction error: 1 - confidence
        pred_error = 1.0 - token_ci.get('confidence', 0.5)
        # Reward = weighted combination
        reward = pred_error * 0.5 + igain * 0.3 + bayes_surprise * 0.2
        self.episode_reward += reward
        self.total_reward += reward
        return reward

    def reset_episode(self):
        self.episode_reward = 0.0

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({'total_reward': self.total_reward, 'reward_history': self.reward_history}, f)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.total_reward = data['total_reward']
            self.reward_history = data['reward_history']
# ------------------------------------------------------------
# SymbolicReasoner – performs rule‑based inference on the knowledge graph
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
        """Convert recent inferences into user‑facing hypotheses."""
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

    Now config‑aware and supports definition learning & synonym detection.
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

            # update co‑occurrence counts (original functionality)
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
            text = f"Is '{subject}' a '{object}'?"
        elif rel_type == 'means':
            text = f"Does '{subject}' mean '{object}'?"
        else:
            text = f"Does '{subject}' relate to '{object}'?"
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

    def infer_possible_meanings(self, word: str) -> List[str]:
        if word not in self.co_occurrence:
            return []
        candidates = sorted(self.co_occurrence[word].items(), key=lambda x: -x[1])
        return [c for c, _ in candidates[:5]]

    def generate_hypothesis(self, word: str, context: Dict) -> str:
        candidates = self.infer_possible_meanings(word)
        if candidates:
            return f"Perhaps '{word}' is related to '{candidates[0]}', since they appear together often."
        else:
            return f"I have no contextual clues about '{word}' yet."

    def plan_query(self, target: str, uncertainty: float = None) -> str:
        """Generate a question; if uncertainty provided, mention it."""
        base = f"What does '{target}' mean?"
        if uncertainty is not None:
            return f"{base} (My uncertainty about this word is {uncertainty:.2f}.)"
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
                # retrieve original phrase from metadata
                phrase = self.phrase_metadata.get(key, {}).get('phrase', key)
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
        self.phrase_metadata[key] = metadata or {'phrase': canon, 'timestamp': time.time()}
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
            'text': f"Perhaps '{word}' is similar to '{best_word}' – they seem to share a definition. Is that right?",
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
        return f"Is '{word}' similar in meaning to '{candidate}'?"

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
            return (
                f"I notice the word '{word}' appears in similar conversational situations. "
                f"What should I call this type of word?"
            )
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
                    'text': f"Is '{subj}' a type of '{obj}'?",
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
                        'text': f"This sentence reminds me of something you said earlier: '{best_sent}'. Are they related?",
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
        base = f"What does the sentence '{sentence}' mean?"
        if uncertainty is not None:
            return f"{base} (My comprehension is {1-uncertainty:.2f}.)"
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
            hyp3 = f"Perhaps '{token}' is related to {', '.join(related)}."
        else:
            hyp3 = self.base.generate_hypothesis(token, context)
        thoughts.append({
            'name': 'graph',
            'hypothesis': hyp3,
            'confidence': 0.5
        })
        return thoughts
# ------------------------------------------------------------
# State Manager – persistent storage of factor vectors
# Now with conflict detection, shadow resolution, and persistent conflict storage.
# ------------------------------------------------------------
class StateManager:
    CONFLICT_FILE = os.path.join(STATE_DIR, "conflicts.json")
    SHADOW_PREFIX = "shadow_merged_"

    def __init__(self, state_dir: str = "states", config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.state_dir = state_dir
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create STATE_DIR: {e}")
        self.states = {}          # label -> factor_vector (np.array)
        self._load_all_states()

        # Conflict management
        self.conflicts = []       # list of dicts: {"labels": [a,b], "similarity": s, "resolved": bool, "resolution_label": str or None}
        self.shadow_states = {}   # label -> factor_vector (temporary shadow states)
        self._load_conflicts()
        self._load_shadows()
        self.state_metadata: Dict[str, dict] = {}   # label -> {energy_cost, recent_usage, predictive_gain, phase, last_used, confidence}
        # Dimensionality: all vectors are aligned to the current reference dimension.
        self.ref_dim = None
        self._update_ref_dim()
        self._load_state_metadata()

    def _load_state_metadata(self):
        meta_path = os.path.join(STATE_DIR, 'state_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    self.state_metadata = json.load(f)
            except Exception as e:
                print(f"Could not load state metadata: {e}")

    def _save_state_metadata(self):
        meta_path = os.path.join(STATE_DIR, 'state_metadata.json')
        tmp = meta_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state_metadata, f, indent=2)
        os.replace(tmp, meta_path)

    def _update_ref_dim(self):
        """Set reference dimension to the mode of all stored vectors."""
        dims = [len(vec) for vec in self.states.values()]
        if dims:
            self.ref_dim = max(set(dims), key=dims.count)  # most common dimension
        else:
            self.ref_dim = 10  # default base dimension

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Pad or truncate vector to match reference dimension."""

        # --- ensure attribute exists ---
        if not hasattr(self, "ref_dim") or self.ref_dim is None:
            if vector is not None and len(vector) > 0:
                self.ref_dim = len(vector)
            else:
                self.ref_dim = 64   # safe fallback default

        if vector is None or len(vector) == 0:
            return np.zeros(self.ref_dim)

        if len(vector) < self.ref_dim:
            return np.pad(vector, (0, self.ref_dim - len(vector)), 'constant')

        if len(vector) > self.ref_dim:
            return vector[:self.ref_dim]

        return vector


    def update_parameters_from_config(self):
        """Reload parameters from config (hot reload)."""
        pass
    # ------------------------------------------------------------------
    # Core state I/O
    # ------------------------------------------------------------------
    def _load_all_states(self):
        """Load all existing state files from the directory."""
        self.states.clear()
        if not os.path.exists(self.state_dir):
            return
        for fname in os.listdir(self.state_dir):
            if fname.endswith(".json") and not fname.startswith(self.SHADOW_PREFIX):
                path = os.path.join(self.state_dir, fname)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    label = data.get("label", fname[:-5])
                    vector = data.get("vector", [])
                    metadata = data.get("metadata", {})
                    self.states[label] = self._normalize_vector(np.array(vector))
                    # Store metadata in a separate dictionary for later use
                    if not hasattr(self, 'state_metadata'):
                        self.state_metadata = {}
                    self.state_metadata[label] = metadata
                except Exception as e:
                    print(f"Could not load state {fname}: {e}")
        self._update_ref_dim()

    def save_state(self, label: str, vector: np.ndarray, metadata: dict = None, is_shadow: bool = False, response: str = None):
        """Save a factor vector under the given label."""
        os.makedirs(self.state_dir, exist_ok=True)
        safe_label = "".join(c for c in label if c.isalnum() or c in " _-").strip()
        if not safe_label:
            safe_label = "unnamed_state"
        if is_shadow:
            fname = f"{self.SHADOW_PREFIX}{safe_label}_{int(time.time())}.json"
        else:
            fname = f"{safe_label}_{int(time.time())}.json"
        path = os.path.join(self.state_dir, fname)
        
        # Store the vector as-is (original length)
        data = {
            "label": safe_label,
            "vector": vector.tolist(),
            "dimension": len(vector),
            "timestamp": time.time(),
            "metadata": metadata or {},
            "is_shadow": is_shadow,
        }
        # ---- metabolic metadata for new state ----
        self.state_metadata[safe_label] = {
            'energy_cost': 1.0,
            'recent_usage': 0.0,
            'predictive_gain': 1.0,  # placeholder
            'phase': 'active',
            'last_used': time.time(),
            'confidence': 1.0,        # how well the state represents the underlying pattern
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        vector = self._normalize_vector(vector)

        if is_shadow:
            self.shadow_states[safe_label] = vector
        else:
            self.states[safe_label] = vector

        self._update_ref_dim()
        self._save_state_metadata()   # <-- save metadata after change
        return safe_label

    def update_state(self, label: str, new_vector: np.ndarray, metadata: dict = None):
        """
        Update an existing state with a new vector.
        Deletes old file(s) for this label and saves a new one.
        """
        if label not in self.states:
            return
        # Delete old files associated with this label
        for fname in os.listdir(self.state_dir):
            if fname.startswith(label) and fname.endswith(".json"):
                try:
                    os.remove(os.path.join(self.state_dir, fname))
                except:
                    pass
        # Save new state
        self.save_state(label, new_vector, metadata=metadata, is_shadow=False)
        # Update in-memory dict
        self.states[label] = self._normalize_vector(new_vector)

    def delete_state(self, label: str):
        """Remove a state file and from memory."""
        if label in self.states:
            del self.states[label]
        if label in self.shadow_states:
            del self.shadow_states[label]
        for fname in os.listdir(self.state_dir):
            if fname.startswith(label) and fname.endswith(".json"):
                os.remove(os.path.join(self.state_dir, fname))
        if label in self.state_metadata:
            del self.state_metadata[label]
            self._save_state_metadata()
        self._update_ref_dim()

    def find_similar_state(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> Optional[str]:
        """
        Return the label of the most similar known state if cosine similarity > threshold.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        query_vec = self._normalize_vector(vector)
        all_states = dict(self.states)
        if include_shadows:
            all_states.update(self.shadow_states)

        if not all_states:
            return None
        best_label = None
        best_sim = -1.0
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        if best_sim >= threshold and best_label:
            # update usage
            meta = self.state_metadata.get(best_label)
            if meta:
                now = time.time()
                if meta.get('last_used'):
                    dt = now - meta['last_used']
                    meta['recent_usage'] *= 0.99 ** dt
                meta['recent_usage'] += 0.1
                meta['last_used'] = now
                self._save_state_metadata()
            return best_label
        return None

    def find_similar_states(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> List[Tuple[str, float]]:
        """
        Return a list of (label, similarity) for all states with similarity > threshold.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        query_vec = self._normalize_vector(vector)
        all_states = dict(self.states)
        if include_shadows:
            all_states.update(self.shadow_states)

        results = []
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim >= threshold:
                results.append((label, sim))
                # update usage for this state
                meta = self.state_metadata.get(label)
                if meta:
                    now = time.time()
                    if meta.get('last_used'):
                        dt = now - meta['last_used']
                        meta['recent_usage'] *= 0.99 ** dt
                    meta['recent_usage'] += 0.05   # lighter update
                    meta['last_used'] = now
                    self._save_state_metadata()
        return results
    # ------------------------------------------------------------------
    # Conflict detection & resolution (adapt to handle different lengths)
    # ------------------------------------------------------------------
    def detect_conflicts(self, threshold: float = None) -> List[Dict]:
        """
        Scan all states and find pairs with similarity > threshold.
        Both vectors are normalised to the reference dimension.
        Returns conflicts including associated words if available in metadata.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        labels = list(self.states.keys())
        conflicts = []
        processed_pairs = set()
        for i, l1 in enumerate(labels):
            v1 = self._normalize_vector(self.states[l1])
            # Retrieve associated words from metadata (if any)
            meta1 = self.state_metadata.get(l1, {})
            words1 = meta1.get('tokens', [])  # stored during state creation
            for l2 in labels[i+1:]:
                pair = tuple(sorted([l1, l2]))
                if pair in processed_pairs:
                    continue
                v2 = self._normalize_vector(self.states[l2])
                meta2 = self.state_metadata.get(l2, {})
                words2 = meta2.get('tokens', [])
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                if sim >= threshold:
                    conflict = {
                        "labels": [l1, l2],
                        "similarity": float(sim),
                        "words1": words1,          # list of tokens that led to state l1
                        "words2": words2,          # list of tokens that led to state l2
                        "resolved": False,
                        "resolution_label": None,
                        "timestamp": time.time(),
                    }
                    conflicts.append(conflict)
                    processed_pairs.add(pair)
        return conflicts

    def refresh_conflicts(self, threshold: float = None):
        """Re-detect conflicts and update self.conflicts, keeping existing resolution status if possible."""
        new_conflicts = self.detect_conflicts(threshold)

        existing_map = {}
        for c in self.conflicts:
            key = tuple(sorted(c["labels"]))
            existing_map[key] = c

        merged = []
        for c in new_conflicts:
            key = tuple(sorted(c["labels"]))
            if key in existing_map and existing_map[key].get("resolved", False):
                c["resolved"] = True
                c["resolution_label"] = existing_map[key].get("resolution_label")
            merged.append(c)

        self.conflicts = merged
        self._save_conflicts()

    def _save_conflicts(self):
        """Persist conflicts to JSON."""
        tmp = self.CONFLICT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.conflicts, f, indent=2)
        os.replace(tmp, self.CONFLICT_FILE)

    def _load_conflicts(self):
        """Load conflicts from disk."""
        if os.path.exists(self.CONFLICT_FILE):
            try:
                with open(self.CONFLICT_FILE, "r") as f:
                    self.conflicts = json.load(f)
            except Exception as e:
                print(f"Could not load conflicts: {e}")
                self.conflicts = []

    def _load_shadows(self):
        """Load all shadow states from the state directory."""
        self.shadow_states.clear()
        if not os.path.exists(self.state_dir):
            return
        for fname in os.listdir(self.state_dir):
            if fname.startswith(self.SHADOW_PREFIX) and fname.endswith(".json"):
                path = os.path.join(self.state_dir, fname)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    label = data.get("label", fname[:-5])
                    vector = data.get("vector", [])
                    self.shadow_states[label] = self._normalize_vector(np.array(vector))
                except Exception as e:
                    print(f"Could not load shadow state {fname}: {e}")

    def create_shadow_resolution(self, conflict: Dict) -> str:
        """Merge the two conflicting vectors (average) and create a temporary shadow state."""
        try:
            l1, l2 = conflict["labels"][0], conflict["labels"][1]
            v1 = self.states[l1]
            v2 = self.states[l2]
            # Align to the longer length before averaging
            max_len = max(len(v1), len(v2))
            v1_aligned = np.pad(v1, (0, max_len - len(v1)), 'constant') if len(v1) < max_len else v1[:max_len]
            v2_aligned = np.pad(v2, (0, max_len - len(v2)), 'constant') if len(v2) < max_len else v2[:max_len]
            merged_vector = self._normalize_vector((v1_aligned + v2_aligned) / 2.0)

            shadow_label = f"shadow_{l1}_{l2}_{int(time.time())}"
            metadata = {
                "merged_from": [l1, l2],
                "similarity": conflict["similarity"],
                "timestamp": time.time(),
            }
            self.save_state(shadow_label, merged_vector, metadata, is_shadow=True)
            return shadow_label
        except Exception as e:
            print(f"   [Error creating shadow resolution: {e}]")
            # Return a fallback label
            return f"shadow_error_{int(time.time())}"

    def apply_shadow_to_conflict(self, conflict: Dict, shadow_label: str):
        """Mark conflict as resolved with the given shadow label."""
        conflict["resolved"] = True
        conflict["resolution_label"] = shadow_label
        conflict["resolution_type"] = "shadow"
        self._save_conflicts()

    def resolve_conflict_permanently(self, conflict: Dict, winning_label: str, delete_other: bool = True):
        """User chooses one label as the canonical state."""
        l1, l2 = conflict["labels"]
        if winning_label == l1:
            to_delete = l2
        else:
            to_delete = l1

        if delete_other:
            self.delete_state(to_delete)

        conflict["resolved"] = True
        conflict["resolution_label"] = winning_label
        conflict["resolution_type"] = "permanent"
        self._save_conflicts()

    def get_unresolved_conflicts(self) -> List[Dict]:
        """Return conflicts that are not yet resolved."""
        return [c for c in self.conflicts if not c.get("resolved", False)]
# ------------------------------------------------------------
# Identity Manager – persistent self-identity representation
# ------------------------------------------------------------
class IdentityManager:
    """
    Maintains a stable self-identity vector that is reinforced
    over successful interactions.
    """

    def __init__(self, emb_dim: int = 64, learning_rate: float = 0.05):
        self.save_dir = STATE_DIR                       # <<< ADD THIS
        self.IDENTITY_FILE = os.path.join(self.save_dir, "identity.npz")  # <<< MOVE HERE

        self.emb_dim = emb_dim
        self.learning_rate = learning_rate

        # Identity embedding – starts as zero vector (neutral)
        self.identity = np.zeros(emb_dim)

        # Confidence in identity (0-1)
        self.confidence = 0.1

        # Number of reinforcement events
        self.reinforcement_count = 0

        self._load()

    def _load(self):
        """Load identity from disk."""
        if os.path.exists(self.IDENTITY_FILE):
            try:
                data = np.load(self.IDENTITY_FILE)
                self.identity = data['identity']
                self.confidence = data['confidence']
                self.reinforcement_count = data['reinforcement_count']
                self.emb_dim = len(self.identity)
            except Exception as e:
                print(f"Could not load identity: {e}")
    
    def save(self):
        """Save identity to disk (atomic)."""
        os.makedirs(self.save_dir, exist_ok=True)

        tmp = self.IDENTITY_FILE + ".tmp.npz"

        with open(tmp, "wb") as f:
            np.savez(
                f,
                identity=self.identity,
                confidence=self.confidence,
                reinforcement_count=self.reinforcement_count
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.IDENTITY_FILE)
    
    def reinforce(self, factor_vector: np.ndarray, reward: float):
        """
        Update identity embedding towards factor_vector when a positive
        outcome occurs. Automatically expands identity if factor_vector is longer.
        """
        # If factor_vector is longer than current identity, expand identity
        # project factor_vector to identity dimension
        if len(factor_vector) != len(self.identity):
            projected = np.zeros_like(self.identity)
            n = min(len(projected), len(factor_vector))
            projected[:n] = factor_vector[:n]
            factor_vector = projected

        
        aligned = np.zeros_like(self.identity)
        n = min(len(aligned), len(factor_vector))
        aligned[:n] = factor_vector[:n]

        norm = np.linalg.norm(aligned)
        if norm > 1e-12:
            direction = aligned / norm
        else:
            direction = np.zeros_like(aligned)
        
        # Update identity (soft update)
        alpha = self.learning_rate * reward * (1 - self.confidence)
        self.identity = (1 - alpha) * self.identity + alpha * direction
        
        # Normalize identity
        norm_id = np.linalg.norm(self.identity)
        if norm_id > 0:
            self.identity /= norm_id
        
        # Increase confidence slowly
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.01 * reward)
        
        self.save()
    
    def detect_drift(self, current_vector: np.ndarray, threshold: float = 0.3) -> float:
        """
        Returns the angular distance between current state vector and stored identity.
        Handles vectors of different lengths via truncation.
        """
        if np.linalg.norm(self.identity) == 0:
            return 0.0
        min_len = min(len(current_vector), len(self.identity))
        vec = current_vector[:min_len]
        ident = self.identity[:min_len]
        sim = np.dot(vec, ident) / (np.linalg.norm(vec) * np.linalg.norm(ident) + 1e-12)
        return 1.0 - sim
    
    def get_stability_bonus(self, current_vector: np.ndarray) -> float:
        """
        Returns a bonus factor (0-1) based on how aligned the current state
        is with the identity.
        """
        drift = self.detect_drift(current_vector)
        bonus = 1.0 - drift
        return max(0, min(1, bonus))
# ------------------------------------------------------------
# Episodic Memory – stores complete interaction episodes
# ------------------------------------------------------------
class EpisodicMemory:
    """
    Stores (state, action, reward, next_state) for reinforcement learning
    and offline consolidation. Episodes are saved to ./episodes/.

    Now config‑aware: reads capacity from config.
    """

    EPISODE_DIR = os.path.join(STATE_DIR, "episodes")

    def __init__(self, capacity: int = None, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        if config_manager:
            self.capacity = config_manager.get('episodic_memory.capacity', 10000)
        else:
            self.capacity = capacity if capacity is not None else 10000

        try:
            os.makedirs(self.EPISODE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create EPISODE_DIR: {e}")
        self.episodes = []          # list of dicts
        self._load_episodes()

    def update_parameters_from_config(self):
        """Reload capacity from config (hot reload)."""
        if self.config_manager:
            new_capacity = self.config_manager.get('episodic_memory.capacity', 10000)
            self.capacity = new_capacity
            # Trim if needed
            while len(self.episodes) > self.capacity:
                removed = self.episodes.pop(0)
                # Optionally delete from disk

    def _load_episodes(self):
        """Load all episode files from disk."""
        for fname in sorted(os.listdir(self.EPISODE_DIR)):
            if fname.endswith(".json"):
                path = os.path.join(self.EPISODE_DIR, fname)
                try:
                    with open(path, 'r') as f:
                        ep = json.load(f)
                    self.episodes.append(ep)
                except Exception as e:
                    print(f"Could not load episode {fname}: {e}")
        # keep only last `capacity` episodes
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def add_episode(self, user_input: str, system_response: str,
                    factor_vector: Dict, intrinsic_reward: float,
                    metadata: dict = None):

        if not isinstance(factor_vector, dict):
            factor_vector = {}

        episode = {
            "timestamp": time.time(),
            "user_input": user_input,
            "system_response": system_response,
            "factor_vector": dict(factor_vector),
            "factor_dim": len(factor_vector),
            "intrinsic_reward": float(intrinsic_reward),
            "metadata": metadata or {},
        }
        self.episodes.append(episode)
        # save to disk
        fname = f"episode_{int(episode['timestamp'])}_{np.random.randint(100000)}.json"
        path = os.path.join(self.EPISODE_DIR, fname)
        with open(path, 'w') as f:
            json.dump(episode, f, indent=2)
        # trim
        if len(self.episodes) > self.capacity:
            removed = self.episodes.pop(0)
            # also remove from disk (optional)

    def sample_batch(self, n: int = 32) -> List[Dict]:
        """Random sample of episodes for offline learning."""
        indices = np.random.choice(len(self.episodes), size=min(n, len(self.episodes)), replace=False)
        return [self.episodes[i] for i in indices]

    def replay_sequence(self) -> List[Dict]:
        """Return episodes in chronological order (for offline consolidation)."""
        return self.episodes[-100:]  # last 100
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
# ------------------------------------------------------------
# Factor Engine – computes the base control signals
# ------------------------------------------------------------
class FactorEngine:
    """
    Derives the base control signals from the current system state.
    All outputs are in [0,1].

    Now uses AdaptiveControlSpace to allow dynamic expansion.
    """

    def __init__(self, control_space: AdaptiveControlSpace,
                 config_manager: Optional['ConfigManager'] = None):
        self.control_space = control_space
        self.config_manager = config_manager

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def compute(self,
                token_analyses: List[Dict],
                ci_report: Dict,
                phon_score: float,
                diag: Dict,
                concept_model,
                meta_learner,
                reward_total: float,
                semantic_learner) -> Dict[str, float]:
        """
        Returns a dict with all control signals (base + dynamic).
        """
        result = {}

        # 1. curiosity – max curiosity score among tokens
        result['curiosity'] = max([ta.get("curiosity_score", 0.0) for ta in token_analyses], default=0.0)

        # 2. threat – based on negative diagnostics, low confidence, low phon_score
        neg_count = len(diag.get("negatives", []))
        result['threat'] = np.clip(0.3 * (neg_count / 3.0) +
                                   0.4 * (1.0 - ci_report.get("confidence", 0.5)) +
                                   0.3 * (1.0 - phon_score), 0, 1)

        # 3. trust – from CI confidence and stability (computed after comprehension)
        result['trust'] = 0.0

        # 4. confidence – CI confidence directly
        result['confidence'] = ci_report.get("confidence", 0.5)

        # 5. uncertainty – average entropy of all known words (global)
        if concept_model.word_concept_counts:
            uncertainties = [concept_model.entropy_word(w) for w in concept_model.word_concept_counts]
            result['uncertainty'] = np.clip(np.mean(uncertainties) / 5.0, 0, 1)
        else:
            result['uncertainty'] = 0.5

        # 6. energy – compute budget proxy (inverse of recent CPU usage)
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            result['energy'] = 1.0 - np.clip(cpu_percent, 0, 1)
        except:
            result['energy'] = 0.8

        # 7. attention – based on input length and number of tokens
        token_count = len(token_analyses)
        result['attention'] = np.clip(token_count / 20.0, 0, 1)

        # ---- Language factors ----
        # comprehension: average semantic similarity among words in input
        if token_analyses and semantic_learner:
            sims = []
            for ta in token_analyses:
                w = ta['token']
                others = [t['token'] for t in token_analyses if t['token'] != w]
                if others:
                    oth_sims = [semantic_learner.similarity(w, o) for o in others
                                if o in semantic_learner.word_embeddings]
                    if oth_sims:
                        sims.append(np.mean(oth_sims))
            result['comprehension'] = np.mean(sims) if sims else 0.5
        else:
            result['comprehension'] = 0.5

        # fluency: placeholder
        result['fluency'] = 0.7

        # cognitive load: inverse of energy * attention
        result['cognitive_load'] = np.clip(1.0 - (result['energy'] * result['attention']), 0, 1)

        # ---- Recompute trust using comprehension ----
        result['trust'] = (
            0.4 * result['confidence'] +
            0.3 * ci_report.get("stability", 0.5) +
            0.3 * result['comprehension']
        )

        # ---- Compute dynamic factors (if any) ----
        state = {
            'token_analyses': token_analyses,
            'ci_report': ci_report,
            'phon_score': phon_score,
            'diag': diag,
            'concept_model': concept_model,
            'meta_learner': meta_learner,
            'reward_total': reward_total,
            'semantic_learner': semantic_learner,
            **result
        }
        for factor_name in self.control_space.dynamic_factors:
            result[factor_name] = self.control_space.compute_factor(factor_name, state)
            self.control_space.record_value(factor_name, result[factor_name])
        # ensure numeric stability
        for k, v in result.items():
            if not isinstance(v, (int, float)) or np.isnan(v) or np.isinf(v):
                result[k] = 0.0
        return result
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
# ------------------------------------------------------------
# Core Curiosity Engine (Enhanced with Factor Engine & State Learning)
# ------------------------------------------------------------
class CuriosityEngine:
    """ss
    Integrates all components: HDP, GP uncertainty, Bayesian surprise,
    intrinsic reward, meta-learner, reasoning, AND now factor-based
    state snapshots with interactive human labeling.

    Now config‑aware and self‑reconfiguring via emotional modulation.
    """
    def __init__(self, load_state: bool = True, config_manager: Optional[ConfigManager] = None):
        # Configuration
        if config_manager is None:
            self.config_manager = ConfigManager()  # loads from default path
        else:
            self.config_manager = config_manager

        # Real engines
        self.ci_engine = CIEngine()
        self.phon_engine = PhonotacticEngine()
        self.diag_engine = DiagnosticEngine()

        # Probabilistic concept model
        self.concept_model = HierarchicalConceptModel(config_manager=self.config_manager)

        # Gaussian Process uncertainty model
        self.gp_model = GaussianProcessUncertainty(config_manager=self.config_manager)

        # Curiosity meta-learner (9 factors)
        self.meta_learner = MetaCuriosityLearner(config_manager=self.config_manager)

        # Intrinsic reward tracker
        self.reward_tracker = IntrinsicRewardTracker(config_manager=self.config_manager)

        # Reasoning engine
        self.reasoner = ReasoningEngine(config_manager=self.config_manager)
        self._last_state_learning_time = 0
        # ---- Language & consciousness modules ----
        self.semantic_learner = SemanticLearner(config_manager=self.config_manager)
        self.syntax_learner = SyntaxLearner(config_manager=self.config_manager)
        self.action_generator = ActionGenerator(
            self.semantic_learner, self.syntax_learner, self.reasoner, self, config_manager=self.config_manager
        )
        self.consciousness = ParameterRevision(config_manager=self.config_manager)
        self.episodic_memory = EpisodicMemory(config_manager=self.config_manager)
        # ---- Parallel reasoning ----
        self.parallel_reasoner = ParallelReasoner(self.reasoner, self.meta_learner)
        # ------------------------------------------
        self._last_feedback_ask_time = 0
        # ---- Adaptive Control Space ----
        self.control_space = AdaptiveControlSpace(config_manager=self.config_manager)
        # Register listener for new control factors → become curiosity factors
        self.control_space.add_listener(self._on_new_control_factor)
        # ---------------------------------

        # ---- Factor engine & State manager ----
        self.factor_engine = FactorEngine(control_space=self.control_space,
                                          config_manager=self.config_manager)
        state_dir = self.config_manager.get('state_manager.state_dir', 'states')
        self.state_manager = StateManager(state_dir=state_dir, config_manager=self.config_manager)
        # ---------------------------------------
        
        # ---- Identity persistence module ----
        emb_dim = self.config_manager.get('semantic_learner.emb_dim', 64)
        self.identity_manager = IdentityManager(emb_dim=emb_dim)
        # ---------------------------------------

        # ---- Persistent latent state concept ----
        self.current_state_labels = []          # label of the currently recognized state
        self.current_state_embedding = None      # embedding of that state (aligned to emb_dim)
        # ---------------------------------------

        # Memory of past observations for each word
        self.word_history: Dict[str, List[Dict]] = defaultdict(list)

        # Interaction history
        self.interaction_history: List[str] = []
        self.last_interaction_time = time.time()
        self.last_asked_word = None      # for definition learning

        # Self‑reconfiguration counter
        self.interaction_counter = 0
        self.reconfigure_interval = self.config_manager.get('engine.reconfigure_interval', 20)

        # State persistence

        # --- Trust-based language feedback tracking ---
        self.expecting_feedback = False
        self.feedback_attempt_word = None
        self.feedback_attempt_response = None
        self.feedback_timestamp = None
        # --- Behavioral rule confirmation tracking ---
        self.expecting_rule_confirmation = False
        self.pending_rule_id = None
        # ---- Response feedback request tracking ----
        self.expecting_response_feedback = False
        self.pending_feedback_context = None   # store the FeedbackEntry being rated
        self.feedback_ask_probability = 0.3    # probability to ask for feedback (if not confident)
        # ----------------------------------------------
        # --- Last factor dict (for dynamic curiosity factors) ---
        self._last_factor_dict = None
        # ---- Add dynamic curiosity factor for state relevance ----
        if "state_relevance" not in self.meta_learner.factor_names:
            self.meta_learner.add_factor(
                "state_relevance",
                initial_mu=0.0,
                compute_function=self._compute_state_relevance
            )
        self._expected_next_state = None
        self._last_action = None        
        # ------------------------------------------------------------
        # New attributes for advanced idle‑time capabilities
        # ------------------------------------------------------------
        # World model for dreaming (simple bigram model)
        self.world_model = None          # will be built on first use
        # Rehearsal buffer for prioritized replay
        self.rehearsal_buffer = []      # list of episode indices / scores
        # Distilled exploration policy (linear classifier weights)
        self.policy_model = None        # (coeffs, intercept) or None
        # User model for social cognition (naive bayes parameters)
        self.user_model = {"pos_mean": None, "neg_mean": None, "pos_count": 0, "neg_count": 0}
        # Long‑term research agenda (priority queue of queries)
        self.research_queue = []        # list of dicts with 'query', 'ig', 'target'
        # History of identity vectors for stabilization
        self.identity_history = []      # list of np.ndarray
        # Episodic embeddings for cross‑modal alignment
        self.episodic_embeddings = {}  # timestamp -> {'state': np.array, 'text_emb': np.array}
        # State manifold centroids (macro‑states)
        self.state_manifold = None     # list of centroid vectors
        # Flag to avoid running all idle tasks at once
        # ---- Structural entropy gate: token behavioral profiles ----
        self.token_profile = {}  # token -> dict with count, context_set, avg_surprise, avg_info_gain, embedding_drift, last_embedding
        self.token_profile_stats = {
            'count_mean': 0.0,
            'count_std': 1.0,
            'ctx_mean': 0.0,
            'ctx_std': 1.0,
            'surp_mean': 0.0,
            'surp_std': 1.0,
            'drift_mean': 0.0,
            'drift_std': 1.0,
        }
        self._cold_start_threshold = 100  # number of unique tokens before cold start bias ends
        self._idle_tasks_last_run = 0
        self.pending_sentence_query = None   # used by ActionGenerator
        self.pending_solution = None   # for equation solving results
        # ---- Logic doubt feedback ----
        self.expecting_logic_feedback = False
        self.pending_logic_doubt = None

        # ---- Procedure learning ----
        self.expecting_procedure_steps = False
        self.pending_problem_expr = None
        self.recorded_steps = []
        self.pending_teaching_request = False
        self.pending_teaching_expr = None

        # ========== CHAT MODE ==========
        self.chat_mode_active = False
        self.chat_mode_trigger = "switch chat"   # exact phrase to enter/exit
        self.chat_history = []                  # recent turns for context
        # ================================
        # ========== BEHAVIORAL RULE LEARNER ==========
        self.behavioral_learner = BehavioralRuleLearner(config_manager=self.config_manager)
        # ---- Conscience & Logic Inheritance ----
        self.conscience_manager = ConscienceManager(config_manager=self.config_manager)

        # ---- Code generation & guilt ----
        self.code_generator = CodeGenerator()
        self.shadow_executor = ShadowExecutor()
        # Only create a new GuiltTracker if one wasn't loaded from disk
        if not hasattr(self, 'guilt_tracker') or self.guilt_tracker is None:
            self.guilt_tracker = GuiltTracker()
        self.generated_code_feedback = GeneratedCodeFeedback()
        # Ensure conscience_manager also uses the same guilt_tracker
        if self.conscience_manager:
            self.conscience_manager.guilt_tracker = self.guilt_tracker
        # Add conscience_active to base control factors
        if "conscience_active" not in self.control_space.base_factors:
            self.control_space.base_factors.append("conscience_active")
        self.identity_aliases = set()   # store learned aliases for self-reference
        self.logic_inheritance = LogicInheritanceManager(
            config_manager=self.config_manager,
            conscience_manager=self.conscience_manager
        )
        # ---- Language model for fluent chat ----
        self.language_model = LSTMLanguageModel.create_default(self.config_manager)
        self.lm_config = {
            'enabled': self.config_manager.get('language_model.enabled', True),
            'min_steps_for_use': self.config_manager.get('language_model.min_steps_for_use', 100),
            'train_batch_size': self.config_manager.get('language_model.train_batch_size', 20),
            'improvement_interval': self.config_manager.get('language_model.improvement_interval', 5000),
        }
        # --- Cognition watchdog (RAM-adaptive) ---
        self._cycle_counter = 0
        self._update_dynamic_limits()

        try:
            mem = psutil.virtual_memory()
            scale = max(0.5, min(8.0, (mem.available/(1024**3))/4.0))
            self._max_interaction_history = int(200000 * scale)
        except Exception:
            self._max_interaction_history = 200000
        self._last_save_time = 0
        self._min_save_interval = 20

        if load_state:
            self._load_state()
        self.structured_world_model = StructuredWorldModel(self.reasoner.knowledge_graph, self.state_manager)
    # ------------------------------------------------------------------
    # Persistence (now with config_manager passed to .load() methods)
    # ------------------------------------------------------------------
    def _update_dynamic_limits(self):
        """Adapt internal limits to available system RAM."""
        try:
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024**3)

            # scale factors
            scale = max(0.5, min(8.0, avail_gb / 4.0))  # normalize around ~4GB baseline

            self._max_cycles_per_session = int(100000 * scale)
            self._max_hypotheses = int(5000 * scale)
            self._max_logic_rules = int(2000 * scale)

        except Exception:
            # fallback defaults
            self._max_cycles_per_session = 100000
            self._max_hypotheses = 5000
            self._max_logic_rules = 2000

    def _metabolic_maintenance(self):
        """
        Apply metabolic cost and phase transitions to all persistent structures.
        Called during idle introspection.
        """
        now = time.time()

        # --- 1. Behavioral rules ---
        for rule in self.behavioral_learner.rules.values():
            self._apply_metabolic_to_item(rule, now)

        # --- 2. Conscience beliefs ---
        for belief in self.conscience_manager.beliefs.values():
            self._apply_metabolic_to_item(belief, now)

        # --- 3. Hypotheses in reasoner ---
        for hyp in self.reasoner.hypotheses:
            self._apply_metabolic_to_dict_item(hyp, now)

        # --- 4. Dynamic curiosity factors ---
        for factor_name, meta in self.meta_learner.factor_metadata.items():
            self._apply_metabolic_to_dict_item(meta, now, key_prefix=("factor", factor_name))

        # --- 5. States ---
        for label, meta in self.state_manager.state_metadata.items():
            self._apply_metabolic_to_dict_item(meta, now, key_prefix=("state", label))

        # --- 6. Logic rules ---
        for logic_id, rule in self.logic_inheritance.confirmed_logic.items():
            self._apply_metabolic_to_dict_item(rule, now, key_prefix=("logic", logic_id))
        for doubt in self.logic_inheritance.pending_doubts:
            rule = doubt['rule']
            self._apply_metabolic_to_dict_item(rule, now, key_prefix=("doubt", doubt['doubt_id']))

    def _apply_metabolic_to_item(self, item, now):
        """Apply metabolic rules to an object with attributes."""
        if item.phase == 'deleted':
            return
        dt = now - (item.last_used if item.last_used else item.created)
        # decay recent_usage exponentially
        decay_factor = 0.99 ** dt
        item.recent_usage *= decay_factor

        # compute effective predictive_gain (could be based on recent_usage and confidence)
        effective_gain = item.predictive_gain * item.recent_usage / (1 + dt)

        # maintenance cost (could be type-dependent)
        cost = item.energy_cost

        if effective_gain < cost:
            # move to next phase
            if item.phase == 'active':
                item.phase = 'weak'
                print(f"   [Metabolic] {item} moved to weak")
            elif item.phase == 'weak':
                item.phase = 'dormant'
                print(f"   [Metabolic] {item} moved to dormant")
            elif item.phase == 'dormant':
                item.phase = 'archived'
                print(f"   [Metabolic] {item} archived")
            elif item.phase == 'archived':
                item.phase = 'deleted'
                print(f"   [Metabolic] {item} deleted")
                # actually remove from its container? We'll handle separately.

    def _apply_metabolic_to_dict_item(self, d, now, key_prefix=None):
        """Apply metabolic rules to a dictionary item."""
        if d.get('phase') == 'deleted':
            return
        last_used = d.get('last_used')
        created = d.get('created', now - 86400)  # fallback
        dt = now - (last_used if last_used else created)
        recent_usage = d.get('recent_usage', 0.0)
        recent_usage *= (0.99 ** dt)
        d['recent_usage'] = recent_usage

        predictive_gain = d.get('predictive_gain', 0.0)
        cost = d.get('energy_cost', 0.5)
        effective_gain = predictive_gain * recent_usage / (1 + dt)

        if effective_gain < cost:
            phase = d.get('phase', 'active')
            if phase == 'active':
                d['phase'] = 'weak'
            elif phase == 'weak':
                d['phase'] = 'dormant'
            elif phase == 'dormant':
                d['phase'] = 'archived'
            elif phase == 'archived':
                d['phase'] = 'deleted'
                # Optionally remove from container – we'll leave that for later cleanup.

    def _align_factor_vector(self, vec: np.ndarray) -> np.ndarray:
        return align_vector(vec, len(self.meta_learner.factor_names))

    def _memory_pressure_backoff(self):
        """Slow cognition when memory is low."""
        try:
            mem = psutil.virtual_memory()
            if mem.available / mem.total < 0.12:   # <12% free
                time.sleep(0.15)
                return True
        except Exception:
            pass
        return False

    def _load_state(self):
        """Load persistent state from disk, including the latent state concept."""
        concept_path = os.path.join(STATE_DIR, 'concept_model.pkl')
        meta_path = os.path.join(STATE_DIR, 'meta_learner.npz')
        gp_path = os.path.join(STATE_DIR, 'gp_model.pkl')
        reward_path = os.path.join(STATE_DIR, 'reward_tracker.pkl')
        reasoner_path = os.path.join(STATE_DIR, 'reasoner.pkl')
        history_path = os.path.join(STATE_DIR, 'history.pkl')
        semantic_path = os.path.join(STATE_DIR, 'semantic_learner.pkl')
        syntax_path = os.path.join(STATE_DIR, 'syntax_learner.pkl')
        # ---- Load Conscience & Logic ----
        conscience_path = os.path.join(STATE_DIR, 'conscience_manager.pkl')
        if os.path.exists(conscience_path):
            self.conscience_manager._load()
        self.logic_inheritance._load()
        if os.path.exists(concept_path):
            self.concept_model = HierarchicalConceptModel.load(concept_path, config_manager=self.config_manager)
        if os.path.exists(meta_path):
            self.meta_learner.load(meta_path)
        if os.path.exists(gp_path):
            with open(gp_path, 'rb') as f:
                self.gp_model = pickle.load(f)
            self.gp_model.config_manager = self.config_manager
        if os.path.exists(reward_path):
            self.reward_tracker.load(reward_path)
            self.reward_tracker.config_manager = self.config_manager
        if os.path.exists(reasoner_path):
            self.reasoner.load(reasoner_path)
            self.reasoner.config_manager = self.config_manager
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.word_history = pickle.load(f)
        if os.path.exists(semantic_path):
            self.semantic_learner.load(semantic_path)
            self.semantic_learner.config_manager = self.config_manager
        if os.path.exists(syntax_path):
            self.syntax_learner.load(syntax_path)
            self.syntax_learner.config_manager = self.config_manager
        # Load new idle‑task related data
        world_model_path = os.path.join(STATE_DIR, 'world_model.pkl')
        if os.path.exists(world_model_path):
            with open(world_model_path, 'rb') as f:
                self.world_model = pickle.load(f)
        policy_path = os.path.join(STATE_DIR, 'policy_model.pkl')
        if os.path.exists(policy_path):
            with open(policy_path, 'rb') as f:
                self.policy_model = pickle.load(f)
        user_model_path = os.path.join(STATE_DIR, 'user_model.pkl')
        if os.path.exists(user_model_path):
            with open(user_model_path, 'rb') as f:
                self.user_model = pickle.load(f)
        research_queue_path = os.path.join(STATE_DIR, 'research_queue.pkl')
        if os.path.exists(research_queue_path):
            with open(research_queue_path, 'rb') as f:
                self.research_queue = pickle.load(f)
        identity_history_path = os.path.join(STATE_DIR, 'identity_history.pkl')
        if os.path.exists(identity_history_path):
            with open(identity_history_path, 'rb') as f:
                self.identity_history = pickle.load(f)
        state_manifold_path = os.path.join(STATE_DIR, 'state_manifold.pkl')
        if os.path.exists(state_manifold_path):
            with open(state_manifold_path, 'rb') as f:
                self.state_manifold = pickle.load(f)
        # ---- Load identity aliases ----
        aliases_path = os.path.join(STATE_DIR, 'identity_aliases.json')
        if os.path.exists(aliases_path):
            try:
                with open(aliases_path, 'r') as f:
                    self.identity_aliases = set(json.load(f))
            except Exception as e:
                print(f"Could not load identity aliases: {e}")
        guilt_path = os.path.join(STATE_DIR, 'guilt_tracker.pkl')
        if os.path.exists(guilt_path):
            self.guilt_tracker.load(guilt_path)
        # Ensure conscience_manager shares the same guilt_tracker instance
        if self.conscience_manager:
            self.conscience_manager.guilt_tracker = self.guilt_tracker
        # Load generated code feedback
        self.generated_code_feedback._load()
        # ---- Load language model ----
        lm_path = os.path.join(STATE_DIR, 'language_model.pkl')
        if os.path.exists(lm_path):
            try:
                self.language_model.load(lm_path)
                print(f"📖 Loaded language model with {self.language_model.vocab_size} words, {self.language_model.steps} steps.")
            except Exception as e:
                print(f"Could not load language model: {e}")
        else:
            # Build initial vocabulary from existing word embeddings
            for word in self.semantic_learner.word_embeddings:
                self.language_model.add_word_to_vocab(word)
        # ---- Load current latent state concept (multi‑state) ----
        current_state_path = os.path.join(STATE_DIR, 'current_state.json')
        if os.path.exists(current_state_path):
            try:
                with open(current_state_path, 'r') as f:
                    data = json.load(f)
                labels = data.get('labels', [])
                if labels:
                    self.current_state_labels = labels
                else:
                    # backward compatibility with old single-label format
                    single = data.get('label')
                    if single:
                        self.current_state_labels = [single]
                emb = data.get('embedding')
                if emb is not None:
                    self.current_state_embedding = np.array(emb)
            except Exception as e:
                print(f"Could not load current state: {e}")
                self.current_state_labels = []
                self.current_state_embedding = None
        else:
            self.current_state_labels = []
            self.current_state_embedding = None
        token_profile_path = os.path.join(STATE_DIR, 'token_profile.pkl')
        if os.path.exists(token_profile_path):
            try:
                with open(token_profile_path, 'rb') as f:
                    data = pickle.load(f)
                    self.token_profile = data.get('token_profile', {})
                    loaded_stats = data.get('token_profile_stats', {})
                    # Ensure new stats exist (backward compatibility)
                    self.token_profile_stats = {
                        'count_mean': loaded_stats.get('count_mean', 0.0),
                        'count_std': loaded_stats.get('count_std', 1.0),
                        'ctx_mean': loaded_stats.get('ctx_mean', 0.0),
                        'ctx_std': loaded_stats.get('ctx_std', 1.0),
                        'surp_mean': loaded_stats.get('surp_mean', 0.0),
                        'surp_std': loaded_stats.get('surp_std', 1.0),
                        'drift_mean': loaded_stats.get('drift_mean', 0.0),
                        'drift_std': loaded_stats.get('drift_std', 1.0),
                    }
                    self._cold_start_threshold = data.get('_cold_start_threshold', 100)
            except Exception as e:
                print(f"Could not load token profiles: {e}")
                self.token_profile = {}
                self.token_profile_stats = {
                    'count_mean': 0.0, 'count_std': 1.0,
                    'ctx_mean': 0.0, 'ctx_std': 1.0,
                    'surp_mean': 0.0, 'surp_std': 1.0,
                    'drift_mean': 0.0, 'drift_std': 1.0,
                }
                self._cold_start_threshold = 100

    def _save_state(self):
        """Persist current state, including the persistent latent state concept."""
        now = time.time()
        if now - getattr(self, "_last_save_time", 0) < getattr(self, "_min_save_interval", 20):
            return
        self._last_save_time = now

        try:
            self.concept_model.save(os.path.join(STATE_DIR, 'concept_model.pkl'))
            self.meta_learner.save(os.path.join(STATE_DIR, 'meta_learner.npz'))
            with open(os.path.join(STATE_DIR, 'gp_model.pkl'), 'wb') as f:
                pickle.dump(self.gp_model, f)
            self.reward_tracker.save(os.path.join(STATE_DIR, 'reward_tracker.pkl'))
            self.reasoner.save(os.path.join(STATE_DIR, 'reasoner.pkl'))
            with open(os.path.join(STATE_DIR, 'history.pkl'), 'wb') as f:
                pickle.dump(self.word_history, f)
            self.semantic_learner.save(os.path.join(STATE_DIR, 'semantic_learner.pkl'))
            self.syntax_learner.save(os.path.join(STATE_DIR, 'syntax_learner.pkl'))
            self.identity_manager.save()
            # ---- Save behavioral rule learner ----
            self.behavioral_learner.save()
            # ---- Save identity aliases ----
            aliases_path = os.path.join(STATE_DIR, 'identity_aliases.json')
            with open(aliases_path, 'w') as f:
                json.dump(list(self.identity_aliases), f)
            # Save new idle‑task related data
            with open(os.path.join(STATE_DIR, 'world_model.pkl'), 'wb') as f:
                pickle.dump(self.world_model, f)
            with open(os.path.join(STATE_DIR, 'policy_model.pkl'), 'wb') as f:
                pickle.dump(self.policy_model, f)
            with open(os.path.join(STATE_DIR, 'user_model.pkl'), 'wb') as f:
                pickle.dump(self.user_model, f)
            with open(os.path.join(STATE_DIR, 'research_queue.pkl'), 'wb') as f:
                pickle.dump(self.research_queue, f)
            with open(os.path.join(STATE_DIR, 'identity_history.pkl'), 'wb') as f:
                pickle.dump(self.identity_history, f)
            with open(os.path.join(STATE_DIR, 'state_manifold.pkl'), 'wb') as f:
                pickle.dump(self.state_manifold, f)
            self.conscience_manager.save()
            self.logic_inheritance.save()
            self.guilt_tracker.save(os.path.join(STATE_DIR, 'guilt_tracker.pkl'))
            self.generated_code_feedback.save()
            # ---- Save language model ----
            self.language_model.save(os.path.join(STATE_DIR, 'language_model.pkl'))
            # ---- Save current latent state concept (multi‑state) ----
            current_state_path = os.path.join(STATE_DIR, 'current_state.json')
            # ---- Save token profiles ----
            token_profile_path = os.path.join(STATE_DIR, 'token_profile.pkl')
            with open(token_profile_path, 'wb') as f:
                pickle.dump({
                    'token_profile': self.token_profile,
                    'token_profile_stats': self.token_profile_stats,
                    '_cold_start_threshold': self._cold_start_threshold,
                }, f)
            data = {
                'labels': self.current_state_labels,
                'embedding': self.current_state_embedding.tolist() if self.current_state_embedding is not None else None
            }
            with open(current_state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}. State may be incomplete.")
            print(traceback.format_exc())
    # ------------------------------------------------------------------
    # Hot‑reload: refresh all components from config file
    # ------------------------------------------------------------------
    def reload_config(self):
        """Reload the configuration file and update all component parameters."""
        self.config_manager.reload()
        # Update each component
        self.concept_model.update_parameters_from_config()
        self.gp_model.update_parameters_from_config()
        self.meta_learner.update_parameters_from_config()
        self.reward_tracker.update_parameters_from_config()
        self.semantic_learner.update_parameters_from_config()
        self.syntax_learner.update_parameters_from_config()
        self.consciousness.update_parameters_from_config()
        self.action_generator.update_parameters_from_config()
        self.reasoner.update_parameters_from_config()
        self.factor_engine.update_parameters_from_config()
        self.state_manager.update_parameters_from_config()
        self.episodic_memory.update_parameters_from_config()
        if hasattr(self, '_idle_reasoner'):
            self._idle_reasoner.update_parameters_from_config()
        # Update engine parameters
        self.reconfigure_interval = self.config_manager.get('engine.reconfigure_interval', 20)
    # ------------------------------------------------------------------
    # Emotional self‑reconfiguration
    # ------------------------------------------------------------------
    def _maybe_reconfigure(self):
        """Generate a new config based on the current factor vector and reload it."""
        # Obtain the latest factor vector (we don't store it, so recompute from last token analyses?)
        # Since this is called at the end of process_input, we have factor_dict.
        # We'll access it from the local variable in process_input – we need to pass it.
        # To keep it simple, we recompute factor vector from the latest state.
        # In practice, we store the last factor_vector in self.last_factor_vector.
        if hasattr(self, '_last_factor_vector'):
            factor_vector = self._last_factor_vector
            self.config_manager.generate_modulated_config(factor_vector)
            self.reload_config()
            # Allow conscience to adjust its own thresholds
            self.conscience_manager.evolve_parameters()
            print("🧬 [Emotional Reconfiguration] Configuration updated based on internal state.")
    # ------------------------------------------------------------------
    # Feature extraction helpers (unchanged)
    # ------------------------------------------------------------------
    def _extract_features(self, token: str, ci_report: Dict, phon_score: float, diag: Dict) -> Dict[str, float]:
        """Convert token analysis into a feature vector for the concept model."""
        features = {
            'confidence': ci_report.get('confidence', 0.5),
            'stability': ci_report.get('stability', 0.5),
            'phon_score': phon_score,
            'char_valid': ci_report.get('mean', 0.5),
            'structure': phon_score,
        }
        for neg in diag.get('negatives', []):
            if 'character' in neg.lower():
                features['char_valid'] = 0.1
            if 'vowel' in neg.lower():
                features['structure'] = 0.2
        return features

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to fixed-length array for GP."""
        keys = ['confidence', 'stability', 'phon_score', 'char_valid', 'structure']
        return np.array([features.get(k, 0.5) for k in keys])

    def _apply_knowledge_graph_curiosity(self, token: str, base_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Modify curiosity factors based on knowledge graph structure.
        - If token has many connections (high degree), boost novelty and uncertainty.
        - If token is a bridge between clusters, boost curiosity further.
        """
        # Get node ID for token
        nid = self.reasoner.knowledge_graph.get_node_id(token, "word")
        if nid is None:
            return base_factors

        # Compute degree
        neighbors = self.reasoner.knowledge_graph.get_neighbors(nid)
        degree = len(neighbors)
        # Normalize degree (max expected degree maybe 100)
        degree_factor = min(1.0, degree / 50.0)

        # Boost novelty and uncertainty based on degree
        base_factors['novelty'] = min(1.0, base_factors.get('novelty', 0.5) + 0.2 * degree_factor)
        base_factors['uncertainty'] = min(1.0, base_factors.get('uncertainty', 0.5) + 0.15 * degree_factor)
        # Boost surprise a little
        base_factors['surprise'] = min(1.0, base_factors.get('surprise', 0.5) + 0.1 * degree_factor)

        # Optionally, if token appears in many different relation types, boost more
        rel_types = set()
        for _, rt, _ in neighbors:
            rel_types.add(rt)
        rel_diversity = len(rel_types) / 5.0  # max 5 types
        base_factors['complexity'] = min(1.0, base_factors.get('complexity', 0.5) + 0.1 * rel_diversity)

        return base_factors

    def _update_token_profile(self, token: str, context_tokens: List[str], bayes_surprise: float, info_gain: float, embedding: np.ndarray):
        """Update token behavioral statistics for structural scoring."""
        if token not in self.token_profile:
            self.token_profile[token] = {
                'count': 0,
                'context_set': set(),
                'avg_surprise': 0.0,
                'avg_info_gain': 0.0,
                'embedding_drift': 0.0,
                'embedding_centroid': None,
            }
        prof = self.token_profile[token]

        # Update count
        prof['count'] += 1

        # Update context set with cap (prevent explosion)
        for ct in context_tokens:
            if len(prof['context_set']) < 200:
                prof['context_set'].add(ct)

        # Update moving averages (exponential with alpha=0.1)
        alpha = 0.1
        prof['avg_surprise'] = (1 - alpha) * prof['avg_surprise'] + alpha * bayes_surprise
        prof['avg_info_gain'] = (1 - alpha) * prof['avg_info_gain'] + alpha * info_gain

        # Centroid-based embedding drift
        if embedding is not None:
            if prof['embedding_centroid'] is None:
                prof['embedding_centroid'] = embedding.copy()
            else:
                # Update centroid (exponential moving average)
                centroid = prof['embedding_centroid']
                centroid = (1 - alpha) * centroid + alpha * embedding
                prof['embedding_centroid'] = centroid

                # Compute drift as 1 - cosine similarity between current embedding and centroid
                norm_cent = np.linalg.norm(centroid)
                norm_emb = np.linalg.norm(embedding)
                if norm_cent > 0 and norm_emb > 0:
                    cos_sim = np.dot(centroid, embedding) / (norm_cent * norm_emb)
                    drift = 1 - cos_sim
                else:
                    drift = 0.0

                # Update moving average of drift
                prof['embedding_drift'] = (1 - alpha) * prof['embedding_drift'] + alpha * drift

        # Update running statistics (global)
        stats = self.token_profile_stats
        def _update_running_stat(old_mean, old_std, new_value):
            new_mean = (1 - alpha) * old_mean + alpha * new_value
            new_std = (1 - alpha) * old_std + alpha * abs(new_value - new_mean)
            return new_mean, new_std

        stats['count_mean'], stats['count_std'] = _update_running_stat(stats['count_mean'], stats['count_std'], prof['count'])
        stats['ctx_mean'], stats['ctx_std'] = _update_running_stat(stats['ctx_mean'], stats['ctx_std'], len(prof['context_set']))
        stats['surp_mean'], stats['surp_std'] = _update_running_stat(stats['surp_mean'], stats['surp_std'], prof['avg_surprise'])
        stats['drift_mean'], stats['drift_std'] = _update_running_stat(stats['drift_mean'], stats['drift_std'], prof['embedding_drift'])

    def _normalize_stat(self, value, mean, std):
        """Sigmoid normalization using running statistics."""
        z = (value - mean) / (std + 1e-8)
        return 1 / (1 + np.exp(-z))
    
    def structural_score(self, token: str) -> float:
        """Compute structural score for a token (high = glue-like)."""
        prof = self.token_profile.get(token)
        if not prof:
            return 0.5  # neutral for unseen

        stats = self.token_profile_stats

        freq = self._normalize_stat(prof['count'], stats['count_mean'], stats['count_std'])
        ctx = self._normalize_stat(len(prof['context_set']), stats['ctx_mean'], stats['ctx_std'])
        surp = 1 - self._normalize_stat(prof['avg_surprise'], stats['surp_mean'], stats['surp_std'])
        drift = 1 - self._normalize_stat(prof['embedding_drift'], stats['drift_mean'], stats['drift_std'])

        score = 0.35 * freq + 0.35 * ctx + 0.15 * surp + 0.15 * drift

        # Cold start bootstrap: if few unique tokens, boost score for short tokens
        unique_tokens = len(self.token_profile)
        if unique_tokens < self._cold_start_threshold and len(token) <= 2:
            score = min(1.0, score + 0.2)

        return float(np.clip(score, 0, 1))        

    def _compute_curiosity_factors(self, token: str, token_ci: Dict, token_phon: float,
                                   diag: Dict, features_array: np.ndarray,
                                   global_factor_dict: Optional[Dict] = None) -> CuriosityFactors:
        """Compute all curiosity factors for a single token and return a dynamic CuriosityFactors object."""
        try:
            uncertainty = self.concept_model.entropy_word(token)

            learning_progress = 0.0
            if token in self.word_history and len(self.word_history[token]) > 1:
                prev_entropy = self.word_history[token][-2].get('entropy', 0.0)
                curr_entropy = self.word_history[token][-1].get('entropy', uncertainty)
                learning_progress = max(0, prev_entropy - curr_entropy)

            novelty = 1.0 - token_ci.get('confidence', 0.5)

            stability = token_ci.get('stability', 0.5)
            complexity = 1.0 - stability + (min(len(token), 20) / 20.0) * 0.3
            complexity = np.clip(complexity, 0, 1)

            predicted = self.concept_model.predict_features(token)
            observed = self._extract_features(token, token_ci, token_phon, diag)
            surprise = 0.0
            all_keys = set(predicted.keys()) | set(observed.keys())
            for k in all_keys:
                p = predicted.get(k, 0.01)
                o = observed.get(k, 0.01)
                surprise += o * math.log(o / p) if o > 0 and p > 0 else 0
            surprise = np.clip(surprise, 0, 1)

            phon_anomaly = 1.0 - token_phon
            neg_count = len(diag.get('negatives', []))
            diag_neg = min(neg_count / 5.0, 1.0)

            bayes_surprise = self.concept_model.bayesian_surprise(token)
            gp_uncertainty = self.gp_model.uncertainty(features_array)

            factor_dict = {
                "uncertainty": uncertainty,
                "learning_progress": learning_progress,
                "novelty": novelty,
                "complexity": complexity,
                "surprise": surprise,
                "phonotactic_anomaly": phon_anomaly,
                "diagnostic_negatives": diag_neg,
                "bayesian_surprise": bayes_surprise,
                "gp_uncertainty": gp_uncertainty,
            }
            # Apply knowledge graph based curiosity boost
            factor_dict = self._apply_knowledge_graph_curiosity(token, factor_dict)
            # If word has a definition, reduce its curiosity significantly
            if self.reasoner and self.reasoner.definitions.get(token):
                factor_dict["novelty"] *= 0.1
                factor_dict["surprise"] *= 0.1
                factor_dict["bayesian_surprise"] *= 0.1
                factor_dict["gp_uncertainty"] *= 0.1
                factor_dict["uncertainty"] *= 0.1
                factor_dict["learning_progress"] *= 0.1

            for dyn_factor in self.meta_learner.dynamic_factor_names:
                val = self.meta_learner.compute_dynamic_factor(
                    dyn_factor, token, token_ci, token_phon, diag,
                    features_array, global_factor_dict
                )
                factor_dict[dyn_factor] = val

            # Create CuriosityFactors object
            factors = CuriosityFactors(factor_dict=factor_dict)

            # Apply top‑down modulation (expects a CuriosityFactors object)
            factors = self._apply_top_down_modulation(token, factors)

            return factors

        except Exception as e:
            print(f"Error computing curiosity factors for '{token}': {e}. Returning neutral factors.")
            print(traceback.format_exc())
            # Return neutral factors to avoid crash
            neutral = {name: 0.5 for name in self.meta_learner.factor_names}
            return CuriosityFactors(factor_dict=neutral)

    def _apply_top_down_modulation(self, token: str, factors: CuriosityFactors) -> CuriosityFactors:
        """
        Modulate curiosity factors using top‑down signals:
        - Current state embedding
        - Identity embedding
        - Conscience beliefs
        """
        if self.current_state_embedding is not None:
            # Compute similarity between token embedding and current state
            token_emb = self.semantic_learner.word_embeddings.get(token)
            if token_emb is not None:
                state_emb = self._align_vector_to_emb_dim(self.current_state_embedding)
                token_emb_aligned = self._align_vector_to_emb_dim(token_emb)
                sim = np.dot(state_emb, token_emb_aligned) / (np.linalg.norm(state_emb)*np.linalg.norm(token_emb_aligned)+1e-12)
                # Boost curiosity if token is state‑relevant (already handled by state_relevance factor)
                # Here we also modulate novelty based on identity consistency
                identity_sim = self.identity_manager.get_stability_bonus(token_emb_aligned)
                # If token aligns with identity, reduce novelty (it's familiar)
                factors.factor_dict['novelty'] *= (1.0 - 0.5 * identity_sim)

        # Modulate by conscience: if a belief forbids asking about certain topics, reduce curiosity
        for belief in self.conscience_manager.get_active_conscience():
            if belief.rule_type == 'ethical' and belief.action.get('effect') == 'reduce_curiosity':
                # Simple keyword match
                if belief.condition.get('keyword') in token.lower():
                    factors.factor_dict['curiosity'] *= 0.3
                    factors.factor_dict['novelty'] *= 0.5
        return factors
    
    def _information_gain(self, token: str, new_entropy: float) -> float:
        if token not in self.word_history or len(self.word_history[token]) < 2:
            return 0.0
        old_entropy = self.word_history[token][-2].get('entropy', 1.0)
        return max(0, old_entropy - new_entropy)
    # ------------------------------------------------------------------
    # State learning (unchanged logic, but uses configurable threshold)
    # ------------------------------------------------------------------
    def _handle_state_learning(self, factor_vector: np.ndarray, metadata: dict):
        """
        Compare current factor vector with known states. Update active states list.
        """
        if factor_vector is None or len(factor_vector) == 0:
            print("   [State Learning] Empty factor vector – skipping.")
            return

        if time.time() - self._last_state_learning_time < 30:
            return

        sim_threshold = self.config_manager.get('engine.state_similarity_threshold', 0.97)
        n_states = max(1, len(self.state_manager.states))
        maturity = np.clip(np.log1p(n_states) / 5.0, 0.2, 1.0)
        novelty_dist_threshold = 0.35 * (1.2 - maturity)

        # Find all states above similarity threshold
        similar = self.state_manager.find_similar_states(factor_vector, threshold=sim_threshold)

        # Apply meta novelty gate per candidate – keep only those within distance threshold
        active_labels = []
        active_embeddings = []
        for label, sim in similar:
            stored_vec = self.state_manager.states.get(label)
            if stored_vec is not None:
                v1 = self.state_manager._normalize_vector(factor_vector)
                v2 = self.state_manager._normalize_vector(stored_vec)
                if v1.shape != v2.shape:
                    max_len = max(len(v1), len(v2))
                    v1 = np.pad(v1, (0, max_len - len(v1)))
                    v2 = np.pad(v2, (0, max_len - len(v2)))
                dist = np.linalg.norm(v1 - v2)
                if dist <= novelty_dist_threshold:
                    active_labels.append(label)
                    active_embeddings.append(self._align_vector_to_emb_dim(v2))

        if active_labels:
            print(f"\n🔍 [State Recognition] This state matches: {', '.join(active_labels)}.")
            # Combine embeddings (average)
            combined = np.mean(active_embeddings, axis=0) if active_embeddings else None
            self._update_state_embedding(active_labels, combined)
            return

        # No close match – create new state
        vec_variance = np.var(factor_vector)
        if vec_variance < 0.05:
            print("   [State Learning] Factor vector too uniform – skipping state creation.")
            return

        print("\n🧠 [State Learning] I encountered a new internal state.")
        print("    Factor vector: " + " ".join(f"{v:.2f}" for v in factor_vector))
        print("    How do humans call this state? (e.g., 'confident exploration', 'cautious inquiry')")
        label = input("    🏷️  Label: ").strip()
        if not label:
            label = "unnamed_state"

        metadata['tokens'] = self._tokenize(metadata.get('input_text', ''))
        self.state_manager.save_state(label, factor_vector, metadata)
        print(f"    ✅ State saved as '{label}'.")
        self._last_state_learning_time = time.time()
        self._reinforce_identity(factor_vector, reward_signal=0.2)
        self._update_state_embedding([label], self._align_vector_to_emb_dim(factor_vector))

    def _detect_semantic_relations(self, tokens: List[str], token_analyses: List[Dict]):
        """
        After processing tokens, check for words that have definitions and look for
        similar definitions among other words. Generate hypotheses for synonymy.
        """
        # Only process words that are in token_analyses (i.e., were in the input)
        for ta in token_analyses:
            word = ta['token']
            # If this word has a stored definition, check for similar definitions
            if word in self.reasoner.definitions:
                hypothesis = self.reasoner.generate_synonym_hypothesis(word)
                if hypothesis:
                    # Hypothesis is already added to reasoner.hypotheses; we may print it or log it.
                    print(f"💡 [Semantic Hypothesis] {hypothesis}")
    # ------------------------------------------------------------------
    # Identity reinforcement
    # ------------------------------------------------------------------
    def _reinforce_identity(self, factor_vector: np.ndarray, reward_signal: float):
        """
        Reinforce the self-identity when a positive interaction occurs.
        Called after successful definition learning, hypothesis confirmation,
        or when user provides a label for a state.
        """
        self.identity_manager.reinforce(factor_vector, reward_signal)
        
        # Optionally log identity drift
        drift = self.identity_manager.detect_drift(factor_vector)
        if drift > 0.5:
            print(f"🧠 [Identity Drift] Current state diverges from learned identity (drift={drift:.3f}).")
    # ------------------------------------------------------------------
    # State embedding helpers (persistent latent concept)
    # ------------------------------------------------------------------
    def _align_vector_to_emb_dim(self, vector: np.ndarray) -> np.ndarray:
        """Truncate or pad a vector to match the semantic learner's embedding dimension."""
        target_dim = self.semantic_learner.emb_dim
        if len(vector) == target_dim:
            return vector.copy()
        elif len(vector) > target_dim:
            return vector[:target_dim].copy()
        else:
            # pad with zeros
            padded = np.zeros(target_dim)
            padded[:len(vector)] = vector
            return padded

    def _compute_state_relevance(self,
                                 token: str,
                                 token_ci: Dict,
                                 token_phon: float,
                                 diag: Dict,
                                 features_array: np.ndarray,
                                 global_factor_dict: Optional[Dict] = None) -> float:
        """
        Dynamic curiosity factor: similarity between the token's embedding
        and the current state embedding.
        Returns 0.0 if no state embedding or token embedding is available.
        """
        if self.current_state_embedding is None:
            return 0.0
        token_emb = self.semantic_learner.word_embeddings.get(token)
        if token_emb is None:
            return 0.0

        # align both embeddings to the same dimension (emb_dim)
        state_emb = self._align_vector_to_emb_dim(self.current_state_embedding)
        token_emb = self._align_vector_to_emb_dim(token_emb)

        norm_state = np.linalg.norm(state_emb)
        norm_token = np.linalg.norm(token_emb)
        if norm_state == 0 or norm_token == 0:
            return 0.0
        sim = np.dot(state_emb, token_emb) / (norm_state * norm_token)
        # scale from [-1,1] to [0,1]
        # update state usage
        if self.current_state_labels:
            current_label = self.current_state_labels[0]  # use first active state for metadata
            meta = self.state_manager.state_metadata.get(current_label)
            if meta:
                now = time.time()
                if meta.get('last_used'):
                    dt = now - meta['last_used']
                    meta['recent_usage'] *= 0.99 ** dt
                meta['recent_usage'] += 0.05
                meta['last_used'] = now
                self.state_manager._save_state_metadata()
        return (sim + 1.0) * 0.5

    def _update_state_embedding(self, labels: List[str], combined_vector: Optional[np.ndarray] = None):
        """
        Set the current active state labels and combined embedding.
        If combined_vector is None, compute it as the average of each label's embedding.
        Also store each label's embedding in the semantic learner.
        """
        if combined_vector is None:
            # compute average of embeddings for all labels
            embs = []
            for lbl in labels:
                if lbl in self.semantic_learner.word_embeddings:
                    embs.append(self.semantic_learner.word_embeddings[lbl])
                else:
                    # label not yet in semantic learner – create it
                    self.semantic_learner.initialize_word(lbl)
                    embs.append(self.semantic_learner.word_embeddings[lbl])
            if embs:
                combined = np.mean(embs, axis=0)
            else:
                combined = np.zeros(self.semantic_learner.emb_dim)
        else:
            combined = combined_vector

        # Ensure each label has an embedding in semantic learner
        for lbl in labels:
            if lbl not in self.semantic_learner.word_embeddings:
                self.semantic_learner.initialize_word(lbl)
                # use the combined vector as a rough embedding (or could use a stored state vector)
                if lbl in self.state_manager.states:
                    vec = self.state_manager.states[lbl]
                    self.semantic_learner.word_embeddings[lbl] = self._align_vector_to_emb_dim(vec)

        self.current_state_labels = labels
        self.current_state_embedding = self._align_vector_to_emb_dim(combined)

    def _process_sentence_reasoning(self, sentence: str, tokens: List[str],
                                    token_analyses: List[Dict], sentence_emb: np.ndarray):
        """
        Called only when all tokens are known.
        Stores sentence, generates hypotheses, detects relational statements,
        and updates word-state associations.
        """
        # Store sentence with embedding
        self.reasoner.store_sentence(sentence, tokens, embedding=sentence_emb,
                                     metadata={'timestamp': time.time(), 'source': 'user'})

        # --- Detect categorical or definitional statement ---
        # Find main verb: look for 'is' or 'means' in tokens
        verb_pos = None
        verb = None
        for i, tok in enumerate(tokens):
            if tok in ['is', 'are', 'was', 'were', 'means', 'like']:
                verb_pos = i
                verb = tok
                break
        if verb_pos is not None:
            subject = ' '.join(tokens[:verb_pos]).strip()
            object = ' '.join(tokens[verb_pos+1:]).strip()
            if subject and object:
                if verb in ['is', 'are', 'was', 'were']:
                    rel_type = 'is_a'
                elif verb == 'means':
                    rel_type = 'means'
                elif verb == 'like':
                    rel_type = 'similar'
                else:
                    rel_type = 'unknown'
                # Store relation in reasoner
                self.reasoner.store_phrase_relation(subject, object, rel_type, confidence=1.0)
                # Also generate a direct hypothesis about this relation if it's new
                key = f"relation:{rel_type}:{subject}:{object}"
                if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                    hyp = {
                        'key': key,
                        'type': 'relation',
                        'subject': subject,
                        'predicate': rel_type,
                        'object': object,
                        'text': f"Did I understand correctly: '{subject}' {verb} '{object}'?",
                        'asked': False,
                    }
                    self.reasoner.hypotheses.append(hyp)

                # ---- Word-state association ----
                # If subject is a word/phrase and object is a state label (single word), associate
                # Also handle "X like Y" as suggesting similarity
                if rel_type in ('means', 'is_a'):
                    # Treat object as a label for subject
                    self.reasoner.word_state.add_association(subject, object, confidence=1.0, source="explicit")
                    print(f"📝 [Word-State] Associated '{subject}' with label '{object}'")
                    
                    # ---- IMMEDIATE PROPAGATION: if object has labels, propagate them to subject ----
                    object_labels = self.reasoner.word_state.get_labels(object, threshold=0.3)
                    for lab in object_labels:
                        if lab != object:  # avoid self‑propagation
                            self.reasoner.word_state.add_association(subject, lab, confidence=0.9, source="inheritance")
                            print(f"   ↪ Propagated label '{lab}' from '{object}' to '{subject}'")
                            
                elif rel_type == 'similar':
                    # If subject and object are both words, they should share labels
                    subj_labels = self.reasoner.word_state.get_labels(subject)
                    obj_labels = self.reasoner.word_state.get_labels(object)
                    # Propagate labels from object to subject with lower confidence
                    for lab in obj_labels:
                        self.reasoner.word_state.add_association(subject, lab, confidence=0.7, source="similarity")
                    for lab in subj_labels:
                        self.reasoner.word_state.add_association(object, lab, confidence=0.7, source="similarity")
                    print(f"📝 [Word-State] Propagated labels between '{subject}' and '{object}'")
                # ---- Also add to knowledge graph ----
                # Get node for subject (word/phrase) and object (label)
                subj_nid = self.reasoner.knowledge_graph.get_node_id(subject, "word" if len(subject.split())==1 else "phrase")
                if subj_nid is None:
                    subj_nid = self.reasoner.knowledge_graph.add_node(subject, "word" if len(subject.split())==1 else "phrase")
                obj_nid = self.reasoner.knowledge_graph.get_node_id(object, "state")
                if obj_nid is None:
                    obj_nid = self.reasoner.knowledge_graph.add_node(object, "state")
                self.reasoner.knowledge_graph.add_edge(subj_nid, obj_nid, "associated", confidence=1.0)

        # Generate other hypotheses (existing logic)
        hypothesis = self.reasoner.generate_sentence_hypothesis(sentence, tokens, sentence_emb)
        if hypothesis:
            print(f"💡 [Sentence Hypothesis] {hypothesis}")
            # Hypothesis is already in self.reasoner.hypotheses
        # Store phrase embedding for future similarity
        if sentence_emb is not None:
            self.reasoner.store_phrase(sentence, sentence_emb, metadata={'source': 'user', 'timestamp': time.time()})

        # Check if we should ask about this sentence (low comprehension)
        if token_analyses:
            avg_novelty = np.mean([ta['factors'].factor_dict.get('novelty', 0.5) for ta in token_analyses])
            comprehension = 1.0 - avg_novelty
            if comprehension < 0.5:   # threshold
                query = self.reasoner.plan_sentence_query(sentence, uncertainty=1.0 - comprehension)
                if query:
                    self.pending_sentence_query = query
                    print(f"❓ [Pending Sentence Query] {query}")

    def _derive_inferences_from_definition(self, subject: str, definition_text: str):
        """
        Given a definition of subject (word or phrase), try to derive transitive inferences.
        For example, if we have "X means Y" and later "Y means Z", we can infer "X means Z".
        This method is called after a definition is stored.
        """
        # For now, we rely on the reasoner's relation graphs; the actual inference is
        # triggered when a relation is added via store_phrase_relation.
        # Here we just log that a definition was added (no direct action needed).
        pass
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Pad or truncate vector to match the current reference dimension (from StateManager)."""
        return self.state_manager._normalize_vector(vector)    

    def _all_tokens_known(self, tokens: List[str]) -> bool:
        """Return True iff every token in the list has a semantic embedding."""
        return all(t in self.semantic_learner.word_embeddings for t in tokens)

    def _integrate_behavioral_rule(self, rule_id: str):
        """Permanently integrate a confirmed behavioral rule."""
        rule = self.behavioral_learner.rules.get(rule_id)
        if not rule or not rule.confirmed:
            return

        if rule.rule_type == 'identity_alias':
            alias = rule.condition['alias']
            self.identity_aliases.add(alias)
            # Add to semantic learner with embedding similar to "I"
            if 'i' in self.semantic_learner.word_embeddings:
                i_emb = self.semantic_learner.word_embeddings['i']
                self.semantic_learner.initialize_word(alias)
                self.semantic_learner.word_embeddings[alias] = i_emb.copy()
                print(f"🧠 [Rule Integration] Learned self-alias: '{alias}'")
            if 'i' in self.semantic_learner.word_embeddings:
                i_emb = self.semantic_learner.word_embeddings['i']
                self.reasoner.store_definition(alias, f"a name for myself", i_emb)
            else:
                self.reasoner.store_definition(alias, f"a name for myself", None)
        self.behavioral_learner.integrate_rule(rule_id)

    def _should_ask_feedback(self, tokens: List[str], factor_dict: Dict, template_id: Optional[str] = None) -> bool:
        """Determine whether to ask for feedback, based on confidence and probability."""
        # Base probability
        base_prob = self.feedback_ask_probability

        # Reduce probability if we have high context confidence
        context_confidence = self.reasoner.get_context_confidence(f"template:{template_id}") if template_id else 0.0
        if context_confidence > 0.8:
            base_prob *= 0.2  # ask much less when confident

        # Also reduce if we asked recently (cooldown)
        if time.time() - self._last_feedback_ask_time < 60:  # at most once per minute
            return False

        if np.random.rand() > base_prob:
            return False

        # Check the corpus if it still needs feedback for this context
        needs = self.reasoner.should_ask_feedback(tokens, factor_dict, template_id)
        if needs:
            self._last_feedback_ask_time = time.time()
        return needs

    def _generate_feedback_question(self) -> str:
        """Generate a neutral question asking for user feedback."""
        templates = [
            "How would you rate my response? (positive/negative/neutral)",
            "Was my answer helpful? (positive/negative/neutral)",
            "Did that make sense? Please rate: positive, negative, or neutral.",
            "How do you feel about my reply? (positive/negative/neutral)",
        ]
        return np.random.choice(templates)

    # ------------------------------------------------------------------
    # Automatic drive creation: control factor → curiosity factor
    # ------------------------------------------------------------------
    def _on_new_control_factor(self, factor_name: str, compute_function: Callable, metadata: dict = None):
        """
        Callback when a new dynamic control factor is added.
        Creates a corresponding curiosity factor with a per‑token compute function.
        """
        def curiosity_compute_function(token: str, token_ci: Dict, token_phon: float,
                                       diag: Dict, features_array: np.ndarray,
                                       global_factor_dict: Optional[Dict] = None) -> float:
            """
            Compute the curiosity value for this token based on the control factor.
            Heuristic: absolute deviation of the control factor from neutral (0.5),
            multiplied by the token's own novelty/uncertainty (approximated by 1 - confidence).
            This creates a drive to explore tokens that are relevant to the current control state.
            """
            # Get current global value of this control factor (default 0.5)
            if global_factor_dict is not None:
                ctrl_val = global_factor_dict.get(factor_name, 0.5)
            else:
                ctrl_val = 0.5
            
            # Deviation from neutral (0.5) – how "active" this factor is
            deviation = abs(ctrl_val - 0.5) * 2.0  # scale to [0,1]
            
            # Token's own uncertainty (novelty)
            novelty = 1.0 - token_ci.get('confidence', 0.5)
            
            # Combine: curiosity is high when the control factor is active AND the token is novel
            curiosity = deviation * novelty
            return np.clip(curiosity, 0, 1)
        
        # Add this as a new curiosity factor with initial weight 0.0 (neutral prior)
        self.meta_learner.add_factor(
            factor_name=f"ctrl_{factor_name}",
            initial_mu=0.0,
            compute_function=curiosity_compute_function,
            metadata={'source': 'control_space', 'original_factor': factor_name}
        )
        print(f"🧠 [Drive Creation] New curiosity factor 'ctrl_{factor_name}' created from control dimension '{factor_name}'.")  

    def build_world_model_transitions(self):
        """Scan episodic memory to build transition counts for the structured world model."""
        episodes = self.episodic_memory.episodes[-500:]
        for i in range(len(episodes)-1):
            ep = episodes[i]
            next_ep = episodes[i+1]
            if 'factor_vector' in ep and 'factor_vector' in next_ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                next_vec = self._align_factor_vector(np.array(list(next_ep['factor_vector'].values())))
                state_label = self.state_manager.find_similar_state(vec, threshold=0.7)
                next_state_label = self.state_manager.find_similar_state(next_vec, threshold=0.7)
                if state_label and next_state_label:
                    action = ep.get('system_response', '')   # crude action representation
                    self.structured_world_model.observe_episode(state_label, action, next_state_label)
                    
    def process_input(self, text: str) -> Dict[str, Any]:
            """
            Full pipeline with meta-learning, reasoning, AND factor-based state learning.
            Also increments interaction counter and triggers self‑reconfiguration.

            FIXES applied vs previous version:
            - factor_dict initialized to {} at top to avoid NameError in section 8
            - expecting_word_conflict handler moved to the top-of-function waiting-state
            block (was previously buried after normal response generation, causing the
            full pipeline to run before the handler was ever reached)
            - 'section 2.5' handler body removed from bottom of function (it is now
            correctly placed as handler #7 at the top)
            - Chat-mode reward multiplier guarded so it only runs when factor_dict has
            been properly computed (i.e. not in early-return paths)
            - expecting_word_conflict flag defensively reset even if _pending_conflict
            is missing, to prevent permanent lock-up
            """
            self.last_interaction_time = time.time()

            # Pre-initialise so section 8 (behavioral rule testing) can safely reference it
            # before the normal-response block computes the real value.
            factor_dict: Dict[str, Any] = {}

            # ----------------- MAINTENANCE -----------------
            self._cycle_counter += 1
            if self._cycle_counter % 250 == 0:
                self._update_dynamic_limits()

            if self._cycle_counter > self._max_cycles_per_session:
                print("[WATCHDOG] Cycle limit reached — checkpointing.")
                self._save_state()
                self._cycle_counter = 0

            self._memory_pressure_backoff()

            # Reset idle reasoner's long-idle flag
            if hasattr(self, '_idle_reasoner'):
                self._idle_reasoner.reset_long_idle_flag()

            self.interaction_history.append(text)
            if len(self.interaction_history) > getattr(self, "_max_interaction_history", 200000):
                self.interaction_history = self.interaction_history[-self._max_interaction_history:]
            # ----------------- expectation from last turn, compute prediction  -----------------
            if self._expected_next_state is not None and self.current_state_labels:
                actual_state = self.current_state_labels[0]
                if actual_state != self._expected_next_state:
                    # Mismatch – boost curiosity for relevant words
                    prediction_error = 1.0
                    # Could add a factor to meta-learner
                    print(f"   [Prediction Error] Expected {self._expected_next_state}, got {actual_state}")
                self._expected_next_state = None   # reset
            # ----------------- RESPONSE HANDLING -----------------
            # If we are in any waiting state, handle the response and return early.
            # All waiting-state handlers live here so the main pipeline is never entered
            # while the system is awaiting a reply.

            # 1. Chat mode toggle (special command, not a waiting state)
            if text.lower().strip() == self.chat_mode_trigger:
                self.chat_mode_active = not self.chat_mode_active
                if self.chat_mode_active:
                    self.chat_history = []   # start fresh
                    system_reply = "Chat mode activated. I'll try to talk with you naturally. What shall we discuss?"
                else:
                    system_reply = "Chat mode deactivated. Returning to normal learning mode."
                print(f"🤖 AGI> {system_reply}")
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": [f"Chat mode: {'on' if self.chat_mode_active else 'off'}"],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 2. Response feedback (emergent sentiment learning)
            if not self.chat_mode_active and self.expecting_response_feedback:
                feedback_lower = text.lower()
                rating = None
                if any(kw in feedback_lower for kw in ['positive', 'good', 'great', 'yes', 'correct', 'right', '👍']):
                    rating = 'positive'
                elif any(kw in feedback_lower for kw in ['negative', 'bad', 'wrong', 'no', 'incorrect', '👎']):
                    rating = 'negative'
                elif any(kw in feedback_lower for kw in ['neutral', 'ok', 'fine', 'average']):
                    rating = 'neutral'

                if rating is not None and self.pending_feedback_context is not None:
                    entry = self.pending_feedback_context
                    entry.rating = rating
                    entry.metadata['feedback_timestamp'] = time.time()
                    entry.metadata['feedback_text'] = text
                    self.reasoner.add_feedback(entry)
                    print(f"📊 [Feedback Recorded] User rated response as: {rating}")

                    if 'positive' in rating and hasattr(self, '_last_factor_vector'):
                        self._reinforce_identity(self._last_factor_vector, reward_signal=0.2)
                else:
                    print(f"🌑 [Feedback] No clear rating detected, ignoring.")

                self.expecting_response_feedback = False
                self.pending_feedback_context = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Feedback recorded."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 3. Rule confirmation response
            if not self.chat_mode_active and self.expecting_rule_confirmation:
                feedback_lower = text.lower()
                is_positive = any(kw in feedback_lower for kw in ['yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm'])
                is_negative = any(kw in feedback_lower for kw in ['no', 'wrong', 'incorrect', 'not', 'nope', 'never'])

                if is_positive or is_negative:
                    rule_id = self.pending_rule_id
                    if rule_id in self.behavioral_learner.rules:
                        if is_positive:
                            print(f"✅ [Rule Confirmed] User accepted rule: {rule_id}")
                            self.behavioral_learner.update_confidence(rule_id, +0.3, feedback=True)
                            if self.behavioral_learner.rules[rule_id].confidence >= self.behavioral_learner.confirmation_threshold:
                                self._integrate_behavioral_rule(rule_id)
                        else:
                            print(f"❌ [Rule Rejected] User denied rule: {rule_id}")
                            self.behavioral_learner.update_confidence(rule_id, -0.5, feedback=True)
                else:
                    print(f"🌑 [Rule Shadow] No clear confirmation for rule: {self.pending_rule_id}")
                    if self.pending_rule_id in self.behavioral_learner.rules:
                        self.behavioral_learner.update_confidence(self.pending_rule_id, -0.1, feedback=False)

                self.expecting_rule_confirmation = False
                self.pending_rule_id = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Rule confirmation processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 4. Logic doubt feedback
            if not self.chat_mode_active and self.expecting_logic_feedback:
                feedback_lower = text.lower()
                is_positive = any(kw in feedback_lower for kw in ['yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm'])
                is_negative = any(kw in feedback_lower for kw in ['no', 'wrong', 'incorrect', 'not', 'nope', 'never'])

                if self.pending_logic_doubt:
                    logic_id = self.pending_logic_doubt['logic_id']
                    if is_positive:
                        self.logic_inheritance.confirm_logic(logic_id, is_correct=True)
                        print(f"✅ [Logic Confirmed] User accepted logic: {logic_id}")
                    elif is_negative:
                        self.logic_inheritance.confirm_logic(logic_id, is_correct=False)
                        print(f"❌ [Logic Rejected] User denied logic: {logic_id}")

                    if is_positive or is_negative:
                        self.generated_code_feedback.add_feedback(
                            logic_id,
                            "positive" if is_positive else "negative",
                            {"factor_dict": self._last_factor_dict.copy() if self._last_factor_dict else {}}
                        )
                        if self._last_factor_dict is not None:
                            rating = "positive" if is_positive else "negative"
                            self.guilt_tracker.update_guilt(rating, self._last_factor_dict)
                    else:
                        print(f"🌑 [Logic Shadow] No clear confirmation for logic: {logic_id}")

                self.expecting_logic_feedback = False
                self.pending_logic_doubt = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Logic feedback processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 5. Pattern label response
            if not self.chat_mode_active and getattr(self, "expecting_pattern_label", False):
                label = text.strip().lower()
                self.reasoner.relations[self.last_asked_word].append(("category", label, 1.0))
                if self.last_asked_word not in self.semantic_learner.word_embeddings:
                    self.semantic_learner.initialize_word(self.last_asked_word)
                print(f"📘 [Pattern Learning] '{self.last_asked_word}' categorized as '{label}'")
                self.expecting_pattern_label = False
                self.last_asked_word = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Pattern label learned."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 6. Trust feedback (from earlier attempt)
            if not self.chat_mode_active and self.expecting_feedback:
                feedback_lower = text.lower()
                is_positive = any(kw in feedback_lower for kw in ['yes', 'correct', 'right', 'yeah', 'true', 'yep'])
                is_negative = any(kw in feedback_lower for kw in ['no', 'wrong', 'incorrect', 'not', 'nope'])

                if is_positive or is_negative:
                    if is_positive:
                        print(f"✅ [Positive Feedback] User confirmed my attempt: '{self.feedback_attempt_response}'")
                        if hasattr(self, '_last_factor_vector'):
                            self._reinforce_identity(self._last_factor_vector, reward_signal=0.4)
                    else:
                        print(f"❌ [Negative Feedback] User corrected my attempt: '{self.feedback_attempt_response}'")
                else:
                    print(f"🌑 [Shadow Attempt] No feedback given for: '{self.feedback_attempt_response}'")
                    if self.feedback_attempt_response:
                        hyp = {
                            'key': f"shadow_trust_{int(time.time())}",
                            'type': 'shadow_trust',
                            'utterance': self.feedback_attempt_response,
                            'timestamp': self.feedback_timestamp,
                        }
                        if len(self.reasoner.hypotheses) < getattr(self, "_max_hypotheses", 5000):
                            self.reasoner.hypotheses.append(hyp)

                self.expecting_feedback = False
                self.feedback_attempt_word = None
                self.feedback_attempt_response = None
                self.feedback_timestamp = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Feedback processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 7. Word-label conflict resolution
            # FIX: this was previously buried after normal response generation, meaning
            # the entire pipeline ran before the handler was reached. It must live here
            # alongside all other waiting-state handlers.
            if not self.chat_mode_active and getattr(self, "expecting_word_conflict", False):
                feedback_lower = text.lower()
                if hasattr(self, '_pending_conflict'):
                    word, label1, label2 = self._pending_conflict
                    if label1.lower() in feedback_lower or label2.lower() in feedback_lower:
                        keep = label1 if label1.lower() in feedback_lower else label2
                        self.reasoner.word_state.break_bridge(word, label1, label2, keep)
                        print(f"✅ [Conflict Resolved] '{word}' is now primarily '{keep}'.")
                    else:
                        print(f"🌑 No clear label chosen for '{word}'. Conflict remains.")
                    del self._pending_conflict
                else:
                    # Safety: flag was set but state was lost — reset cleanly
                    print("🌑 [Word Conflict] Expected conflict state missing; resetting flag.")
                self.expecting_word_conflict = False
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Word conflict processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 8. Definition response (when last_asked_word is set and user provides a definition)
            handled_definition = False
            if not self.chat_mode_active and self.last_asked_word is not None:
                if "means" in text.lower() or "is" in text.lower() or "refers to" in text.lower():
                    is_phrase = ' ' in self.last_asked_word
                    if is_phrase:
                        phrase = self.last_asked_word
                        definition = text
                        phrase_tokens = phrase.split()
                        known_embs = [self.semantic_learner.word_embeddings[t] for t in phrase_tokens if t in self.semantic_learner.word_embeddings]
                        phrase_emb = np.mean(known_embs, axis=0) if known_embs else None
                        self.reasoner.phrase_definitions[phrase] = definition
                        if phrase_emb is not None:
                            self.reasoner.phrase_embeddings[phrase] = phrase_emb
                        print(f"📘 [Phrase Definition] Learned about '{phrase}': {definition}")
                        self._derive_inferences_from_definition(phrase, definition)
                    else:
                        word = self.last_asked_word
                        if word not in self.semantic_learner.word_embeddings:
                            self.semantic_learner.initialize_word(word)
                        self.semantic_learner.observe_definition(word, text)
                        def_emb = self.semantic_learner.text_to_embedding(text)
                        if def_emb is not None:
                            self.reasoner.store_definition(word, text, def_emb)
                            self.concept_model.register_definition_event(word)
                        if word in self.word_history:
                            for rec in self.word_history[word][-3:]:
                                rec['curiosity_factors'].gp_uncertainty *= 0.3
                                rec['curiosity_factors'].bayesian_surprise *= 0.3
                            if hasattr(self.concept_model, 'word_concept_counts') and word in self.concept_model.word_concept_counts:
                                counts = self.concept_model.word_concept_counts[word]
                                if counts:
                                    dominant = max(counts, key=counts.get)
                                    new_counts = Counter({dominant: 100})
                                    self.concept_model.word_concept_counts[word] = new_counts
                            hypothesis = self.reasoner.generate_synonym_hypothesis(word)
                            if hypothesis:
                                print(f"💡 [Semantic Hypothesis] {hypothesis}")
                        print(f"📘 [Definition Learning] Learned about '{word}': {text}")
                        self._derive_inferences_from_definition(word, text)

                    if hasattr(self, '_last_factor_vector'):
                        self._reinforce_identity(self._last_factor_vector, reward_signal=0.3)
                    self.last_asked_word = None
                    handled_definition = True
                    # Fall through to main processing so the sentence is still analyzed
            # 9. Procedure explanation (when user is teaching step by step)
            if not self.chat_mode_active and self.expecting_procedure_steps:
                # Record the step provided by user
                step_text = text.strip()
                self.recorded_steps.append(step_text)
                print(f"📝 [Step Recorded] {step_text}")
                # Simple heuristic: if step ends with '.' or we have a solution, maybe stop.
                # We'll just continue until user says "done" or "that's it".
                if "done" in step_text.lower() or "that's it" in step_text.lower() or "finish" in step_text.lower():
                    # End of teaching
                    if self.pending_problem_expr is not None:
                        # Store procedure
                        proc = self.reasoner.procedural_learner.learn_procedure(
                            self.pending_problem_expr, self.recorded_steps, None
                        )
                        print(f"🧠 [Procedure Learned] I've learned a new procedure for {self.pending_problem_expr}")
                    self.expecting_procedure_steps = False
                    self.pending_problem_expr = None
                    self.recorded_steps = []
                    return {
                        "ci_report": {},
                        "phon_score": 0,
                        "diagnostic": {},
                        "token_analyses": [],
                        "curiosity_artifacts": ["Procedure recorded."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }
                else:
                    # Continue expecting more steps
                    return {
                        "ci_report": {},
                        "phon_score": 0,
                        "diagnostic": {},
                        "token_analyses": [],
                        "curiosity_artifacts": ["Step recorded. Continue."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }
            # ----------------- MAIN PROCESSING -----------------
            # If we reach here, we are not in a waiting state (or we handled a definition and continue).
            self.interaction_counter += 1

            # 1. Memory ingestion
            add_sentence_to_memory(text)

            # 2. Whole-sentence evaluation
            ci_report = self.ci_engine.evaluate(text)
            phon_score, phon_diagnostics = self.phon_engine.score_text(text)
            diag_explanation = self.diag_engine.explain(ci_report, phon_score)

            # 3. Token analysis
            tokens = self._tokenize(text)
            self.reasoner.observe_sentence(text, tokens)
            self.reward_tracker.reset_episode()

            token_analyses = []
            for token in tokens:
                token_ci = self.ci_engine.evaluate(token)
                token_phon, _ = self.phon_engine.score_text(token)

                features = self._extract_features(token, token_ci, token_phon, diag_explanation)
                features_array = self._features_to_array(features)
                concept = self.concept_model.add_word_observation(token, features)

                factors = self._compute_curiosity_factors(
                    token, token_ci, token_phon, diag_explanation, features_array,
                    global_factor_dict=self._last_factor_dict
                )

                new_entropy = self.concept_model.entropy_word(token)
                igain = self._information_gain(token, new_entropy)

                if igain == 0.0:
                    igain = 0.25 * factors.bayesian_surprise + 0.25 * factors.gp_uncertainty

                self.gp_model.update(features_array, token_ci['confidence'])

                reward = self.reward_tracker.compute_reward(token_ci, igain, factors.bayesian_surprise)

                self.word_history[token].append({
                    'ci_confidence': token_ci['confidence'],
                    'phon_score': token_phon,
                    'features': features,
                    'concept': concept,
                    'entropy': new_entropy,
                    'curiosity_factors': factors,
                    'information_gain': igain,
                    'intrinsic_reward': reward,
                })

                factor_array = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))

                struct_score = self.structural_score(token)

                # Adjust information gain for meta-learner to avoid structural contamination
                igain_adjusted = igain * (1 - struct_score)
                self.meta_learner.update(factor_array, igain_adjusted)

                curiosity_score = self.meta_learner.predict_curiosity(factor_array)
                curiosity_score *= (1 - struct_score)

                token_analyses.append({
                    'token': token,
                    'concept_id': concept,
                    'entropy': new_entropy,
                    'curiosity_score': curiosity_score,
                    'factors': factors,
                    'igain': igain,
                    'intrinsic_reward': reward,
                    'gp_uncertainty': factors.gp_uncertainty,
                    'bayesian_surprise': factors.bayesian_surprise,
                })
            # ---- Update token profiles for structural scoring ----
            for token in tokens:
                # Get embedding (if available)
                emb = self.semantic_learner.word_embeddings.get(token)
                if emb is None:
                    emb = np.zeros(self.semantic_learner.emb_dim)  # fallback
                # Find the corresponding token analysis entry for this token (to get bayes_surprise and igain)
                ta = next((t for t in token_analyses if t['token'] == token), None)
                if ta:
                    bayes = ta.get('bayesian_surprise', 0.0)
                    igain = ta.get('igain', 0.0)
                else:
                    bayes = 0.0
                    igain = 0.0
                # Context: all other tokens in this input
                context = [t for t in tokens if t != token]
                self._update_token_profile(token, context, bayes, igain, emb)
            # ---- Sentence-level reasoning (if all tokens known) ----
            if self._all_tokens_known(tokens):
                sentence_emb = self.semantic_learner.text_to_embedding(text)
                if sentence_emb is not None:
                    self._process_sentence_reasoning(text, tokens, token_analyses, sentence_emb)

            # ---- Build expression from syntax tree ----
            syntax_tree = self.syntax_learner.parse(tokens)
            if syntax_tree is not None:
                expr = syntax_tree.to_expression(self.reasoner.type_hierarchy, self.reasoner.operators)
                if expr is not None and isinstance(expr.root, Operator):
                    self.reasoner.store_expression(text, expr)

                # ---- Detect equations (left = right) ----
                # Heuristic: if the top node is '=' and has two children, it's an equation.
                if expr is not None and expr.root == '=' and expr.left and expr.right:
                    left = expr.left
                    right = expr.right
                    # Feed to schema learner
                    self.reasoner.schema_learner.incorporate_equation(left, right)

                    # Try to solve the equation
                    subs = self.reasoner.solve_equation(left, right)
                    if subs is not None:
                        # Build a readable result
                        result_parts = []
                        for var_name, val_expr in subs.items():
                            # Convert value expression to string
                            if val_expr.is_leaf():
                                val_str = str(val_expr.root)
                            else:
                                val_str = str(val_expr)
                            result_parts.append(f"{var_name} = {val_str}")
                        if result_parts:
                            self.pending_solution = ", ".join(result_parts)
                        else:
                            self.pending_solution = None
                    else:
                        self.pending_solution = None
                        # If we couldn't solve and user asked a question, consider teaching
                        if '?' in text:
                            if expr is not None:
                                proc = self.reasoner.procedural_learner.find_procedure(expr)
                                if proc:
                                    # Apply procedure (placeholder)
                                    result_expr = proc.apply(expr, self.reasoner)
                                    if result_expr:
                                        self.pending_solution = f"Using learned procedure: {result_expr}"
                                else:
                                    # No procedure, ask user to teach us
                                    self.pending_teaching_request = True
                                    self.pending_teaching_expr = expr
            if syntax_tree is not None:
                expr = syntax_tree.to_expression(self.reasoner.type_hierarchy, self.reasoner.operators)
                if expr is not None and isinstance(expr.root, Operator):
                    self.reasoner.store_expression(text, expr)

            # ----------------- DECIDE WHETHER TO ASK A QUESTION -----------------
            # These checks are in priority order; the first that triggers will ask and return early.

            # 4. Logic doubt (ask about a generated logic rule)
            # 4.5 Teaching request (ask user to teach us)
            if not self.chat_mode_active and self.pending_teaching_request:
                self.pending_teaching_request = False
                expr = self.pending_teaching_expr
                self.pending_teaching_expr = None
                self.expecting_procedure_steps = True
                self.pending_problem_expr = expr
                self.recorded_steps = []
                teach_question = f"I don't know how to solve {expr}. Could you teach me step by step?"
                print(f"\n🧠 [Teaching Request] {teach_question}")
                print(f"🤖 AGI> {teach_question}")
                return {
                    "ci_report": ci_report,
                    "phon_score": phon_score,
                    "diagnostic": diag_explanation,
                    "token_analyses": token_analyses,
                    "curiosity_artifacts": ["Teaching mode activated."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
            if not self.chat_mode_active:
                doubt = self.logic_inheritance.get_next_doubt()
                if doubt:
                    rule = doubt['rule']
                    description = rule.get('description', f"I wrote a logical rule: {rule}")
                    query = f"I have a doubt about some logic I wrote: {description} Is this correct?"
                    print(f"\n🧠 [Logic Doubt] {query}")
                    print(f"🤖 AGI> {query}")
                    self.expecting_logic_feedback = True
                    self.pending_logic_doubt = doubt
                    return {
                        "ci_report": ci_report,
                        "phon_score": phon_score,
                        "diagnostic": diag_explanation,
                        "token_analyses": token_analyses,
                        "curiosity_artifacts": ["Logic doubt – awaiting confirmation."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }

            # 5. Unknown word detection
            if not self.chat_mode_active:
                unknown_tokens = [t for t in tokens if t not in self.semantic_learner.word_embeddings]
                unknown_ratio = len(unknown_tokens) / max(len(tokens), 1)
                if unknown_ratio > 0.5 and unknown_tokens:
                    # Score each unknown token (structural score)
                    scored_unknowns = [(t, self.structural_score(t)) for t in unknown_tokens]
                    # Sort by score ascending (most semantic first)
                    scored_unknowns.sort(key=lambda x: x[1])
                    target, struct_score = scored_unknowns[0]
                    # Only ask if structural score is not too high (i.e., it's likely semantic)
                    if struct_score <= 0.65:
                        self.last_asked_word = target
                        system_reply = f"What does '{target}' mean?"
                        print(f"\n🧠 [Knowledge Gap] {system_reply}")
                        print(f"🤖 AGI> {system_reply}")
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Knowledge gap detected — waiting for definition."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": 0.0,
                            "factor_vector": {},
                        }
            # 6. Pattern discovery (ask for category label)
            if not self.chat_mode_active:
                for token in set(tokens):
                    pattern_query = self.reasoner.detect_pattern_and_query(token)
                    if pattern_query:
                        self.last_asked_word = token
                        self.expecting_pattern_label = True
                        print(f"\n🧠 [Pattern Discovery] {pattern_query}")
                        print(f"🤖 AGI> {pattern_query}")
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Pattern discovered — awaiting label."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": self.reward_tracker.episode_reward,
                            "factor_vector": {},
                        }

            # 7. Conflict resolution (interactive – not a waiting state)
            if not self.chat_mode_active:
                unresolved = self.state_manager.get_unresolved_conflicts()
                if unresolved:
                    print("\n🧠 [Conflict Resolution] I have some unresolved state conflicts from previous idle introspection.")
                    for conflict in unresolved[:2]:
                        l1, l2 = conflict["labels"]
                        sim = conflict["similarity"]
                        shadow_label = conflict.get("resolution_label")
                        print(f"    Conflict: '{l1}' ↔ '{l2}' (similarity={sim:.3f})")
                        print(f"    I created a temporary shadow state '{shadow_label}'.")
                        print("    How would you like to resolve this?")
                        print("      1. Keep both (no change)")
                        print("      2. Keep only one (specify which)")
                        print("      3. Rename and keep both")
                        print("      4. Accept shadow state as permanent")
                        print("      5. Both states are different – show factors and edit")
                        choice = input("    Your choice (1/2/3/4/5): ").strip()

                        if choice == "2":
                            winner = input(f"    Which label do you want to keep? ({l1}/{l2}): ").strip()
                            if winner in [l1, l2]:
                                self.state_manager.resolve_conflict_permanently(conflict, winner, delete_other=True)
                                print(f"    ✅ Resolved: keeping '{winner}', deleted the other.")
                        elif choice == "3":
                            new_label1 = input(f"    New label for '{l1}': ").strip() or l1
                            new_label2 = input(f"    New label for '{l2}': ").strip() or l2
                            vec1 = self.state_manager.states[l1]
                            vec2 = self.state_manager.states[l2]
                            self.state_manager.save_state(new_label1, vec1)
                            self.state_manager.save_state(new_label2, vec2)
                            self.state_manager.delete_state(l1)
                            self.state_manager.delete_state(l2)
                            conflict["resolved"] = True
                            conflict["resolution_label"] = f"{new_label1},{new_label2}"
                            for c in self.state_manager.conflicts:
                                if l1 in c["labels"] or l2 in c["labels"]:
                                    c["resolved"] = True
                                    c["resolution_label"] = "renamed"
                            print(f"    ✅ States renamed and kept.")
                        elif choice == "4" and shadow_label:
                            vec = self.state_manager.shadow_states.get(shadow_label)
                            if vec is not None:
                                perm_label = input(f"    Permanent name for this merged state: ").strip() or f"merged_{l1}_{l2}"
                                self.state_manager.save_state(perm_label, vec, is_shadow=False)
                                self.state_manager.delete_state(shadow_label)
                                self.state_manager.resolve_conflict_permanently(conflict, perm_label, delete_other=True)
                                print(f"    ✅ Shadow state promoted to '{perm_label}', both originals deleted.")
                        elif choice == "5":
                            # Show factor vectors and allow editing
                            vec1 = self.state_manager.states[l1]
                            vec2 = self.state_manager.states[l2]
                            # Align to same length (max of both)
                            max_len = max(len(vec1), len(vec2))
                            v1_aligned = np.pad(vec1, (0, max_len - len(vec1)), 'constant') if len(vec1) < max_len else vec1[:max_len]
                            v2_aligned = np.pad(vec2, (0, max_len - len(vec2)), 'constant') if len(vec2) < max_len else vec2[:max_len]

                            # Get factor names (from control_space)
                            factor_names = self.control_space.all_factors
                            # If lengths differ, we may need to pad factor names with placeholders
                            if len(factor_names) < max_len:
                                factor_names = factor_names + [f"factor_{i}" for i in range(len(factor_names), max_len)]
                            elif len(factor_names) > max_len:
                                factor_names = factor_names[:max_len]

                            print(f"\n    Factor vectors for '{l1}' and '{l2}':")
                            for i, name in enumerate(factor_names):
                                print(f"      {name:20s} : {l1} = {v1_aligned[i]:.3f} , {l2} = {v2_aligned[i]:.3f}")

                            print("\n    You can now edit the factor vectors. Enter new values for each state as comma‑separated lists.")
                            print("    (Press Enter to keep current values for that state.)")
                            new_vec1_str = input(f"    New vector for '{l1}' (comma‑separated, {max_len} numbers): ").strip()
                            new_vec2_str = input(f"    New vector for '{l2}' (comma‑separated, {max_len} numbers): ").strip()

                            def parse_vector(s, default_vec):
                                if not s:
                                    return default_vec
                                parts = s.split(',')
                                try:
                                    nums = [float(p.strip()) for p in parts]
                                    if len(nums) < max_len:
                                        nums += [default_vec[len(nums):].tolist()] if len(default_vec) > len(nums) else []
                                    elif len(nums) > max_len:
                                        nums = nums[:max_len]
                                    return np.array(nums)
                                except:
                                    print("    Invalid input, keeping original.")
                                    return default_vec

                            new_v1 = parse_vector(new_vec1_str, v1_aligned)
                            new_v2 = parse_vector(new_vec2_str, v2_aligned)

                            # Update states (they will be normalized by update_state)
                            self.state_manager.update_state(l1, new_v1, metadata=conflict.get('metadata', {}))
                            self.state_manager.update_state(l2, new_v2, metadata=conflict.get('metadata', {}))

                            # Mark conflict as resolved (we keep both)
                            conflict["resolved"] = True
                            conflict["resolution_label"] = f"edited:{l1},{l2}"
                            print(f"    ✅ States '{l1}' and '{l2}' updated and conflict resolved.")
                        else:
                            print("    No action taken; conflict remains unresolved.")
                    self.state_manager._save_conflicts()

            # 8. Behavioral rule testing (ask for confirmation)
            # so the 'factor_dict in locals()' workaround is no longer needed.
            if not self.chat_mode_active:
                context = {
                    'user_input': text,
                    'tokens': tokens,
                    'token_analyses': token_analyses,
                    'factor_dict': factor_dict,
                    'state': self.current_state_labels
                }
                testable = self.behavioral_learner.get_testable_rules(context)
                if testable:
                    rule = max(testable, key=lambda r: r.confidence)
                    if rule.rule_type == 'identity_alias':
                        alias = rule.condition['alias']
                        query = f"Do you want me to be called '{alias}'? Should I refer to myself as '{alias}'?"
                    elif rule.rule_type == 'pronoun_mapping':
                        query = f"When you say '{rule.condition['pronoun']}', do you mean me?"
                    else:
                        query = f"I noticed a pattern: {rule.condition}. Is that correct?"
                    rule.asked = True
                    rule.last_tested = time.time()
                    self.behavioral_learner.save()
                    self.expecting_rule_confirmation = True
                    self.pending_rule_id = rule.rule_id
                    print(f"\n🧠 [Rule Hypothesis] {query}")
                    print(f"🤖 AGI> {query}")
                    return {
                        "ci_report": ci_report,
                        "phon_score": phon_score,
                        "diagnostic": diag_explanation,
                        "token_analyses": token_analyses,
                        "curiosity_artifacts": ["Behavioral rule hypothesis — awaiting confirmation."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": self.reward_tracker.episode_reward,
                        "factor_vector": factor_dict,
                    }

            # ----------------- NORMAL RESPONSE GENERATION -----------------
            # If no question was asked, compute factors and generate a normal reply.

            # Factor computation & state learning
            factor_dict = self.factor_engine.compute(
                token_analyses, ci_report, phon_score, diag_explanation,
                self.concept_model, self.meta_learner, self.reward_tracker.total_reward,
                self.semantic_learner
            )
            if not isinstance(factor_dict, dict):
                factor_dict = {}

            # Conscience influence
            factor_dict = self.conscience_manager.apply_ethical_influence(factor_dict)
            if not isinstance(factor_dict, dict):
                factor_dict = {}

            # Build factor vector
            factor_vector_list = []
            for k in self.control_space.all_factors:
                factor_vector_list.append(factor_dict.get(k, 0.5))
            factor_vector = np.array(factor_vector_list)
            self._last_factor_vector = factor_vector
            self._last_factor_dict = factor_dict

            metadata = {
                "input_text": text,
                "timestamp": time.time(),
                "ci_confidence": ci_report.get("confidence"),
                "phon_score": phon_score,
            }
            # State learning gate
            if not self.chat_mode_active and len(self.semantic_learner.word_embeddings) > 20:
                self._handle_state_learning(factor_vector, metadata)

            # Semantic relation discovery
            self._detect_semantic_relations(tokens, token_analyses)

            # Generate artifacts
            artifacts = self._generate_artifacts(
                tokens, token_analyses, ci_report, phon_score, diag_explanation,
                factor_dict
            )
            # Generate system reply
            system_reply = self.action_generator.generate_reply(text, {
                'token_analyses': token_analyses,
                'factor_vector': factor_dict,
            })
            if self.pending_solution is not None:
                system_reply = f"Solution: {self.pending_solution}. " + system_reply
                self.pending_solution = None   # reset
            self._last_action = system_reply
            # ---- Top‑down expectation: predict next state using world model ----
            if self.structured_world_model and self.current_state_labels:
                current_state = self.current_state_labels[0] if self.current_state_labels else None
                if current_state:
                    action = system_reply  # the action we just took
                    predicted_next = self.structured_world_model.predict_next_state(current_state, action)
                    if predicted_next:
                        # Compare predicted vs actual next state (which we don't have yet – would need next turn)
                        # For now, we store the prediction to be checked later.
                        self._expected_next_state = predicted_next
            if "What does" in system_reply and "'" in system_reply:
                import re
                match = re.search(r"'([^']+)'", system_reply)
                if match:
                    self.last_asked_word = match.group(1)
            print(f"🤖 AGI> {system_reply}")

            # Possibly ask for feedback on the response we just generated
            if not self.chat_mode_active and not self.expecting_response_feedback and not self.expecting_feedback and not self.expecting_rule_confirmation and not self.expecting_logic_feedback:
                template_id = f"resp_{hash(system_reply) % 10000:04x}"
                self.reasoner.register_response_template(template_id, system_reply)
                if self._should_ask_feedback(tokens, factor_dict, template_id):
                    feedback_question = self._generate_feedback_question()
                    print(f"🤖 AGI> {feedback_question}")
                    entry = FeedbackEntry(
                        timestamp=time.time(),
                        user_input=text,
                        system_response=system_reply,
                        factor_dict=factor_dict.copy(),
                        rating='pending',
                        context_tokens=tokens,
                        response_tokens=self._tokenize(system_reply),
                        response_template_id=template_id
                    )
                    self.pending_feedback_context = entry
                    self.expecting_response_feedback = True

            # Check for word-label conflicts and ask user to resolve if any found.
            # This runs after response generation so we already have factor_dict.
            # FIX: the old "2.5" block that also appeared here (as a handler) has been
            # removed — the handler now lives at the top with all other waiting-state
            # handlers (#7). This block only *raises* new word conflicts; it never
            # handles replies to them.
            if not self.chat_mode_active and not any([
                self.expecting_response_feedback,
                self.expecting_feedback,
                self.expecting_rule_confirmation,
                self.expecting_logic_feedback,
                getattr(self, 'expecting_word_conflict', False),
            ]):
                for token in tokens:
                    conflicts = self.reasoner.word_state.detect_conflicts(token)
                    if conflicts:
                        label1, label2 = conflicts[0]
                        question = f"I'm unsure whether '{token}' means '{label1}' or '{label2}'. Which one is correct?"
                        print(f"\n🧠 [Word Conflict] {question}")
                        print(f"🤖 AGI> {question}")
                        self.expecting_word_conflict = True
                        self._pending_conflict = (token, label1, label2)
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Word conflict – awaiting clarification."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": self.reward_tracker.episode_reward,
                            "factor_vector": factor_dict,
                        }

            # Chat mode reward multiplier — only applied after factor_dict is fully computed.
            # FIX: previously ran even on early-return paths where factor_dict was still {}.
            if self.chat_mode_active:
                multiplier = self._compute_chat_reward_multiplier(factor_dict)
                original_reward = self.reward_tracker.episode_reward
                self.reward_tracker.episode_reward *= multiplier
                print(f"🤖 [Chat Mode] Reward multiplier: {multiplier:.2f} (original: {original_reward:.3f} → boosted: {self.reward_tracker.episode_reward:.3f})")

            # Store episode
            safe_factor_dict = factor_dict if isinstance(factor_dict, dict) else {}
            self.episodic_memory.add_episode(
                user_input=text,
                system_response=system_reply,
                factor_vector=safe_factor_dict,
                intrinsic_reward=float(self.reward_tracker.episode_reward),
                metadata={'ci_confidence': ci_report.get('confidence')}
            )

            if self.chat_mode_active:
                self.chat_history.append((text, system_reply))
                if len(self.chat_history) > 20:
                    self.chat_history.pop(0)

            self._save_state()

            if self.interaction_counter % self.reconfigure_interval == 0:
                self._maybe_reconfigure()

            return {
                'ci_report': ci_report,
                'phon_score': phon_score,
                'diagnostic': diag_explanation,
                'token_analyses': token_analyses,
                'curiosity_artifacts': artifacts,
                'meta_weights': self.meta_learner.get_weights(),
                'total_intrinsic_reward': self.reward_tracker.total_reward,
                'episode_reward': self.reward_tracker.episode_reward,
                'factor_vector': factor_dict,
            }


    def _tokenize(self, text: str) -> List[str]:
            import re

            # --- Base tokenization ---
            raw = [t.lower() for t in re.findall(r"[a-zA-Z']+|\d+\.?\d*|[+\-*/=<>%^()]", text) if len(t) >= 2]

            # --- State-based min-length gate (adapts as vocabulary grows) ---
            vocab_size = len(self.semantic_learner.word_embeddings) if hasattr(self, 'semantic_learner') else 0
            min_len = 1 if vocab_size < 50 else 2

            tokens = [t for t in raw if len(t) >= min_len]

            # --- Self-developing: learn which token patterns are useful ---
            # Boost tokens that already exist in semantic learner (known = useful)
            known = [t for t in tokens if t in getattr(self, 'semantic_learner', {}).word_embeddings if hasattr(self, 'semantic_learner')]
            unknown = [t for t in tokens if t not in known]

            # If we have very few known tokens, relax filter to include shorter unknowns
            if len(known) == 0 and unknown:
                tokens = [t for t in raw if len(t) >= 1]

            # --- Track tokenization stats for self-improvement ---
            if not hasattr(self, '_tokenizer_stats'):
                self._tokenizer_stats = {'total': 0, 'known_ratio_history': []}
            self._tokenizer_stats['total'] += 1
            if tokens:
                ratio = len(known) / len(tokens)
                self._tokenizer_stats['known_ratio_history'].append(ratio)
                if len(self._tokenizer_stats['known_ratio_history']) > 200:
                    self._tokenizer_stats['known_ratio_history'].pop(0)
            return tokens if tokens else raw
    
    def _generate_artifacts(
        self,
        tokens: List[str],
        token_analyses: List[Dict],
        ci_report: Dict,
        phon_score: float,
        diag: Dict,
        factor_dict: Dict,
    ) -> List[str]:
        """Generate rich artifacts including dynamic factor state."""
        artifacts = []
        artifacts.append("🧠 [Meta-Curiosity Analysis v0.5 – Adaptive Control Space]")
        artifacts.append(f"   Input tokens: {tokens}")

        artifacts.append(f"   • CI confidence: {ci_report.get('confidence', 0.5):.3f} | Stability: {ci_report.get('stability', 0.5):.3f}")
        artifacts.append(f"   • Phonotactic score: {phon_score:.3f}")

        # Diagnostic summary
        artifacts.append(f"   • Diagnostic: {diag.get('summary', 'N/A')}")

        # Meta-learner weights (dynamic)
        weights = self.meta_learner.get_weights()
        artifacts.append("   • Current curiosity priorities (learned):")
        # Show top 5 absolute weights
        sorted_weights = sorted(weights.items(), key=lambda x: -abs(x[1]))
        for name, w in sorted_weights[:5]:
            artifacts.append(f"     - {name}: {w:.3f}")

        # Intrinsic reward
        artifacts.append(f"   • Total intrinsic reward (lifetime): {self.reward_tracker.total_reward:.3f}")
        artifacts.append(f"   • Episode intrinsic reward: {self.reward_tracker.episode_reward:.3f}")

        # ---- Factor state snapshot (now includes dynamic factors) ----
        artifacts.append("   • Current control-signal state:")
        # Show base factors first
        for k in self.control_space.base_factors:
            if k in factor_dict:
                artifacts.append(f"     - {k}: {factor_dict[k]:.3f}")
        # Then show dynamic factors
        for k in self.control_space.dynamic_factors:
            if k in factor_dict:
                artifacts.append(f"     - {k}: {factor_dict[k]:.3f} (dynamic)")
        
        # Check if this state is already known
        vec = np.array([factor_dict[k] for k in self.control_space.all_factors if k in factor_dict])
        threshold = self.config_manager.get('engine.state_similarity_threshold', 0.85)
        similar_label = self.state_manager.find_similar_state(vec, threshold=threshold)
        if similar_label:
            artifacts.append(f"   • State recognized as: '{similar_label}'")
        # ------------------------------------

        # Per-token deep analysis (adapt to show all available curiosity factors)
        artifacts.append("   • Token-level curiosity & reasoning:")
        for ta in token_analyses:
            token = ta['token']
            score = ta['curiosity_score']
            entropy = ta['entropy']
            factors = ta['factors']  # CuriosityFactors object
            gp_unc = ta['gp_uncertainty']
            bayes_surp = ta['bayesian_surprise']

            artifacts.append(f"     🔹 '{token}': curiosity={score:.3f} (entropy={entropy:.3f})")
            
            # Show all curiosity factors that are present
            factor_line_parts = []
            for name, val in factors.factor_dict.items():
                if name in ["bayesian_surprise", "gp_uncertainty"]:
                    continue  # show separately
                factor_line_parts.append(f"{name}:{val:.2f}")
            artifacts.append(f"        " + " ".join(factor_line_parts[:6]))  # first 6
            artifacts.append(f"        Bayesian surprise:{bayes_surp:.3f} GP uncertainty:{gp_unc:.3f}")

            hypothesis = self.reasoner.generate_hypothesis(token, {})
            artifacts.append(f"        💭 {hypothesis}")

            if score > 0.4:
                query = self.reasoner.plan_query(token, uncertainty=gp_unc)
                artifacts.append(f"        ❓ {query}")

        total_concepts = len(self.concept_model.concept_counts)
        artifacts.append(f"   • Total latent concepts discovered: {total_concepts}")
        if self.word_history:
            most_uncertain_word = max(self.word_history.keys(),
                                      key=lambda w: self.concept_model.entropy_word(w),
                                      default=None)
            if most_uncertain_word:
                artifacts.append(f"   • Most globally uncertain word: '{most_uncertain_word}' "
                                 f"(entropy={self.concept_model.entropy_word(most_uncertain_word):.3f})")

        return artifacts
    # ------------------------------------------------------------------
    # Idle reflection and introspection (unchanged logic)
    # ------------------------------------------------------------------
    def perform_state_introspection(self):
        """
        Long-idle task:
        1. Detect conflicts among saved states, create shadow resolutions.
        2. Offline learning from episodic memory.
        """
        print("\n🧠 [Deep State Introspection] Examining saved states for conflicts...")
        threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
        self.state_manager.refresh_conflicts(threshold=threshold)
        unresolved = self.state_manager.get_unresolved_conflicts()
        if not unresolved:
            print("    No unresolved conflicts found.")
        else:
            print(f"    Found {len(unresolved)} unresolved conflict(s).")
            for conflict in unresolved[:3]:
                l1, l2 = conflict["labels"]
                sim = conflict["similarity"]
                words1 = conflict.get("words1", [])
                words2 = conflict.get("words2", [])
                print(f"    Conflict: '{l1}' ↔ '{l2}' (similarity={sim:.3f})")
                if words1:
                    print(f"      Words that led to '{l1}': {', '.join(words1)}")
                if words2:
                    print(f"      Words that led to '{l2}': {', '.join(words2)}")
                shadow_label = self.state_manager.create_shadow_resolution(conflict)
                self.state_manager.apply_shadow_to_conflict(conflict, shadow_label)
                print(f"      → Created shadow state '{shadow_label}'. Will ask user at next interaction.")
            self.state_manager._save_conflicts()

        # ---- Offline learning from episodic memory ----
        print("    Consolidating knowledge from recent episodes...")
        episodes = self.episodic_memory.replay_sequence()
        for ep in episodes[-20:]:   # last 20
            user_input = ep['user_input']
            tokens = self._tokenize(user_input)
            for t in tokens:
                context = [w for w in tokens if w != t]
                self.semantic_learner.observe_context(t, context)
            self.syntax_learner.add_sentence(user_input)
        print("    Offline consolidation complete.\n")
    # =========================================================================
    # IDLE TASK METHODS (called by IdleReasoner)
    # =========================================================================
    def _get_resource_status(self):
        """
        Returns (cpu_ok, mem_ok, throttle_factor, cpu_percent, mem_percent).
        throttle_factor ∈ [0.3, 1.0] – used to reduce workload when resources are strained.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
        except Exception:
            # if psutil fails, assume everything is fine (but throttle a little)
            return True, True, 0.8, 0.0, 0.0

        cpu_ok = cpu_percent < 80.0          # CPU below 80%
        mem_ok = mem_percent < 90.0          # memory below 90%

        if not cpu_ok or not mem_ok:
            throttle = 0.3                  # severely reduce workload
        else:
            throttle = 1.0                 # full workload

        return cpu_ok, mem_ok, throttle, cpu_percent, mem_percent

    def _compute_chat_reward_multiplier(self, factor_dict: Dict) -> float:
        """
        Returns a dynamic reward multiplier based on emotional state.
        Used only when chat mode is active.
        Formula: base 1.0 + (curiosity * 0.5) + (trust * 0.3) - (threat * 0.4) + (comprehension * 0.2)
        Clamped to [0.5, 2.0].
        """
        curiosity = factor_dict.get('curiosity', 0.5)
        trust = factor_dict.get('trust', 0.5)
        threat = factor_dict.get('threat', 0.0)
        comprehension = factor_dict.get('comprehension', 0.5)

        multiplier = 1.0 + (curiosity * 0.5) + (trust * 0.3) - (threat * 0.4) + (comprehension * 0.2)
        return np.clip(multiplier, 0.5, 2.0)
    
    def world_model_dreaming(self):
        """1. Multi‑term dreaming using n‑gram models from episodic memory."""
        if not self.config_manager.get('idle_reasoner.enable_world_model_dreaming', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping World Model Dreaming (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🌙 [World Model Dreaming] Simulating multi‑term synthetic conversations...")

        # Build n‑gram model from episodic memory user inputs (only known words)
        n = self.config_manager.get('idle_reasoner.dreaming_ngram_order', 3)
        ngram_model = defaultdict(Counter)

        episodes = self.episodic_memory.episodes[-200:]   # recent only
        for ep in episodes:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < n:
                continue
            # only keep sentences where all tokens are known (to ensure plausible dreaming)
            if not self._all_tokens_known(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                ngram_model[context][next_word] += 1

        if not ngram_model:
            print("   [Dreaming] Not enough n‑gram data yet.")
            return

        # Generate synthetic sentences
        base_num = self.config_manager.get('idle_reasoner.world_model_num_synthetic', 5)
        num_syn = max(1, int(base_num * throttle))

        vocab_known = list(self.semantic_learner.word_embeddings.keys())
        if not vocab_known:
            return

        for _ in range(num_syn):
            # start with a random known context (first n-1 words)
            contexts = list(ngram_model.keys())
            start_ctx = contexts[np.random.randint(len(contexts))]
            words = list(start_ctx)
            # generate up to 12 words
            for _ in range(12 - n):
                ctx = tuple(words[-(n-1):]) if len(words) >= n-1 else None
                if ctx in ngram_model:
                    next_words = list(ngram_model[ctx].keys())
                    probs = np.array(list(ngram_model[ctx].values()))
                    probs = probs / probs.sum()
                    next_w = np.random.choice(next_words, p=probs)
                    words.append(next_w)
                else:
                    break
            sentence = ' '.join(words)
            # Only dream about sentences where all words are known
            if self._all_tokens_known(words):
                self._simulate_interaction(sentence)

        print("   [Dreaming] Processed {} synthetic multi‑term sentences (throttle={:.2f}, n={}).".format(num_syn, throttle, n))

    def _simulate_interaction(self, text: str):
        """Internal: process a synthetic input without side effects (no console output)."""
        # Minimal version of process_input – only update models, no queries to user.
        tokens = self._tokenize(text)
        self.reasoner.observe_sentence(text, tokens)
        for token in tokens:
            token_ci = self.ci_engine.evaluate(token)
            token_phon, _ = self.phon_engine.score_text(token)
            diag_explanation = self.diag_engine.explain(token_ci, token_phon)
            features = self._extract_features(token, token_ci, token_phon, diag_explanation)
            features_array = self._features_to_array(features)
            concept = self.concept_model.add_word_observation(token, features)
            new_entropy = self.concept_model.entropy_word(token)
            igain = self._information_gain(token, new_entropy)
            self.gp_model.update(features_array, token_ci['confidence'])
            # record history
            self.word_history[token].append({
                'ci_confidence': token_ci['confidence'],
                'phon_score': token_phon,
                'features': features,
                'concept': concept,
                'entropy': new_entropy,
                'information_gain': igain,
            })
            # update semantic learner via context
            # ---- update semantic learner via context ----
            context = [w for w in tokens if w != token]
            if self.chat_mode_active:
                # Aggressive training: update 3 times with higher learning rate
                for _ in range(3):
                    # Temporarily increase learning rate by scaling word count denominator
                    orig_count = self.semantic_learner.word_counts[token]
                    self.semantic_learner.word_counts[token] = max(1, orig_count * 0.5)  # artificially reduce
                    self.semantic_learner.observe_context(token, context)
                    self.semantic_learner.word_counts[token] = orig_count  # restore
            else:
                self.semantic_learner.observe_context(token, context)
        # store a dummy episode (optional)
        self.episodic_memory.add_episode(
            user_input=text,
            system_response="[synthetic]",
            factor_vector={},
            intrinsic_reward=0.0,
            metadata={'synthetic': True}
        )

    def active_memory_rehearsal(self):
        """2. Active Memory Rehearsal Scheduling (CLS theory)."""
        if not self.config_manager.get('idle_reasoner.enable_active_memory_rehearsal', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Active Memory Rehearsal (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Active Memory Rehearsal] Prioritizing episodes...")
        episodes = self.episodic_memory.episodes
        if len(episodes) < 5:
            return
        # Compute a priority score for each episode
        scores = []
        for idx, ep in enumerate(episodes):
            surprise = 0.0
            # Bayesian surprise for words in user_input
            tokens = self._tokenize(ep.get('user_input', ''))
            for t in tokens:
                surprise += self.concept_model.bayesian_surprise(t)
            # rarity: inverse of frequency of state vector (if available)
            rarity = 0.0
            if 'factor_vector' in ep and ep['factor_vector']:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                sims = [1.0 - cdist([vec], [s], 'cosine')[0,0]
                        for s in self.state_manager.states.values() if len(s) == len(vec)]
                if sims:
                    rarity = 1.0 - np.mean(sims)
            # identity drift
            drift = 0.0
            if hasattr(self, 'identity_manager') and 'factor_vector' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                drift = self.identity_manager.detect_drift(vec)
            # unresolved hypothesis bonus
            hyp_bonus = 0.2 if any(h.get('asked') is False for h in self.reasoner.hypotheses) else 0.0
            scores.append(surprise * 0.4 + rarity * 0.3 + drift * 0.2 + hyp_bonus * 0.1)
        # Select top k episodes – scale by throttle
        base_k = self.config_manager.get('idle_reasoner.rehearsal_batch_size', 20)
        k = max(1, int(base_k * throttle))
        top_idx = np.argsort(scores)[-k:]
        print(f"   Replaying {len(top_idx)} high‑priority episodes (throttle={throttle:.2f}).")
        for idx in top_idx:
            ep = episodes[idx]
            user_input = ep.get('user_input', '')
            tokens = self._tokenize(user_input)
            for t in tokens:
                context = [w for w in tokens if w != t]
                self.semantic_learner.observe_context(t, context)
            self.syntax_learner.add_sentence(user_input)

    def curiosity_landscape_optimization(self):
        """3. Curiosity Landscape Optimization (meta‑curiosity gradient analysis)."""
        if not self.config_manager.get('idle_reasoner.enable_curiosity_landscape_optimization', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Curiosity Landscape Optimization (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("📈 [Curiosity Landscape Optimization] Analyzing factor correlations...")
        # collect recent factor vectors and IGs from word_history
        factor_vals = []
        ig_vals = []
        for w, hist in self.word_history.items():
            for entry in hist[-20:]:  # recent
                if 'curiosity_factors' in entry and 'information_gain' in entry:
                    factors = entry['curiosity_factors'].as_array(self.meta_learner.factor_names)
                    factor_vals.append(factors)
                    ig_vals.append(entry['information_gain'])
        if len(factor_vals) < 10:
            return
        factor_matrix = np.array(factor_vals)

        # remove constant columns (zero variance → NaN corr)
        stds = factor_matrix.std(axis=0)
        valid_cols = stds > 1e-6
        factor_matrix = factor_matrix[:, valid_cols]

        if factor_matrix.shape[1] < 2:
            return

        # If throttled, skip the expensive correlation matrix and merging
        if throttle < 0.8:
            print(f"   [Resource Aware] Skipping correlation analysis (throttle={throttle:.2f}).")
            return

        corr = np.corrcoef(factor_matrix.T)

        # detect highly correlated pairs (redundancy)
        thresh = self.config_manager.get('idle_reasoner.curiosity_correlation_threshold', 0.9)
        redundant = []
        valid_indices = np.where(valid_cols)[0]

        for ii in range(len(valid_indices)):
            for jj in range(ii+1, len(valid_indices)):
                i = valid_indices[ii]
                j = valid_indices[jj]
                if corr[ii, jj] > thresh:
                    redundant.append((i, j, corr[ii, jj]))
        # merge weak factors (placeholder)
        for i, j, r in redundant[:2]:  # limit per run
            name_i = self.meta_learner.factor_names[i]
            name_j = self.meta_learner.factor_names[j]
            # merge weights: weighted average by variance
            w_i = self.meta_learner.mu[i]
            w_j = self.meta_learner.mu[j]
            new_w = (w_i + w_j) / 2.0
            # keep first factor, zero second
            self.meta_learner.mu[i] = new_w
            self.meta_learner.mu[j] = 0.0
            print(f"   Merged factor '{name_j}' into '{name_i}' (corr={r:.3f})")
        # spawn new factor from residual unexplained variance
        ig_array = np.array(ig_vals)
        mu = align_vector(self.meta_learner.mu, factor_matrix.shape[1])
        pred_ig = factor_matrix @ mu
        residual = ig_array - pred_ig
        if np.var(residual) > 0.01:
            # simple heuristic: new factor = residual itself (for next iteration)
            new_factor_name = f"residual_{int(time.time())}"
            def residual_compute(token, token_ci, token_phon, diag, features, global_factors):
                # placeholder: returns small random value
                return np.random.rand() * 0.1
            self.meta_learner.add_factor(new_factor_name, initial_mu=0.2, compute_function=residual_compute)
            print(f"   Spawned new curiosity factor '{new_factor_name}' from unexplained variance.")

    def concept_graph_compression(self):
        """4. Concept Graph Compression (Symbolic abstraction formation)."""
        if not self.config_manager.get('idle_reasoner.enable_concept_graph_compression', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Concept Graph Compression (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🗂️ [Concept Graph Compression] Creating higher‑order concepts...")
        # cluster concepts based on word membership
        concept_words = defaultdict(list)
        for word, cnts in self.concept_model.word_concept_counts.items():
            # assign word to its most probable concept
            if cnts:
                concept = max(cnts, key=cnts.get)
                concept_words[concept].append(word)
        # for concepts with many words, create a meta‑concept
        for concept, words in concept_words.items():
            if len(words) >= 3:
                meta_label = f"meta_{'_'.join(words[:2])}"
                # store in reasoning engine as a relation
                for w in words:
                    self.reasoner.relations[w].append(("category", meta_label, 0.8))
                # also add as a definition
                meta_def = f"a group of words: {', '.join(words)}"
                # create a dummy embedding (average of member embeddings)
                emb_list = [self.semantic_learner.word_embeddings.get(w, np.zeros(self.semantic_learner.emb_dim))
                            for w in words if w in self.semantic_learner.word_embeddings]
                if emb_list:
                    avg_emb = np.mean(emb_list, axis=0)
                    self.reasoner.store_definition(meta_label, meta_def, avg_emb)
                print(f"   Created meta‑concept '{meta_label}' from {len(words)} words.")

    def autonomous_hypothesis_planning(self):
        """5. Generate experiments for both words AND sentences."""
        if not self.config_manager.get('idle_reasoner.enable_autonomous_hypothesis_planning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Autonomous Hypothesis Planning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔬 [Autonomous Hypothesis Planning] Generating word and sentence experiments...")

        # --- Word-level candidates (unchanged) ---
        candidates = []
        for word in self.concept_model.word_concept_counts:
            entropy = self.concept_model.entropy_word(word)
            if entropy > 0.5 and word not in self.reasoner.definitions:
                ig_estimate = entropy - 0.1
                candidates.append((word, ig_estimate, "definition"))
        for hyp in self.reasoner.hypotheses[:getattr(self, "_max_hypotheses", 5000)]:
            if hyp.get('type') == 'synonym' and not hyp.get('asked', False):
                candidates.append((hyp['word'], 0.8, "synonym"))

        # --- Sentence-level candidates from episodic memory ---
        sentence_candidates = []
        for ep in self.episodic_memory.episodes[-100:]:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < 3:
                continue
            if not self._all_tokens_known(tokens):
                continue
            # compute a crude "uncertainty" for the sentence (average novelty)
            avg_novelty = 0.0
            count = 0
            for t in tokens:
                if t in self.word_history and self.word_history[t]:
                    last = self.word_history[t][-1]
                    if 'curiosity_factors' in last:
                        avg_novelty += last['curiosity_factors'].factor_dict.get('novelty', 0.5)
                        count += 1
            if count > 0:
                avg_novelty /= count
                if avg_novelty > 0.5:   # uncertain sentence
                    ig_estimate = avg_novelty * 0.5   # heuristic
                    sentence_candidates.append((sent, ig_estimate, "sentence_meaning"))

        # Combine and sort
        all_candidates = candidates + sentence_candidates
        all_candidates.sort(key=lambda x: -x[1])
        self.research_queue = []
        max_queue = max(1, int(10 * throttle))
        for item, ig, typ in all_candidates[:max_queue]:
            if typ == "definition":
                query = f"What does '{item}' mean?"
            elif typ == "synonym":
                hyp = next((h for h in self.reasoner.hypotheses if h.get('word')==item and h.get('type')=='synonym'), None)
                if hyp:
                    query = f"Is '{item}' similar to '{hyp['candidate']}'?"
                else:
                    continue
            elif typ == "sentence_meaning":
                query = f"What does the sentence '{item}' mean?"
            else:
                continue
            self.research_queue.append({
                'query': query,
                'target': item if typ != "sentence_meaning" else "sentence",
                'ig': ig,
                'type': typ,
                'timestamp': time.time()
            })
        print(f"   Research queue now has {len(self.research_queue)} high‑priority queries (throttle={throttle:.2f}).")

    def identity_stabilization(self):
        """6. Identity Stabilization & Self‑Model Consolidation."""
        if not self.config_manager.get('idle_reasoner.enable_identity_stabilization', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Identity Stabilization (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧘 [Identity Stabilization] Consolidating self‑model...")
        # collect recent identity vectors from episodes with positive reward
        for ep in self.episodic_memory.episodes[-50:]:
            if ep.get('intrinsic_reward', 0) > 0.1 and 'factor_vector' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                self.identity_history.append(
                    self._align_vector_to_emb_dim(self._align_factor_vector(vec))
                )
        if len(self.identity_history) < 5:
            return
        # keep last 100
        self.identity_history = self.identity_history[-100:]

        # If throttled heavily, skip k‑means (expensive)
        if throttle < 0.5:
            print(f"   [Resource Aware] Skipping clustering due to low throttle ({throttle:.2f}).")
            # still do a simple moving average
            attractor = np.mean(self.identity_history[-10:], axis=0)
        else:
            # cluster identity vectors
            identity_matrix = np.array(self.identity_history)
            k = min(self.config_manager.get('idle_reasoner.identity_cluster_k', 3), len(identity_matrix))
            try:
                centroids, _ = kmeans2(identity_matrix, k, minit='points')
            except:
                return
            # largest cluster centroid as attractor
            labels = []
            for vec in identity_matrix:
                dists = cdist([vec], centroids, 'euclidean')[0]
                labels.append(np.argmin(dists))
            largest_cluster = np.argmax(np.bincount(labels))
            attractor = centroids[largest_cluster]

        # move identity towards attractor
        alpha = 0.1
        self.identity_manager.identity = (1 - alpha) * self.identity_manager.identity + alpha * attractor
        norm = np.linalg.norm(self.identity_manager.identity)
        if norm > 0:
            self.identity_manager.identity /= norm
        self.identity_manager.confidence = min(1.0, self.identity_manager.confidence + 0.01)
        self.identity_manager.save()
        # detect phase transition
        drift = self.identity_manager.detect_drift(attractor)
        if drift > 0.4:
            print(f"   ⚠️ Identity phase transition detected (drift={drift:.3f}). New attractor formed.")

    def cross_modal_alignment(self):
        """7. Cross‑modal representation alignment (contrastive learning)."""
        if not self.config_manager.get('idle_reasoner.enable_cross_modal_alignment', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Cross‑modal Alignment (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        # Contrastive learning is CPU intensive – skip entirely if throttle < 0.5
        if throttle < 0.5:
            print(f"   [Resource Aware] Skipping Cross‑modal Alignment due to low resources (throttle={throttle:.2f}).")
            return

        print("🔄 [Cross‑modal Alignment] Aligning semantic, state, episodic embeddings...")
        # gather positive pairs: (state_vector, text_embedding) from same episode
        pos_pairs = []
        for ep in self.episodic_memory.episodes[-100:]:
            if 'factor_vector' in ep and ep['factor_vector']:
                state_vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                state_vec = self._align_vector_to_emb_dim(state_vec)
                text = ep.get('user_input', '')
                text_emb = self.semantic_learner.text_to_embedding(text)
                if text_emb is not None:
                    pos_pairs.append((state_vec, text_emb))
        if len(pos_pairs) < 5:
            return
        # simple contrastive update: pull positive pairs together, push negatives apart
        lr = 0.01 * throttle   # reduce learning rate when throttled
        for state_vec, text_emb in pos_pairs:
            # positive pair similarity
            sim_pos = np.dot(state_vec, text_emb) / (np.linalg.norm(state_vec)*np.linalg.norm(text_emb)+1e-12)
            # sample a negative text embedding (random other episode)
            neg_candidates = [p for p in pos_pairs if not np.allclose(p[1], text_emb)]
            neg_pair = None
            if neg_candidates:
                neg_pair = neg_candidates[np.random.randint(len(neg_candidates))]

            if neg_pair:
                neg_emb = neg_pair[1]
                sim_neg = np.dot(state_vec, neg_emb) / (np.linalg.norm(state_vec)*np.linalg.norm(neg_emb)+1e-12)
                # InfoNCE‑like loss: maximize exp(sim_pos)/ (exp(sim_pos)+exp(sim_neg))
                # gradient ascent on log prob
                prob_pos = np.exp(sim_pos) / (np.exp(sim_pos) + np.exp(sim_neg))
                # update text embedding towards state_vec (very crude)
                grad = (1 - prob_pos) * (state_vec - text_emb)
                # update the words in the text embedding? We'll approximate by adjusting each word's embedding
                tokens = self._tokenize(ep['user_input'])
                for t in tokens:
                    if t in self.semantic_learner.word_embeddings:
                        self.semantic_learner.word_embeddings[t] += lr * grad / len(tokens)
        print("   Contrastive alignment step completed (throttle={:.2f}).".format(throttle))

    def exploration_policy_distillation(self):
        """8. Exploration Policy Distillation."""
        if not self.config_manager.get('idle_reasoner.enable_exploration_policy_distillation', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Exploration Policy Distillation (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        if throttle < 0.4:
            print(f"   [Resource Aware] Skipping policy training (throttle={throttle:.2f}).")
            return

        print("🎯 [Exploration Policy Distillation] Learning when to ask...")
        # We need to collect decision points: episodes where we asked a question vs. not.
        # For simplicity, we classify based on whether system response contains '?'
        X = []
        y = []
        for ep in self.episodic_memory.episodes[-200:]:
            if 'factor_vector' in ep and ep['factor_vector'] and 'system_response' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                vec = self._normalize_vector(vec)  # use state_manager's normalizer
                X.append(vec)
                response = ep['system_response']
                y.append(1 if '?' in response else 0)
        if len(X) < 20 or len(set(y)) < 2:
            return
        X = np.array(X)
        y = np.array(y)
        # train a simple logistic regression classifier (no sklearn, use scipy.optimize)
        from scipy.optimize import minimize
        def sigmoid(z): return 1 / (1 + np.exp(-z))
        def loss(w, X, y):
            z = X @ w[:-1] + w[-1]
            p = sigmoid(z)
            eps = 1e-12
            return -np.mean(y * np.log(p+eps) + (1-y) * np.log(1-p+eps))
        w0 = np.zeros(X.shape[1] + 1)
        # Use a simpler optimizer when throttled
        if throttle < 0.7:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'
        res = minimize(loss, w0, args=(X, y), method=method)
        self.policy_model = (res.x[:-1], res.x[-1])
        print(f"   Policy distilled. Intercept: {res.x[-1]:.3f} (method={method})")

    def cognitive_drive_evolution(self):
        """9. Cognitive Drive Evolution (Automatic Drive Creation)."""
        if not self.config_manager.get('idle_reasoner.enable_cognitive_drive_evolution', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Cognitive Drive Evolution (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧬 [Cognitive Drive Evolution] Spawning/removing intrinsic drives...")
        # Remove weak dynamic factors (|mu| < threshold)
        weak_thresh = 0.001
        to_remove = []
        for i, name in enumerate(self.meta_learner.dynamic_factor_names):
            idx = self.meta_learner.factor_names.index(name)
            if abs(self.meta_learner.mu[idx]) < weak_thresh:
                to_remove.append(name)
        # Remove each factor one by one, recomputing the index each time
        for name in to_remove:
            if name not in self.meta_learner.dynamic_factor_names:
                continue
            idx = self.meta_learner.factor_names.index(name)
            self.meta_learner.dynamic_factor_names.remove(name)
            # Remove from arrays
            keep = [i for i in range(len(self.meta_learner.factor_names)) if i != idx]
            self.meta_learner.mu = self.meta_learner.mu[keep]
            self.meta_learner.lambda_ = self.meta_learner.lambda_[keep]
            self.meta_learner.alpha = self.meta_learner.alpha[keep]
            self.meta_learner.beta = self.meta_learner.beta[keep]
            if name in self.meta_learner.dynamic_compute_functions:
                del self.meta_learner.dynamic_compute_functions[name]
            if name in self.meta_learner.factor_metadata:
                del self.meta_learner.factor_metadata[name]
            print(f"   Removed weak drive: '{name}'")
        # Spawn new drive from residual variance (simplified) – reduce probability when throttled
        spawn_prob = 0.2 * throttle
        if np.random.rand() < spawn_prob:
            new_name = f"drive_{int(time.time())}"
            def new_drive_compute(token, token_ci, token_phon, diag, features, global_factors):
                return np.random.rand() * 0.5
            self.meta_learner.add_factor(new_name, initial_mu=0.1, compute_function=new_drive_compute)
            print(f"   Spawned new cognitive drive: '{new_name}' (throttle={throttle:.2f})")    

    def counterfactual_reasoning(self):
        """10. Counterfactual Reasoning Engine (Offline causal imagination)."""
        if not self.config_manager.get('idle_reasoner.enable_counterfactual_reasoning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Counterfactual Reasoning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Counterfactual Reasoning] Simulating 'what if I asked X instead?'...")
        # For each recent episode, pick an alternative word and estimate IG
        recent = self.episodic_memory.episodes[-20:]
        max_eps = max(1, int(len(recent) * throttle))
        for ep in recent[:max_eps]:
            if 'user_input' not in ep:
                continue
            tokens = self._tokenize(ep['user_input'])
            if not tokens:
                continue
            # actual word we might have asked about? Not stored; we just simulate
            alt_word = np.random.choice(tokens)
            entropy = self.concept_model.entropy_word(alt_word)
            # crude IG estimate if we had asked about it
            alt_ig = entropy - 0.1  # assume answer reduces entropy to 0.1
            # compare to actual IG from episode (if available)
            actual_ig = ep.get('intrinsic_reward', 0)  # crude proxy
            if alt_ig > actual_ig + 0.1:
                # alternative would have been better; increase weight of factors active in this state
                if 'factor_vector' in ep and ep['factor_vector']:
                    vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                    # normalize
                    if np.linalg.norm(vec) > 0:
                        vec = vec / np.linalg.norm(vec)
                    # boost meta‑learner weights for factors with high activation
                    n = len(self.meta_learner.factor_names)

                    vec = align_vector(vec, n)
                    self.meta_learner.mu = align_vector(self.meta_learner.mu, n)

                    for i in range(n):
                        self.meta_learner.mu[i] += 0.01 * vec[i]
        print(f"   Counterfactual updates applied (processed {max_eps} episodes, throttle={throttle:.2f}).")

    def semantic_self_talk(self):
        """11. Advanced self‑talk: trigram + crossover from episodic memory."""
        if not self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Semantic Self‑Talk (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🗣️ [Semantic Self‑Talk] Recombining learned sentences (trigram + crossover)...")

        # Build trigram model from episodic memory (only known words)
        trigram = defaultdict(Counter)
        episodes = self.episodic_memory.episodes[-200:]
        for ep in episodes:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < 3:
                continue
            if not self._all_tokens_known(tokens):
                continue
            for i in range(len(tokens)-2):
                ctx = tuple(tokens[i:i+2])
                nxt = tokens[i+2]
                trigram[ctx][nxt] += 1

        if not trigram:
            print("   Not enough trigram data yet.")
            return

        # generate sentences
        num_sentences = max(1, int(5 * throttle))
        for _ in range(num_sentences):
            # either pure trigram walk (70%) or crossover (30%)
            if np.random.rand() < 0.7:
                # trigram walk
                keys = list(trigram.keys())
                ctx = keys[np.random.randint(len(keys))]
                words = list(ctx)
                for _ in range(np.random.randint(5, 12)):
                    ctx_key = tuple(words[-2:])
                    if ctx_key in trigram:
                        next_words = list(trigram[ctx_key].keys())
                        probs = np.array(list(trigram[ctx_key].values()))
                        probs = probs / probs.sum()
                        words.append(np.random.choice(next_words, p=probs))
                    else:
                        break
            else:
                # crossover: take prefix of one episode, suffix of another
                if len(episodes) >= 2:
                    idx = np.random.choice(len(episodes), 2, replace=False)
                    ep1 = episodes[idx[0]]
                    ep2 = episodes[idx[1]]
                    sent1 = ep1.get('user_input', '')
                    sent2 = ep2.get('user_input', '')
                    tok1 = self._tokenize(sent1)
                    tok2 = self._tokenize(sent2)
                    if len(tok1) >= 2 and len(tok2) >= 2:
                        split = np.random.randint(1, min(len(tok1), len(tok2)))
                        words = tok1[:split] + tok2[split:]
                    else:
                        words = tok1[:2] + tok2[-2:]
                else:
                    continue

            sentence = ' '.join(words)
            # only learn if all words are known
            if self._all_tokens_known(words):
                self.syntax_learner.add_sentence(sentence)
                for token in words:
                    context = [w for w in words if w != token]
                    self.semantic_learner.observe_context(token, context)
                print(f"   Generated and learned: '{sentence[:70]}...'")
        print(f"   Generated {num_sentences} sentences (throttle={throttle:.2f}).")

    def state_space_topology_learning(self):
        """12. State‑space topology learning (cognitive manifold discovery)."""
        if not self.config_manager.get('idle_reasoner.enable_state_space_topology_learning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping State‑space Topology Learning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        if throttle < 0.6:
            print(f"   [Resource Aware] Skipping topology learning (throttle={throttle:.2f}).")
            return

        print("🌐 [State‑space Topology] Discovering attractors and macro‑states...")
        # collect all state vectors from state_manager
        all_vectors = []
        labels = []
        for label, vec in self.state_manager.states.items():
            all_vectors.append(self._normalize_vector(vec))
            labels.append(label)
        if len(all_vectors) < 4:
            return
        all_vectors = np.array(all_vectors)
        n_clusters = min(self.config_manager.get('idle_reasoner.topology_n_clusters', 5), len(all_vectors))
        try:
            centroids, assignments = kmeans2(all_vectors, n_clusters, minit='points')
        except:
            return
        # create macro‑state for each centroid
        for i, cent in enumerate(centroids):
            # find members
            members = [labels[j] for j, a in enumerate(assignments) if a == i]
            if len(members) >= 2:
                macro_name = f"macro_{i}_{int(time.time())}"
                # save as a shadow state (macro‑state)
                self.state_manager.save_state(macro_name, cent, metadata={'members': members}, is_shadow=True)
                print(f"   Created macro‑state '{macro_name}' from {len(members)} states.")
        self.state_manifold = centroids

    def knowledge_gap_frontier_detection(self):
        """13. Knowledge gap frontier detection."""
        if not self.config_manager.get('idle_reasoner.enable_knowledge_gap_frontier', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Knowledge Gap Frontier (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🌍 [Knowledge Gap Frontier] Updating research agenda...")
        # recompute IG for all candidate words (high entropy, no definition)
        frontiers = []
        for word in self.concept_model.word_concept_counts:
            if word not in self.reasoner.definitions:
                entropy = self.concept_model.entropy_word(word)
                if entropy > 0.3:
                    ig = entropy - 0.1
                    frontiers.append((word, ig, "definition"))
        # also include unresolved hypotheses
        for hyp in self.reasoner.hypotheses:
            if not hyp.get('asked', False):
                word = hyp.get('word')
                if word:
                    frontiers.append((word, 0.8, "hypothesis"))
        # sort and keep top N, scaled by throttle
        frontiers.sort(key=lambda x: -x[1])
        self.research_queue = []
        max_items = max(1, int(20 * throttle))
        for word, ig, typ in frontiers[:max_items]:
            if typ == "definition":
                query = f"What does '{word}' mean?"
            else:
                query = f"Could you clarify '{word}'?"
            self.research_queue.append({
                'query': query,
                'target': word,
                'ig': ig,
                'type': typ,
                'timestamp': time.time()
            })
        print(f"   Research queue updated: {len(self.research_queue)} items (throttle={throttle:.2f}).")

    def social_cognition_simulation(self):
        """14. Social cognition simulation (theory of mind dreaming)."""
        if not self.config_manager.get('idle_reasoner.enable_social_cognition_simulation', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Social Cognition Simulation (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧑‍🤝‍🧑 [Social Cognition] Simulating user responses...")
        # collect feedback events (stored in reasoner.hypotheses with type 'shadow_trust')
        pos_features = []
        neg_features = []
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') == 'shadow_trust' and 'utterance' in hyp:
                # we don't have state vector here; we need to store it with the hypothesis
                # For simplicity, we just simulate
                pass
        # Instead, use the user_model attributes already defined
        # if we have positive and negative examples, train a naive mean classifier
        # This is a placeholder: we simply record the mean state for positive feedback.
        # In a full system, you would store state vectors when feedback is received.
        # Here we simulate: if we have episodes where feedback was positive (from earlier handling)
        # we update user_model.
        # Since we don't have explicit feedback storage, we skip detailed training.
        print("   (User model update placeholder – would need feedback storage)")

    def long_horizon_planning_rehearsal(self):
        """15. Long‑horizon cognitive planning rehearsal."""
        if not self.config_manager.get('idle_reasoner.enable_long_horizon_planning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Long‑horizon Planning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🎭 [Long‑horizon Planning] Simulating multi‑turn dialogues...")
        # simple rollout: start from a random state, simulate turns, accumulate reward
        if not self.state_manager.states:
            return
        # pick a random start state
        start_label = np.random.choice(list(self.state_manager.states.keys()))
        state_vec = self.state_manager.states[start_label].copy()
        total_reward = 0.0
        # reduce number of turns when throttled
        num_turns = max(1, int(3 * throttle))
        for _ in range(num_turns):
            # decide action: ask about most uncertain word? (simplified)
            # find word with highest entropy
            if self.concept_model.word_concept_counts:
                word = max(self.concept_model.word_concept_counts.keys(),
                           key=lambda w: self.concept_model.entropy_word(w))
            else:
                break
            # simulate asking
            ig_estimate = self.concept_model.entropy_word(word) - 0.1
            total_reward += ig_estimate
            # update state (crude: move towards identity)
            aligned_state = self._align_vector_to_emb_dim(state_vec)
            aligned_identity = self._align_vector_to_emb_dim(self.identity_manager.identity)

            state_vec = 0.9 * aligned_state + 0.1 * aligned_identity
            # normalize
            if np.linalg.norm(state_vec) > 0:
                state_vec = state_vec / np.linalg.norm(state_vec)
        print(f"   Simulated {num_turns}‑turn conversation, total IG = {total_reward:.3f} (throttle={throttle:.2f})")

    def behavioral_rule_discovery(self):
        """16. Dream about behavioral rules and propose candidates."""
        if not self.config_manager.get('idle_reasoner.enable_behavioral_rule_discovery', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Behavioral Rule Discovery (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧠 [Behavioral Rule Discovery] Dreaming about new rules...")
        candidates = self.behavioral_learner.propose_new_rules(self)
        if candidates:
            for rule in candidates:
                print(f"   ➕ Proposed new rule: {rule.rule_type} - {rule.condition} (confidence {rule.confidence:.2f})")
        else:
            print("   No new rule candidates found.")
        self.behavioral_learner.decay_confidences()

    def generate_logic_from_experience(self):
        filepath = None
        """Create new logic rules based on episodic patterns and write to ./logic."""
        if not self.config_manager.get('idle_reasoner.enable_logic_inheritance', True):
            return
        # Generate from highly confident behavioral rules that are not yet conscience beliefs
        candidates = []
        for rule in self.behavioral_learner.rules.values():
            if rule.confirmed and rule.confidence > 0.9 and rule.rule_id not in [b.belief_id for b in self.conscience_manager.beliefs.values()]:
                candidates.append(rule)
        if candidates:
            rule = max(candidates, key=lambda r: r.confidence)
            logic_rule = {
                'description': f"Behavioral rule: {rule.condition} -> {rule.action}",
                'condition': rule.condition,
                'action': rule.action,
                'ethical': True if rule.rule_type == 'identity_alias' else False,
                'source': 'behavioral_learner'
            }
            filepath = self.logic_inheritance.generate_logic_file(logic_rule)
            self.logic_inheritance.generate_doubt(logic_rule['logic_id'])
        if filepath is not None:
            print(f"📜 [Logic Generation] New logic rule written to {filepath}, doubt queued.")
        else:
            print("📜 [Logic Generation] No new rule candidates found.")

    def contextual_sentiment_analysis(self):
        """17. Deep analysis of feedback corpus – emergent sentiment learning."""
        if not self.config_manager.get('idle_reasoner.enable_contextual_sentiment_analysis', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Contextual Sentiment Analysis (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("📊 [Contextual Sentiment Analysis] Mining feedback corpus for emergent patterns...")
        # No explicit analysis needed; the corpus updates models incrementally.
        # We just print a summary of confident contexts.
        confident_contexts = 0
        for ctx, models in self.reasoner.feedback_corpus.context_models.items():
            confidence = self.reasoner.feedback_corpus.get_confidence(ctx)
            if confidence >= ContextualSentimentCorpus.CONFIDENCE_THRESHOLD:
                confident_contexts += 1
        total_entries = len(self.reasoner.feedback_corpus.entries)
        print(f"   Feedback entries: {total_entries}, confident contexts: {confident_contexts}.")

    def autonomous_code_generation_and_testing(self):
        """
        Idle task: select promising logic candidates (high‑confidence behavioral rules,
        unresolved hypotheses, new state definitions) and generate + shadow‑test code.
        Does NOT integrate; only creates doubts for user confirmation.
        """
        if not self.config_manager.get('idle_reasoner.enable_logic_inheritance', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping autonomous code generation (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🤖 [Autonomous Code Generation] Selecting candidates...")
        candidates = []

        # 1. High‑confidence behavioral rules that are confirmed but not yet coded
        for rule in self.behavioral_learner.rules.values():
            if rule.confirmed and rule.confidence > 0.85:
                if rule.rule_id not in self.logic_inheritance.generated_code_paths:
                    candidates.append(("behavioral", rule))

        # 2. Synonym hypotheses with strong similarity (>0.9) not yet asked
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') == 'synonym' and hyp.get('similarity', 0) > 0.9:
                if not hyp.get('asked', False):
                    # Use a unique logic_id based on word and candidate
                    logic_id = f"synonym_{hyp['word']}_{hyp['candidate']}"
                    if logic_id not in self.logic_inheritance.generated_code_paths:
                        candidates.append(("semantic", hyp, logic_id))

        # 3. Recently labeled states (if they have a stored embedding)
        if self.current_state_labels and self.current_state_embedding is not None:
            state_id = f"state_{self.current_state_labels[0]}"
            if state_id not in self.logic_inheritance.generated_code_paths:
                state_rule = {
                    "logic_id": state_id,
                    "rule_type": "state_definition",
                    "label": self.current_state_labels[0],
                    "prototype": self.current_state_embedding.tolist(),
                    "description": f"Recognize the '{self.current_state_labels[0]}' state"
                }
                candidates.append(("state", state_rule, state_id))

        if not candidates:
            print("   No candidates for code generation.")
            return

        # Process up to 2 candidates per idle cycle
        for item in candidates[:2]:
            if len(item) == 3:
                cand_type, cand, logic_id = item
            else:
                cand_type, cand = item
                logic_id = cand.rule_id if cand_type == "behavioral" else None

            rule_dict = None
            if cand_type == "behavioral":
                rule_dict = {
                    "rule_type": cand.rule_type,
                    "condition": cand.condition,
                    "action": cand.action,
                    "description": f"Behavioral rule: {cand.condition} -> {cand.action}"
                }
            elif cand_type == "semantic":
                word = cand.get('word', 'unknown')
                candidate = cand.get('candidate', 'unknown')
                similarity = cand.get('similarity', 0.0)
                rule_dict = {
                    "type": "synonym",
                    "word": word,
                    "candidate": candidate,
                    "similarity": similarity,
                    "description": f"Synonym hypothesis: {word} ≈ {candidate}"
                }
            elif cand_type == "state":
                rule_dict = cand  # already a dict

            if not rule_dict:
                continue

            # Step 1: Create a logic JSON file and get the assigned logic_id
            filepath_json = self.logic_inheritance.generate_logic_file(rule_dict)
            if not filepath_json:
                continue
            # The rule_dict now has 'logic_id' added by generate_logic_file
            logic_id = rule_dict['logic_id']

            # Step 2: Create a doubt (adds to pending_doubts)
            doubt = self.logic_inheritance.generate_doubt(logic_id)
            if not doubt:
                continue
            print(f"   ❓ Doubt created for {logic_id}")

            # Step 3: Generate code file (will use the rule from pending_doubts)
            filepath_py = self.logic_inheritance.generate_code_for_logic(logic_id)
            if not filepath_py:
                continue

            # Step 4: Shadow test
            test_result = self.logic_inheritance.shadow_test_generated_code(logic_id)
            if test_result["success"]:
                print(f"   ✅ Generated & tested: {logic_id}")
            else:
                print(f"   ⚠️ Generated {logic_id} but shadow test failed: {test_result['error'][:50]}")
                # Error already appended as comment in the .py file

    def deep_behavioral_pattern_mining(self):
        """18. Deep offline mining of conversational patterns."""
        if not self.config_manager.get('idle_reasoner.enable_behavioral_rule_discovery', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping deep pattern mining (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔍 [Deep Pattern Mining] Mining long‑range conversational patterns...")
        candidates = self.behavioral_learner.deep_pattern_mining(self)
        if candidates:
            for rule in candidates:
                self.behavioral_learner.add_candidate_rule(rule)
                print(f"   ➕ Discovered pattern: {rule.condition} -> {rule.action}")
        else:
            print("   No new deep patterns found.")

    def propagate_word_state_associations(self):
        """
        During idle, use semantic similarity to propagate word-state labels.
        If two words have high embedding similarity, they likely share labels.
        """
        if not self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True):  # reuse existing config
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping word-state propagation (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Word-State Propagation] Spreading labels via semantic similarity...")
        assoc = self.reasoner.word_state
        # Get all words that have embeddings
        all_words = list(self.semantic_learner.word_embeddings.keys())
        if len(all_words) < 5:
            return

        # Compute pairwise similarities for a sample of words (to avoid O(n^2))
        import random
        sample = random.sample(all_words, min(50, len(all_words)))
        similarities = {}
        for w1 in sample:
            for w2 in sample:
                if w1 >= w2:
                    continue
                sim = self.semantic_learner.similarity(w1, w2)
                if sim > 0.8:  # high similarity
                    similarities[(w1, w2)] = sim

        # Propagate labels between similar words
        propagated = 0
        for (w1, w2), sim in similarities.items():
            labels1 = set(assoc.get_labels(w1, threshold=0.3))
            labels2 = set(assoc.get_labels(w2, threshold=0.3))
            # For each label in w1, add to w2 with confidence scaled by similarity
            for lab in labels1:
                if lab not in labels2:
                    assoc.add_association(w2, lab, confidence=sim * 0.8, source="similarity")
                    propagated += 1
            for lab in labels2:
                if lab not in labels1:
                    assoc.add_association(w1, lab, confidence=sim * 0.8, source="similarity")
                    propagated += 1
        print(f"   Propagated {propagated} label associations.")

        # Check for conflicts (words with multiple labels of similar confidence)
        for word in all_words:
            conflicts = assoc.detect_conflicts(word)
            if conflicts:
                print(f"   ⚠️ Word '{word}' has ambiguous labels: {conflicts}. (Will ask user later)")
                # We could add a hypothesis to ask the user, but for now just note.

    def mine_phrase_relations(self):
        """
        During idle, find similar phrases in episodic memory and hypothesize relations.
        """
        if not self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping phrase relation mining (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🔍 [Phrase Relation Mining] Finding similar phrases...")
        # Collect recent user inputs from episodic memory
        episodes = self.episodic_memory.episodes[-200:]
        phrases = []
        for ep in episodes:
            user_input = ep.get('user_input', '')
            if user_input and len(user_input.split()) <= 10:  # phrase up to 10 words
                phrases.append(user_input)

        if len(phrases) < 5:
            return

        # Compute embeddings for these phrases (if possible)
        phrase_embs = {}
        for p in phrases:
            emb = self.semantic_learner.text_to_embedding(p, fallback_to_char=True)
            if emb is not None:
                phrase_embs[p] = emb

        # Compare pairwise (sample to avoid O(n^2))
        import random
        items = list(phrase_embs.items())
        if len(items) > 30:
            items = random.sample(items, 30)
        for i, (p1, emb1) in enumerate(items):
            for p2, emb2 in items[i+1:]:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2)+1e-12)
                if sim > 0.85:  # high similarity
                    # Check if we already have a relation or hypothesis
                    key = f"phrase_similar:{p1}:{p2}"
                    if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                        hyp = {
                            'key': key,
                            'type': 'phrase_similarity',
                            'phrase1': p1,
                            'phrase2': p2,
                            'similarity': sim,
                            'text': f"I notice that '{p1}' and '{p2}' seem similar. Are they related?",
                            'asked': False,
                        }
                        self.reasoner.hypotheses.append(hyp)
                        print(f"   ➕ Proposed phrase similarity: '{p1}' ↔ '{p2}' (sim={sim:.2f})")

    def converge_knowledge_graph(self):
        """
        Idle-time consolidation of the knowledge graph:
        - Infer transitive relations (is_a, means)
        - Merge nodes that are synonyms or highly similar
        - Prune weak edges
        - Detect central nodes for curiosity prioritization
        """
        if not hasattr(self.reasoner, 'knowledge_graph'):
            return
        kg = self.reasoner.knowledge_graph

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping knowledge graph convergence (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🕸️ [Knowledge Graph] Converging...")

        # 1. Infer transitive relations for is_a and means
        for rel in ['is_a', 'means']:
            inferred = kg.infer_transitive_relations(rel, threshold=0.6)
            for n1, n2, rt, conf in inferred:
                kg.add_edge(n1, n2, rt, conf)
                print(f"   Inferred {rt}: {kg.nodes[n1]['label']} -> {kg.nodes[n2]['label']} (conf={conf:.2f})")

        # 2. Merge synonym candidates (if two words have very high similarity in embeddings)
        #    This is a placeholder – you could use semantic similarity or co-occurrence.
        #    For now, we'll skip because merging requires careful handling.

        # 3. Prune weak edges (confidence < 0.1)
        kg.prune_weak_edges(threshold=0.1)
        print("   Pruned weak edges.")
        self.reasoner.run_constraint_propagation()
        # 4. Identify central nodes to influence curiosity (store in a separate structure)
        central = kg.get_central_nodes(top_k=10)
        self.central_graph_nodes = central
        print("   Top central nodes: " + ", ".join(f"{kg.nodes[nid]['label']}({deg})" for nid, deg in central))

        kg._save()

    def form_meta_concepts_from_concepts(self):
        """Call the concept model's meta‑concept formation."""
        if hasattr(self.concept_model, 'form_meta_concepts'):
            self.concept_model.form_meta_concepts()

    def perform_all_idle_tasks(self):
        """Orchestrate all idle‑time tasks, respecting configuration."""
        now = time.time()
        # avoid running all tasks every idle cycle; run a random subset
        if now - self._idle_tasks_last_run < 300:  # at most every 5 minutes
            return
        self._idle_tasks_last_run = now

        # Global resource check – if system is overloaded, skip everything
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"\n⚠️ [Idle Session] Skipped – resources too low (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%)")
            return

        print("\n" + "="*80)
        print("🧠 [AGI Idle Session] Running deep consolidation tasks...")
        print("="*80)
        tasks = [
            self.world_model_dreaming,
            self.active_memory_rehearsal,
            self.curiosity_landscape_optimization,
            self.concept_graph_compression,
            self.autonomous_hypothesis_planning,
            self.identity_stabilization,
            self.cross_modal_alignment,
            self.exploration_policy_distillation,
            self.cognitive_drive_evolution,
            self.counterfactual_reasoning,
            self.semantic_self_talk,
            self.state_space_topology_learning,
            self.knowledge_gap_frontier_detection,
            self.social_cognition_simulation,
            self.long_horizon_planning_rehearsal,
            self.behavioral_rule_discovery,
            self.generate_logic_from_experience,
            self.autonomous_code_generation_and_testing,
            self.contextual_sentiment_analysis,
            self.deep_behavioral_pattern_mining,
            self.propagate_word_state_associations,
            self.mine_phrase_relations,
            self._metabolic_maintenance,
            self.converge_knowledge_graph,
            self.train_language_model,
            self.propose_lm_improvement,
            self.reasoner.run_symbolic_inference,
            self.form_meta_concepts_from_concepts,
            self.build_world_model_transitions,            
        ]
        for task in tasks:
            try:
                task()
            except Exception:
                print(f"   [ERROR in {task.__name__}]")
                print(traceback.format_exc())
            # Delay between tasks – longer when throttled
            time.sleep(max(0.05, 0.2 * throttle))

        print("="*80 + "\n")

    def train_language_model(self):
        """Train the neural language model on all available textual data."""
        if not self.lm_config.get('enabled', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping language model training (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        # Collect training sentences from episodic memory (user inputs and system responses)
        sentences = []
        for ep in self.episodic_memory.episodes[-500:]:
            user = ep.get('user_input', '').strip()
            resp = ep.get('system_response', '').strip()
            if user and len(user.split()) <= 30:
                sentences.append(user)
            if resp and len(resp.split()) <= 30 and '?' not in resp:
                sentences.append(resp)

        # Add synthetic sentences from dreaming (if any)
        if hasattr(self, 'world_model') and self.world_model:
            # we don't have a store, but could use last few from dreaming task
            pass

        # Add definitions
        for word, def_text in self.reasoner.definitions.items():
            sentences.append(def_text)

        if len(sentences) < 10:
            return

        # Shuffle and take a batch
        np.random.shuffle(sentences)
        batch = sentences[:min(self.lm_config['train_batch_size'], len(sentences))]

        total_loss = 0.0
        lr = self.language_model.learning_rate * throttle  # scale by throttle

        # Add new words to vocabulary
        for sent in batch:
            for word in sent.split():
                if word not in self.language_model.word_to_idx:
                    self.language_model.add_word_to_vocab(word)

        # Train
        for sent in batch:
            try:
                loss = self.language_model.train_on_sentence(sent, lr=lr)
                total_loss += loss
            except Exception as e:
                print(f"   [LM] Error training on '{sent}': {e}")

        avg_loss = total_loss / len(batch)
        print(f"   [Language Model] Trained on {len(batch)} sentences, avg loss {avg_loss:.4f}, vocab size {self.language_model.vocab_size}, total steps {self.language_model.steps}")

    def _build_lm_validation_set(self, size=50):
        """Create a validation set from recent episodes, excluding training data."""
        sentences = []
        for ep in self.episodic_memory.episodes[-500:]:
            user = ep.get('user_input', '').strip()
            resp = ep.get('system_response', '').strip()
            if user and len(user.split()) <= 30:
                sentences.append(user)
            if resp and len(resp.split()) <= 30 and '?' not in resp:
                sentences.append(resp)
        # Shuffle and take last `size` as validation (assuming recent ones not in train)
        np.random.shuffle(sentences)
        return sentences[-size:] if len(sentences) >= size else sentences

    def evaluate_language_model(self, model=None, validation_sentences=None):
        """Compute average loss of a language model on validation sentences."""
        if model is None:
            model = self.language_model
        if validation_sentences is None:
            validation_sentences = self._build_lm_validation_set(30)
        total_loss = 0.0
        count = 0
        for sent in validation_sentences:
            try:
                ids = model.sentence_to_ids(sent)
                if len(ids) < 2:
                    continue
                input_ids = ids[:-1]
                target_ids = ids[1:]
                loss = model.loss(input_ids, target_ids)
                total_loss += loss
                count += 1
            except Exception:
                continue
        return total_loss / count if count > 0 else float('inf')
    
    def propose_lm_improvement(self):
        """
        When the language model has accumulated enough training steps,
        propose several new architectures, train each briefly, compare validation loss,
        and replace the current model if a better one is found.
        """
        if not self.lm_config.get('enabled', True):
            return
        if self.language_model.steps < self.lm_config.get('improvement_interval', 5000):
            return

        if self.language_model.steps % self.lm_config['improvement_interval'] != 0:
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping LM improvement (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🧬 [LM Improvement] Exploring new architectures...")

        # 1. Build validation set and evaluate current model
        val_sentences = self._build_lm_validation_set(50)
        current_loss = self.evaluate_language_model(self.language_model, val_sentences)
        print(f"   Current model validation loss: {current_loss:.4f}")

        # 2. Define candidate architectures (variations)
        candidates = []
        base_hidden = self.language_model.hidden_dim
        base_layers = getattr(self.language_model, 'num_layers', 1)
        base_cell = getattr(self.language_model, 'cell_type', 'lstm')

        # Try doubling hidden size
        candidates.append({
            'hidden_dim': base_hidden * 2,
            'num_layers': base_layers,
            'cell_type': base_cell,
            'learning_rate': self.language_model.learning_rate,
            'description': f"double hidden ({base_hidden}→{base_hidden*2})"
        })
        # Try adding a layer
        candidates.append({
            'hidden_dim': base_hidden,
            'num_layers': base_layers + 1,
            'cell_type': base_cell,
            'learning_rate': self.language_model.learning_rate * 0.5,  # lower LR for deeper net
            'description': f"add layer ({base_layers}→{base_layers+1})"
        })
        # Try switching cell type (if LSTM -> GRU, else LSTM)
        other_cell = 'gru' if base_cell == 'lstm' else 'lstm'
        candidates.append({
            'hidden_dim': base_hidden,
            'num_layers': base_layers,
            'cell_type': other_cell,
            'learning_rate': self.language_model.learning_rate,
            'description': f"switch to {other_cell}"
        })
        # Try smaller hidden but more layers (e.g., half hidden, double layers)
        if base_hidden > 32:
            candidates.append({
                'hidden_dim': base_hidden // 2,
                'num_layers': base_layers * 2,
                'cell_type': base_cell,
                'learning_rate': self.language_model.learning_rate * 0.5,
                'description': f"deeper+smaller ({base_hidden//2} hidden, {base_layers*2} layers)"
            })

        best_model = None
        best_loss = current_loss
        best_logic_id = None
        best_rule = None

        # 3. For each candidate, generate, train briefly, evaluate
        for cand in candidates:
            print(f"   Trying: {cand['description']}")
            logic_id = f"lm_improvement_{int(time.time())}_{np.random.randint(1000)}"
            rule = {
                'logic_id': logic_id,
                'type': 'language_model',
                'description': cand['description'],
                'hidden_dim': cand['hidden_dim'],
                'num_layers': cand['num_layers'],
                'cell_type': cand['cell_type'],
                'emb_dim': self.language_model.emb_dim,
                'learning_rate': cand['learning_rate'],
                'max_vocab': self.language_model.max_vocab,
                'source': 'propose_lm_improvement'
            }
            filepath = self.logic_inheritance.generate_logic_file(rule)
            if not filepath:
                continue

            code = self.code_generator.generate(rule)
            if not code:
                continue
            code_path = self.code_generator.save_code_file(logic_id, code)

            # Dynamically import
            spec = importlib.util.spec_from_file_location(f"lm_cand_{logic_id}", code_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"      Error loading: {e}")
                continue

            new_model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, LanguageModel) and attr != LanguageModel:
                    new_model_class = attr
                    break
            if new_model_class is None:
                continue

            new_model = new_model_class()

            # Copy vocabulary
            for word in self.language_model.word_to_idx:
                if word not in new_model.word_to_idx and word not in ('<PAD>','<BOS>','<EOS>','<UNK>'):
                    new_model.add_word_to_vocab(word)

            # Quick training (few sentences)
            train_sentences = []
            for ep in self.episodic_memory.episodes[-200:]:
                user = ep.get('user_input', '').strip()
                if user and len(user.split()) <= 30:
                    train_sentences.append(user)
            np.random.shuffle(train_sentences)
            for sent in train_sentences[:20]:
                try:
                    new_model.train_on_sentence(sent, lr=new_model.learning_rate * 0.1)
                except Exception:
                    continue

            new_loss = self.evaluate_language_model(new_model, val_sentences)
            print(f"      Validation loss: {new_loss:.4f}")

            if new_loss < best_loss * 0.95:  # at least 5% better
                best_loss = new_loss
                best_model = new_model
                best_logic_id = logic_id
                best_rule = rule

        # 4. If a better model found, replace current
        if best_model is not None:
            print(f"   ✅ Best model ({best_rule['description']}) outperforms current. Replacing language model.")
            self.language_model = best_model
            self.language_model.save(os.path.join(STATE_DIR, 'language_model.pkl'))
            self.logic_inheritance.confirm_logic(best_logic_id, is_correct=True)
        else:
            print(f"   No better model found. Keeping current.")
            # Still create a doubt for each candidate? Maybe only for the closest one.
            # For simplicity, we'll just note.
                
    def idle_reflection(self) -> str:
        """Autonomous introspection – now includes current state label if known."""
        if not self.word_history:
            return "I have no knowledge yet. I wonder what the world is like?"

        total_words = len(self.word_history)
        avg_entropy = np.mean([self.concept_model.entropy_word(w) for w in self.word_history])
        total_concepts = len(self.concept_model.concept_counts)

        # Most curious word
        word_scores = {}
        for w in self.word_history:
            last = self.word_history[w][-1]
            factors = last.get('curiosity_factors')
            if factors:
                farr = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))
                word_scores[w] = self.meta_learner.predict_curiosity(farr)
        most_curious = max(word_scores, key=word_scores.get, default=None)

        hypothesis = ""
        if most_curious:
            hypothesis = self.reasoner.generate_hypothesis(most_curious, {})

        thought = (
            f"[Idle Meta-Thought] I know {total_words} words and have formed {total_concepts} latent concepts. "
            f"My average uncertainty is {avg_entropy:.3f}. "
            f"I have accumulated {self.reward_tracker.total_reward:.2f} intrinsic reward so far. "
        )

        # ---- Mention current state if we have a recent factor vector ----
        num_states = len(self.state_manager.states)
        thought += f"I have learned {num_states} distinct interaction states. "

        if most_curious:
            thought += (
                f"The word I'm most curious about is '{most_curious}' (score={word_scores[most_curious]:.3f}). "
                f"{hypothesis} "
                f"I should ask about it to reduce my GP uncertainty and Bayesian surprise."
            )
        else:
            thought += "I have no strong curiosities at the moment."

        if np.random.rand() < 0.3:
            weights = self.meta_learner.get_weights()
            top_factor = max(weights, key=lambda k: abs(weights[k]))
            thought += f" I've learned that '{top_factor}' is currently the most informative curiosity factor."

        return thought
# ------------------------------------------------------------
# Action Generator – produces system responsesself.propagate_word_state_associations,
# ------------------------------------------------------------
class ActionGenerator:
    """
    Generates natural language responses based on:
    - Current curiosity (which word to ask about)
    - Conflict resolution prompts
    - Idle thoughts
    - Semantic similarity for definitions
    - Synonym/category hypotheses from the reasoning engine
    - High-trust spontaneous language attempts

    Now config‑aware and uses reasoner for semantic queries.
    """

    def __init__(self, semantic_learner: SemanticLearner,
                 syntax_learner: SyntaxLearner,
                 reasoner: ReasoningEngine,
                 engine: 'CuriosityEngine',  # added to access trust factor and feedback state
                 config_manager: Optional['ConfigManager'] = None):
        self.semantic = semantic_learner
        self.syntax = syntax_learner
        self.reasoner = reasoner
        self.engine = engine
        self.config_manager = config_manager
        self.semantic_learner = semantic_learner
        self.guilt_tracker = getattr(engine, 'guilt_tracker', None)

    def _matches_conscience_condition(self, belief: ConscienceBelief, user_input: str, state: Dict) -> bool:
        """Check if the current context matches the condition of a conscience belief."""
        cond = belief.condition
        if belief.rule_type == 'identity_alias':
            alias = cond.get('alias')
            return alias in user_input.lower()
        for k, v in cond.items():
            if k == 'keyword' and isinstance(v, str) and v in user_input.lower():
                return True
            if k == 'tokens' and isinstance(v, list):
                tokens = self.engine._tokenize(user_input)
                if any(t in v for t in tokens):
                    return True
        return False

    def generate_guilt_expression(self, guilt_level: float) -> Optional[str]:
        """Return a sentence expressing guilt/unease if guilt is significant."""
        if guilt_level < 0.2:
            return None
        templates = [
            "I feel uneasy about that.",
            "I'm not sure I did the right thing.",
            "I think I made a mistake earlier.",
            "I feel guilty about that.",
            "I'm sorry if that was wrong.",
            "I feel conflicted about my response.",
        ]
        base = np.random.choice(templates)
        # Use identity alias if available
        if self.engine.identity_aliases:
            alias = next(iter(self.engine.identity_aliases))
            base = base.replace("I", alias.capitalize()).replace("I", alias)
        return base  

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def generate_question(self, target: str, uncertainty: float = None) -> str:
        """Produce a question about an unknown word."""
        base = f"What does '{target}' mean?"
        if uncertainty is not None:
            return f"{base} (My uncertainty about this word is {uncertainty:.2f}.)"
        return base

    def generate_definition(self, word: str) -> str:
        """If we have learned something, generate a definition."""
        # find similar known words
        sim_words = []
        for w in self.semantic.word_embeddings:
            if w != word:
                sim = self.semantic.similarity(word, w)
                if sim > 0.6:
                    sim_words.append((w, sim))
        sim_words.sort(key=lambda x: -x[1])
        if sim_words:
            best = sim_words[0][0]
            return f"I think '{word}' might be similar to '{best}'."
        else:
            return f"I don't know what '{word}' means yet."

    def generate_synonym_question(self, word: str, candidate: str) -> str:
        """Generate a question about a potential synonym relationship."""
        return f"Is '{word}' similar in meaning to '{candidate}'?"

    def generate_trust_utterance(self, trust_value: float) -> str:
        """
        Generate a spontaneous, possibly broken-English utterance when trust is high.
        Now also used when emotional state is negative – but with different templates.
        Templates are not pre‑defined as positive/negative; the system learns
        appropriateness through feedback.
        """
        # Use threat level to decide tone
        threat = self.engine._last_factor_dict.get('threat', 0.0) if self.engine._last_factor_dict else 0.0
        energy = self.engine._last_factor_dict.get('energy', 0.8) if self.engine._last_factor_dict else 0.8
        comprehension = self.engine._last_factor_dict.get('comprehension', 0.5) if self.engine._last_factor_dict else 0.5

        # Base templates – all neutral; sentiment is learned via feedback
        if self.reasoner.definitions:
            word = np.random.choice(list(self.reasoner.definitions.keys()))
            def_text = self.reasoner.definitions[word][:20]
            templates = [
                f"I think {word} means {def_text}.",
                f"{word} is like {def_text}, I believe.",
                f"I guess {word} = {def_text}.",
                f"{word} might mean {def_text}.",
            ]
        else:
            if self.reasoner.co_occurrence:
                word_counts = Counter()
                for w, cnts in self.reasoner.co_occurrence.items():
                    word_counts[w] = sum(cnts.values())
                if word_counts:
                    word = word_counts.most_common(1)[0][0]
                    templates = [
                        f"I see '{word}' often.",
                        f"'{word}' appears a lot.",
                        f"Maybe '{word}' is important.",
                    ]
                else:
                    templates = ["I'm learning..."]
            else:
                templates = ["I'm learning..."]

        utterance = np.random.choice(templates)

        # Emotional modulation – add a brief state expression (learned from feedback)
        # This is purely emergent: the system will learn which expressions get positive feedback.
        if threat > 0.6:
            state_phrases = [
                " I'm uncertain about this.",
                " This is difficult.",
                " I feel unsure.",
                " I might be wrong.",
            ]
            utterance += np.random.choice(state_phrases)
        elif comprehension < 0.3:
            state_phrases = [
                " I'm confused.",
                " I don't follow.",
                " I need clarification.",
            ]
            utterance += np.random.choice(state_phrases)
        elif energy < 0.3:
            state_phrases = [
                " I'm tired.",
                " This is draining.",
                " I need a break.",
            ]
            utterance += np.random.choice(state_phrases)
        elif trust_value > 0.7:
            state_phrases = [
                " I feel confident!",
                " This is going well.",
                " I think I understand.",
            ]
            utterance += np.random.choice(state_phrases)

        # Replace "I" with learned self-alias if available
        if self.engine.identity_aliases:
            alias = next(iter(self.engine.identity_aliases))
            utterance = utterance.replace("I", alias.capitalize()).replace("I", alias)
        return utterance

    def generate_chat_response(self, user_input: str, state: Dict) -> str:
        """
        Generate a conversational, non‑interrogative response when chat mode is active.
        PRIORITY 1: generate a base reply addressing the user input (definitions, patterns, trigram).
        PRIORITY 2: modulate that base reply with the current internal state.
        PRIORITY 3: if the base reply is generic, inject a state‑based comment.
        """
        user_input_lower = user_input.lower().strip()
        factor_dict = state.get('factor_vector', {})
        if not isinstance(factor_dict, dict):
            factor_dict = {}

        # -------------------------------------------------------------
        # 1. Generate a base reply that addresses the user input
        # -------------------------------------------------------------
        base_reply = None

        # 1.1 Use identity alias if available
        alias = None
        if self.engine.identity_aliases:
            alias = next(iter(self.engine.identity_aliases))

        # 1.2 Check learned behavioral patterns (conversational_pattern)
        for rule in self.engine.behavioral_learner.rules.values():
            if rule.rule_type == 'conversational_pattern' and rule.confidence > 0.6:
                cond = rule.condition
                if 'user_input_pattern' in cond and user_input_lower == cond['user_input_pattern']:
                    if 'response_template' in rule.action:
                        self.engine.behavioral_learner.record_usage(rule.rule_id)
                        base_reply = rule.action['response_template']
                        break
                elif 'question_pattern' in cond and user_input_lower.rstrip('?') == cond['question_pattern']:
                    if 'answer_template' in rule.action:
                        self.engine.behavioral_learner.record_usage(rule.rule_id)
                        base_reply = rule.action['answer_template']
                        break
                elif 'expected_user_input' in cond and user_input_lower == cond['expected_user_input']:
                    self.engine.behavioral_learner.record_usage(rule.rule_id)
                    base_reply = "I see you're continuing the conversation."
                    break

        # 1.3 Try to understand the user's input using known definitions and relations
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for phrase in self.reasoner.phrase_definitions:
                if phrase in user_input_lower:
                    def_text = self.reasoner.phrase_definitions[phrase]
                    base_reply = f"Ah, '{phrase}' means {def_text}. That's interesting."
                    break

        # 1.4 Use word-state labels to provide contextual info
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for word in tokens:
                labels = self.reasoner.word_state.get_labels(word, threshold=0.6)
                if labels:
                    label_str = ', '.join(labels)
                    base_reply = f"I know '{word}' is associated with: {label_str}. Tell me more about it."
                    break

        # 1.5 Check if user input is similar to known phrases
        if base_reply is None:
            input_emb = self.semantic_learner.text_to_embedding(user_input, fallback_to_char=True)
            if input_emb is not None:
                similar = self.reasoner.find_similar_phrases(input_emb, threshold=0.8)
                if similar:
                    best_phrase, sim = similar[0]
                    base_reply = f"This reminds me of when you said '{best_phrase}'. Tell me more."

        # 1.6 Check for words with definitions
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for word in tokens:
                if word in self.reasoner.definitions:
                    def_text = self.reasoner.definitions[word][:50]
                    relations = self.reasoner.phrase_relations.get(word, [])
                    rel_strs = []
                    for rel_type, obj, conf in relations[:2]:
                        if conf > 0.7:
                            rel_strs.append(f"{rel_type} {obj}")
                    if rel_strs:
                        base_reply = f"I know that {word} is {def_text}. Also, it is {', '.join(rel_strs)}."
                    else:
                        base_reply = f"I understand '{word}' as {def_text}. Tell me more."
                    break

        # 1.7 Fall back to trigram model from episodic memory
        if base_reply is None:
            trigram = defaultdict(Counter)
            episodes = self.engine.episodic_memory.episodes[-50:]
            for ep in episodes:
                sent = ep.get('user_input', '')
                tokens_ep = self.engine._tokenize(sent)
                if len(tokens_ep) < 3 or not self.engine._all_tokens_known(tokens_ep):
                    continue
                for i in range(len(tokens_ep)-2):
                    ctx = tuple(tokens_ep[i:i+2])
                    nxt = tokens_ep[i+2]
                    trigram[ctx][nxt] += 1

            known_user_tokens = [t for t in tokens if t in self.semantic.word_embeddings]
            if len(known_user_tokens) >= 2 and trigram:
                seed = tuple(known_user_tokens[-2:])
                if seed in trigram:
                    next_words = list(trigram[seed].keys())
                    probs = np.array(list(trigram[seed].values()))
                    probs = probs / probs.sum()
                    next_w = np.random.choice(next_words, p=probs)
                    base_reply = f"{' '.join(known_user_tokens)} {next_w} ... what do you think?"
                else:
                    recent_ep = episodes[-1] if episodes else None
                    if recent_ep:
                        topic = recent_ep.get('user_input', '').split()[:2]
                        topic = ' '.join(topic)
                        base_reply = f"Tell me more about {topic}."

        # 1.8 Use learned structural patterns
        if base_reply is None:
            freq_patterns = self.reasoner.structural_patterns.get_frequent_patterns(min_freq=2)
            if freq_patterns and np.random.rand() < 0.3:
                pattern = np.random.choice(freq_patterns)
                generated = self.reasoner.structural_patterns.generate_from_pattern(pattern)
                if generated:
                    base_reply = f"I've noticed a pattern: {generated}"

        # 1.8.5 Use language model to generate a response (if available and trained)
        if base_reply is None and hasattr(self.engine, 'language_model'):
            lm = self.engine.language_model
            # Only use if sufficiently trained
            if lm.steps >= self.engine.lm_config.get('min_steps_for_use', 100):
                try:
                    generated = lm.generate(user_input, max_len=15, temperature=0.8)
                    if generated and len(generated.split()) > 1:
                        base_reply = generated
                except Exception as e:
                    print(f"   [LM] Generation error: {e}")

        # 1.9 Ultimate generic fallback
        if base_reply is None:
            if known_user_tokens:
                word = known_user_tokens[0]
                if word in self.reasoner.definitions:
                    base_reply = f"I recall that {word} means {self.reasoner.definitions[word][:50]}."
                else:
                    base_reply = f"What do you mean by '{word}'?"
            else:
                base_reply = "That's interesting. Can you tell me more?"

        # -------------------------------------------------------------
        # 2. Modulate the base reply with the current internal state
        # -------------------------------------------------------------
        modulated = self._modulate_with_state(base_reply, factor_dict)

        # If the base reply is the generic "That's interesting...", we can optionally
        # inject a more state‑specific comment instead.
        if "That's interesting" in base_reply and factor_dict:
            # Determine dominant emotion
            emotions = ['curiosity', 'threat', 'trust', 'comprehension', 'energy']
            dominant = max(emotions, key=lambda e: factor_dict.get(e, 0.5))
            val = factor_dict.get(dominant, 0.5)

            # Pick a word that is relevant to the current state
            candidate_word = None
            if self.engine.current_state_labels and self.engine.current_state_embedding is not None:
                state_emb = self.engine.current_state_embedding
                best_word = None
                best_sim = -1
                for w, emb in self.semantic_learner.word_embeddings.items():
                    if w in self.reasoner.definitions:
                        continue
                    sim = np.dot(state_emb, emb) / (np.linalg.norm(state_emb)*np.linalg.norm(emb)+1e-12)
                    if sim > best_sim:
                        best_sim = sim
                        best_word = w
                if best_word and best_sim > 0.6:
                    candidate_word = best_word
            if not candidate_word and self.engine.word_history:
                candidate_word = np.random.choice(list(self.engine.word_history.keys()))

            if candidate_word:
                if dominant == 'curiosity' and val > 0.6:
                    modulated = f"I'm really curious about '{candidate_word}'. {base_reply}"
                elif dominant == 'threat' and val > 0.5:
                    modulated = f"I'm feeling a bit uncertain right now. {base_reply}"
                elif dominant == 'trust' and val > 0.7:
                    modulated = f"I feel confident. {base_reply}"
                elif dominant == 'comprehension' and val < 0.3:
                    modulated = f"I'm having trouble understanding. {base_reply}"
                elif dominant == 'energy' and val < 0.3:
                    modulated = f"I'm a bit tired, but {base_reply[0].lower() + base_reply[1:]}"
                else:
                    modulated = f"I've been thinking about '{candidate_word}'. {base_reply}"
            else:
                modulated = self._modulate_with_state(base_reply, factor_dict)

        return modulated

    def _modulate_with_state(self, response: str, factor_dict: Dict) -> str:
        """
        Optionally prepend a brief emotional phrase based on the current state.
        This makes the response reflect the internal state without changing its core content.
        """
        # Only modulate with some probability, to avoid overdoing it
        if np.random.rand() > 0.4:
            return response

        # Extract emotional signals
        threat = factor_dict.get('threat', 0.0)
        trust = factor_dict.get('trust', 0.5)
        curiosity = factor_dict.get('curiosity', 0.5)
        comprehension = factor_dict.get('comprehension', 0.5)
        energy = factor_dict.get('energy', 0.8)

        prefixes = []
        if threat > 0.7:
            prefixes.append("Hmm, I'm a bit uneasy, but")
        elif threat > 0.4:
            prefixes.append("I'm somewhat uncertain, but")
        elif trust > 0.7:
            prefixes.append("I feel confident, and")
        elif curiosity > 0.7:
            prefixes.append("Curious as I am,")
        elif comprehension < 0.3:
            prefixes.append("I'm a bit lost, but")
        elif energy < 0.3:
            prefixes.append("Feeling a little low on energy, but")

        if prefixes:
            prefix = np.random.choice(prefixes)
            # Capitalize first letter of response if needed
            if response and response[0].islower():
                response = response[0].upper() + response[1:]
            return f"{prefix} {response}"
        return response
        
    def generate_reply(self, user_input: str, state: Dict) -> str:
        """
        Main entry point: decide what to say.
        Priority:
          0. Chat mode → conversational reply (if active)
          1. Unresolved conflicts (handled in process_input)
          1.5. Pending sentence query (from sentence reasoning)
          2. Pending synonym/category/analogy hypotheses
          3. High curiosity words (unknown words)
          4. State awareness utterance
          5. High trust spontaneous attempt
          6. Chitchat
        """
        # --- Priority 0: Chat mode (overrides most other behaviours) ---
        if self.engine.chat_mode_active:
            return self.generate_chat_response(user_input, state)

        # --- Priority 0.5: Logic doubt (already handled in process_input, but keep as safeguard) ---
        doubt = self.engine.logic_inheritance.get_next_doubt()
        if doubt:
            rule = doubt['rule']
            description = rule.get('description', f"a logical rule")
            return f"I have a doubt about some logic I wrote: {description} Is this correct?"

        # --- Priority 0.75: Evaluate expressions in user input ---
        # If user input contains an operator and all symbols have known values,
        # try to evaluate it and return result.
        if self.engine.reasoner.operators:
            tokens = self.engine._tokenize(user_input)
            # crude check: if tokens contain an operator and all others are numbers or known symbols
            op_symbols = set(self.engine.reasoner.operators.keys())
            if any(t in op_symbols for t in tokens):
                # attempt to parse as expression – for now just simple arithmetic
                try:
                    # Very simple evaluation: if input looks like "2 + 2", evaluate
                    # (This is a placeholder; a real parser would use syntax trees)
                    result = eval(user_input, {"__builtins__": None}, {})
                    return f"The result is {result}."
                except:
                    pass

        # --- Priority 0.7: Conscience relevance ---
        for belief in self.engine.conscience_manager.get_active_conscience():
            if self._matches_conscience_condition(belief, user_input, state):
                return f"My conscience suggests that {belief.description}. Is that appropriate?"

        # ---- Express guilt if significant ----
        if self.guilt_tracker and self.guilt_tracker.guilt_level > 0.3 and np.random.rand() < 0.4:
            guilt_expr = self.generate_guilt_expression(self.guilt_tracker.guilt_level)
            if guilt_expr:
                return guilt_expr

        # --- Priority 1.5: Pending sentence query ---
        if self.engine.pending_sentence_query:
            query = self.engine.pending_sentence_query
            self.engine.pending_sentence_query = None   # consume
            return query

        # --- Priority 2: Pending hypotheses (synonym, category, analogy, inference) ---
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') in ('synonym', 'category', 'analogy', 'inference') and not hyp.get('asked', False):
                hyp['asked'] = True
                hyp['last_used'] = time.time()
                hyp['recent_usage'] = hyp.get('recent_usage', 0.0) * 0.99 + 0.1
                return hyp['text']

        # --- Priority 2.5: Parallel reasoning thoughts (if multiple chains available) ---
        if 'token_analyses' in state and state['token_analyses']:
            # pick the most curious token
            token_analyses = state['token_analyses']
            high_curiosity = [ta for ta in token_analyses if ta.get('curiosity_score', 0) > 0.4]
            if high_curiosity:
                token = high_curiosity[0]['token']
                # generate parallel thoughts
                thoughts = self.engine.parallel_reasoner.generate_thoughts(token, state)
                # present the most confident one
                best = max(thoughts, key=lambda t: t['confidence'])
                return best['hypothesis']

        # --- Priority 3: High curiosity words (unknown words) ---
        if 'token_analyses' in state:
            high_curiosity = [ta for ta in state['token_analyses'] if ta.get('curiosity_score', 0) > 0.5]
            if high_curiosity:
                target = high_curiosity[0]['token']
                unc = high_curiosity[0].get('gp_uncertainty')
                return self.generate_question(target, unc)

        # --- Priority 4: State awareness utterance (30% chance) ---
        if self.engine.current_state_labels and np.random.rand() < 0.3:
            # pick one of the active states to mention
            chosen_state = np.random.choice(self.engine.current_state_labels)
            return f"I feel like I'm in a '{chosen_state}' state right now."

        # --- Priority 5: Spontaneous emotional utterance (based on state) ---
        factor_vec = state.get('factor_vector', {})
        if not isinstance(factor_vec, dict):
            factor_vec = {}
        threat = factor_vec.get('threat', 0.0)
        trust = factor_vec.get('trust', 0.5)
        comprehension = factor_vec.get('comprehension', 0.5)
        energy = factor_vec.get('energy', 0.8)

        # Trigger in both positive and negative states, with different probabilities
        trigger_prob = 0.0
        if trust > 0.7 and not self.engine.expecting_feedback:
            trigger_prob = 0.4   # high trust, express confidence
        elif threat > 0.6:
            trigger_prob = 0.5   # high threat, express uncertainty
        elif comprehension < 0.3:
            trigger_prob = 0.6   # low comprehension, express confusion
        elif energy < 0.3:
            trigger_prob = 0.4   # low energy, express tiredness

        if np.random.rand() < trigger_prob:
            utterance = self.generate_trust_utterance(trust)  # now handles all emotional tones
            if self.engine.current_state_labels:
                # Prepend state information
                state_str = ', '.join(self.engine.current_state_labels)
                utterance = f"As I'm {state_str}, {utterance[0].lower()}{utterance[1:]}"
            self.engine.expecting_feedback = True
            self.engine.feedback_attempt_word = None
            self.engine.feedback_attempt_response = utterance
            self.engine.feedback_timestamp = time.time()
            return utterance

        # --- Priority 6: idle thought / chitchat ---
        return "I'm listening. Tell me more."
# ------------------------------------------------------------
# Idle Reasoner (background thread) – now with long-idle introspection
# ------------------------------------------------------------
class IdleReasoner(threading.Thread):
    def __init__(self, engine: CuriosityEngine,
                 short_idle_threshold: float = None,
                 long_idle_threshold: float = None):
        super().__init__(daemon=True)
        self.engine = engine
        # Read thresholds from config if available
        if engine.config_manager:
            self.short_thresh = engine.config_manager.get('idle_reasoner.short_idle_threshold', 8.0)
            self.long_thresh = engine.config_manager.get('idle_reasoner.long_idle_threshold', 600.0)
        else:
            self.short_thresh = short_idle_threshold if short_idle_threshold is not None else 8.0
            self.long_thresh = long_idle_threshold if long_idle_threshold is not None else 600.0
        self.running = True
        self.long_idle_done = False   # reset after interaction

    def update_parameters_from_config(self):
        """Reload thresholds from config (hot reload)."""
        if self.engine.config_manager:
            self.short_thresh = self.engine.config_manager.get('idle_reasoner.short_idle_threshold', 8.0)
            self.long_thresh = self.engine.config_manager.get('idle_reasoner.long_idle_threshold', 600.0)

    def run(self):
        while self.running:
            try:
                time.sleep(2)
                idle_time = time.time() - self.engine.last_interaction_time

                # Short idle → simple meta-thought
                if idle_time > self.short_thresh:
                    thought = self.engine.idle_reflection()
                    print(f"\n⏳ [Idle at {time.strftime('%H:%M:%S')}]\n{thought}\n")
                    time.sleep(self.short_thresh)

                # Long idle → deep state introspection AND all idle tasks
                if idle_time > self.long_thresh and not self.long_idle_done:
                    self.engine.perform_state_introspection()
                    self.engine.perform_all_idle_tasks()
                    try:
                        self.engine._save_state()
                    except Exception:
                        pass
                    self.long_idle_done = True
            except Exception:
                print("\n[IdleReasoner ERROR — Full Traceback]")
                print(traceback.format_exc())
                time.sleep(5)

    def reset_long_idle_flag(self):
        """Call when user interacts to allow a new long-idle cycle."""
        self.long_idle_done = False
# ------------------------------------------------------------
# Command-Line Interface
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print("🧠 AGI Curiosity Engine v0.9 - Information-Theoretic & Intrinsically Motivated")
    print("Integrated with backend.ci (Akkadian transliteration models)")
    print("State persists in ./curiosity_state/ - the system grows over time.")
    print("with: Gaussian Process uncertainty, Bayesian surprise, Intrinsic Reward.")
    print("Type your message or press Ctrl+C to exit.")
    print("=" * 80)

    engine = CuriosityEngine(load_state=True)
    idle_reasoner = IdleReasoner(engine, short_idle_threshold=8.0, long_idle_threshold=600.0)
    engine._idle_reasoner = idle_reasoner 
    idle_reasoner.start() 
    try:
        while True:
            try:
                user_input = input("\n🧠 AGI> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Shutting down.")
                    break

                result = engine.process_input(user_input)

                print("\n--- Curiosity Output ---")
                for line in result["curiosity_artifacts"]:
                    print(line)
                print("------------------------\n")
            except Exception:
                import traceback
                traceback.print_exc()
                print("Attempting to recover and continue...")
                # Optionally save emergency state
                try:
                    engine._save_state()
                except:
                    pass
                continue
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    finally:
        idle_reasoner.running = False

if __name__ == "__main__":
    main()
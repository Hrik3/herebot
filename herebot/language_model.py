from .importglobal import np,pickle
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

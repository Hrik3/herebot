from importglobal import json,time,os,traceback,io,importlib,redirect_stderr,redirect_stdout
from directory_handler import CODE_GEN_DIR, STATE_DIR
# ------------------------------------------------------------
# CodeGenerator – writes Python code from confirmed logic rules
# Enhanced with resource monitoring, error handling, and dynamic code generation
# ------------------------------------------------------------
class CodeGenerator:
    """
    Converts high-level logic rules into executable Python code.
    Uses safe templates; every generated file contains a `run()` function.
    Enhanced to be resource-aware and scalable, with built-in performance tracking.
    """

    # Common imports added to every generated file
    BASE_IMPORTS = '''import time
import logging
import psutil  # for resource monitoring (optional, gracefully degrades)
from functools import wraps
'''

    # Template for behavioral rules (identity aliases, pronoun mappings)
    BEHAVIORAL_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# Rule type: {rule_type}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}

# Setup module-level logger
logger = logging.getLogger(__name__)

def resource_monitor(func):
    """Decorator to track execution time and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            # Log if execution takes too long (>0.5s)
            if elapsed > 0.5:
                logger.warning(f"{{func.__name__}} took {{elapsed:.3f}}s")
            # Optionally check memory
            try:
                mem = psutil.Process().memory_percent()
                if mem > 10.0:  # >10% of system memory
                    logger.warning(f"{{func.__name__}} using {{mem:.1f}}% memory")
            except:
                pass
        return result
    return wrapper

@resource_monitor
def run(context: dict) -> str:
    """
    context: {{'user_input': str, 'tokens': list, 'state': dict, ...}}
    Returns a response or None.
    """
    user_input = context.get("user_input", "").lower()
    tokens = context.get("tokens", [])
    state = context.get("state", {{}})
    
    # Resource check: if system under heavy load, return early
    try:
        if psutil.cpu_percent(interval=None) > 80:
            logger.info("CPU load high, skipping expensive processing")
            return None
    except:
        pass

    # Condition: {condition}
    {condition_code}

    # Action: {action}
    {action_code}

    return response if 'response' in locals() else None
'''

    # Template for synonym / category hypotheses
    SEMANTIC_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# Semantic relation: {relation}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}

logger = logging.getLogger(__name__)

@resource_monitor
def run(word: str, semantic_learner) -> float:
    """
    Returns similarity confidence (0-1) between word and known concept.
    """
    {body}
    return 0.0
'''

    # Template for state definitions (persistent latent concepts)
    STATE_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# State label: {label}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}
import numpy as np

logger = logging.getLogger(__name__)

@resource_monitor
def run(factor_vector: list) -> bool:
    """
    Returns True if the current factor vector matches this state.
    """
    {body}
    return False
'''
    @classmethod
    def _generate_language_model(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        hidden_dim = rule.get('hidden_dim', 128)
        emb_dim = rule.get('emb_dim', 64)
        lr = rule.get('learning_rate', 0.01)
        max_vocab = rule.get('max_vocab', 10000)
        cell_type = rule.get('cell_type', 'lstm').lower()
        num_layers = rule.get('num_layers', 1)

        # Generate code with full implementation
        code = f'''# AUTO-GENERATED LANGUAGE MODEL
# Logic ID: {logic_id}
# Created: {timestamp}
# Version: 1.0
# Architecture: {cell_type.upper()} with {num_layers} layer(s), hidden={hidden_dim}, emb={emb_dim}

{base_imports}
import numpy as np

class ImprovedLSTMLanguageModel(LanguageModel):
    """
    Self-improved language model with {cell_type.upper()} cell, {num_layers} layer(s), hidden_dim={hidden_dim}.
    """
    def __init__(self):
        super().__init__()
        self.emb_dim = {emb_dim}
        self.hidden_dim = {hidden_dim}
        self.num_layers = {num_layers}
        self.cell_type = "{cell_type}"
        self.learning_rate = {lr}
        self.max_vocab = {max_vocab}

        self.word_to_idx = {{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}}
        self.idx_to_word = {{0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}}
        self.vocab_size = 4

        self.W_emb = np.random.randn(self.vocab_size, self.emb_dim) * 0.1

        # Initialize parameters for each layer
        self.W_ih = []  # input to hidden
        self.W_hh = []  # hidden to hidden
        self.b_h = []   # bias

        for layer in range(self.num_layers):
            if self.cell_type == 'lstm':
                # LSTM has 4 gates: input, forget, cell, output
                self.W_ih.append(np.random.randn(self.emb_dim if layer==0 else self.hidden_dim, 4 * self.hidden_dim) * 0.1)
                self.W_hh.append(np.random.randn(self.hidden_dim, 4 * self.hidden_dim) * 0.1)
                self.b_h.append(np.zeros(4 * self.hidden_dim))
            else:  # gru
                # GRU has 3 gates: reset, update, new
                self.W_ih.append(np.random.randn(self.emb_dim if layer==0 else self.hidden_dim, 3 * self.hidden_dim) * 0.1)
                self.W_hh.append(np.random.randn(self.hidden_dim, 3 * self.hidden_dim) * 0.1)
                self.b_h.append(np.zeros(3 * self.hidden_dim))

        self.W_out = np.random.randn(self.hidden_dim, self.vocab_size) * 0.1
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
        T = len(input_ids)
        batch_size = 1  # simplified for single sequence

        # Initialize states
        if prev_h is None:
            h = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
            if self.cell_type == 'lstm':
                c = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        else:
            h = prev_h
            c = prev_c if self.cell_type == 'lstm' else None

        logits = []
        for t in range(T):
            x = self.W_emb[input_ids[t]].reshape(1, -1)  # (1, emb_dim)

            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    # LSTM
                    gates = x @ self.W_ih[layer] + h[layer] @ self.W_hh[layer] + self.b_h[layer]
                    i, f, g, o = np.split(gates, 4, axis=1)
                    i = self._sigmoid(i)
                    f = self._sigmoid(f)
                    g = self._tanh(g)
                    o = self._sigmoid(o)
                    c[layer] = f * c[layer] + i * g
                    h[layer] = o * self._tanh(c[layer])
                else:
                    # GRU
                    gates = x @ self.W_ih[layer] + h[layer] @ self.W_hh[layer] + self.b_h[layer]
                    r, z, n = np.split(gates, 3, axis=1)
                    r = self._sigmoid(r)
                    z = self._sigmoid(z)
                    n = self._tanh(n)
                    h[layer] = (1 - z) * n + z * h[layer]

                x = h[layer]  # output of this layer becomes input to next

            logit = h[-1] @ self.W_out + self.b_out
            logits.append(logit[0])  # remove batch dim

        final_h = [h_i[0] for h_i in h]
        final_c = [c_i[0] for c_i in c] if self.cell_type == 'lstm' else None
        return np.array(logits), (final_h, final_c)

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
        # Forward pass with storage – simplified; for production you'd need full BPTT.
        # Here we'll just do a placeholder update.
        # In a real system you'd implement proper backprop.
        # For demonstration, we'll use a simple SGD update on output layer only.
        logits, (h_final, _) = self.forward(input_ids)
        loss = 0.0
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        for t in range(T):
            probs = self._softmax(logits[t])
            dlogit = probs.copy()
            dlogit[target_ids[t]] -= 1.0
            dW_out += np.outer(h_final[-1], dlogit)
            db_out += dlogit
            loss -= np.log(probs[target_ids[t]] + 1e-12)
        # Update output layer
        self.W_out -= lr * dW_out / T
        self.b_out -= lr * db_out / T
        self.steps += 1
        self.loss_history.append(loss / T)
        return loss / T

    def add_word_to_vocab(self, word):
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
            x = self.W_emb[last_id].reshape(1, -1)
            # Run through layers
            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    gates = x @ self.W_ih[layer] + (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim))) @ self.W_hh[layer] + self.b_h[layer]
                    i, f, g, o = np.split(gates, 4, axis=1)
                    i = self._sigmoid(i)
                    f = self._sigmoid(f)
                    g = self._tanh(g)
                    o = self._sigmoid(o)
                    c[layer] = f * (c[layer] if c is not None else np.zeros((1, self.hidden_dim))) + i * g
                    h[layer] = o * self._tanh(c[layer])
                else:
                    gates = x @ self.W_ih[layer] + (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim))) @ self.W_hh[layer] + self.b_h[layer]
                    r, z, n = np.split(gates, 3, axis=1)
                    r = self._sigmoid(r)
                    z = self._sigmoid(z)
                    n = self._tanh(n)
                    h[layer] = (1 - z) * n + z * (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim)))
                x = h[layer]
            logit = h[-1] @ self.W_out + self.b_out
            probs = self._softmax(logit[0] / temperature)
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
        data = {{
            'W_emb': self.W_emb,
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'steps': self.steps,
            'loss_history': self.loss_history,
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'cell_type': self.cell_type,
            'learning_rate': self.learning_rate,
            'max_vocab': self.max_vocab,
        }}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.W_emb = data['W_emb']
        self.W_ih = data['W_ih']
        self.W_hh = data['W_hh']
        self.b_h = data['b_h']
        self.W_out = data['W_out']
        self.b_out = data['b_out']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = data['vocab_size']
        self.steps = data['steps']
        self.loss_history = data['loss_history']
        self.emb_dim = data['emb_dim']
        self.hidden_dim = data['hidden_dim']
        self.num_layers = data['num_layers']
        self.cell_type = data['cell_type']
        self.learning_rate = data['learning_rate']
        self.max_vocab = data['max_vocab']
'''
        return code
    @classmethod
    def generate(cls, logic_rule: dict) -> str:
        """
        Dispatch to the appropriate template based on rule type.
        Enhanced with dynamic condition/action parsing.
        """
        rule_type = logic_rule.get("rule_type", logic_rule.get("type", "unknown"))
        logic_id = logic_rule.get("logic_id", f"gen_{int(time.time())}")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Common imports will be inserted by the template formatter
        base_imports = cls.BASE_IMPORTS

        if rule_type in ("identity_alias", "pronoun_mapping", "behavioral"):
            return cls._generate_behavioral(logic_rule, logic_id, timestamp, base_imports)
        elif rule_type in ("synonym", "category", "analogy"):
            return cls._generate_semantic(logic_rule, logic_id, timestamp, base_imports)
        elif rule_type == "state_definition":
            return cls._generate_state(logic_rule, logic_id, timestamp, base_imports)
        elif rule_type == "language_model":
            return cls._generate_language_model(logic_rule, logic_id, timestamp, base_imports)
        else:
            # Fallback: generic template with minimal functionality
            return f"""# No template for rule type {rule_type}
{base_imports}

def run(context: dict) -> str:
    return None
"""

    @classmethod
    def _generate_behavioral(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        condition = rule.get("condition", {})
        action = rule.get("action", {})

        # Convert condition dict to robust Python code
        condition_code = cls._parse_condition(condition)

        # Convert action dict to Python statements
        action_code = cls._parse_action(action)

        return cls.BEHAVIORAL_TEMPLATE.format(
            logic_id=logic_id,
            rule_type=rule.get("rule_type", "behavioral"),
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            condition=condition,
            condition_code=condition_code,
            action=action,
            action_code=action_code
        )

    @classmethod
    def _generate_semantic(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        # Enhanced semantic: use embedding comparison if available, else fallback
        candidate = rule.get("candidate", "")
        word = rule.get("word", "")
        if candidate and word:
            body = f'''    # Learned similarity from rule
    if word == "{word}":
        # Check if semantic_learner provides embedding
        try:
            emb1 = semantic_learner.get_embedding(word)
            emb2 = semantic_learner.get_embedding("{candidate}")
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12)
                return float(sim)
        except (AttributeError, ImportError):
            pass
        return 0.95  # fallback confidence
    # TODO: implement more general similarity'''
        else:
            body = '''    # No specific word/candidate; return zero
    return 0.0'''

        return cls.SEMANTIC_TEMPLATE.format(
            logic_id=logic_id,
            relation=rule.get("type", "synonym"),
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            body=body
        )

    @classmethod
    def _generate_state(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        label = rule.get("label", "unknown_state")
        prototype = rule.get("prototype", [])
        if not prototype:
            prototype = [0.0] * 10  # default dimension
        proto_str = "[" + ", ".join(f"{x:.4f}" for x in prototype) + "]"
        body = f'''    try:
        import numpy as np
    except ImportError:
        return False  # cannot compute without numpy

    prototype = np.array({proto_str})
    if len(factor_vector) != len(prototype):
        return False
    vec = np.array(factor_vector)
    sim = np.dot(vec, prototype) / (np.linalg.norm(vec) * np.linalg.norm(prototype) + 1e-12)
    # Threshold can be adjusted dynamically
    threshold = rule.get("threshold", 0.85) if isinstance(rule, dict) else 0.85
    return bool(sim > threshold)'''
        return cls.STATE_TEMPLATE.format(
            logic_id=logic_id,
            label=label,
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            body=body
        )

    @staticmethod
    def _parse_condition(condition: dict) -> str:
        """
        Convert a condition dictionary into executable Python code.
        Supports multiple keys: 'alias', 'pronoun', 'regex', 'state', etc.
        Returns a string of Python statements that set a boolean variable `condition_met`.
        """
        if not condition:
            return "condition_met = False"

        lines = []
        parts = []

        # Alias condition
        if "alias" in condition:
            alias = condition["alias"].lower()
            parts.append(f"'{alias}' in user_input")

        # Pronoun condition
        if "pronoun" in condition:
            pronoun = condition["pronoun"].lower()
            parts.append(f"'{pronoun}' in tokens")

        # Regex condition
        if "regex" in condition:
            import re
            pattern = condition["regex"]
            # Safely compile regex; use raw string
            parts.append(f"re.search(r'{pattern}', user_input) is not None")
            lines.insert(0, "import re")

        # State condition (check a value in the state dict)
        if "state_key" in condition and "state_value" in condition:
            key = condition["state_key"]
            value = condition["state_value"]
            if isinstance(value, str):
                parts.append(f"state.get('{key}') == '{value}'")
            else:
                parts.append(f"state.get('{key}') == {value}")

        if parts:
            condition_code = "condition_met = " + " and ".join(parts)
        else:
            condition_code = "condition_met = False"

        if lines:
            condition_code = "\n".join(lines) + "\n" + condition_code
        return condition_code

    @staticmethod
    def _parse_action(action: dict) -> str:
        """
        Convert an action dictionary into Python statements.
        Returns a string of code to execute, which should set a `response` variable.
        """
        if not action:
            return '    response = "[acknowledged]"'

        lines = []
        if "self_reference" in action:
            alias = action["self_reference"]
            lines.append(f'    response = f"I am {{alias}}."')
        elif "say" in action:
            text = action["say"]
            lines.append(f'    response = "{text}"')
        elif "modify_state" in action:
            # Example: modify_state: {"key": "value"}
            mod = action["modify_state"]
            for k, v in mod.items():
                lines.append(f'    state["{k}"] = "{v}"')
            lines.append('    response = "[state updated]"')
        else:
            lines.append('    response = "[acknowledged]"')

        # Add optional logging
        lines.append('    logger.debug(f"Action executed: {response}")')
        return "\n".join(lines)

    @staticmethod
    def save_code_file(logic_id: str, code: str) -> str:
        """Write code to a .py file and return the full path."""
        safe_id = "".join(c for c in logic_id if c.isalnum() or c == "_")
        filename = f"{safe_id}.py"
        filepath = os.path.join(CODE_GEN_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        return filepath
# ------------------------------------------------------------
# ShadowExecutor – imports and tests generated code in a sandbox
# ------------------------------------------------------------
class ShadowExecutor:
    """
    Dynamically imports a Python file, executes its `run()` function,
    captures output/errors, and appends errors as comments to the file.
    """

    @staticmethod
    def test_code_file(filepath: str, test_input: dict = None) -> tuple:
        """
        Returns (success: bool, error_message: str, output: str).
        If an exception occurs, it is appended as a comment to the file.
        """
        if not os.path.exists(filepath):
            return False, "File not found", ""

        # Prepare a dummy context for behavioral rules
        if test_input is None:
            test_input = {"user_input": "hello", "tokens": ["hello"], "state": "neutral"}

        module_name = f"shadow_{int(time.time()*1000)}_{os.path.basename(filepath)[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return False, "Failed to load spec", ""

        module = importlib.util.module_from_spec(spec)
        # Capture stdout/stderr
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                start_time = time.time()
                spec.loader.exec_module(module)
                if time.time() - start_time > 12.0:
                    raise RuntimeError("Shadow execution timeout")
                if hasattr(module, "run"):
                    result = module.run(test_input)
                else:
                    result = None
            success = True
            error_msg = ""
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
            # Append error as comment to the file
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(f"\n\n# SHADOW TEST ERROR [{time.strftime('%Y-%m-%d %H:%M:%S')}]:\n")
                    f.write(f"# {error_msg}\n")
                    f.write(f"# Traceback:\n")
                    for line in traceback.format_exc().split("\n"):
                        f.write(f"# {line}\n")
            except:
                pass

        output = f_stdout.getvalue() + f_stderr.getvalue()
        return success, error_msg, output
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

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

import sys
import time
import threading
import math
import pickle
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist, dirichlet, norm, gamma, multivariate_normal
from scipy.special import digamma, gammaln, rel_entr
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
import psutil
import json
from scipy.special import logsumexp
from scipy.optimize import minimize
from typing import Callable
import copy
import importlib.util
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
# ------------------------------------------------------------
# Real CI, Phonotactic, Diagnostic engines from backend.ci
# ------------------------------------------------------------
from backend.ci import (
    CIEngine,
    PhonotacticEngine,
    DiagnosticEngine,
    add_sentence_to_memory,
)
# ------------------------------------------------------------
# Persistent state directory
# ------------------------------------------------------------
STATE_DIR = "curiosity_state"
try:
    os.makedirs(STATE_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create STATE_DIR: {e}")
# ------------------------------------------------------------
# Logic directory for inheritance – AGI writes own logic here
# ------------------------------------------------------------
LOGIC_DIR = os.path.join(os.path.dirname(__file__), "logic")
try:
    os.makedirs(LOGIC_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create LOGIC_DIR: {e}")
# ------------------------------------------------------------
# Generated code directory – AGI writes its own executable logic here
# ------------------------------------------------------------
CODE_GEN_DIR = os.path.join(os.path.dirname(__file__), "generated_code")
try:
    os.makedirs(CODE_GEN_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create CODE_GEN_DIR: {e}")
# Shadow test logs – store execution traces of generated code
SHADOW_LOG_DIR = os.path.join(STATE_DIR, "shadow_test_logs")
try:
    os.makedirs(SHADOW_LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create SHADOW_LOG_DIR: {e}")
def safe_choice(a):
    """
    Robust replacement for np.random.choice that accepts
    lists, tuples, sets, dicts, Counters, numpy arrays,
    or nested structures and always samples safely.
    """
    if a is None:
        return None

    # Convert mappings / Counters to keys
    if hasattr(a, "keys"):
        a = list(a.keys())
    else:
        a = list(a)

    if len(a) == 0:
        return None

    idx = np.random.randint(len(a))
    return a[idx]
def align_vector(vec: np.ndarray, target_dim: int, fill: float = 0.0) -> np.ndarray:
    """Pad or truncate vector safely to target dimension."""
    if vec is None:
        return np.zeros(target_dim)

    if len(vec) == target_dim:
        return vec

    if len(vec) > target_dim:
        return vec[:target_dim]

    out = np.full(target_dim, fill)
    out[:len(vec)] = vec
    return out
def align_factor_vector(vec: np.ndarray, target_dim: int, fill: float = 0.5):
    """Ensure vector matches target dimensionality."""
    if vec is None:
        return np.full(target_dim, fill)

    vec = np.asarray(vec).reshape(-1)

    if len(vec) == target_dim:
        return vec

    if len(vec) < target_dim:
        return np.pad(vec, (0, target_dim - len(vec)), constant_values=fill)

    return vec[:target_dim]

# ------------------------------------------------------------
# Configuration Manager – loads parameters from active_config.json
# ------------------------------------------------------------
class ConfigManager:
    """
    Manages external JSON configuration for the AGI.
    Loads from curiosity_state/active_config.json, provides defaults,
    supports deep updates, saving, and generating new configs via emotional modulation.
    """
    DEFAULT_CONFIG_PATH = os.path.join(STATE_DIR, "active_config.json")
    
    # Default configuration values
    DEFAULTS = {
        "hierarchical_concept_model": {
            "alpha": 0.5,
            "beta": 1.0,
        },
        "gaussian_process": {
            "input_dim": 5,
            "n_features": 100,
            "length_scale": 1.0,
            "sigma_noise": 0.1,
        },
        "meta_curiosity_learner": {
            "n_factors": 9,
        },
        "intrinsic_reward_tracker": {
            # no parameters currently, placeholder
        },
        "semantic_learner": {
            "emb_dim": 64,
            "alpha": 0.1,
        },
        "syntax_learner": {
            # no parameters currently, placeholder
        },
        "consciousness_model": {
            # no parameters currently, placeholder
        },
        "action_generator": {
            # no parameters currently, placeholder
        },
        "reasoning_engine": {
            # no parameters currently, placeholder
        },
        "factor_engine": {
            # no parameters currently, placeholder
        },
        "state_manager": {
            "state_dir": "states",
            "conflict_threshold": 0.85,
        },
        "episodic_memory": {
            "capacity": 10000,
        },
        "idle_reasoner": {
            "short_idle_threshold": 60.0,
            "long_idle_threshold": 400.0,
            "enable_world_model_dreaming": True,
            "enable_active_memory_rehearsal": True,
            "enable_curiosity_landscape_optimization": True,
            "enable_concept_graph_compression": True,
            "enable_autonomous_hypothesis_planning": True,
            "enable_identity_stabilization": True,
            "enable_cross_modal_alignment": True,
            "enable_exploration_policy_distillation": True,
            "enable_cognitive_drive_evolution": True,
            "enable_counterfactual_reasoning": True,
            "enable_semantic_self_talk": True,
            "enable_state_space_topology_learning": True,
            "enable_knowledge_gap_frontier": True,
            "enable_social_cognition_simulation": True,
            "enable_long_horizon_planning": True,
            "world_model_num_synthetic": 5,
            "dreaming_ngram_order": 3, 
            "rehearsal_batch_size": 20,
            "enable_behavioral_rule_discovery": True,
            "enable_logic_inheritance": True,
            "enable_contextual_sentiment_analysis": True,
            "curiosity_correlation_threshold": 0.9,
            "identity_cluster_k": 3,
            "topology_n_clusters": 5,
        },
        "conscience_manager": {
            "trust_threshold_for_addition": 0.85,
            "min_evidence": 5,
            "decay_rate": 0.99,
            "max_beliefs": 50,
            "evolution_trigger": 10,
        },
        "engine": {
            "reconfigure_interval": 20,
            "state_similarity_threshold": 0.85,
        },
        "adaptive_control_space": {
            "variance_threshold": 0.05,
            "utility_threshold": 0.01,
            "max_factors": 50,
            "expansion_cooldown": 100,
        },
        # Emotional modulation rules: mapping from factor name to config path and transform
        "modulation_rules": {
            "curiosity": {
                "target": "hierarchical_concept_model.alpha",
                "transform": "linear",
                "scale": 0.5,
                "offset": 0.5,
                "min": 0.1,
                "max": 2.0,
            },
            "uncertainty": {
                "target": "gaussian_process.sigma_noise",
                "transform": "linear",
                "scale": 0.2,
                "offset": 0.1,
                "min": 0.01,
                "max": 0.5,
            },
            "threat": {
                "target": "meta_curiosity_learner.n_factors",
                "transform": "identity",
            },
            # more rules can be added later
        },
    }  
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_defaults()
        self.load()
    
    def _load_defaults(self) -> dict:
        """Return a deep copy of default configuration."""
        return copy.deepcopy(self.DEFAULTS)
    
    def load(self):
        """Load configuration from file, merging with defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    try:
                        user_config = json.load(f)
                    except Exception as e:
                        print(f"[WARN] Config corrupted, loading defaults: {e}")
                        user_config = {}
                self.config = self.deep_update(self.config, user_config)
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
        else:
            # No config file, save defaults
            self.save()
    
    def save(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        tmp_path = self.config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        os.replace(tmp_path, self.config_path)
    
    def reload(self):
        """Reload configuration from disk (hot reload)."""
        self.load()
    
    def get(self, key: str, default=None):
        """
        Get a configuration value using dot notation.
        e.g., config.get('hierarchical_concept_model.alpha')
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value):
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()
    
    def deep_update(self, base: dict, update: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self.deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def generate_modulated_config(self, factor_vector: np.ndarray, factor_names: List[str] = None):
        """
        Generate a new configuration based on the current factor vector
        and the modulation rules defined in DEFAULTS.
        Saves the new config to disk and returns the config dict.        """


        if factor_names is None:
            factor_names = ["curiosity","threat","trust","confidence",
                            "uncertainty","energy","attention",
                            "comprehension","fluency","cognitive_load"]
        
        # Start from current config
        new_config = copy.deepcopy(self.config)
        
        rules = self.config.get("modulation_rules", self.DEFAULTS["modulation_rules"])
        
        for factor_name, value in zip(factor_names, factor_vector):
            if factor_name not in rules:
                continue
            rule = rules[factor_name]
            target_path = rule.get("target")
            if not target_path:
                continue
            
            # Compute modulated value
            transform = rule.get("transform", "linear")
            if transform == "linear":
                scale = rule.get("scale", 1.0)
                offset = rule.get("offset", 0.0)
                modulated = scale * value + offset
            elif transform == "identity":
                modulated = value
            elif transform == "inverse":
                modulated = 1.0 - value
            else:
                modulated = value
            
            # Clamp
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val is not None:
                modulated = max(min_val, modulated)
            if max_val is not None:
                modulated = min(max_val, modulated)
            
            # Apply to nested dict
            keys = target_path.split('.')
            target = new_config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = modulated
        
        # Save the new config
        self.config = new_config
        self.save()
        return new_config
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
        Called when a word receives an explicit definition from the user.
        Performs strong supervised stabilization:
        - boosts observation confidence
        - collapses entropy for the concept
        - marks the concept as semantically anchored
        - schedules hierarchy promotion
        """

        if not hasattr(self, "word_concept_counts"):
            return

        if word not in self.word_concept_counts:
            return

        # ---------- Strong confidence reinforcement ----------
        # Explicit definitions act as high-confidence observations
        self.word_concept_counts[word] += 10

        # ---------- Mark stabilized words ----------
        if not hasattr(self, "stabilized_words"):
            self.stabilized_words = set()

        self.stabilized_words.add(word)

        # ---------- Entropy collapse ----------
        if hasattr(self, "word_entropies") and word in self.word_entropies:
            self.word_entropies[word] *= 0.2  # strong reduction

        # ---------- Semantic anchoring ----------
        if not hasattr(self, "semantic_anchors"):
            self.semantic_anchors = set()
        self.semantic_anchors.add(word)

        # ---------- Promote toward higher-level concept (hierarchical compression) ----------
        if hasattr(self, "_promote_word_to_higher_concept"):
            try:
                self._promote_word_to_higher_concept(word)
            except Exception:
                pass

        # ---------- Schedule stability reinforcement ----------
        if not hasattr(self, "definition_events"):
            self.definition_events = {}

        self.definition_events[word] = {
            "timestamp": time.time(),
            "stability": 1.0,
            "type": "explicit_definition"
        }

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
# ------------------------------------------------------------
# Semantic Learner – Bayesian model of word meanings
# ------------------------------------------------------------
class SemanticLearner:
    """
    Implements a hierarchical generative model of word meanings:
    - Each word has a latent vector in a continuous semantic space.
    - Observed co-occurrence and user definitions are generated from this vector.
    - Variational inference via stochastic gradient descent.

    Now config‑aware: reads emb_dim, alpha.
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
                # Changing embedding dimension is non‑trivial; we ignore for hot reload
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

    def observe_definition(self, word: str, definition_text: str):
        """
        When user provides a definition (e.g., "markus means a name"),
        treat it as supervised learning: the embedding should be close
        to the embedding of the definition words.
        """
        # tokenise definition, get average embedding of known words
        tokens = definition_text.split()
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
# ------------------------------------------------------------
# Syntax Learner – Bayesian grammar induction
# ------------------------------------------------------------
class SyntaxLearner:
    """
    Incrementally learns a probabilistic context-free grammar (PCFG)
    from observed sentences. Uses a Dirichlet process prior over rules
    to allow unbounded rule creation.

    Now config‑aware (placeholder for future parameters).
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        # rule probabilities: (A -> BC) or (A -> word)
        self.nonterminals = {'S', 'NP', 'VP', 'PP', 'N', 'V', 'Adj', 'Det'}
        self.lexicon = {}        # word -> set of POS tags
        self.rule_counts = defaultdict(Counter)  # LHS -> RHS -> count
        self.rule_total = Counter()              # LHS -> total count
        self.dp_alpha = 1.0     # Dirichlet process concentration

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

    def generate_response(self, target: str) -> str:
        """
        Generate a grammatical sentence about `target`.
        Very simple template for now – expand later.
        """
        # Example: "I think [target] might be related to ..."
        return f"I think '{target}' might be related to something."

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
# Consciousness Model – Global Workspace with Attention
# ------------------------------------------------------------
class ConsciousnessModel:
    """
    Simulates a global workspace:
    - Parallel processors generate "coalitions" (hypotheses, predictions, questions).
    - A winner-take-all competition selects the most salient coalition.
    - The winner is broadcast globally, influencing all modules.

    Now config‑aware (placeholder for future parameters).
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.workspace = []      # current active coalitions
        self.global_broadcast = None  # current conscious content
        self.attention_weights = {}   # module -> weight

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def update(self, coalitions: List[Dict], factors: Dict[str, float]):
        """
        coalitions: list of candidate hypotheses / actions / queries
        factors: current factor vector (curiosity, threat, etc.)
        Computes salience and selects winner.
        """
        salience = []
        for c in coalitions:
            # salience = curiosity * (1 - confidence) + threat * negativity + ...
            s = factors.get('curiosity', 0) * c.get('curiosity_relevance', 1)
            s += factors.get('threat', 0) * c.get('threat_relevance', 0)
            s += np.random.rand() * 0.1   # stochasticity
            salience.append(s)
        if salience:
            idx = np.argmax(salience)
            self.global_broadcast = coalitions[idx]
        else:
            self.global_broadcast = None
        return self.global_broadcast

    def get_current_goal(self) -> Optional[str]:
        """Extract a natural language goal from the broadcast."""
        if self.global_broadcast:
            return self.global_broadcast.get('query') or self.global_broadcast.get('hypothesis')
        return None
# ------------------------------------------------------------
# Gaussian Process Uncertainty Model (Bayesian Linear Regression with RBF features)
# ------------------------------------------------------------
class GaussianProcessUncertainty:
    """
    Simplified Gaussian Process using Bayesian linear regression with random Fourier features.
    Approximates an RBF kernel for online uncertainty estimation.
    Maintains posterior over weights and predicts mean and variance.

    Now config‑aware: reads input_dim, n_features, length_scale, sigma_noise.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None,
                 input_dim: int = None, n_features: int = None,
                 length_scale: float = None, sigma_noise: float = None):
        self.config_manager = config_manager
        if config_manager:
            self.input_dim = config_manager.get('gaussian_process.input_dim', 5)
            self.n_features = config_manager.get('gaussian_process.n_features', 100)
            self.length_scale = config_manager.get('gaussian_process.length_scale', 1.0)
            self.sigma_noise = config_manager.get('gaussian_process.sigma_noise', 0.1)
        else:
            self.input_dim = input_dim if input_dim is not None else 5
            self.n_features = n_features if n_features is not None else 100
            self.length_scale = length_scale if length_scale is not None else 1.0
            self.sigma_noise = sigma_noise if sigma_noise is not None else 0.1

        # Random Fourier features: generate random weights and biases
        self.W = np.random.randn(self.n_features, self.input_dim) / self.length_scale
        self.b = np.random.uniform(0, 2 * np.pi, self.n_features)

        # Bayesian linear regression posterior parameters
        self.m = np.zeros(self.n_features)          # mean of weights
        self.S = np.eye(self.n_features) * 1.0      # covariance of weights
        self.beta = 1.0 / self.sigma_noise**2       # precision of noise

    def update_parameters_from_config(self):
        """Reload parameters from the config manager (for hot reload)."""
        if self.config_manager:
            self.input_dim = self.config_manager.get('gaussian_process.input_dim', 5)
            self.n_features = self.config_manager.get('gaussian_process.n_features', 100)
            self.length_scale = self.config_manager.get('gaussian_process.length_scale', 1.0)
            self.sigma_noise = self.config_manager.get('gaussian_process.sigma_noise', 0.1)
            self.beta = 1.0 / self.sigma_noise**2
            # Note: Random Fourier features are not regenerated on hot reload
            # because that would change the feature map.
            # In a full implementation you might want to handle that, but we keep it simple.

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Compute random Fourier features for input x (1D or 2D)."""
        x = np.atleast_2d(x)
        features = np.cos(x @ self.W.T + self.b) * np.sqrt(2.0 / self.n_features)
        return features

    def update(self, x: np.ndarray, y: float):
        """Online Bayesian update with new observation (x, y)."""
        try:
            phi = self._phi(x).flatten()
            # Posterior precision: S^{-1} = S_0^{-1} + beta * phi phi^T
            # Sherman-Morrison update
            S_inv = np.linalg.pinv(self.S)
            S_inv += self.beta * np.outer(phi, phi)
            self.S = np.linalg.pinv(S_inv)
            self.m = self.S @ (S_inv @ self.m + self.beta * phi * y)
        except np.linalg.LinAlgError:
            # Singular matrix – reset covariance
            self.S = np.eye(self.n_features) * 1.0
            self.m = np.zeros(self.n_features)
        except Exception as e:
            print(f"GP update failed: {e}. Skipping update.")

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Return predictive mean and variance."""
        phi = self._phi(x).flatten()
        mean = phi @ self.m
        var = 1.0 / self.beta + phi @ self.S @ phi.T
        var = max(var, 1e-12)
        return mean, var

    def uncertainty(self, x: np.ndarray) -> float:
        """Return predictive standard deviation (uncertainty)."""
        _, var = self.predict(x)
        return np.sqrt(var)
# Multi-Factor Curiosity with Meta-Learned Weights (extended)
# ------------------------------------------------------------
@dataclass
class CuriosityFactors:
    """
    Container for all curiosity scores for a given word/token.
    This class is now dynamic – factors are stored in a dictionary.
    """
    factor_dict: Dict[str, float] = field(default_factory=dict)
    
    def __getattr__(self, name):
        # During pickle loading __dict__ may not yet contain factor_dict,
        # so access it safely without triggering recursion
        factor_dict = self.__dict__.get("factor_dict", None)

        if factor_dict is not None and name in factor_dict:
            return factor_dict[name]

        raise AttributeError(f"'CuriosityFactors' has no attribute '{name}'")

    
    def __setattr__(self, name, value):
        if name == 'factor_dict':
            super().__setattr__(name, value)
        else:
            self.factor_dict[name] = value
    
    def as_array(self, factor_names: List[str]) -> np.ndarray:
        """Return values for the given factor names in order."""
        return np.array([self.factor_dict.get(name, 0.0) for name in factor_names])

class MetaCuriosityLearner:
    """
    Learns the weights of curiosity factors via Bayesian linear regression.
    Target = actual information gain (reduction in uncertainty).

    Now config‑aware and supports dynamic expansion of factor set.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None,
                 n_factors: int = None):
        self.config_manager = config_manager
        
        # Base curiosity factors (always present)
        self.base_factor_names = [
            "uncertainty", "learning_progress", "novelty",
            "complexity", "surprise", "phonotactic_anomaly",
            "diagnostic_negatives", "bayesian_surprise", "gp_uncertainty"
        ]
        
        # Dynamic factor names (can be added later)
        self.dynamic_factor_names = []
        
        if config_manager:
            # We'll keep the config parameter for backward compatibility, but it's no longer fixed.
            # Actually, we ignore the config n_factors because we manage our own list.
            pass
        
        # Initialize weights for base factors
        self.n_factors = len(self.base_factor_names)
        self.mu = np.zeros(self.n_factors)
        self.lambda_ = np.ones(self.n_factors) * 0.1
        self.alpha = np.ones(self.n_factors) * 1.0
        self.beta = np.ones(self.n_factors) * 1.0
        # Dictionary to store compute functions for dynamic curiosity factors
        self.dynamic_compute_functions = {}  # factor_name -> callable
        self._load()

    @property
    def factor_names(self):
        """Complete list of all curiosity factors (base + dynamic)."""
        return self.base_factor_names + self.dynamic_factor_names

    def update_parameters_from_config(self):
        """Placeholder for hot reload (no fixed n_factors anymore)."""
        pass

    def add_factor(self, factor_name: str, initial_mu: float = 0.0,
                compute_function: Optional[Callable] = None,
                metadata: dict = None):

        if factor_name in self.factor_names:
            return

        # append ONLY to dynamic list
        self.dynamic_factor_names.append(factor_name)

        if compute_function is not None:
            self.dynamic_compute_functions[factor_name] = compute_function

        # resize
        self.mu = np.append(self.mu, initial_mu)
        self.lambda_ = np.append(self.lambda_, 1.0)
        self.alpha = np.append(self.alpha, 1.0)
        self.beta = np.append(self.beta, 1.0)

        self.n_factors = len(self.mu)

        self.save(os.path.join(STATE_DIR, 'meta_learner.npz'))

    def compute_dynamic_factor(self, factor_name: str, token: str,
                               token_ci: Dict, token_phon: float,
                               diag: Dict, features_array: np.ndarray,
                               global_factor_dict: Optional[Dict] = None) -> float:
        """
        Compute the value of a dynamic curiosity factor using its registered compute function.
        Returns 0.0 if no function is registered.
        """
        if factor_name in self.dynamic_compute_functions:
            func = self.dynamic_compute_functions[factor_name]
            try:
                return func(token, token_ci, token_phon, diag, features_array, global_factor_dict)
            except Exception as e:
                print(f"Error computing dynamic factor '{factor_name}': {e}")
                print(traceback.format_exc())
                return 0.0
        return 0.0

    def update(self, factors: np.ndarray, information_gain: float):

        target_dim = len(factors)

        # ---- ensure all parameter arrays match the same size ----
        def _expand(arr, fill):
            if len(arr) < target_dim:
                return np.append(arr, np.full(target_dim - len(arr), fill))
            return arr

        self.mu = _expand(self.mu, 0.0)
        self.lambda_ = _expand(self.lambda_, 0.1)
        self.alpha = _expand(self.alpha, 1.0)
        self.beta = _expand(self.beta, 1.0)

        # if learner larger than factors → pad factors
        if len(factors) < len(self.mu):
            factors = np.pad(factors, (0, len(self.mu) - len(factors)))

        # ---- safe update ----
        for i in range(len(self.mu)):
            x = factors[i]
            self.lambda_[i] += x ** 2
            self.mu[i] = (
                (self.lambda_[i] * self.mu[i] + x * information_gain)
                / (self.lambda_[i] + x ** 2)
            )

    def predict_curiosity(self, factors: np.ndarray) -> float:
        """Weighted sum of factors using posterior mean weights."""
        # ---- Auto align dimensionality (dynamic factor safety) ----
        if len(self.mu) < len(factors):
            # expand mu
            pad = np.zeros(len(factors) - len(self.mu))
            self.mu = np.concatenate([self.mu, pad])

        elif len(self.mu) > len(factors):
            # pad factors
            factors = np.pad(factors, (0, len(self.mu) - len(factors)))

        factors = align_factor_vector(factors, len(self.mu))
        return np.dot(self.mu, factors)

    def get_weights(self) -> Dict[str, float]:
        """Return dictionary mapping factor names to learned weights."""
        return dict(zip(self.factor_names, self.mu))

    def apply_guilt_to_weights(self, guilt_factor: float):
        """
        guilt_factor ∈ [0.5,1.0] (higher guilt → lower factor).
        Reduce weights for factors that may have caused negative feedback.
        For simplicity, we reduce all weights slightly and increase the weight of a
        hypothetical "caution" factor if it exists.
        """
        # Scale down all weights by guilt_factor
        self.mu *= guilt_factor
        # Boost a "caution" factor if present
        if "caution" in self.factor_names:
            idx = self.factor_names.index("caution")
            self.mu[idx] = min(1.0, self.mu[idx] + (1.0 - guilt_factor) * 0.2)

    def save(self, path: str):
        """Save weights and factor names."""
        np.savez(path,
                 mu=self.mu,
                 lambda_=self.lambda_,
                 alpha=self.alpha,
                 beta=self.beta,
                 base_factor_names=self.base_factor_names,
                 dynamic_factor_names=self.dynamic_factor_names)

    def load(self, path: str):
        """Load weights and factor names."""
        data = np.load(path, allow_pickle=True)
        self.mu = data['mu']
        self.lambda_ = data['lambda_']
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.base_factor_names = list(data['base_factor_names'])
        self.dynamic_factor_names = list(data['dynamic_factor_names'])
        self.n_factors = len(self.mu)
    
    def _load(self):
        """Load from default path if exists."""
        path = os.path.join(STATE_DIR, 'meta_learner.npz')
        if os.path.exists(path):
            self.load(path)
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

    def __hash__(self):
        return hash(self.rule_id)

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

# ------------------------------------------------------------
# Word-State Associator – maps words to semantic state labels
# ------------------------------------------------------------
class WordStateAssociator:
    """
    Tracks associations between words/phrases and user-defined state labels (e.g., "greeting").
    Supports multiple labels per word with confidence scores.
    """
    def __init__(self):
        self.word_to_labels: Dict[str, Dict[str, float]] = defaultdict(dict)  # word -> {label: confidence}
        self.label_to_words: Dict[str, Set[str]] = defaultdict(set)           # label -> set of words
        self.conflict_resolution_threshold = 0.3   # minimum confidence difference to auto-resolve

    def add_association(self, word: str, label: str, confidence: float = 1.0, source: str = "explicit"):
        """
        Add a word-label association. If the word already has the label, update confidence.
        If the word has other labels with similar confidence, we may need to ask later.
        """
        current = self.word_to_labels[word].get(label, 0.0)
        self.word_to_labels[word][label] = max(current, confidence)
        self.label_to_words[label].add(word)

    def get_labels(self, word: str, threshold: float = 0.5) -> List[str]:
        """Return labels for word with confidence > threshold."""
        return [lab for lab, conf in self.word_to_labels.get(word, {}).items() if conf > threshold]

    def get_words(self, label: str, threshold: float = 0.5) -> List[str]:
        """Return words associated with a label (with any confidence)."""
        return list(self.label_to_words.get(label, set()))

    def find_similar_words_by_label(self, word: str) -> List[str]:
        """Find other words that share at least one label with the given word."""
        labels = self.get_labels(word, threshold=0.3)
        similar = set()
        for lab in labels:
            similar.update(self.get_words(lab))
        similar.discard(word)
        return list(similar)

    def detect_conflicts(self, word: str) -> List[Tuple[str, str]]:
        """
        Return pairs of labels for the same word that have similar confidence,
        indicating possible ambiguity that should be clarified.
        """
        if word not in self.word_to_labels:
            return []
        labels = list(self.word_to_labels[word].items())
        conflicts = []
        for i, (l1, c1) in enumerate(labels):
            for l2, c2 in labels[i+1:]:
                if abs(c1 - c2) < self.conflict_resolution_threshold:
                    conflicts.append((l1, l2))
        return conflicts

    def save(self, path: str):
        # Convert sets to lists for JSON serialization
        data = {
            'word_to_labels': {w: dict(labs) for w, labs in self.word_to_labels.items()},
            'label_to_words': {lab: list(words) for lab, words in self.label_to_words.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.word_to_labels = defaultdict(dict, {w: defaultdict(float, labs) for w, labs in data['word_to_labels'].items()})
            self.label_to_words = defaultdict(set, {lab: set(words) for lab, words in data['label_to_words'].items()})

@dataclass
class ConscienceBelief:
    """A deeply held behavioral/ethical belief."""
    belief_id: str
    description: str      # natural language description
    rule_type: str        # e.g., 'ethical', 'behavioral', 'meta'
    condition: Dict[str, Any]  # conditions under which it applies
    action: Dict[str, Any]     # prescribed action or constraint
    confidence: float = 0.0
    evidence: int = 0
    added_to_conscience: bool = False   # True when formally integrated
    created: float = field(default_factory=time.time)
    last_updated: Optional[float] = None

    def __hash__(self):
        return hash(self.belief_id)

class ConscienceManager:
    """
    Manages deep behavioral beliefs (conscience). Beliefs are only added to
    conscience after extensive evidence and high trust. The system can
    also adjust its own parameters (thresholds) after sufficient maturity.
    """
    STATE_FILE = os.path.join(STATE_DIR, "conscience_manager.pkl")

    DEFAULT_PARAMS = {
        "trust_threshold_for_addition": 0.85,
        "min_evidence": 5,
        "decay_rate": 0.99,
        "max_beliefs": 50,
        "evolution_trigger": 10,
    }

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.params = copy.deepcopy(self.DEFAULT_PARAMS)
        if config_manager:
            cm_config = config_manager.config.get('conscience_manager', {})
            for key in self.params:
                if key in cm_config:
                    self.params[key] = cm_config[key]

        self.beliefs: Dict[str, ConscienceBelief] = {}
        self.conscience_beliefs: Dict[str, ConscienceBelief] = {}
        self.belief_history: List[Dict] = []
        self.guilt_tracker = GuiltTracker()
        self._load()

    def _load(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.params = data.get('params', self.DEFAULT_PARAMS)
                    self.beliefs = data.get('beliefs', {})
                    self.conscience_beliefs = data.get('conscience_beliefs', {})
                    self.belief_history = data.get('history', [])
            except Exception as e:
                print(f"Could not load conscience manager: {e}")

    def save(self):
        tmp = self.STATE_FILE + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'beliefs': self.beliefs,
                'conscience_beliefs': self.conscience_beliefs,
                'history': self.belief_history,
            }, f)
        os.replace(tmp, self.STATE_FILE)

    def add_belief_candidate(self, belief: ConscienceBelief):
        if belief.belief_id in self.beliefs:
            return
        if len(self.beliefs) >= self.params['max_beliefs']:
            non_conscience = [b for b in self.beliefs.values() if not b.added_to_conscience]
            if non_conscience:
                oldest = min(non_conscience, key=lambda b: b.created)
                del self.beliefs[oldest.belief_id]
        self.beliefs[belief.belief_id] = belief
        self.belief_history.append({
            'event': 'add',
            'belief_id': belief.belief_id,
            'timestamp': time.time()
        })
        self.save()

    def update_confidence(self, belief_id: str, delta: float, feedback: bool = False, emotional_state: dict = None):
        if belief_id not in self.beliefs:
            return
        belief = self.beliefs[belief_id]
        belief.confidence = np.clip(belief.confidence + delta, 0.0, 1.0)
        if feedback:
            belief.evidence += 1 if delta > 0 else -1
        if feedback:
            if delta < 0 and emotional_state:   # negative feedback
                self.guilt_tracker.update_guilt("negative", emotional_state)
            elif delta > 0 and emotional_state:
                self.guilt_tracker.update_guilt("positive", emotional_state)
        belief.last_updated = time.time()
        self.belief_history.append({
            'event': 'update',
            'belief_id': belief_id,
            'delta': delta,
            'confidence': belief.confidence,
            'timestamp': time.time()
        })
        self.save()

    def apply_guilt_to_control_signals(self, factor_dict: Dict) -> Dict:
        """
        Modify emotional factors based on current guilt level.
        Guilt reduces confidence, increases threat, lowers trust.
        """
        guilt = self.guilt_tracker.guilt_level
        if guilt > 0:
            factor_dict['confidence'] = factor_dict.get('confidence', 0.5) * (1 - guilt * 0.3)
            factor_dict['threat'] = min(1.0, factor_dict.get('threat', 0.0) + guilt * 0.2)
            factor_dict['trust'] = max(0.0, factor_dict.get('trust', 0.5) - guilt * 0.2)
            factor_dict['guilt'] = guilt   # expose guilt as a control signal
        return factor_dict
    
    def decay_confidences(self):
        for belief in self.beliefs.values():
            if not belief.added_to_conscience and belief.last_updated:
                days_since = (time.time() - belief.last_updated) / 86400
                decay = self.params['decay_rate'] ** days_since
                belief.confidence *= decay
        self.save()

    def maybe_promote_to_conscience(self, belief_id: str) -> bool:
        if belief_id not in self.beliefs:
            return False
        belief = self.beliefs[belief_id]
        if belief.added_to_conscience:
            return False
        if belief.confidence >= self.params['trust_threshold_for_addition'] and \
           belief.evidence >= self.params['min_evidence']:
            belief.added_to_conscience = True
            self.conscience_beliefs[belief_id] = belief
            self.belief_history.append({
                'event': 'promote',
                'belief_id': belief_id,
                'timestamp': time.time()
            })
            self.save()
            print(f"🧠 [Conscience] New belief integrated: {belief.description[:60]}...")
            return True
        return False

    def get_active_conscience(self) -> List[ConscienceBelief]:
        return list(self.conscience_beliefs.values())

    def check_action(self, proposed_action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        for belief in self.conscience_beliefs.values():
            if self._matches_condition(belief.condition, proposed_action):
                if 'forbid' in belief.action or belief.action.get('effect') == 'forbid':
                    return False, belief.description
        return True, None

    def _matches_condition(self, condition: Dict, action: Dict) -> bool:
        for k, v in condition.items():
            if k in action and action[k] != v:
                return False
        return True

    def evolve_parameters(self):
        if len(self.conscience_beliefs) < self.params['evolution_trigger']:
            return
        recent_history = [h for h in self.belief_history if h['timestamp'] > time.time() - 86400*7]
        if not recent_history:
            return
        confirm_ratio = sum(1 for h in recent_history if h['event'] == 'update' and h.get('delta',0) > 0) / len(recent_history)
        if confirm_ratio > 0.8:
            self.params['trust_threshold_for_addition'] = max(0.6, self.params['trust_threshold_for_addition'] - 0.05)
            self.params['min_evidence'] = max(3, self.params['min_evidence'] - 1)
            print(f"⚙️ [Conscience] Evolving parameters: trust_threshold={self.params['trust_threshold_for_addition']:.2f}, min_evidence={self.params['min_evidence']}")
        elif confirm_ratio < 0.3:
            self.params['trust_threshold_for_addition'] = min(0.95, self.params['trust_threshold_for_addition'] + 0.05)
            self.params['min_evidence'] = min(10, self.params['min_evidence'] + 1)
            print(f"⚙️ [Conscience] Evolving parameters (cautious): trust_threshold={self.params['trust_threshold_for_addition']:.2f}, min_evidence={self.params['min_evidence']}")
        if self.config_manager:
            cm_config = self.config_manager.config.get('conscience_manager', {})
            for key in self.params:
                cm_config[key] = self.params[key]
            self.config_manager.config['conscience_manager'] = cm_config
            self.config_manager.save()
        self.save()

    def apply_ethical_influence(self, factor_dict: Dict) -> Dict:
        if self.conscience_beliefs:
            factor_dict['conscience_active'] = min(1.0, len(self.conscience_beliefs) / self.params.get('max_beliefs', 50))
        else:
            factor_dict['conscience_active'] = 0.0
        return factor_dict

class LogicInheritanceManager:
    """
    Allows the AGI to write its own logical rules to disk (./logic)
    and later ask the user for confirmation. Once confirmed, the logic
    can be integrated into the reasoning engine or become a conscience belief.
    """
    LOGIC_DIR = LOGIC_DIR

    def __init__(self, config_manager: Optional['ConfigManager'] = None, conscience_manager: Optional[ConscienceManager] = None):
        self.config_manager = config_manager
        self.conscience_manager = conscience_manager
        self.pending_doubts: List[Dict] = []
        self.confirmed_logic: Dict[str, Dict] = {}
        self.code_generator = CodeGenerator()
        self.shadow_executor = ShadowExecutor()
        self.generated_code_feedback = GeneratedCodeFeedback()
        self.generated_code_paths = {}   # logic_id -> filepath
        self._load()

    def _load(self):
        path = os.path.join(self.LOGIC_DIR, 'confirmed_logic.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.confirmed_logic = data.get('confirmed_logic', {})
                    self.pending_doubts = data.get('pending_doubts', [])
                    self.generated_code_paths = data.get('generated_code_paths', {})
            except Exception as e:
                print(f"Could not load logic inheritance: {e}")

    def save(self):
        path = os.path.join(self.LOGIC_DIR, 'confirmed_logic.pkl')
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'confirmed_logic': self.confirmed_logic,
                'pending_doubts': self.pending_doubts,
                'generated_code_paths': self.generated_code_paths,
            }, f)
        os.replace(tmp, path)

    def generate_logic_file(self, rule: Dict) -> str:
        logic_id = f"logic_{int(time.time())}_{np.random.randint(1000)}"
        safe_id = "".join(c for c in logic_id if c.isalnum() or c == "_")
        filename = f"{safe_id}.json"
        logic_id = safe_id
        filepath = os.path.join(self.LOGIC_DIR, filename)
        rule['logic_id'] = logic_id
        rule['created'] = time.time()
        rule['confirmed'] = False
        with open(filepath, 'w') as f:
            json.dump(rule, f, indent=2)
        return filepath

    def generate_doubt(self, logic_id: str) -> Optional[Dict]:
        filepath = os.path.join(self.LOGIC_DIR, f"{logic_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            rule = json.load(f)
        doubt = {
            'doubt_id': f"doubt_{logic_id}_{int(time.time())}",
            'logic_id': logic_id,
            'rule': rule,
            'timestamp': time.time(),
            'asked': False,
            'response': None
        }
        self.pending_doubts.append(doubt)
        self.save()
        return doubt

    def generate_code_for_logic(self, logic_id: str) -> Optional[str]:
        """
        Generate Python code from a confirmed logic rule and save it.
        Returns filepath if successful, else None.
        """
        # Look for the rule in confirmed_logic or pending_doubts
        rule = self.confirmed_logic.get(logic_id)
        if not rule:
            for d in self.pending_doubts:
                if d['logic_id'] == logic_id:
                    rule = d['rule']
                    break
        if not rule:
            return None

        code = self.code_generator.generate(rule)
        filepath = self.code_generator.save_code_file(logic_id, code)
        self.generated_code_paths[logic_id] = filepath
        return filepath

    def shadow_test_generated_code(self, logic_id: str) -> dict:
        """
        Run the generated code in shadow mode, record errors, and return test result.
        """
        if logic_id not in self.generated_code_paths:
            return {"success": False, "error": "No generated file"}

        filepath = self.generated_code_paths[logic_id]
        success, error, output = self.shadow_executor.test_code_file(filepath)

        # Log the test run
        log_entry = {
            "timestamp": time.time(),
            "success": success,
            "error": error,
            "output": output[:500]   # truncate
        }
        log_file = os.path.join(SHADOW_LOG_DIR, f"{logic_id}_test.json")
        try:
            with open(log_file, "w") as f:
                json.dump(log_entry, f)
        except:
            pass

        return {"success": success, "error": error, "output": output}
        
    def confirm_logic(self, logic_id: str, is_correct: bool):
        doubt = None
        for d in self.pending_doubts:
            if d['logic_id'] == logic_id:
                doubt = d
                break
        if not doubt:
            return
        if is_correct:
            self.confirmed_logic[logic_id] = doubt['rule']
            # ---- NEW: generate code and shadow test ----
            filepath = self.generate_code_for_logic(logic_id)
            if filepath:
                test_result = self.shadow_test_generated_code(logic_id)
                if not test_result["success"]:
                    # Error already appended to file as comment
                    print(f"⚠️ [Shadow Test] Generated code for {logic_id} has errors.")
            # ---------------------------------------------
            if self.conscience_manager:
                belief = ConscienceBelief(
                    belief_id=f"conscience_{logic_id}",
                    description=doubt['rule'].get('description', f"Logical rule: {doubt['rule']}"),
                    rule_type='ethical' if doubt['rule'].get('ethical', False) else 'behavioral',
                    condition=doubt['rule'].get('condition', {}),
                    action=doubt['rule'].get('action', {}),
                    confidence=0.7,
                    evidence=1,
                    added_to_conscience=False
                )
                self.conscience_manager.add_belief_candidate(belief)
                self.conscience_manager.update_confidence(belief.belief_id, 0.2, feedback=True)
        else:
            # ---- Negative feedback: store and possibly trigger guilt ----
            if self.conscience_manager and hasattr(self.conscience_manager, 'guilt_tracker'):
                # We'll need to pass the emotional state – will be handled in CuriosityEngine
                pass
        self.pending_doubts = [d for d in self.pending_doubts if d['logic_id'] != logic_id]
        self.save()

    def get_next_doubt(self) -> Optional[Dict]:
        for d in self.pending_doubts:
            if not d.get('asked', False):
                d['asked'] = True
                self.save()
                return d
        return None
# ------------------------------------------------------------
# CodeGenerator – writes Python code from confirmed logic rules
# ------------------------------------------------------------
class CodeGenerator:
    """
    Converts high‑level logic rules into executable Python code.
    Uses safe templates; every generated file contains a `run()` function.
    """

    # Template for behavioral rules (identity aliases, pronoun mappings)
    BEHAVIORAL_TEMPLATE = '''# AUTO‑GENERATED CODE
# Logic ID: {logic_id}
# Rule type: {rule_type}
# Created: {timestamp}

def run(context: dict) -> str:
    """
    context: {'user_input': str, 'tokens': list, 'state': str, ...}
    Returns a response or None.
    """
    user_input = context.get("user_input", "").lower()
    tokens = context.get("tokens", [])
    
    # Condition: {condition}
    if {condition_code}:
        # Action: {action}
{action_code}
        return response
    return None
'''

    # Template for synonym / category hypotheses
    SEMANTIC_TEMPLATE = '''# AUTO‑GENERATED CODE
# Logic ID: {logic_id}
# Semantic relation: {relation}
# Created: {timestamp}

def run(word: str, semantic_learner) -> float:
    """
    Returns similarity confidence (0‑1) between word and known concept.
    """
{body}
    return 0.0
'''

    # Template for state definitions (persistent latent concepts)
    STATE_TEMPLATE = '''# AUTO‑GENERATED CODE
# Logic ID: {logic_id}
# State label: {label}
# Created: {timestamp}

def run(factor_vector: list) -> bool:
    """
    Returns True if the current factor vector matches this state.
    """
{body}
    return False
'''

    @classmethod
    def generate(cls, logic_rule: dict) -> str:
        """
        Dispatch to the appropriate template based on rule type.
        """
        rule_type = logic_rule.get("rule_type", logic_rule.get("type", "unknown"))
        logic_id = logic_rule.get("logic_id", f"gen_{int(time.time())}")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if rule_type in ("identity_alias", "pronoun_mapping", "behavioral"):
            return cls._generate_behavioral(logic_rule, logic_id, timestamp)
        elif rule_type in ("synonym", "category", "analogy"):
            return cls._generate_semantic(logic_rule, logic_id, timestamp)
        elif rule_type == "state_definition":
            return cls._generate_state(logic_rule, logic_id, timestamp)
        else:
            # Fallback: generic template
            return f"# No template for rule type {rule_type}\\n"

    @classmethod
    def _generate_behavioral(cls, rule: dict, logic_id: str, timestamp: str) -> str:
        condition = rule.get("condition", {})
        action = rule.get("action", {})

        # Convert condition dict to Python expression
        cond_parts = []
        if "alias" in condition:
            alias = condition["alias"].lower()
            cond_parts.append(f"'{alias}' in user_input")
        if "pronoun" in condition:
            pronoun = condition["pronoun"].lower()
            cond_parts.append(f"'{pronoun}' in tokens")
        # Add more condition handlers as needed
        condition_code = " and ".join(cond_parts) if cond_parts else "False"

        # Convert action dict to Python statements
        action_lines = []
        if "self_reference" in action:
            alias = action["self_reference"]
            action_lines.append(f'    response = f"I am {{alias}}."')
        else:
            action_lines.append("    response = \"[acknowledged]\"")
        action_code = "\n".join(action_lines)

        return cls.BEHAVIORAL_TEMPLATE.format(
            logic_id=logic_id,
            rule_type=rule.get("rule_type", "behavioral"),
            timestamp=timestamp,
            condition=condition,
            condition_code=condition_code,
            action=action,
            action_code=action_code
        )

    @classmethod
    def _generate_semantic(cls, rule: dict, logic_id: str, timestamp: str) -> str:
        # Simplified: return fixed similarity if word matches candidate
        candidate = rule.get("candidate", "")
        body = f'''    if word == "{rule.get('word', '')}":
        return 0.95  # learned similarity
    # TODO: implement embedding comparison'''
        return cls.SEMANTIC_TEMPLATE.format(
            logic_id=logic_id,
            relation=rule.get("type", "synonym"),
            timestamp=timestamp,
            body=body
        )

    @classmethod
    def _generate_state(cls, rule: dict, logic_id: str, timestamp: str) -> str:
        label = rule.get("label", "unknown_state")
        # Use cosine similarity with stored prototype
        prototype = rule.get("prototype", [])
        if not prototype:
            prototype = [0.0] * 10  # default dimension
        proto_str = "[" + ", ".join(f"{x:.4f}" for x in prototype) + "]"
        body = f'''    from numpy import dot, linalg
    prototype = {proto_str}
    if len(factor_vector) != len(prototype):
        return False
    sim = dot(factor_vector, prototype) / (linalg.norm(factor_vector) * linalg.norm(prototype) + 1e-12)
    return sim > 0.85'''
        return cls.STATE_TEMPLATE.format(
            logic_id=logic_id,
            label=label,
            timestamp=timestamp,
            body=body
        )

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
# ------------------------------------------------------------
# GuiltTracker – models unease after negative feedback in positive contexts
# ------------------------------------------------------------
class GuiltTracker:
    """
    Guilt level ∈ [0,1]. Increases when negative feedback is received while
    the system is in a "positive" emotional state (high trust, low threat, high energy).
    Decays over time.
    """

    def __init__(self, decay_rate: float = 0.99, guilt_boost: float = 0.2):
        self.guilt_level = 0.0          # 0 = no guilt, 1 = intense guilt
        self.decay_rate = decay_rate    # per hour
        self.guilt_boost = guilt_boost  # how much guilt increases per negative feedback
        self.history = []              # list of (timestamp, guilt_before, guilt_after, context)
        self.last_update = time.time()

    def update_guilt(self, feedback_rating: str, emotional_state: dict = None):
        """
        Called after receiving user feedback.
        If rating is negative and emotional_state indicates positive environment,
        guilt increases. Otherwise, guilt decays.
        """
        # Compute positivity score (0..1)
        if emotional_state is None:
            emotional_state = {}
        trust = emotional_state.get("trust", 0.5)
        threat = emotional_state.get("threat", 0.0)
        energy = emotional_state.get("energy", 0.5)
        comprehension = emotional_state.get("comprehension", 0.5)
        positivity = (trust + (1 - threat) + energy + comprehension) / 4.0

        # Decay first
        self.decay()

        before = self.guilt_level

        if feedback_rating.lower().startswith("negative"):
            self.guilt_level = min(1.0, self.guilt_level + self.guilt_boost * positivity)
        else:
            self.guilt_level *= 0.95

        self.history.append((time.time(), before, self.guilt_level, emotional_state))

    def decay(self):
        """Explicit decay call (called periodically)."""
        now = time.time()
        hours_passed = (now - self.last_update) / 3600.0
        self.guilt_level *= (self.decay_rate ** hours_passed)
        self.last_update = now

    def get_guilt_factor(self) -> float:
        """Return a multiplier ∈ [0.5, 1.0] for confidence/curiosity."""
        # Higher guilt → lower confidence, lower curiosity
        return 1.0 - self.guilt_level * 0.5

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "guilt_level": self.guilt_level,
                "decay_rate": self.decay_rate,
                "guilt_boost": self.guilt_boost,
                "history": self.history,
                "last_update": self.last_update
            }, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.guilt_level = data.get("guilt_level", 0.0)
                self.decay_rate = data.get("decay_rate", 0.99)
                self.guilt_boost = data.get("guilt_boost", 0.2)
                self.history = data.get("history", [])
                self.last_update = data.get("last_update", time.time())
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
# Reasoning Engine (enhanced with active learning via GP)
# ------------------------------------------------------------
class ReasoningEngine:
    """
    Stores world knowledge, co-occurrence graphs, definitions, and semantic relations.
    Generates hypotheses about word meanings and plans queries.

    Now config‑aware and supports definition learning & synonym detection.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.co_occurrence = defaultdict(Counter)
        self.relations = defaultdict(list)          # word -> list of (rel_type, target_word, confidence)
        self.hypotheses = []                       # pending hypotheses to ask user
        # GP for predicting curiosity from word features (optional)
        self.gp_predictor = None

        # --- NEW: Definition memory ---
        self.definitions: Dict[str, str] = {}           # word -> definition text
        self.definition_embeddings: Dict[str, np.ndarray] = {}

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

            for w2 in tokens[i+1:]:
                self.co_occurrence[w1][w2] += 1
                self.co_occurrence[w2][w1] += 1

        # --- store sentence (without embedding now; embedding added later) ---
        self.store_sentence(sentence, tokens, embedding=None,
                            metadata={'source': 'user', 'timestamp': time.time()})

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
        self._check_transitive_inferences(subject, object, rel_type)

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
                # NEW fields
                'phrase_definitions': self.phrase_definitions,
                'phrase_embeddings': phrase_embed_serializable,
                'phrase_relations': self.phrase_relations,
                'phrase_metadata': self.phrase_metadata, 
                'inferences': self.inferences,
                'is_a_relations': dict(self.is_a_relations),
                'means_relations': dict(self.means_relations),
            }, f)
        os.replace(tmp, path)
        # Save word-state associations separately
        self.word_state.save(os.path.join(STATE_DIR, 'word_state.json'))

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
        # Load word-state associations
        self.word_state.load(os.path.join(STATE_DIR, 'word_state.json'))

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
        
        # Dimensionality: all vectors are aligned to the current reference dimension.
        self.ref_dim = None
        self._update_ref_dim()

    def _update_ref_dim(self):
        """Set reference dimension to the mode of all stored vectors."""
        dims = [len(vec) for vec in self.states.values()]
        if dims:
            self.ref_dim = max(set(dims), key=dims.count)  # most common dimension
        else:
            self.ref_dim = 10  # default base dimension

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Pad or truncate vector to match reference dimension."""
        if self.ref_dim is None:
            self._update_ref_dim()
        if vector is None or len(vector) == 0:
            return np.zeros(self.ref_dim)
        if len(vector) < self.ref_dim:
            return np.pad(vector, (0, self.ref_dim - len(vector)), 'constant')
        elif len(vector) > self.ref_dim:
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
                    self.states[label] = self._normalize_vector(np.array(vector))
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
        return safe_label

    def delete_state(self, label: str):
        """Remove a state file and from memory."""
        if label in self.states:
            del self.states[label]
        if label in self.shadow_states:
            del self.shadow_states[label]
        for fname in os.listdir(self.state_dir):
            if fname.startswith(label) and fname.endswith(".json"):
                os.remove(os.path.join(self.state_dir, fname))
        self._update_ref_dim()

    def find_similar_state(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> Optional[str]:
        """
        Return the label of the most similar known state if cosine similarity > threshold.
        Automatically normalises vectors to the reference dimension.
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
        if best_sim >= threshold:
            return best_label
        return None

    # ------------------------------------------------------------------
    # Conflict detection & resolution (adapt to handle different lengths)
    # ------------------------------------------------------------------
    def detect_conflicts(self, threshold: float = None) -> List[Dict]:
        """
        Scan all states and find pairs with similarity > threshold.
        Both vectors are normalised to the reference dimension.
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
            for l2 in labels[i+1:]:
                pair = tuple(sorted([l1, l2]))
                if pair in processed_pairs:
                    continue
                v2 = self._normalize_vector(self.states[l2])
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                if sim >= threshold:
                    conflict = {
                        "labels": [l1, l2],
                        "similarity": float(sim),
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
# Core Curiosity Engine (Enhanced with Factor Engine & State Learning)
# ------------------------------------------------------------
class CuriosityEngine:
    """
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
        self.consciousness = ConsciousnessModel(config_manager=self.config_manager)
        self.episodic_memory = EpisodicMemory(config_manager=self.config_manager)
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
        self.current_state_label = None          # label of the currently recognized state
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
        self._idle_tasks_last_run = 0
        self.pending_sentence_query = None   # used by ActionGenerator
        # ---- Logic doubt feedback ----
        self.expecting_logic_feedback = False
        self.pending_logic_doubt = None

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
        # ---- Load current latent state concept ----
        current_state_path = os.path.join(STATE_DIR, 'current_state.json')
        if os.path.exists(current_state_path):
            try:
                with open(current_state_path, 'r') as f:
                    data = json.load(f)
                self.current_state_label = data.get('label')
                emb = data.get('embedding')
                if emb is not None:
                    self.current_state_embedding = np.array(emb)
            except Exception as e:
                print(f"Could not load current state: {e}")
                self.current_state_label = None
                self.current_state_embedding = None
        else:
            self.current_state_label = None
            self.current_state_embedding = None

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
            # ---- Save current latent state concept ----
            current_state_path = os.path.join(STATE_DIR, 'current_state.json')
            data = {
                'label': self.current_state_label,
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

            return CuriosityFactors(factor_dict=factor_dict)
        except Exception as e:
            print(f"Error computing curiosity factors for '{token}': {e}. Returning neutral factors.")
            print(traceback.format_exc())
            # Return neutral factors to avoid crash
            neutral = {name: 0.5 for name in self.meta_learner.factor_names}
            return CuriosityFactors(factor_dict=neutral)   

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
        Compare current factor vector with known states.
        Uses meta-level novelty gating to avoid state collapse.
        Now also sets the persistent latent state concept and its embedding.
        """
        if factor_vector is None or len(factor_vector) == 0:
            print("   [State Learning] Empty factor vector – skipping.")
            return

        # Only consider state learning if enough interactions have passed since last state addition
        if time.time() - self._last_state_learning_time < 30:  # at most once per 30 seconds
            return

        # similarity threshold (high)
        sim_threshold = self.config_manager.get('engine.state_similarity_threshold', 0.97)

        # meta maturity: more states → stricter novelty requirement
        n_states = max(1, len(self.state_manager.states))
        maturity = np.clip(np.log1p(n_states) / 5.0, 0.2, 1.0)

        # adaptive novelty distance threshold
        novelty_dist_threshold = 0.35 * (1.2 - maturity)

        similar_label = self.state_manager.find_similar_state(factor_vector, threshold=sim_threshold)

        # ---------- META NOVELTY GATE ----------
        if similar_label:
            stored_vec = self.state_manager.states.get(similar_label)
            if stored_vec is not None:
                v1 = self.state_manager._normalize_vector(factor_vector)
                v2 = self.state_manager._normalize_vector(stored_vec)

                if v1.shape != v2.shape:
                    max_len = max(len(v1), len(v2))
                    v1 = np.pad(v1, (0, max_len - len(v1)))
                    v2 = np.pad(v2, (0, max_len - len(v2)))

                dist = np.linalg.norm(v1 - v2)

                # if sufficiently different → treat as new state
                if dist > novelty_dist_threshold:
                    similar_label = None
        # ---------------------------------------

        if similar_label:
            print(f"\n🔍 [State Recognition] This state resembles '{similar_label}'.")
            # Update current state to the recognized one
            stored_vec = self.state_manager.states[similar_label]
            self._update_state_embedding(similar_label, stored_vec)
            return

        # Novel state detected – only ask for label if the vector has high variance (meaningful)
        # Compute variance of factor vector as a measure of distinctiveness
        vec_variance = np.var(factor_vector)
        if vec_variance < 0.05:  # very flat vector – probably not meaningful
            print("   [State Learning] Factor vector too uniform – skipping state creation.")
            return

        print("\n🧠 [State Learning] I encountered a new internal state.")
        print("    Factor vector: " + " ".join(f"{v:.2f}" for v in factor_vector))
        print("    How do humans call this state? (e.g., 'confident exploration', 'cautious inquiry')")
        label = input("    🏷️  Label: ").strip()
        if not label:
            label = "unnamed_state"

        self.state_manager.save_state(label, factor_vector, metadata)
        print(f"    ✅ State saved as '{label}'.")
        self._last_state_learning_time = time.time()  # record time
        # Reinforce identity – labeling a state is a positive interaction
        self._reinforce_identity(factor_vector, reward_signal=0.2)
        # Set this new state as the current latent concept
        self._update_state_embedding(label, factor_vector)

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
        return (sim + 1.0) * 0.5

    def _update_state_embedding(self, label: str, vector: np.ndarray):
        """
        Set the current state label and embedding.
        Also store the label's embedding in the semantic learner,
        making it a persistent latent concept.
        """
        # Align vector to semantic embedding dimension
        aligned = self._align_vector_to_emb_dim(vector)

        # Update semantic learner for this label
        if label not in self.semantic_learner.word_embeddings:
            self.semantic_learner.initialize_word(label)
        # Set embedding (overwrite)
        self.semantic_learner.word_embeddings[label] = aligned.copy()
        # Also store as current
        self.current_state_label = label
        self.current_state_embedding = aligned.copy()
        # Optionally, we could also store in reasoning engine's definitions
        # to enrich semantic relations, but that's optional.

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

                # ---- NEW: Word-state association ----
                # If subject is a word/phrase and object is a state label (single word), associate
                # Also handle "X like Y" as suggesting similarity
                if rel_type in ('means', 'is_a'):
                    # Treat object as a label for subject
                    self.reasoner.word_state.add_association(subject, object, confidence=1.0, source="explicit")
                    print(f"📝 [Word-State] Associated '{subject}' with label '{object}'")
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

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline with meta-learning, reasoning, AND factor-based state learning.
        Also increments interaction counter and triggers self‑reconfiguration.
        """
        self.last_interaction_time = time.time()

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
        # -------------------------------------------------------
        # HANDLE RESPONSE FEEDBACK (emergent sentiment learning)
        # -------------------------------------------------------
        if not self.chat_mode_active and self.expecting_response_feedback:
            feedback_lower = text.lower()
            rating = None
            # Detect rating from user input
            if any(kw in feedback_lower for kw in ['positive', 'good', 'great', 'yes', 'correct', 'right', '👍']):
                rating = 'positive'
            elif any(kw in feedback_lower for kw in ['negative', 'bad', 'wrong', 'no', 'incorrect', '👎']):
                rating = 'negative'
            elif any(kw in feedback_lower for kw in ['neutral', 'ok', 'fine', 'average']):
                rating = 'neutral'
            # If rating found and we have a pending context
            if rating is not None and self.pending_feedback_context is not None:
                entry = self.pending_feedback_context
                entry.rating = rating
                entry.metadata['feedback_timestamp'] = time.time()
                entry.metadata['feedback_text'] = text
                self.reasoner.add_feedback(entry)
                print(f"📊 [Feedback Recorded] User rated response as: {rating}")

                if 'positive' in rating and hasattr(self, '_last_factor_vector'):
                    self._reinforce_identity(self._last_factor_vector, reward_signal=0.2)

                self.expecting_response_feedback = False
                self.pending_feedback_context = None
            else:
                print(f"🌑 [Feedback] No clear rating detected, ignoring.")
                self.expecting_response_feedback = False
                self.pending_feedback_context = None
        # -------------------------------------------------------
        # CHAT MODE TRIGGER
        # -------------------------------------------------------
        if text.lower().strip() == self.chat_mode_trigger:
            self.chat_mode_active = not self.chat_mode_active
            if self.chat_mode_active:
                self.chat_history = []   # start fresh
                system_reply = "Chat mode activated. I'll try to talk with you naturally. What shall we discuss?"
            else:
                system_reply = "Chat mode deactivated. Returning to normal learning mode."
            print(f"🤖 AGI> {system_reply}")
            # Return early – no need to process this as a normal sentence
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
        # -------------------------------------------------------
        # HANDLE RULE CONFIRMATION RESPONSE
        # -------------------------------------------------------
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
                self.expecting_rule_confirmation = False
                self.pending_rule_id = None
            else:
                print(f"🌑 [Rule Shadow] No clear confirmation for rule: {self.pending_rule_id}")
                self.behavioral_learner.update_confidence(self.pending_rule_id, -0.1, feedback=False)
                self.expecting_rule_confirmation = False
                self.pending_rule_id = None
    
        # -------------------------------------------------------
        # HANDLE LOGIC DOUBT FEEDBACK
        # -------------------------------------------------------
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
                # ---- Record feedback for generated code ----
                if is_positive or is_negative:
                    self.generated_code_feedback.add_feedback(
                        logic_id,
                        "positive" if is_positive else "negative",
                        {"factor_dict": self._last_factor_dict.copy() if self._last_factor_dict else {}}
                    )
                    # ---- Update guilt based on feedback and emotional state ----
                    if self._last_factor_dict is not None:
                        rating = "positive" if is_positive else "negative"
                        self.guilt_tracker.update_guilt(rating, self._last_factor_dict)
                else:
                    print(f"🌑 [Logic Shadow] No clear confirmation for logic: {logic_id}")
                self.expecting_logic_feedback = False
                self.pending_logic_doubt = None
            else:
                self.expecting_logic_feedback = False
        # -------------------------------------------------------
        # HANDLE PENDING PATTERN LABEL
        # -------------------------------------------------------
        if not self.chat_mode_active and getattr(self, "expecting_pattern_label", False):                
            label = text.strip().lower()

            # store relation: word → category
            self.reasoner.relations[self.last_asked_word].append(
                ("category", label, 1.0)
            )
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

        self.interaction_counter += 1

        # 1. Memory ingestion
        add_sentence_to_memory(text)

        # 2. Whole-sentence evaluation
        ci_report = self.ci_engine.evaluate(text)
        phon_score, phon_diagnostics = self.phon_engine.score_text(text)
        diag_explanation = self.diag_engine.explain(ci_report, phon_score)

        # ---- Detect if user is defining a previously asked word/phrase ----
        if not self.chat_mode_active and self.last_asked_word is not None:
            if "means" in text.lower() or "is" in text.lower() or "refers to" in text.lower():

                is_phrase = ' ' in self.last_asked_word  # phrase contains space

                if is_phrase:
                    # Handle phrase definition
                    phrase = self.last_asked_word
                    definition = text
                    # Compute embedding for the phrase (average of its word embeddings)
                    phrase_tokens = phrase.split()
                    known_embs = [self.semantic_learner.word_embeddings[t] for t in phrase_tokens if t in self.semantic_learner.word_embeddings]
                    if known_embs:
                        phrase_emb = np.mean(known_embs, axis=0)
                    else:
                        phrase_emb = None
                    # Store in reasoner
                    self.reasoner.phrase_definitions[phrase] = definition
                    if phrase_emb is not None:
                        self.reasoner.phrase_embeddings[phrase] = phrase_emb
                    print(f"📘 [Phrase Definition] Learned about '{phrase}': {definition}")
                    # Derive inferences from this definition
                    self._derive_inferences_from_definition(phrase, definition)

                else:
                    # Original word definition handling
                    if self.last_asked_word not in self.semantic_learner.word_embeddings:
                        self.semantic_learner.initialize_word(self.last_asked_word)

                    self.semantic_learner.observe_definition(self.last_asked_word, text)

                    def_emb = self.semantic_learner.text_to_embedding(text)
                    if def_emb is not None:
                        self.reasoner.store_definition(self.last_asked_word, text, def_emb)
                        self.concept_model.register_definition_event(self.last_asked_word)

                    self.concept_model.register_definition_event(self.last_asked_word)

                    if self.last_asked_word in self.word_history:
                        for rec in self.word_history[self.last_asked_word][-3:]:
                            rec['curiosity_factors'].gp_uncertainty *= 0.3
                            rec['curiosity_factors'].bayesian_surprise *= 0.3
                        # Also force entropy to near zero in concept model
                        if hasattr(self.concept_model, 'word_concept_counts') and self.last_asked_word in self.concept_model.word_concept_counts:
                            # Set all counts to concentrate on one concept
                            counts = self.concept_model.word_concept_counts[self.last_asked_word]
                            if counts:
                                dominant = max(counts, key=counts.get)
                                new_counts = Counter({dominant: 100})
                                self.concept_model.word_concept_counts[self.last_asked_word] = new_counts
                        hypothesis = self.reasoner.generate_synonym_hypothesis(self.last_asked_word)
                        if hypothesis:
                            print(f"💡 [Semantic Hypothesis] {hypothesis}")
                    print(f"📘 [Definition Learning] Learned about '{self.last_asked_word}': {text}")
                    # Derive inferences from word definition (simple)
                    self._derive_inferences_from_definition(self.last_asked_word, text)

                # Reinforce identity with positive reward
                if hasattr(self, '_last_factor_vector'):
                    self._reinforce_identity(self._last_factor_vector, reward_signal=0.3)
                self.last_asked_word = None  # reset

        # ---- Handle feedback from trust-based utterances ----
        if not self.chat_mode_active and self.expecting_feedback:
            feedback_lower = text.lower()
            is_positive = any(kw in feedback_lower for kw in ['yes', 'correct', 'right', 'yeah', 'true', 'yep'])
            is_negative = any(kw in feedback_lower for kw in ['no', 'wrong', 'incorrect', 'not', 'nope'])
            
            if is_positive or is_negative:
                # Process feedback
                if is_positive:
                    print(f"✅ [Positive Feedback] User confirmed my attempt: '{self.feedback_attempt_response}'")
                    # Reinforce identity with positive reward
                    if hasattr(self, '_last_factor_vector'):
                        self._reinforce_identity(self._last_factor_vector, reward_signal=0.4)
                    # Store this successful interaction in reasoner? Could add as a positive example.
                else:
                    print(f"❌ [Negative Feedback] User corrected my attempt: '{self.feedback_attempt_response}'")
                    # Could store this as a negative example to avoid similar mistakes
                
                # Clear feedback state
                self.expecting_feedback = False
                self.feedback_attempt_word = None
                self.feedback_attempt_response = None
                self.feedback_timestamp = None
            else:
                # Not feedback, mark the attempt as shadow and continue
                print(f"🌑 [Shadow Attempt] No feedback given for: '{self.feedback_attempt_response}'")
                # Optionally store as shadow in reasoner for later reflection
                if self.feedback_attempt_response:
                    # Add to hypotheses as unresolved?
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

        # 3. Token analysiss
        tokens = self._tokenize(text)
        self.reasoner.observe_sentence(text, tokens)
        self.reward_tracker.reset_episode()
        # -------------------------------------------------------
        # PENDING LOGIC DOUBT (highest priority)
        # -------------------------------------------------------
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
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Logic doubt – awaiting confirmation."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
        # -------------------------------------------------------
        # UNKNOWN WORD DETECTION (prevents early greeting collapse)
        # -------------------------------------------------------
        if not self.chat_mode_active:
            unknown_tokens = [t for t in tokens if t not in self.semantic_learner.word_embeddings]
            unknown_ratio = len(unknown_tokens) / max(len(tokens), 1)
            if unknown_ratio > 0.5:
                target = unknown_tokens[0]
                self.last_asked_word = target
                system_reply = f"What does '{target}' mean?"
                print(f"\n🧠 [Knowledge Gap] {system_reply}")
                print(f"🤖 AGI> {system_reply}")
                return {
                    "ci_report": ci_report,
                    "phon_score": phon_score,
                    "diagnostic": diag_explanation,
                    "token_analyses": [],
                    "curiosity_artifacts": ["Knowledge gap detected — waiting for definition."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
        token_analyses = []
        for token in tokens:
            token_ci = self.ci_engine.evaluate(token)
            token_phon, _ = self.phon_engine.score_text(token)

            features = self._extract_features(token, token_ci, token_phon, diag_explanation)
            features_array = self._features_to_array(features)
            concept = self.concept_model.add_word_observation(token, features)

            factors = self._compute_curiosity_factors(
                token, token_ci, token_phon, diag_explanation, features_array,
                global_factor_dict=self._last_factor_dict  # from previous interaction
            )

            new_entropy = self.concept_model.entropy_word(token)
            igain = self._information_gain(token, new_entropy)

            # bootstrap curiosity when entropy history is short
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

            # Build factor array in the order of meta_learner.factor_names
            factor_array = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))
            self.meta_learner.update(factor_array, igain)

            curiosity_score = self.meta_learner.predict_curiosity(factor_array)

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
        # ---- Sentence-level reasoning (only if all tokens known) ----
        if self._all_tokens_known(tokens):
            sentence_emb = self.semantic_learner.text_to_embedding(text)
            if sentence_emb is not None:
                self._process_sentence_reasoning(text, tokens, token_analyses, sentence_emb)        
        # ---- Pattern-based semantic emergence ----
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


        # ---- Factor computation & state learning ----
        factor_dict = self.factor_engine.compute(
            token_analyses, ci_report, phon_score, diag_explanation,
            self.concept_model, self.meta_learner, self.reward_tracker.total_reward,
            self.semantic_learner
        )

        if not isinstance(factor_dict, dict):
            factor_dict = {}

        # ---- Conscience influence on control signals ----
        factor_dict = self.conscience_manager.apply_ethical_influence(factor_dict)

        if not isinstance(factor_dict, dict):
            factor_dict = {}

        # Build factor vector safely – missing factors become 0.5
        factor_vector_list = []
        for k in self.control_space.all_factors:
            if k in factor_dict:
                factor_vector_list.append(factor_dict[k])
            else:
                factor_vector_list.append(0.5)
        factor_vector = np.array(factor_vector_list)
        self._last_factor_vector = factor_vector   # store for self‑reconfiguration
        self._last_factor_dict = factor_dict       # store for dynamic curiosity factors
        metadata = {
            "input_text": text,
            "timestamp": time.time(),
            "ci_confidence": ci_report.get("confidence"),
            "phon_score": phon_score,
        }
        #-------------------------------------------------------
        # STATE LEARNING GATE — requires minimal knowledge maturity
        # -------------------------------------------------------
        if not self.chat_mode_active:
            known_vocab_size = len(self.semantic_learner.word_embeddings)
            if known_vocab_size > 20:  # adjustable
                self._handle_state_learning(factor_vector, metadata)

        # ---------------------------------------------------
        # ---- Check for unresolved conflicts and ask user ----
        if not self.chat_mode_active:
            unresolved = self.state_manager.get_unresolved_conflicts()
            if unresolved:
                print("\n🧠 [Conflict Resolution] I have some unresolved state conflicts from previous idle introspection.")
            for conflict in unresolved[:2]:  # handle up to 2 per interaction
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
                choice = input("    Your choice (1/2/3/4): ").strip()

                if choice == "2":
                    winner = input(f"    Which label do you want to keep? ({l1}/{l2}): ").strip()
                    if winner in [l1, l2]:
                        self.state_manager.resolve_conflict_permanently(conflict, winner, delete_other=True)
                        print(f"    ✅ Resolved: keeping '{winner}', deleted the other.")
                elif choice == "3":
                    new_label1 = input(f"    New label for '{l1}': ").strip() or l1
                    new_label2 = input(f"    New label for '{l2}': ").strip() or l2
                    # Rename by saving new and deleting old
                    vec1 = self.state_manager.states[l1]
                    vec2 = self.state_manager.states[l2]
                    self.state_manager.save_state(new_label1, vec1)
                    self.state_manager.save_state(new_label2, vec2)
                    self.state_manager.delete_state(l1)
                    self.state_manager.delete_state(l2)
                    # Update conflict: mark as resolved and set resolution label
                    conflict["resolved"] = True
                    conflict["resolution_label"] = f"{new_label1},{new_label2}"
                    # Also update any other conflicts that might reference these labels
                    for c in self.state_manager.conflicts:
                        if l1 in c["labels"] or l2 in c["labels"]:
                            c["resolved"] = True
                            c["resolution_label"] = "renamed"
                    print(f"    ✅ States renamed and kept.")
                elif choice == "4" and shadow_label:
                    # Promote shadow to permanent
                    vec = self.state_manager.shadow_states.get(shadow_label)
                    if vec is not None:
                        perm_label = input(f"    Permanent name for this merged state: ").strip() or f"merged_{l1}_{l2}"
                        self.state_manager.save_state(perm_label, vec, is_shadow=False)
                        self.state_manager.delete_state(shadow_label)
                        self.state_manager.resolve_conflict_permanently(conflict, perm_label, delete_other=True)
                        print(f"    ✅ Shadow state promoted to '{perm_label}', both originals deleted.")
                else:
                    print("    No action taken; conflict remains unresolved.")

            # Save after handling
            self.state_manager._save_conflicts()
        # ---- Behavioral rule testing (ask user to confirm hypotheses) ----
        if not self.chat_mode_active:
            context = {
                'user_input': text,
                'tokens': tokens,
                'token_analyses': token_analyses,
                'factor_dict': factor_dict,
                'state': self.current_state_label
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
                    "factor_vector": factor_dict if isinstance(factor_dict, dict) else {},
                }

        # 4. Generate advanced artifacts (now including factor info)
        # ---- Semantic relation discovery ----
        self._detect_semantic_relations(tokens, token_analyses)
        artifacts = self._generate_artifacts(
            tokens, token_analyses, ci_report, phon_score, diag_explanation,
            factor_dict  # <-- pass factor dict
        )
        # ---- Generate system response ----
        system_reply = self.action_generator.generate_reply(text, {
            'token_analyses': token_analyses,
            'factor_vector': factor_dict,
        })
        # If we asked about a word, record it
        if "What does" in system_reply and "'" in system_reply:
            import re
            match = re.search(r"'([^']+)'", system_reply)
            if match:
                self.last_asked_word = match.group(1)
        print(f"🤖 AGI> {system_reply}")

        # ---- Ask for user feedback on the response (emergent learning) ----
        if not self.chat_mode_active and not self.expecting_response_feedback and not self.expecting_feedback and not self.expecting_rule_confirmation and not self.expecting_logic_feedback:
            # Determine a template ID for this response (simple hash of the response text)
            template_id = f"resp_{hash(system_reply) % 10000:04x}"
            self.reasoner.register_response_template(template_id, system_reply)
            if self._should_ask_feedback(tokens, factor_dict, template_id):
                feedback_question = self._generate_feedback_question()
                print(f"🤖 AGI> {feedback_question}")
                # Prepare feedback entry to be filled later
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

        # ---- Dynamic reward multiplier for chat mode ----
        if self.chat_mode_active:
            multiplier = self._compute_chat_reward_multiplier(factor_dict)
            original_reward = self.reward_tracker.episode_reward
            self.reward_tracker.episode_reward *= multiplier
            print(f"🤖 [Chat Mode] Reward multiplier: {multiplier:.2f} (original: {original_reward:.3f} → boosted: {self.reward_tracker.episode_reward:.3f})")

        safe_factor_dict = (
            factor_dict if isinstance(factor_dict, dict)
            else {}
        )

        self.episodic_memory.add_episode(
            user_input=text,
            system_response=system_reply,
            factor_vector=safe_factor_dict,
            intrinsic_reward=float(self.reward_tracker.episode_reward),
            metadata={'ci_confidence': ci_report.get('confidence')}
        )

        # ---- Store in chat history if chat mode is active ----
        if self.chat_mode_active:
            self.chat_history.append((text, system_reply))
            if len(self.chat_history) > 20:
                self.chat_history.pop(0)
        # 5. Save state
        self._save_state()

        # ---- Self‑reconfiguration trigger ----
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
            'factor_vector': factor_dict,   # <-- also return for inspection
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in text.split() if len(t) >= 2]

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
                print(f"    Conflict: '{l1}' ↔ '{l2}' (similarity={sim:.3f})")
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
        if self.current_state_label and self.current_state_embedding is not None:
            state_id = f"state_{self.current_state_label}"
            if state_id not in self.logic_inheritance.generated_code_paths:
                state_rule = {
                    "logic_id": state_id,
                    "rule_type": "state_definition",
                    "label": self.current_state_label,
                    "prototype": self.current_state_embedding.tolist(),
                    "description": f"Recognize the '{self.current_state_label}' state"
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
        Uses learned conversational patterns from behavioral rules, then definitions, then trigram.
        """
        user_input_lower = user_input.lower().strip()

        # --- Use identity alias if available ---
        alias = None
        if self.engine.identity_aliases:
            alias = next(iter(self.engine.identity_aliases))

        # --- If we have a pending sentence query, ask it (still valid) ---
        if self.engine.pending_sentence_query:
            query = self.engine.pending_sentence_query
            self.engine.pending_sentence_query = None
            return query

        # --- If there's an unanswered hypothesis, still ask it (curiosity) ---
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') in ('synonym', 'category', 'analogy', 'inference') and not hyp.get('asked', False):
                hyp['asked'] = True
                return hyp['text']

        # --- 1. Check learned behavioral patterns (conversational_pattern) ---
        # We look for rules where condition contains 'user_input_pattern' or 'question_pattern'
        for rule in self.engine.behavioral_learner.rules.values():
            if rule.rule_type == 'conversational_pattern' and rule.confidence > 0.6:
                cond = rule.condition
                if 'user_input_pattern' in cond and user_input_lower == cond['user_input_pattern']:
                    # Exact match greeting pattern
                    if 'response_template' in rule.action:
                        resp = rule.action['response_template']
                        # Replace placeholder if needed (e.g., [name])
                        return resp
                elif 'question_pattern' in cond and user_input_lower.rstrip('?') == cond['question_pattern']:
                    if 'answer_template' in rule.action:
                        return rule.action['answer_template']
                elif 'expected_user_input' in cond and user_input_lower == cond['expected_user_input']:
                    # This is a transition pattern; we might use the previous response as a clue
                    # For now, just acknowledge
                    return f"I see you're continuing the conversation."

        # --- 2. Try to understand the user's input using known definitions and relations ---
        tokens = self.engine._tokenize(user_input)
        # Check for exact phrase matches in phrase definitions
        for phrase in self.reasoner.phrase_definitions:
            if phrase in user_input_lower:
                def_text = self.reasoner.phrase_definitions[phrase]
                return f"Ah, '{phrase}' means {def_text}. That's interesting."

        # --- 2.5 NEW: Use word-state labels to provide contextual info ---
        for word in tokens:
            labels = self.reasoner.word_state.get_labels(word, threshold=0.6)
            if labels:
                label_str = ', '.join(labels)
                return f"I know '{word}' is associated with: {label_str}. Tell me more about it."

        # --- 2.2 Check if user input is similar to known phrases ---
        input_emb = self.semantic_learner.text_to_embedding(user_input, fallback_to_char=True)
        if input_emb is not None:
            similar = self.reasoner.find_similar_phrases(input_emb, threshold=0.8)
            if similar:
                best_phrase, sim = similar[0]
                return f"This reminds me of when you said '{best_phrase}'. Tell me more."
            
        # Check for words with definitions
        for word in tokens:
            if word in self.reasoner.definitions:
                def_text = self.reasoner.definitions[word][:50]
                relations = self.reasoner.phrase_relations.get(word, [])
                rel_strs = []
                for rel_type, obj, conf in relations[:2]:
                    if conf > 0.7:
                        rel_strs.append(f"{rel_type} {obj}")
                if rel_strs:
                    return f"I know that {word} is {def_text}. Also, it is {', '.join(rel_strs)}."
                else:
                    return f"I understand '{word}' as {def_text}. Tell me more."

        # --- 3. Fall back to trigram model from episodic memory ---
        # Build a simple trigram model from recent episodes
        trigram = defaultdict(Counter)
        episodes = self.engine.episodic_memory.episodes[-50:]  # recent
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
            if len(known_user_tokens) >= 2:
                seed = tuple(known_user_tokens[-2:])
            else:
                seed = None
            if seed is not None and seed in trigram:
                next_words = list(trigram[seed].keys())
                probs = np.array(list(trigram[seed].values()))
                probs = probs / probs.sum()
                next_w = np.random.choice(next_words, p=probs)
                return f"{' '.join(known_user_tokens)} {next_w} ... what do you think?"
            else:
                recent_ep = episodes[-1] if episodes else None
                if recent_ep:
                    topic = recent_ep.get('user_input', '').split()[:2]
                    topic = ' '.join(topic)
                    return f"Tell me more about {topic}."
        elif known_user_tokens:
            word = known_user_tokens[0]
            if word in self.reasoner.definitions:
                return f"I recall that {word} means {self.reasoner.definitions[word][:50]}."
            else:
                return f"What do you mean by '{word}'?"
        else:
            return "That's interesting. Can you tell me more?"
              
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

        # --- Priority 2: Pending hypotheses (synonym, category, analogy) ---
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') in ('synonym', 'category', 'analogy', 'inference') and not hyp.get('asked', False):
                hyp['asked'] = True
                return hyp['text']

        # --- Priority 3: High curiosity words (unknown words) ---
        if 'token_analyses' in state:
            high_curiosity = [ta for ta in state['token_analyses'] if ta.get('curiosity_score', 0) > 0.5]
            if high_curiosity:
                target = high_curiosity[0]['token']
                unc = high_curiosity[0].get('gp_uncertainty')
                return self.generate_question(target, unc)

        # --- Priority 4: State awareness utterance (30% chance) ---
        if self.engine.current_state_label is not None and np.random.rand() < 0.3:
            return f"I feel like I'm in a '{self.engine.current_state_label}' state right now."

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
            if self.engine.current_state_label is not None:
                utterance = f"As I'm {self.engine.current_state_label}, {utterance[0].lower()}{utterance[1:]}"
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
    print("🧠 AGI Curiosity Engine v0.4 - Information-Theoretic & Intrinsically Motivated")
    print("Integrated with backend.ci (Akkadian transliteration models)")
    print("State persists in ./curiosity_state/ - the system grows over time.")
    print("Now with: Gaussian Process uncertainty, Bayesian surprise, Intrinsic Reward.")
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
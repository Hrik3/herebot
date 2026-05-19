# Generated from: notebookcdfb2599d2.ipynb
# Converted at: 2025-12-21T11:55:01.440Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
from scipy.stats import norm, entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import string
import hashlib
import re
import threading
from collections import Counter
from backend.incremental_csr_lib import IncrementalCSR
from backend.herebot_memory import HerebotMemoryBridge

# =====================================================
# Loading Database (text.csv)
# =====================================================
CSR_MEMORY = IncrementalCSR("config_listner.json")
_DEFAULT_MEMORY_BRIDGE = None
_DEFAULT_MEMORY_LOCK = threading.RLock()



VALID_CHARS = set(string.ascii_lowercase + " -'") | set(string.ascii_uppercase)

# =====================================================
# Utility Functions
# =====================================================
def _safe_div(a, b):
    return a / b if b != 0 else 0.0


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def get_default_memory_bridge():
    global _DEFAULT_MEMORY_BRIDGE
    with _DEFAULT_MEMORY_LOCK:
        if _DEFAULT_MEMORY_BRIDGE is None:
            _DEFAULT_MEMORY_BRIDGE = HerebotMemoryBridge(csr=CSR_MEMORY)
        return _DEFAULT_MEMORY_BRIDGE


def add_sentence_to_memory(text: str):
    try:
        return get_default_memory_bridge().add_runtime_text(
            text,
            source="compat:add_sentence",
            doc_id=hashlib.sha256(str(text).encode("utf-8", errors="ignore")).hexdigest(),
        )
    except Exception:
        return CSR_MEMORY.add(text)

def load_all_shard_texts(memory_bridge=None):
    if memory_bridge is not None:
        try:
            return memory_bridge.get_corpus()
        except Exception:
            pass
    if hasattr(CSR_MEMORY, "get_all_texts"):
        try:
            return CSR_MEMORY.get_all_texts()
        except Exception:
            pass
    import glob, json, os
    texts = []
    shard_meta = glob.glob(os.path.join(CSR_MEMORY.shard_dir, "shard_*.meta.json"))
    for mf in shard_meta:
        try:
            with open(mf) as f:
                meta = json.load(f)
                if "texts" in meta:
                    texts.extend(meta["texts"])
        except:
            pass
    return texts


# =====================================================
# Base Feature Extraction
# =====================================================
class FeatureExtractor:
    """
    Extracts numerical linguistic/statistical features
    from a transliteration string.
    """

    def __init__(self, corpus):
        self.corpus = self._prepare_corpus(corpus)
        self._build_models()

    def _prepare_corpus(self, corpus):
        clean = []
        seen = set()
        for item in corpus or []:
            text = " ".join(str(item).split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(text)
        return clean

    def _build_models(self):
        if not self.corpus:
            raise ValueError("FeatureExtractor requires a non-empty real corpus")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 5)
        )
        self.vectorizer.fit(self.corpus)

        self.lengths = np.array([[len(t)] for t in self.corpus])
        self.outlier_model = None
        if len(self.lengths) >= 2:
            try:
                self.outlier_model = IsolationForest(
                    contamination=0.15,
                    random_state=42
                )
                self.outlier_model.fit(self.lengths)
            except Exception:
                self.outlier_model = None

    def character_validity(self, text):
        valid = sum(c.lower() in VALID_CHARS for c in text)
        return _safe_div(valid, len(text))
    
    def length_score(self, text):
        n = len(text)
        if n < 3:
            return 0.3
        if n > 300:
            return 0.6
        return 1.0



    def vowel_structure(self, text):
        vowels = "aeiouāēīūâêîû"
        v = sum(c in vowels for c in text.lower())
        return _clip01(_safe_div(v, len(text)) / 0.5)

    def tfidf_similarity(self, text):
        vecs = self.vectorizer.transform(self.corpus + [str(text)])
        sims = cosine_similarity(vecs[-1], vecs[:-1])[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_text = self.corpus[best_idx]

        return {
            "score": best_score,
            "index": best_idx,
            "matched_text": best_text
        }


    def outlier_score(self, text):
        if self.outlier_model is None:
            return 1.0
        pred = self.outlier_model.predict([[len(text)]])[0]
        return 1.0 if pred == 1 else 0.4

    def extract(self, text):
        text = "" if text is None else str(text)
        sim = self.tfidf_similarity(text)

        return {
            "char_valid": self.character_validity(text),
            "length": self.length_score(text),
            "structure": self.vowel_structure(text),
            "outlier": self.outlier_score(text),
            "similarity": sim["score"],
            "_match_index": sim["index"],
            "_matched_text": sim["matched_text"],
        }

# =====================================================
# Probabilistic Model
# =====================================================
class ProbabilisticCIModel:
    """
    Converts feature vectors into a probabilistic
    confidence estimate.
    """

    def __init__(self):
        self.weights = {
            "char_valid": 0.20,
            "length": 0.15,
            "structure": 0.20,
            "similarity": 0.30,
            "outlier": 0.15,
        }

    def score(self, features):
        s = 0.0
        for k, w in self.weights.items():
            s += w * features.get(k, 0.0)
        return _clip01(s)

    def uncertainty(self, scores):
        """
        Measures uncertainty as variance
        """
        return float(np.std(scores))

    def to_distribution(self, scores):
        mu = float(np.mean(scores))
        sigma = float(np.std(scores) + 1e-6)
        return mu, sigma


# =====================================================
# Monte Carlo Perturbation Engine
# =====================================================
class MonteCarloEngine:
    """
    Tests stability by random perturbation.
    """

    def __init__(self, extractor, model):
        self.extractor = extractor
        self.model = model

    def perturb(self, text, p=0.08):
        chars = list(text)
        for i in range(len(chars)):
            if np.random.rand() < p:
                chars[i] = ""
        return "".join(chars)

    def sample(self, text, n=50):
        scores = []
        for _ in range(n):
            t = self.perturb(text)
            feats = self.extractor.extract(t)
            scores.append(self.model.score(feats))
        return scores


# =====================================================
# Public CI Engine (MAIN ENTRY POINT)
# =====================================================
class CIEngine:
    """
    High-level interface for competition submission.
    """
    def __init__(self, memory_bridge=None, corpus=None, config_manager=None):
        if corpus is None and memory_bridge is not None and not hasattr(memory_bridge, "get_corpus"):
            corpus = memory_bridge
            memory_bridge = None
        self.memory_bridge = memory_bridge
        self.config_manager = config_manager
        self._explicit_corpus = corpus is not None
        self._lock = threading.RLock()

        # Always define corpus first
        self.corpus = []

        # If corpus explicitly provided
        if corpus is not None:
            self.corpus = self._prepare_corpus(corpus)
        else:
            self.corpus = self._prepare_corpus(self._read_corpus())

        # Build engines
        self._corpus_fingerprint = self._fingerprint(self.corpus)
        self._build_engines()

    def _prepare_corpus(self, corpus):
        clean = []
        seen = set()
        for item in corpus or []:
            text = " ".join(str(item).split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(text)
        return clean

    def _fingerprint(self, corpus):
        h = hashlib.sha256()
        for text in corpus or []:
            h.update(str(text).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    def _read_corpus(self):
        if self.memory_bridge is not None:
            try:
                return self.memory_bridge.get_corpus()
            except Exception:
                return []
        try:
            return load_all_shard_texts()
        except Exception:
            return []

    def _build_engines(self):
        self.extractor = FeatureExtractor(self.corpus) if self.corpus else None
        self.model = ProbabilisticCIModel()
        self.mc = MonteCarloEngine(self.extractor, self.model) if self.extractor is not None else None

    def refresh_corpus(self, force=False):
        if self._explicit_corpus and not force:
            return False
        new_corpus = self._prepare_corpus(self._read_corpus())
        fp = self._fingerprint(new_corpus)
        if not force and fp == self._corpus_fingerprint:
            return False
        with self._lock:
            self.corpus = new_corpus
            self._corpus_fingerprint = fp
            self._build_engines()
        return True

    def evaluate(self, text, samples=60):
        self.refresh_corpus()
        if not self.corpus or self.extractor is None or self.mc is None:
            return self._neutral_report(mode="insufficient_corpus")

        scores = []
        match_indices = []
        matched_texts = []

        for _ in range(samples):
            t = self.mc.perturb(text)
            feats = self.extractor.extract(t)
            scores.append(self.model.score(feats))
            match_indices.append(feats["_match_index"])
            matched_texts.append(feats["_matched_text"])

        mu, sigma = self.model.to_distribution(scores)
        stability = _clip01(1 - sigma)
        confidence = _clip01(mu * stability)
        # Most frequent match
        best_match_idx = int(pd.Series(match_indices).mode()[0]) if match_indices else 0
        best_match_text = pd.Series(matched_texts).mode()[0] if matched_texts else ""

        return {
            "confidence": confidence,
            "mean": mu,
            "std": sigma,
            "stability": stability,
            "matched_index": best_match_idx,
            "matched_text": best_match_text,
            "corpus_size": len(self.corpus),
            "backend_source": "bridge" if self.memory_bridge is not None else "csr",
            "mode": "corpus",
            "diagnostics": {
                "reason": "real_corpus_scoring",
                "corpus_size": len(self.corpus),
            },
            "distribution": {
                "type": "normal",
                "mu": mu,
                "sigma": sigma
            }
        }

    def _neutral_report(self, mode="insufficient_corpus"):
        return {
            "confidence": 0.5,
            "mean": 0.5,
            "std": 0.0,
            "stability": 0.5,
            "matched_index": -1,
            "matched_text": "",
            "corpus_size": len(self.corpus),
            "backend_source": "bridge" if self.memory_bridge is not None else "csr",
            "mode": mode,
            "diagnostics": {
                "reason": "insufficient_real_corpus",
                "corpus_size": len(self.corpus),
            },
            "distribution": {
                "type": "normal",
                "mu": 0.5,
                "sigma": 0.0,
            },
        }

import numpy as np
import pandas as pd
from scipy.stats import entropy


# =====================================================
# Character classes for corpus-adaptive phonotactic scoring
# =====================================================
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

ALLOWED_CHARS = VOWELS | CONSONANTS | {"-", " "}


# =====================================================
# Helper Functions
# =====================================================
def _is_vowel(c):
    return c in VOWELS


def _is_consonant(c):
    return c in CONSONANTS


def _safe_div(a, b):
    return a / b if b else 0.0


# =====================================================
# Syllable Pattern Analyzer
# =====================================================
class SyllableAnalyzer:
    """
    Detects syllable shapes like:
    CV, CVC, VC, V
    """

    def syllabify(self, word):
        patterns = []
        current = ""

        for c in word:
            if _is_vowel(c):
                current += "V"
                patterns.append(current)
                current = ""
            elif _is_consonant(c):
                current += "C"

        if current:
            patterns.append(current)

        return patterns

    def score(self, word):
        syllables = self.syllabify(word)
        if not syllables:
            return 0.0

        valid = {"V", "CV", "CVC", "VC"}
        score = sum(1 for s in syllables if s in valid)

        return _safe_div(score, len(syllables))


# =====================================================
# Consonant Cluster Analysis
# =====================================================
class ClusterAnalyzer:
    """
    Penalizes unnatural consonant stacking.
    """

    def score(self, word):
        max_cluster = 0
        current = 0

        for c in word:
            if _is_consonant(c):
                current += 1
                max_cluster = max(max_cluster, current)
            else:
                current = 0

        if max_cluster <= 3:
            return 1.0
        if max_cluster == 4:
            return 0.6
        return 0.3



# =====================================================
# Vowel Harmony & Ratio
# =====================================================
class VowelAnalyzer:
    """
    Evaluates vowel balance and entropy.
    """

    def score(self, word):
        vowels = [c for c in word if _is_vowel(c)]
        if not vowels:
            return 0.0

        ratio = len(vowels) / len(word)

        counts = pd.Series(vowels).value_counts(normalize=True)
        ent = entropy(counts)

        ratio_score = 1 - abs(ratio - 0.45)
        entropy_score = np.clip(ent / 2.5, 0, 1)

        return np.clip(0.5 * ratio_score + 0.5 * entropy_score, 0, 1)


# =====================================================
# Forbidden Sequence Detector
# =====================================================
class ForbiddenSequenceAnalyzer:
    """
    Penalizes unstable character runs and low-signal clusters.
    """

    FORBIDDEN = [
        "aaa","eee","iii",
        "zzz","qqq",
        "zxq","qzx","jxq"
    ]


    def score(self, word):
        penalties = sum(1 for f in self.FORBIDDEN if f in word)
        if penalties == 0:
            return 1.0
        if penalties == 1:
            return 0.5
        return 0.2


# =====================================================
# Morpheme Boundary Analyzer
# =====================================================
class MorphemeAnalyzer:
    """
    Evaluates realism of hyphen usage.
    """

    def score(self, word):
        if "-" not in word:
            return 1.0

        parts = word.split("-")
        if any(len(p) < 2 for p in parts):
            return 0.4

        return 0.9


class DataAdaptivePhonProfile:
    """Learns simple phonotactic norms from the unified Herebot corpus."""

    def __init__(self, min_words=10):
        self.min_words = min_words
        self.active = False
        self.word_count = 0
        self.ngrams = Counter()
        self.patterns = Counter()
        self.vowel_ratio_mean = 0.45
        self.vowel_ratio_std = 0.2
        self.max_cluster_norm = 3

    def rebuild(self, corpus, syllable_analyzer=None):
        words = self._words_from_corpus(corpus)
        self.word_count = len(words)
        self.active = self.word_count >= self.min_words
        self.ngrams = Counter()
        self.patterns = Counter()
        ratios = []
        clusters = []
        for word in words:
            ratios.append(_safe_div(sum(1 for c in word if _is_vowel(c)), len(word)))
            clusters.append(self._max_cluster(word))
            for n in (2, 3):
                for gram in self._ngrams(word, n):
                    self.ngrams[gram] += 1
            if syllable_analyzer is not None:
                for pattern in syllable_analyzer.syllabify(word):
                    self.patterns[pattern] += 1
        if ratios:
            self.vowel_ratio_mean = float(np.mean(ratios))
            self.vowel_ratio_std = max(float(np.std(ratios)), 0.05)
        if clusters:
            self.max_cluster_norm = max(1, int(np.percentile(clusters, 90)))
        return self.active

    def score(self, word, syllable_patterns=None):
        if not self.active:
            return {
                "score": None,
                "ngram": None,
                "vowel_ratio": None,
                "syllable_pattern": None,
                "cluster_norm": None,
                "profile_words": self.word_count,
            }
        word = self._clean_word(word)
        if not word:
            return {
                "score": 0.0,
                "ngram": 0.0,
                "vowel_ratio": 0.0,
                "syllable_pattern": 0.0,
                "cluster_norm": 0.0,
                "profile_words": self.word_count,
            }
        ngram_score = self._ngram_score(word)
        ratio = _safe_div(sum(1 for c in word if _is_vowel(c)), len(word))
        ratio_score = _clip01(1.0 - abs(ratio - self.vowel_ratio_mean) / (2.0 * self.vowel_ratio_std))
        cluster_score = 1.0 if self._max_cluster(word) <= self.max_cluster_norm else 0.45
        pattern_score = self._pattern_score(syllable_patterns or [])
        combined = _clip01(
            0.35 * ngram_score +
            0.25 * ratio_score +
            0.20 * pattern_score +
            0.20 * cluster_score
        )
        return {
            "score": combined,
            "ngram": ngram_score,
            "vowel_ratio": ratio_score,
            "syllable_pattern": pattern_score,
            "cluster_norm": cluster_score,
            "profile_words": self.word_count,
        }

    def _words_from_corpus(self, corpus):
        words = []
        for text in corpus or []:
            for raw in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", str(text).lower()):
                word = self._clean_word(raw)
                if len(word) >= 2:
                    words.append(word)
        return words

    def _clean_word(self, word):
        return "".join(c for c in str(word).lower() if c in ALLOWED_CHARS).strip()

    def _ngrams(self, word, n):
        if len(word) < n:
            return []
        return [word[i:i + n] for i in range(len(word) - n + 1)]

    def _ngram_score(self, word):
        grams = self._ngrams(word, 2) + self._ngrams(word, 3)
        if not grams:
            return 0.0
        known = sum(1 for gram in grams if self.ngrams.get(gram, 0) > 0)
        return _safe_div(known, len(grams))

    def _pattern_score(self, syllable_patterns):
        if not syllable_patterns:
            return 0.0
        known = sum(1 for pattern in syllable_patterns if self.patterns.get(pattern, 0) > 0)
        return _safe_div(known, len(syllable_patterns))

    def _max_cluster(self, word):
        max_cluster = 0
        current = 0
        for c in word:
            if _is_consonant(c):
                current += 1
                max_cluster = max(max_cluster, current)
            else:
                current = 0
        return max_cluster


# =====================================================
# Phonotactic Engine (PUBLIC)
# =====================================================
class PhonotacticEngine:
    """
    Aggregates phonotactic realism scores.
    """

    def __init__(self, memory_bridge=None, fallback_mode=True):
        self.memory_bridge = memory_bridge
        self.fallback_mode = fallback_mode
        self.syllable = SyllableAnalyzer()
        self.cluster = ClusterAnalyzer()
        self.vowel = VowelAnalyzer()
        self.forbidden = ForbiddenSequenceAnalyzer()
        self.morpheme = MorphemeAnalyzer()
        self.profile = DataAdaptivePhonProfile()
        self.profile_active = False
        self._corpus_fingerprint = None
        self.refresh_profile(force=True)

    def _base_score(self, scores):
        return (
            0.25 * scores["syllable"] +
            0.20 * scores["cluster"] +
            0.20 * scores["vowel"] +
            0.20 * scores["forbidden"] +
            0.15 * scores["morpheme"]
        )

    def refresh_profile(self, force=False):
        if self.memory_bridge is None:
            self.profile_active = False
            return False
        try:
            corpus = self.memory_bridge.get_corpus()
        except Exception:
            corpus = []
        fp = hashlib.sha256("\0".join(str(t) for t in corpus).encode("utf-8", errors="ignore")).hexdigest()
        if not force and fp == self._corpus_fingerprint:
            return False
        self._corpus_fingerprint = fp
        self.profile_active = self.profile.rebuild(corpus, syllable_analyzer=self.syllable)
        return True

    def score_word(self, word):
        syllable_patterns = self.syllable.syllabify(word)
        scores = {
            "syllable": self.syllable.score(word),
            "cluster": self.cluster.score(word),
            "vowel": self.vowel.score(word),
            "forbidden": self.forbidden.score(word),
            "morpheme": self.morpheme.score(word),
        }
        adaptive = self.profile.score(word, syllable_patterns)
        base_score = self._base_score(scores)
        if adaptive["score"] is not None:
            combined = _clip01(0.65 * adaptive["score"] + 0.35 * base_score)
            mode = "data_adaptive"
            fallback_used = False
        else:
            combined = base_score if self.fallback_mode else 0.0
            mode = "fallback_heuristic" if self.fallback_mode else "insufficient_corpus"
            fallback_used = self.fallback_mode
        scores.update({
            "data_adaptive": adaptive["score"],
            "profile_words": adaptive["profile_words"],
            "mode": mode,
            "fallback_used": fallback_used,
            "adaptive_components": {
                "ngram": adaptive["ngram"],
                "vowel_ratio": adaptive["vowel_ratio"],
                "syllable_pattern": adaptive["syllable_pattern"],
                "cluster_norm": adaptive["cluster_norm"],
            },
            "combined": combined,
        })
        return scores

    def score_text(self, text):
        self.refresh_profile()
        words = [w for w in text.split() if w]
        if not words:
            return 0.0, {}

        word_scores = []
        diagnostics = []

        for w in words:
            s = self.score_word(w)
            diagnostics.append(s)
            word_scores.append(s["combined"])

        return float(np.mean(word_scores)), diagnostics

import numpy as np


# =====================================================
# Text Perturbation Operators
# =====================================================
class TextPerturbator:
    """
    Applies controlled noise to text.
    """

    def __init__(self, deletion_prob=0.1, swap_prob=0.05):
        self.deletion_prob = deletion_prob
        self.swap_prob = swap_prob

    def delete_chars(self, text):
        chars = []
        for c in text:
            if np.random.rand() > self.deletion_prob:
                chars.append(c)
        return "".join(chars)

    def swap_chars(self, text):
        chars = list(text)
        for i in range(len(chars) - 1):
            if np.random.rand() < self.swap_prob:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def perturb(self, text):
        text = self.delete_chars(text)
        text = self.swap_chars(text)
        return text


# =====================================================
# Stability Analyzer
# =====================================================
class StabilityAnalyzer:
    """
    Measures confidence stability under perturbation.
    """

    def __init__(self, ci_engine, phonotactic_engine):
        self.ci_engine = ci_engine
        self.phonotactic_engine = phonotactic_engine
        self.perturbator = TextPerturbator()

    def run_trials(self, text, trials=50):
        scores = []
        phon_scores = []

        for _ in range(trials):
            corrupted = self.perturbator.perturb(text)

            ci_report = self.ci_engine.evaluate(corrupted)
            phon_score, _ = self.phonotactic_engine.score_text(corrupted)

            scores.append(ci_report["confidence"])
            phon_scores.append(phon_score)

        return np.array(scores), np.array(phon_scores)

    def stability_metrics(self, text, trials=50):
        ci_scores, phon_scores = self.run_trials(text, trials)

        return {
            "ci_mean": float(np.mean(ci_scores)),
            "ci_std": float(np.std(ci_scores)),
            "ci_min": float(np.min(ci_scores)),
            "ci_max": float(np.max(ci_scores)),
            "phon_mean": float(np.mean(phon_scores)),
            "phon_std": float(np.std(phon_scores)),
            "robustness": float(1 - np.std(ci_scores)),
        }

import numpy as np


# =====================================================
# Explanation Templates
# =====================================================
POSITIVE_REASONS = {
    "char_valid": "High character validity (correct transliteration symbols).",
    "structure": "Natural vowel–consonant balance.",
    "similarity": "Strong similarity to known memory patterns.",
    "phonotactics": "Valid phonotactic structure (syllables and clusters).",
    "stability": "Confidence remains stable under perturbation.",
}

NEGATIVE_REASONS = {
    "char_valid": "Presence of unusual or invalid characters.",
    "structure": "Unnatural vowel–consonant ratio.",
    "similarity": "Low similarity to reference corpus.",
    "phonotactics": "Illegal syllable shapes or consonant clusters.",
    "stability": "Confidence unstable under small perturbations.",
}


# =====================================================
# Feature Attribution Engine
# =====================================================
class FeatureAttribution:
    """
    Determines which features contributed
    positively or negatively to confidence.
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            "good": 0.7,
            "bad": 0.4
        }

    def attribute(self, features):
        positives = []
        negatives = []
        neutral = []

        for name, value in features.items():
            if value >= self.thresholds["good"]:
                positives.append(name)
            elif value <= self.thresholds["bad"]:
                negatives.append(name)
            else:
                neutral.append(name)

        return positives, negatives, neutral


# =====================================================
# Failure Mode Analyzer
# =====================================================
class FailureModeAnalyzer:
    """
    Attempts to infer *why* confidence degraded.
    """

    def analyze(self, features, stability):
        reasons = []

        if features.get("char_valid", 1) < 0.5:
            reasons.append("Character corruption detected.")

        if features.get("structure", 1) < 0.5:
            reasons.append("Abnormal vowel–consonant structure.")

        if features.get("similarity", 1) < 0.4:
            reasons.append("Text diverges from known memory patterns.")

        if features.get("phonotactics", 1) < 0.5:
            reasons.append("Phonotactic violations detected.")

        if stability < 0.6:
            reasons.append("Confidence highly sensitive to noise.")

        if not reasons:
            reasons.append("No dominant failure mode detected.")

        return reasons


# =====================================================
# Human-Readable Explanation Generator
# =====================================================
class ExplanationGenerator:
    """
    Converts diagnostics into natural-language explanations.
    """

    def __init__(self):
        self.attributor = FeatureAttribution()
        self.failure_analyzer = FailureModeAnalyzer()

    def generate(self, features, stability):
        positives, negatives, neutral = self.attributor.attribute(features)

        explanation = {
            "summary": "",
            "positives": [],
            "negatives": [],
            "neutral": [],
            "failure_modes": []
        }

        # Positive contributions
        for p in positives:
            explanation["positives"].append(
                POSITIVE_REASONS.get(p, f"Strong performance in {p}.")
            )

        # Negative contributions
        for n in negatives:
            explanation["negatives"].append(
                NEGATIVE_REASONS.get(n, f"Weak performance in {n}.")
            )

        # Neutral features
        for n in neutral:
            explanation["neutral"].append(
                f"{n} is within acceptable range."
            )

        # Failure analysis
        explanation["failure_modes"] = self.failure_analyzer.analyze(
            features, stability
        )

        # Summary sentence
        if explanation["negatives"]:
            explanation["summary"] = (
                "Confidence reduced due to identifiable linguistic or stability issues."
            )
        else:
            explanation["summary"] = (
                "High confidence supported by strong linguistic and statistical evidence."
            )

        return explanation


# =====================================================
# Public Diagnostics Interface
# =====================================================
class DiagnosticEngine:
    """
    High-level explainability interface.
    """

    def __init__(self):
        self.generator = ExplanationGenerator()

    def explain(self, ci_report, phon_score):
        """
        ci_report: output from CIEngine.evaluate()
        phon_score: phonotactic score from Module 2
        """

        features = {
            "char_valid": ci_report["mean"],       # proxy
            "structure": phon_score,
            "similarity": ci_report["mean"],
            "phonotactics": phon_score,
            "stability": ci_report["stability"]
        }

        return self.generator.generate(
            features,
            ci_report["stability"]
        )

import numpy as np


def _clean_eval_texts(texts):
    clean = []
    for text in texts or []:
        value = " ".join(str(text).split())
        if value:
            clean.append(value)
    return clean


# =====================================================
# Benchmark Evaluator
# =====================================================
class BenchmarkEvaluator:
    """
    Compares confidence scores for caller-provided real positive/negative sets.
    """

    def __init__(self, ci_engine):
        self.ci_engine = ci_engine

    def evaluate(self, positives, negatives):
        positives = _clean_eval_texts(positives)
        negatives = _clean_eval_texts(negatives)
        if not positives or not negatives:
            return {
                "ok": False,
                "mode": "missing_real_comparison_data",
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "reason": "benchmark_requires_real_positive_and_negative_sets",
            }
        pos_scores = [
            self.ci_engine.evaluate(t)["confidence"]
            for t in positives
        ]
        neg_scores = [
            self.ci_engine.evaluate(t)["confidence"]
            for t in negatives
        ]

        return {
            "ok": True,
            "mode": "real_comparison",
            "positive_mean": float(np.mean(pos_scores)),
            "positive_std": float(np.std(pos_scores)),
            "negative_mean": float(np.mean(neg_scores)),
            "negative_std": float(np.std(neg_scores)),
            "separation": float(
                np.mean(pos_scores) - np.mean(neg_scores)
            )
        }


# =====================================================
# Robustness Curve Analyzer
# =====================================================
class RobustnessAnalyzer:
    """
    Measures confidence across caller-provided real text variants.
    """

    def __init__(self, ci_engine):
        self.ci_engine = ci_engine

    def robustness_curve(self, variants_by_level=None):
        if not variants_by_level:
            return {
                "ok": False,
                "mode": "missing_real_variants",
                "reason": "robustness_requires_caller_provided_text_variants",
            }
        curve = {}

        for level, variants in variants_by_level.items():
            real_variants = _clean_eval_texts(variants)
            if not real_variants:
                curve[level] = {
                    "ok": False,
                    "mean": None,
                    "count": 0,
                }
                continue
            scores = [self.ci_engine.evaluate(variant)["confidence"] for variant in real_variants]
            curve[level] = {
                "ok": True,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "count": len(scores),
            }

        return {
            "ok": True,
            "mode": "real_variants",
            "curve": curve,
        }


# =====================================================
# Ablation Study
# =====================================================
class AblationStudy:
    """
    Reports component scores without simulating removed components.
    """

    def __init__(self, ci_engine, phon_engine=None):
        self.ci_engine = ci_engine
        self.phon_engine = phon_engine

    def run(self, text):
        results = {}

        results["full"] = self.ci_engine.evaluate(text)["confidence"]

        if self.phon_engine:
            phon_score, _ = self.phon_engine.score_text(text)
            results["phonotactics"] = phon_score

        results["mode"] = "component_report"
        results["ablation_available"] = False
        results["reason"] = "no_alternate_real_engine_supplied"

        return results


# =====================================================
# Evaluation Orchestrator (PUBLIC)
# =====================================================
class EvaluationSuite:
    """
    High-level evaluation interface.
    """

    def __init__(self, ci_engine, phon_engine, corpus=None):
        self.corpus = _clean_eval_texts(corpus)
        self.benchmark = BenchmarkEvaluator(ci_engine)
        self.robustness = RobustnessAnalyzer(ci_engine)
        self.ablation = AblationStudy(ci_engine, phon_engine)

    def full_evaluation(self, text, corpus=None, negatives=None, robustness_variants=None):
        positives = _clean_eval_texts(corpus) or self.corpus
        benchmark_results = self.benchmark.evaluate(positives, negatives)
        robustness_results = self.robustness.robustness_curve(robustness_variants)

        ablation_results = self.ablation.run(text)

        return {
            "benchmark": benchmark_results,
            "robustness": robustness_results,
            "ablation": ablation_results
        }

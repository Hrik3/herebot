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
from . incremental_csr_lib import IncrementalCSR

# =====================================================
# Loading Database (text.csv)
# =====================================================
CSR_MEMORY = IncrementalCSR("config_listner.json")



VALID_CHARS = set(string.ascii_lowercase + " -'") | set(string.ascii_uppercase)

# =====================================================
# Utility Functions
# =====================================================
def _safe_div(a, b):
    return a / b if b != 0 else 0.0


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def add_sentence_to_memory(text: str):
    CSR_MEMORY.add(text)

def load_all_shard_texts():
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
        self.corpus = corpus
        self._build_models()

    def _build_models(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5)
        )
        self.vectorizer.fit(self.corpus)

        self.lengths = np.array([[len(t)] for t in self.corpus])
        self.outlier_model = IsolationForest(
            contamination=0.15,
            random_state=42
        )
        self.outlier_model.fit(self.lengths)

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
        vecs = self.vectorizer.transform(self.corpus + [text])
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
        pred = self.outlier_model.predict([[len(text)]])[0]
        return 1.0 if pred == 1 else 0.4

    def extract(self, text):
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
    def __init__(self, corpus=None):

        # Always define corpus first
        self.corpus = []

        # If corpus explicitly provided
        if corpus is not None:
            self.corpus = corpus
        else:
            X = CSR_MEMORY.load_full()
            if X.shape[0] > 0:
                self.corpus = load_all_shard_texts()

        # Fallback seed so vectorizer never crashes
        if not self.corpus:
            self.corpus = ["hello world"]

        # Build engines
        self.extractor = FeatureExtractor(self.corpus)
        self.model = ProbabilisticCIModel()
        self.mc = MonteCarloEngine(self.extractor, self.model)

    def evaluate(self, text, samples=60):
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
        new_corpus = load_all_shard_texts()

        if len(new_corpus) > len(self.corpus):
            self.corpus = new_corpus
            self.extractor = FeatureExtractor(self.corpus)
        # Most frequent match
        best_match_idx = int(pd.Series(match_indices).mode()[0])
        best_match_text = pd.Series(matched_texts).mode()[0]

        return {
            "confidence": confidence,
            "mean": mu,
            "std": sigma,
            "stability": stability,
            "matched_index": best_match_idx,
            "matched_text": best_match_text,
            "distribution": {
                "type": "normal",
                "mu": mu,
                "sigma": sigma
            }
        }

import numpy as np
import pandas as pd
from scipy.stats import entropy


# =====================================================
# Phonetic Definitions (Akkadian Transliteration)
# =====================================================
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

ALLOWED_CHARS = VOWELS | CONSONANTS | {"-", " "}


# =====================================================
# Reference Phonotactic Corpus
# =====================================================
REFERENCE_WORDS = [
    "ša", "naqba", "īmuru", "ana", "šarri", "bēliya",
    "mār", "šipri", "šarrum", "awīlum", "ilum",
    "rabûm", "ṭuppi", "bēlum", "šumšu"
]


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
    Penalizes phoneme sequences that almost never
    appear in Akkadian transliteration.
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


# =====================================================
# Phonotactic Engine (PUBLIC)
# =====================================================
class PhonotacticEngine:
    """
    Aggregates phonotactic realism scores.
    """

    def __init__(self):
        self.syllable = SyllableAnalyzer()
        self.cluster = ClusterAnalyzer()
        self.vowel = VowelAnalyzer()
        self.forbidden = ForbiddenSequenceAnalyzer()
        self.morpheme = MorphemeAnalyzer()

    def score_word(self, word):
        return {
            "syllable": self.syllable.score(word),
            "cluster": self.cluster.score(word),
            "vowel": self.vowel.score(word),
            "forbidden": self.forbidden.score(word),
            "morpheme": self.morpheme.score(word),
        }

    def score_text(self, text):
        words = [w for w in text.split() if w]
        if not words:
            return 0.0, {}

        word_scores = []
        diagnostics = []

        for w in words:
            s = self.score_word(w)
            diagnostics.append(s)
            word_scores.append(
                0.25 * s["syllable"] +
                0.20 * s["cluster"] +
                0.20 * s["vowel"] +
                0.20 * s["forbidden"] +
                0.15 * s["morpheme"]
            )

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
    "similarity": "Strong similarity to known Akkadian patterns.",
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
            reasons.append("Text diverges from known Akkadian patterns.")

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
import random


# =====================================================
# Synthetic Data Generator
# =====================================================
class SyntheticDataGenerator:
    """
    Generates corrupted / fake transliterations
    for evaluation.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.chars = list("abcdefghijklmnopqrstuvwxyz")

    def corrupt(self, text, p=0.2):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p:
                chars[i] = random.choice(self.chars)
        return "".join(chars)

    def generate_negative_samples(self, n=50):
        samples = []
        for _ in range(n):
            base = random.choice(self.corpus)
            samples.append(self.corrupt(base))
        return samples


# =====================================================
# Benchmark Evaluator
# =====================================================
class BenchmarkEvaluator:
    """
    Compares confidence scores for
    real vs corrupted transliterations.
    """

    def __init__(self, ci_engine):
        self.ci_engine = ci_engine

    def evaluate(self, positives, negatives):
        pos_scores = [
            self.ci_engine.evaluate(t)["confidence"]
            for t in positives
        ]
        neg_scores = [
            self.ci_engine.evaluate(t)["confidence"]
            for t in negatives
        ]

        return {
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
    Measures confidence degradation under noise.
    """

    def __init__(self, ci_engine, generator):
        self.ci_engine = ci_engine
        self.generator = generator

    def robustness_curve(self, text, noise_levels):
        curve = {}

        for p in noise_levels:
            corrupted = self.generator.corrupt(text, p=p)
            score = self.ci_engine.evaluate(corrupted)["confidence"]
            curve[p] = score

        return curve


# =====================================================
# Ablation Study
# =====================================================
class AblationStudy:
    """
    Evaluates system performance when
    components are removed.
    """

    def __init__(self, ci_engine, phon_engine=None):
        self.ci_engine = ci_engine
        self.phon_engine = phon_engine

    def run(self, text):
        results = {}

        # Full system
        results["full"] = self.ci_engine.evaluate(text)["confidence"]

        # Without phonotactics (simulated)
        if self.phon_engine:
            phon_score, _ = self.phon_engine.score_text(text)
            reduced = max(0.0, results["full"] - 0.3 * phon_score)
            results["without_phonotactics"] = reduced

        # Minimal baseline
        results["baseline"] = 0.5

        return results


# =====================================================
# Evaluation Orchestrator (PUBLIC)
# =====================================================
class EvaluationSuite:
    """
    High-level evaluation interface.
    """

    def __init__(self, ci_engine, phon_engine, corpus):
        self.generator = SyntheticDataGenerator(corpus)
        self.benchmark = BenchmarkEvaluator(ci_engine)
        self.robustness = RobustnessAnalyzer(ci_engine, self.generator)
        self.ablation = AblationStudy(ci_engine, phon_engine)

    def full_evaluation(self, text, corpus):
        negatives = self.generator.generate_negative_samples()
        benchmark_results = self.benchmark.evaluate(corpus, negatives)

        robustness_results = self.robustness.robustness_curve(
            text,
            noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3]
        )

        ablation_results = self.ablation.run(text)

        return {
            "benchmark": benchmark_results,
            "robustness": robustness_results,
            "ablation": ablation_results
        }
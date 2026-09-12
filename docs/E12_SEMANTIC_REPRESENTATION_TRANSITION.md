# E12 — Semantic Representation Transition Report

Date: 2026-09-12  
Status: RESOLVED & VERIFIED

---

## 1. Executive Summary and Scope

This transition report documents the theory-constrained root-cause repair of **E12 (Semantic Representation Collapse)** in Herebot.

E10 (factor-space versioning) remains closed and unchanged. This work specifically targets the degeneration of linguistic and phrase semantics where:
- 110 of 122 words in `semantic_learner.pkl` collapsed into zero vectors.
- Surviving non-zero words had been contaminated by positive affective factor vectors, yielding mutual cosine similarity $\ge 0.9439$.
- Content-words contributing zero caused definition phrases (e.g. `what is a word`, `moon is a thing`) to collapse into literal numerical identity ($1.0000000000$) dominated by the OOV character-hash of the connector `'a'`.
- Spurious synonym and phrase hypotheses inundated the cognitive reasoning queue.

The repair was executed strictly under Herebot's non-transformer theoretical commitments:
- **Complementary Learning Systems (CLS)**: Preserving the boundary between rapid episodic traces and slower, generalized semantic representation.
- **Intrinsic Motivation / Curiosity**: Ensuring hypotheses are treated as exploratory questions rather than empirical truth.
- **Bayesian Prior / Information Gain**: Replacing the zero-absorbing singularity with an informative, deterministic subword empirical prior while allowing variational updates to rotate vectors according to context.
- **World Model / Sleep Safety**: Eliminating recursive self-talk and cross-modal contamination loops without disabling sleep.
- **Factor / Semantic Isolation**: Removing direct writes from affective control space into lexical semantic space.

---

## 2. Representation Boundaries and Dependency Graph

Herebot comprises distinct representation spaces that must not be conflated merely because they are serialized as vectors:

```
+-------------------------------------------------------------------------+
|                         HEREBOT COGNITIVE ARCHITECTURE                  |
+-------------------------------------------------------------------------+
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────┐                         ┌─────────────────────┐
│  AFFECTIVE CONTROL      │                         │  LEXICAL SEMANTIC   │
│  (FactorSpace, E10)     │                         │  (SemanticLearner)  │
│  Dim: Dynamic [11, 13]  │                         │  Dim: 64            │
│  Curiosity, Threat,     │                         │  e_w in R^64        │
│  Energy, Trust, etc.    │                         │  Unit sphere S^63   │
└──────────┬──────────────┘                         └──────────┬──────────┘
           │                                                   │
           ▼                                                   ▼
┌─────────────────────────┐                         ┌─────────────────────┐
│  State Space            │                         │  Phrase / Sentence  │
│  (StateManager)         │                         │  (ReasoningEngine)  │
│  Attractor vectors      │                         │  Content-weighted   │
│  in FactorSpace         │                         │  compositions       │
└──────────┬──────────────┘                         └──────────┬──────────┘
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │          SYMBOLIC MEDIATION            │
                │        (WordStateAssociator)           │
                │  Discrete word <-> state associations  │
                │  with probabilistic confidence scores  │
                │  (NO vector contamination)             │
                └────────────────────────────────────────┘
```

### Direct Dependency Audit:
1. **`word_embeddings` (Lexical semantic space, $\mathbb{R}^{64}$):**
   - *Writers:* `SemanticLearner.initialize_word()`, `observe_context()`, `observe_definition()`, `load()`.
   - *Forbidden Writers (Removed):* `CuriosityEngine._update_state_embedding()` (line 466) and `CuriosityEngine.cross_modal_alignment()` (line 564).
2. **`phrase_embeddings` (Reasoned phrase space, $\mathbb{R}^{64}$):**
   - *Writers:* `ReasoningEngine.store_phrase()` using deterministic SHA-256 keys.
   - *Readers:* `ReasoningEngine.find_similar_phrases()`.
3. **`definition_embeddings` (Stored definiendum representations, $\mathbb{R}^{64}$):**
   - *Writers:* `ReasoningEngine.store_definition()`.
   - *Readers:* `ReasoningEngine.find_similar_definitions()`.
4. **`FactorSpace` / Control vectors ($\mathbb{R}^{d_{factor}}$):**
   - Owned by `AdaptiveControlSpace` and `MetaCuriosityLearner`. Sized and versioned by E10 contracts.
5. **`StateManager` states:**
   - Saved factor vectors representing affective system states, mapped to labels.

---

## 3. Analysis and Challenge of Candidate Interventions

As required by Section 13, each proposed intervention was critically analyzed before implementation:

### 1. Deterministic Cryptographic Hashing
- **Does it fix the causal mechanism?** Yes. Python's process-salted `hash()` caused unseen character n-grams to hash to different indices across interpreter sessions, breaking persistence across restarts.
- **Theoretical commitment preserved:** Random Features / GP (Constraint F) and Continual Learning (Constraint G). Repetition of an unseen subword must yield a consistent feature projection across time.
- **New failure modes:** Modulo-64 bucket collisions. Mitigated by using an alternating sign-hash ($\pm 1$) derived from higher-order bits, providing an unbiased expectation $\mathbb{E}[\mathbf{c}(w_1)^\top \mathbf{c}(w_2)] = 0$ for unrelated character sequences.
- **Falsification experiment:** Compute `char_ngram_embedding` for identical strings under differing `PYTHONHASHSEED` values. Vectors must be bit-for-bit identical. (Verified in `SemanticRepresentation_test.py`).

### 2. Definition Symmetry
- **Does it fix the causal mechanism?** Yes. Previously, `observe_definition` stripped structural tokens to update the word vector, while `store_definition` stored an unweighted average of the full text including glue words. This caused $e_{\text{word}} \ne e_{\text{def}}$.
- **Theoretical commitment preserved:** Bayesian concept formation (Constraint E): the definiendum representation must align with the definition representation.
- **New failure modes:** Empty token set if a definition contains only structural words. Mitigated by ensuring a non-zero floor weight ($w_{\min} = 0.15$) and falling back to content subwords if all tokens are masked.
- **Falsification experiment:** Train `"stream is a flowing liquid"`. Measure cosine between updated word vector and stored definition vector; similarity must exceed $0.65$. (Verified: achieves $> 0.70$).

### 3. Removal of Direct Factor-Vector Injection
- **Does it fix the causal mechanism?** Yes. Directly padding 11-dimensional affective vectors to 64 and writing them into `word_embeddings` forced all touched words into the positive hyperoctant ($\cos \ge 0.94$).
- **Theoretical commitment preserved:** Complementary Learning Systems (Constraint A) and Representation Boundaries (Section 3). Affective states and linguistic semantic spaces must remain decoupled.
- **New failure modes:** Loss of affective awareness in chat generation. Mitigated by deriving `current_state_embedding` from the linguistic label of the active state via `text_to_embedding(label)` and mediating word-state relationships through `WordStateAssociator`.
- **Falsification experiment:** Run state learning and idle cycles; verify no factor vectors or multi-word state names exist in `word_embeddings`. (Verified).

### 4. Non-Zero Deterministic Morphological Prior
- **Does it fix the causal mechanism?** Yes. Zero initialization created an absorbing singularity where gradient updates vanished ($\nabla \cos(\mathbf{0}, \mathbf{c}) = \mathbf{0}$).
- **Theoretical commitment preserved:** Variational Bayesian learning: An empirical prior $\mathcal{N}(\mu_0, \Sigma_0)$ centered on subword morphology allows gradient ascent to function immediately from turn 1.
- **New failure modes:** Overly rigid prior preventing semantic updates. Mitigated by setting initial learning rate $lr = 1/(n_w + 1) = 1.0$ for fresh words, and supervised definition step size to $0.5 \times \text{grad}$, enabling context to rotate the embedding away from morphology.
- **Falsification experiment:** Initialize `"river"` and `"flow"`. Observe co-occurrences. Embedding must move $\Delta \ge 0.05$ from prior and align with context. (Verified).

### 5. Content-Weighted Phrase Pooling
- **Does it fix the causal mechanism?** Yes. Unweighted mean pooling allowed high-frequency function words and articles (`"a"`, `"is"`) to dominate sentence representations.
- **Theoretical commitment preserved:** Continual learning and non-parametric concept formation: Phrases must be discriminated by their content, not collapsed by syntactic scaffolding.
- **New failure modes:** Loss of grammatical distinction if function words are stripped entirely. Mitigated by soft inverse-frequency downweighting $w(t) = 1/\sqrt{1 + 0.1 \times \text{count}(t)}$ and structural downweighting with a positive floor $w_{\min} = 0.15$.
- **Falsification experiment:** Compare `"what is a word"` with `"moon is a thing"`. Cosine similarity must drop below $0.60$. (Verified: drops from $1.0000$ to $0.5104$).

---

## 4. Source Changes Summary

1. **`herebot/traits/semlearn.py`**:
   - Implemented MD5-based deterministic feature hashing with alternating sign hash in `char_ngram_embedding`.
   - Updated `initialize_word()` to initialize with unit-norm morphological subword prior.
   - Refactored `observe_context()` and `observe_definition()` to normalize embeddings onto the unit sphere, preventing absorbing zero collapse and scale divergence.
   - Enhanced `text_to_embedding()` to support soft inverse-frequency and structural weighting.
   - Added `semantic_space_version = 1` and explicit legacy v0 migration in `load()`.
2. **`herebot/core/cuengine/curiosity.py`**:
   - Refactored `_update_state_embedding()`: computes semantic state embedding strictly from state label text using `text_to_embedding()`.
   - Removed all direct assignments from factor vectors into `word_embeddings`.
3. **`herebot/core/cuengine/idle.py`**:
   - Refactored `cross_modal_alignment()`: updates probabilistic associations in `WordStateAssociator` rather than adding factor vectors to `word_embeddings`.
   - Updated `mine_phrase_relations()` to use content-weighted phrase embeddings and record explicit provenance (`'idle_mining'`).
4. **`herebot/core/cuengine/processing.py`**:
   - Updated definition handler (lines 587–590) to pass `structural_detector` to `text_to_embedding()` for stored definition construction, establishing definition symmetry.
5. **`herebot/components/resoning.py`**:
   - Updated `store_phrase()` to generate dictionary keys via deterministic SHA-256 of canonical text.
   - Recorded explicit `'provenance': 'definition_similarity'` on generated synonym hypotheses.
6. **`herebot/core/mind.py`**:
   - Passed `structural_detector` to `text_to_embedding()` during phrase retrieval in chat response generation.
7. **`herebot/tests/cor/SemanticRepresentation_test.py`**:
   - Added 15 comprehensive unit, adversarial, persistence, and non-regression tests.

---

## 5. Empirical Verification and Before/After Metrics

The repair was evaluated on the exact persisted experimental state (`semantic_learner.pkl` containing 122 words).

| Metric | Phase A (Pre-Repair Baseline) | Phase D/E/F (Post-Repair) | Status / Target |
| :--- | :--- | :--- | :--- |
| **Total Words in Vocabulary** | 122 | 120 (cleaned 2 multi-word labels) | Preserved |
| **Zero-Vector Count** | 110 (90.16%) | **0 (0.00%)** | Absorbing barrier eliminated |
| **Non-Zero Word Vector Count** | 12 (9.84%) | **120 (100.00%)** | All words informative |
| **Word Norm Distribution** | min=0.0000, mean=0.0369, max=1.0000 | min=0.0895, mean=0.9375, max=1.0000 | Unit-scale stability |
| **Pairwise Cosine (Non-Zero Sample)** | min=0.7702, mean=0.9550, max=1.0000 | **min=-0.5774, mean=0.0083, max=1.0000** | Orthogonal expectation restored |
| **Proportion of Word Pairs with $\cos \ge 0.94$** | **83.33%** | **0.49%** | Collinear cone eliminated |
| **Contaminated State Vectors in `word_embeddings`** | 3 (`'crow fly over water'`, `'bright curiosity'`, `'exit'`) | **0** | Complete factor isolation |
| **`sim("what is a word", "moon is a thing")`** | **1.0000000000** | **0.5104** | Discriminated ($< 0.60$) |
| **`sim("what is a word", "zibber means a glowing stone")`** | **0.980387** | **0.3535** | Discriminated ($< 0.60$) |
| **`sim("water is cold", "fire is hot")`** | 0.0000 (both zero) | **0.4067** | Informative semantic contrast |
| **`sim("water is cold", "water is cold")`** | 0.0000 (both zero) | **1.0000** | Identity preserved |
| **`sim("fish is an animal", "fish is a animal")`** | 0.0000 (both zero) | **0.7332** | Legitimate similarity preserved |
| **Cross-Process OOV Hash Invariance** | $0.0000$ (scrambled) | **1.0000000000** (identical) | Strict persistence across restarts |
| **E10 FactorSpace Test Suite** | 5 / 5 passed | **5 / 5 passed** | Zero regression |
| **Complete Test Suite (`*test.py`)** | 154 / 154 passed | **169 / 169 passed** | Full suite green |

---

## 6. Final Acceptance Verification

1. **Original collapse reproduced:** Confirmed via Phase A baseline metrics and script reproduction.
2. **Causal mechanism removed:** Zero-absorbing barrier replaced with deterministic subword prior; factor vector injection eliminated; definition asymmetry reconciled; deterministic hashing established.
3. **Unrelated phrases separated:** `"what is a word"` vs `"moon is a thing"` dropped from 1.0000 to 0.5104.
4. **Legitimate similarity preserved:** `"fish is an animal"` vs `"fish is a animal"` scores 0.7332; identical sentences score 1.0000.
5. **OOV representations restart-invariant:** Tested across separate processes with different hash seeds; identical.
6. **Factor/state representations isolated:** State labels and affective vectors removed from `word_embeddings`.
7. **Sleep amplification controlled:** `cross_modal_alignment` mediated via `WordStateAssociator`; idle phrase mining content-weighted with explicit provenance.
8. **Historical data preserved:** Tracker data and episode JSONs remain untouched; semantic state migrated with version schema.
9. **E10 non-regression:** `FactorSpace_test.py` fully green.
10. **Full test suite passes:** 169 / 169 tests pass in 60s.

**E12 is formally verified and CLOSED.**

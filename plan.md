# E12 — Semantic Representation Collapse: Root-Cause Investigation and Architectural Repair Plan

## Context and Executive Summary

Following the resolution and closure of E10 (factor-space versioning), this investigation addresses **E12 (Semantic Representation Collapse)**. 

In prior sessions, Herebot exhibited degenerate phrase and definition similarity scores:
- Definition pairs repeatedly scored cosine similarities of `1.000000` or `0.980387` (e.g. `what is a word` $\leftrightarrow$ `zibber means a glowing stone`).
- In chat mode, phrase recall returned completely unrelated sentences (e.g. answering `are you fine` with `TPL.RECALL.PHRASE phrase='what is zibber'`).
- The synonym generation queue (`TPL.ASK.SYN`) became flooded with spurious hypotheses (`fish/thing`, `moon/thing`, `water/crow`).
- Idle sleep consolidation proposed hundreds of identical phrase relationships.

This investigation was conducted in read-only mode against the live persistent state (`herebot/curiosity_state/semantic_learner.pkl` and `reasoner.pkl`) and the codebase. No production code or persistent weights were modified during this analysis.

The investigation proved that **the 1.00 similarity values are literal, bit-for-bit numerical equality ($L_2$ difference $= 0.0000000000$)**, caused by a chain of 5 interacting architectural bugs centered on a **zero-vector absorbing barrier in `SemanticLearner`** and **positive-hyperoctant cross-modal contamination**.

---

## 1. Trace Phrase Embedding Creation

### Code Locations
- `herebot/traits/semlearn.py:118–139` (`text_to_embedding`)
- `herebot/traits/semlearn.py:140–160` (`char_ngram_embedding`)
- `herebot/traits/semlearn.py:88–97` (`observe_definition` target construction)

### Mathematical Operation Implemented
Given input text string $S$, tokenized by whitespace into raw tokens $[t_1, t_2, \dots, t_n]$:

$$\mathbf{v}_i = \begin{cases} \mathbf{e}(t_i) & \text{if } t_i \in \text{word\_embeddings} \\ \mathbf{c}(t_i) & \text{if } t_i \notin \text{word\_embeddings and } \text{fallback\_to\_char} \end{cases}$$

where $\mathbf{e}(t_i) \in \mathbb{R}^{64}$ is the stored word vector (which may be $\mathbf{0}$), and $\mathbf{c}(t_i)$ is the bag-of-character-trigrams vector:

$$\mathbf{c}(t)_k = \sum_{ng \in \text{trigrams}(\text{' } + t + \text{' '})} \mathbb{I}(\text{hash}(ng) \pmod{64} == k), \quad \mathbf{c}(t) \leftarrow \frac{\mathbf{c}(t)}{\|\mathbf{c}(t)\|}$$

The phrase embedding is then:

$$\mathbf{v}_{\text{raw}} = \frac{1}{n} \sum_{i=1}^n \mathbf{v}_i, \qquad \mathbf{v}_{\text{phrase}} = \begin{cases} \frac{\mathbf{v}_{\text{raw}}}{\|\mathbf{v}_{\text{raw}}\|} & \text{if } \|\mathbf{v}_{\text{raw}}\| > 0 \\ \mathbf{0} & \text{otherwise} \end{cases}$$

### Property Analysis
- **Tokens included/excluded:** Every whitespace-split token is included without exception.
- **Stop words / structural tokens removed:** **No.** `text_to_embedding` has no filtering mechanism, stop-word list, or structural-token detector call.
- **Token frequency / IDF weighting:** **No.** Every token has equal weight ($1/n$).
- **Word embedding pre-normalization:** **No.** Raw vectors are summed directly before averaging.
- **Phrase vector post-normalization:** **Yes.** Unit $L_2$-norm normalization is applied at line 137.
- **Untrained / zero embedding behavior:** If $\mathbf{e}(t) = \mathbf{0}$, it contributes $\mathbf{0}$ to the numerator $\sum \mathbf{v}_i$, but increments the denominator $n$. When normalized, **a zero vector contributes zero direction, meaning the phrase vector points entirely in the direction of whichever tokens are non-zero.**
- **Unknown words:** Routed to `char_ngram_embedding`.
- **Phrase length:** Does not affect direction; only affects dilution before post-normalization.

---

## 2. Trace Definition Learning

### Path Traced
User input: `"zibber means a glowing stone"` vs `"moon is a thing"`  
Code: `processing.py:575–592` $\to$ `semlearn.py:77–105` $\to$ `resoning.py:687–690`.

1. **Target Word Selection:**  
   The target word is `word = self.last_asked_word` (e.g., `zibber` or `moon`).
2. **Word Embedding Update in `semlearn.py:88–104`:**
   ```python
   tokens = definition_text.split()
   if structural_detector is not None:
       tokens = [t for t in tokens if not structural_detector.is_structural(t)]
   known_embs = [self.word_embeddings[t] for t in tokens if t in self.word_embeddings]
   if not known_embs:
       return
   target = np.mean(known_embs, axis=0)
   target = target / (np.linalg.norm(target) + 1e-12)
   emb = self.word_embeddings[word]  # initialised to np.zeros(64)
   grad = (target - emb) - self.alpha * emb
   self.word_embeddings[word] += 0.1 * grad
   ```
3. **The Zero-Target Trap:**  
   When `moon is a thing` is taught:
   - `moon`, `is`, `thing` are all in `word_embeddings`, but all were initialized to $\mathbf{0}$!
   - `known_embs = [0, 0, 0]`.
   - $\text{target} = \mathbf{0} / (0 + 10^{-12}) = \mathbf{0}$.
   - $\text{grad} = (\mathbf{0} - \mathbf{0}) - 0.1 \times \mathbf{0} = \mathbf{0}$.
   - **`word_embeddings['moon']` remains exactly $\mathbf{0}$!**
4. **The Structural Word Dominance Trap:**  
   `StructuralWordDetector` (`pattern_observer.py:140`) has `min_count=3`. During early learning, `is` and `means` have count $< 3$, so they are *not* recognized as structural and are included in `known_embs`.
5. **Connector Updates:**  
   Connector words (`is`, `means`) are *never* updated as part of another word's definition. They only update if defined directly or through `observe_context` during idle.
6. **Convergence Equation:**  
   If any word in a definition has a non-zero vector $\mathbf{u}$, then $\text{target} = \mathbf{u} / \|\mathbf{u}\|$. The gradient $\text{grad} = \text{target} - (1+\alpha)\mathbf{e}$ pulls the new word *collinearly* into the span of $\mathbf{u}$.

---

## 3. Empirical Mathematical Reproduction from Current State

Direct inspection of `herebot/curiosity_state/semantic_learner.pkl` (122 words total):
- **110 words have norm exactly equal to 0.0000.**
- **12 words have norm $> 0$.**

Pairwise cosine comparisons using the current saved semantic state:

| Phrase 1 | Phrase 2 | Cosine Similarity | $L_2$ Vector Difference | Explanation from Code |
| :--- | :--- | :--- | :--- | :--- |
| `what is a word` | `moon is a thing` | **1.000000** | **0.000000** | Both collapse to `char_ngram('a')` |
| `what is a word` | `zibber means a glowing stone` | **0.980387** | **0.198057** | Both dominated by `char_ngram('a')` |
| `what is zibber` | `what is learns` | **1.000000** | **0.000000** | Only non-zero words are collinear |
| `what is zibber` | `are you fine` | **1.000000** | **0.000000** | Only non-zero words are collinear |
| `what is learns` | `are you fine` | **1.000000** | **0.000000** | Only non-zero words are collinear |
| `hello friend` | `good morning` | **1.000000** | **0.000000** | Non-zero words share factor direction |
| `fish is an animal` | `water is cold` | **0.000000** | **0.000000** | Both are literally zero vectors ($\mathbf{0}$) |
| `the bright sun` | `water is cold` | **0.000000** | **0.000000** | Both are literally zero vectors ($\mathbf{0}$) |

### Crucial Finding on the 1.00 Values
The $1.000000$ values are **literal numerical equality** ($\|\mathbf{v}_1 - \mathbf{v}_2\| = 0.0000000000$), not a rounding artifact.

In `what is a word` and `moon is a thing`:
- `what`, `is`, `word`, `moon`, `thing` are all known in `word_embeddings`, but **every one of them has a norm of 0.0000**.
- The token `'a'` was skipped by the tokenizer when vocabulary was small, so `'a'` was never registered in `word_embeddings`.
- Therefore, `'a'` falls back to `char_ngram_embedding('a')`.
- Summing `0 + 0 + char_ngram('a') + 0` yields `char_ngram('a')`.
- Unit-normalizing `char_ngram('a')` yields `char_ngram('a')`.
- The phrase vectors for `"what is a word"` and `"moon is a thing"` are **bit-for-bit identical to `char_ngram_embedding('a')`**.

Empirical verification:
`diff(char_ngram("a"), text_to_embedding("what is a word")) == 0.0000000000`

---

## 4. Inspection of `find_similar_phrases`

### Code Location
`herebot/components/resoning.py:710–723`:
```python
def find_similar_phrases(self, phrase_emb: np.ndarray, threshold: float = 0.85) -> List[Tuple[str, float]]:
    results = []
    for key, emb in self.phrase_embeddings.items():
        sim = np.dot(phrase_emb, emb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(emb) + 1e-12)
        if sim >= threshold:
            phrase = self.phrase_metadata.get(key, {}).get('phrase')
            if not phrase:
                continue
            results.append((phrase, sim))
    results.sort(key=lambda x: -x[1])
    return results
```

### Inspection Findings
- **Objects compared:** Stored phrase vector `emb` from `self.phrase_embeddings` against input `phrase_emb`.
- **Recomputation:** Vectors are **never recomputed**; it uses the frozen vectors stored at the time the phrase was encountered.
- **Direction / word order:** Ignored completely (bag of words).
- **Filtering:** No filtering for structural definitions, stopwords, or common tokens.
- **Frequency penalty:** None.
- **Symmetry:** Cosine similarity is mathematically symmetric.
- **Self-matches:** Not filtered inside `find_similar_phrases` (if the exact phrase was previously stored, it returns itself with similarity 1.0).
- **SYN Queue Source:** The SYN queue is generated in `resoning.py:742–765` via `find_similar_definitions`, which uses the exact same cosine comparison on `definition_embeddings`. **The similarity function is mathematically correct; the vectors it compares are identical.**

---

## 5. Phrase Storage Inspection

### Code Location
`herebot/components/resoning.py:724–736` (`store_phrase`) and `resoning.py:837–844` (`load`).

- **What is stored:**
  - Key: `str(hash(canon))` (salted hash string)
  - `self.phrase_embeddings[key]`: `np.ndarray` of shape `(64,)`
  - `self.phrase_metadata[key]`: dictionary containing:
    - `'phrase'`: normalized string
    - `'timestamp'`: float
    - `'source'`: optional string (e.g., `'user'`)
- **What is NOT stored:** tokens, parse trees, syntactic relations, token frequencies, contextual variance.
- **Legacy hash-only behavior verification:**  
  In `herebot/curiosity_state/reasoner.pkl`, **43 of 77 stored phrase keys have no `'phrase'` in their metadata** (legacy residue from before the fix). The defensive check at line 721 (`if not phrase: continue`) successfully prevents raw hashes like `'51085...'` from leaking into conversational output. However, those 43 orphaned vectors are still computed in the pairwise loop during idle mining.

---

## 6. Classification of the Root Cause

**Classification: E — Multiple interacting causes (hierarchically ordered):**

1. **Primary Cause (The Zero-Attractor):**  
   `semlearn.py:43` initializes all new words to $\mathbf{0}$. In `observe_context` (`semlearn.py:64`), the gradient is guarded by `if norm_e > 0 and norm_c > 0:`. If a word is zero, its gradient is identically zero. In `observe_definition` (`semlearn.py:95`), if context words are zero, the target vector is zero. Zero is an absorbing barrier from which words cannot escape through normal learning.
2. **Amplifying Cause (The Single-Token Fallback Domination):**  
   Because content words remain zero, any phrase containing an OOV token (such as `'a'`) has its entire semantic representation determined solely by that single OOV token.
3. **Cross-Modal Vector Contamination:**  
   `curiosity.py:466` and `idle.py:564` took 11-dimensional control/factor vectors (curiosity, threat, energy) and directly added them to `semantic_learner.word_embeddings`. Because factor vectors are strictly positive, all words touched by cross-modal alignment were pushed into a single collinear ray in the positive hyperoctant (cosine $\ge 0.94$).
4. **Definition Asymmetry:**  
   `observe_definition` applies structural token filtering to the word target, but `text_to_embedding` at `processing.py:588` does not filter structural tokens when constructing the stored `definition_embeddings`.

---

## 7. Non-Transformer Constraint

No transformers, attention mechanisms, sentence-transformers, BERT, or external embedding APIs will be introduced. All architectural solutions will operate within Herebot's native variational/statistical, lifelong-learning framework.

---

## 8. Evaluation of Candidate Fixes

| Candidate Fix | Information Preserved | Information Lost | Comp. Cost | Compatibility with Online Lifelong Learning | Changes Existing Vectors? | Risk to Experiment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Weighted Mean / IDF** | Content token salience | Function word nuance | $O(N)$ trivial | Fully compatible | Yes (on recompute) | Low |
| **2. Structural-Token Masking** | Pure content semantics | Grammatical frame | $O(N)$ trivial | Fully compatible | Yes (on recompute) | Low |
| **3. Positional / Circular Conv** | Word order, syntax | Commutativity | $O(N \log N)$ | Compatible | Yes | Medium |
| **4. Learned Phrase Embeddings** | Holistic phrase idioms | Compositionality | $O(1)$ lookup | Memory grows with phrases | No | Low |
| **5. Relation-Aware Triplet Reps** | Subject-Verb-Object roles | Loose associations | Moderate | Dependent on triplet extractor | Yes | High |
| **6. Separate Def / Word Spaces** | Word meaning vs definition | Cross-space comparison | Double memory | Fully compatible | No | Low |
| **7. TF-IDF Vectors** | Term distinctiveness | Continuous geometry | Low | Incompatible with continuous GP/HDP | Yes | High |
| **8. Hybrid Semantic + Lexical** | Both continuous and discrete | Pure metric elegance | Low | Fully compatible | No | Minimal |
| **9. Raise SYN Threshold** | Nothing | Low-sim candidates | Zero | Treats symptom only | No | Zero |
| **10. Repeated Evidence Gate** | Reliability | One-shot discovery | Low | Fully compatible | No | Minimal |
| **11. Deterministic Morphological Prior + Absorbing Zero Removal** | Individuated word identity, gradient flow | Pure zero-prior assumption | Low | Native to Variational Bayesian design | Yes | Low (clean reset) |

---

## 9. Check for Feedback Loops

Tracing the hypothesized cycle:
$$\text{Phrase Sim} \longrightarrow \text{SYN Hypothesis} \longrightarrow \text{User Confirm} \longrightarrow \text{Semantic Update} \longrightarrow \text{New Embeddings} \longrightarrow \text{More Collapse}$$

- Code trace in `resoning.py` and `processing.py`:
  - When `generate_synonym_hypothesis` creates a hypothesis, it appends it to `self.reasoner.hypotheses`.
  - When uttered, `mind.py:482` sets `hyp['asked'] = True`.
  - **There is no confirmation handler for synonyms in `processing.py`!** Only shadow differentiation links have confirmation handlers (`expecting_shadow_diff_confirmation`).
- **Conclusion:** There is **NO direct semantic weight feedback loop** from synonym generation back into `word_embeddings`.
- The damage is an **output-turn bottleneck**: the noisy SYN queue preempts normal conversational responses because pending hypotheses have Priority 2 in `ActionGenerator.generate_reply` (`mind.py:480`).

---

## 10. Check Whether Sleep Makes It Worse

During idle sleep (`idle.py:1200–1240` `mine_phrase_relations`):
1. **Corpus Scanned:** Recent 200 episodes, including synthetic sentences from `semantic_self_talk` (e.g. `'feeling is thing...'`, `'animal is friend...'`).
2. **Re-embedding:** Every phrase is re-embedded using `text_to_embedding(p, fallback_to_char=True)`.
3. **Execution Order:**
   - Line 1314: `cross_modal_alignment` injects factor vectors into words.
   - Line 1318: `semantic_self_talk` injects trigram structural sentences into memory.
   - Line 1329: `mine_phrase_relations` compares all phrases.
4. **Mathematical Result:** Sleep runs `cross_modal_alignment` *before* `mine_phrase_relations`. This pulls more words into the collinear positive hyperoctant, directly increasing the number of phrases that score $1.0000$ and generating dozens of spurious `phrase_similarity` hypotheses every sleep cycle.

---

## 11. Scientific Interpretation

- **OBSERVED FACT:**
  1. 110 of 122 words in `semantic_learner.pkl` have $L_2$ norm identically equal to $0.0000$.
  2. The 12 non-zero words have non-zero values strictly in factor dimensions (0..9 and 12) with mutual cosine similarity $\ge 0.9439$.
  3. `diff(char_ngram("a"), text_to_embedding("what is a word")) == 0.0000000000`.
  4. Cross-modal alignment directly adds factor vectors to word embeddings.
  5. Python's built-in `hash()` produces different character-trigram indices on every process launch.
- **ROOT-CAUSE INFERENCE:**
  1. The zero-attractor in `SemanticLearner` prevents words from acquiring distinct semantic representations.
  2. Because content words remain zero, definition phrases collapse to the single OOV function word `'a'`.
  3. Spurious synonym hypotheses are the mathematical consequence of comparing identical 1-hot vectors.
- **SPECULATION:**
  1. Whether character n-grams alone provide sufficient lexical differentiation for small vocabularies without frequency downweighting.

---

## 12. Final Architecture Map and Recommendations

### Exact Root Cause
The collapse is caused by **zero-vector absorbing dynamics in `SemanticLearner`**: words initialize to $\mathbf{0}$ and gradients vanish when $\|\mathbf{e}\| = 0$. Consequently, content words contribute $\mathbf{0}$ to phrase embeddings, causing definition phrases to collapse to the OOV hash vector of the connector/article `'a'`. This is further corrupted by `cross_modal_alignment` injecting positive factor-space control vectors directly into semantic word vectors.

### Exact Source Locations
1. `herebot/traits/semlearn.py:43`: `self.word_embeddings[word] = np.zeros(self.emb_dim)`
2. `herebot/traits/semlearn.py:64–66`: `if norm_e > 0 and norm_c > 0:` suppresses gradients when embeddings are zero.
3. `herebot/traits/semlearn.py:95–96`: `target / (np.linalg.norm(target) + 1e-12)` produces $\mathbf{0}$ target when context words are zero.
4. `herebot/traits/semlearn.py:153`: `idx = hash(ng) % emb_dim` uses non-deterministic process-salted hashing.
5. `herebot/core/cuengine/idle.py:564`: `self.semantic_learner.word_embeddings[t] += lr * grad / len(tokens)` injects factor vectors into semantic vectors.
6. `herebot/core/cuengine/curiosity.py:466`: `self.semantic_learner.word_embeddings[lbl] = self._align_vector_to_emb_dim(vec)` injects factor state vectors into word embeddings.
7. `herebot/core/cuengine/processing.py:588`: `def_emb = self.semantic_learner.text_to_embedding(text)` creates definition embeddings without structural word filtering.

### Data-Flow Diagram of the Failure
```
initialize_word("water") ──> word_embeddings["water"] = 0.0
                                      │
"water is a nature" ──────────────────┼──> known content words have norm 0.0
                                      │         │
                                      │         v
                                      │    text_to_embedding()
                                      │    "water"(0) + "is"(0) + "nature"(0) + "a"(OOV)
                                      │         │
                                      │         v
                                      │    phrase vector = char_ngram("a") [1-hot at index 33]
                                      │         │
                                      v         v
                      definition_embeddings["water"] = [0, ..., 1.0 (idx 33), ... 0]
                                      │
                                      │   (Identical for "crow is a animal", "stone is a nature")
                                      v
                      find_similar_definitions("water", "crow")
                                      │
                                      v
                      sim = dot(vec_33, vec_33) = 1.0000000000
                                      │
                                      v
                      TPL.ASK.SYN word='water' candidate='crow'?
```

### Minimal Reproduction Script
```python
import numpy as np
from traits.semlearn import SemanticLearner

sl = SemanticLearner(emb_dim=64)
# Current code behavior:
sl.initialize_word("water")
sl.initialize_word("crow")
sl.initialize_word("is")
sl.initialize_word("nature")
sl.initialize_word("animal")

# "a" is omitted from vocab, falling back to char_ngram
v1 = sl.text_to_embedding("water is a nature")
v2 = sl.text_to_embedding("crow is a animal")
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
assert cos_sim == 1.0000000000  # Confirms exact collapse
```

### Candidate Fixes Ranked by Invasiveness
1. **Fix 1 (Non-invasive):** Replace Python `hash()` with stable deterministic cryptographic hashing (`hashlib.md5(ng.encode()).digest()`) in `char_ngram_embedding`.
2. **Fix 2 (Low invasiveness):** Fix definition asymmetry in `processing.py:588`: filter structural tokens when generating stored definition embeddings, matching `observe_definition`.
3. **Fix 3 (Low invasiveness):** Remove direct injection of factor vectors into `word_embeddings` in `curiosity.py:466` and `idle.py:564`.
4. **Fix 4 (Medium invasiveness):** Replace zero-initialization in `initialize_word` with an informative subword/morphological prior derived deterministically from the word's character n-grams, and remove the zero-absorbing guard in `observe_context`.
5. **Fix 5 (Medium invasiveness):** Content-weighted phrase pooling: weight tokens in `text_to_embedding` by inverse structural frequency ($1 - \text{structural\_score}(t)$).

### Recommendation
Implement **Fixes 1, 2, 3, 4, and 5 together**. They directly address each link in the collapse chain while respecting the non-transformer architecture:
- Fix 1 makes representation persistent across sessions.
- Fix 2 establishes definition symmetry.
- Fix 3 stops cross-modal factor contamination.
- Fix 4 eliminates the zero-attractor while adhering to Bayesian prior principles.
- Fix 5 ensures phrases are dominated by their distinct semantic content rather than glue articles.

### Proving Tests Required
1. `test_deterministic_oov_hashing`: Verify that `char_ngram_embedding` produces identical vectors across different Python processes with different hash seeds.
2. `test_nonzero_word_initialization`: Verify that freshly initialized words have non-zero norm, distinct directional angles, and that `observe_context` updates them away from their prior.
3. `test_definition_symmetry`: Verify that word target and stored definition vectors have cosine similarity $> 0.90$.
4. `test_phrase_discriminability`: Verify that `sim("what is a word", "moon is a thing") < 0.60` and `sim("water is cold", "fire is hot") < 0.85`.
5. `test_no_factor_space_contamination`: Verify that state/factor updates never write into `semantic_learner.word_embeddings`.

### Experimental Integrity
- **Should existing experiment remain frozen?** Yes. Historical data (`reports.md` and `tracker_data.json`) must remain frozen.
- **Can the fix be applied without rewriting historical data?** Yes. Historical episodes and tracker records remain completely intact. Semantic state can be migrated cleanly by initializing newly learned words with the new prior and allowing lifelong learning to differentiate existing vocabulary without destroying the session logs.

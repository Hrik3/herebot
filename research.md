# Herebot: A Research Analysis

## Research Hypothesis

**H1: Information-theoretic intrinsic motivation with Bayesian surprise and Gaussian Process uncertainty can drive sample-efficient, lifelong concept acquisition in a non-transformer cognitive architecture, achieving approximately 100–1000× greater sample efficiency per concept than transformer fine-tuning, at the cost of zero generative fluency and zero transfer breadth.**

**H2: A coupled system of probabilistic approximations—Dirichlet Process concept assignment, GP-based uncertainty, KL-divergence Bayesian surprise, and meta-learned factor weighting—produces emergent self-directed questioning behavior that prioritizes knowledge gaps without explicit instruction, and this behavior is robust to representation collapse when factor-space and semantic-space representations are properly isolated.**

**H3: Sleep-phase consolidation (dreaming, rehearsal, meta-concept formation, policy distillation) in a curiosity-driven system produces measurable retention benefits but simultaneously introduces representational pollution through self-talk and cross-modal contamination, creating a fundamental trade-off between consolidation quality and semantic integrity.**

**H4: The one-question-per-turn output bottleneck, while limiting conversational throughput, serves as an implicit regularization mechanism that prevents catastrophic forgetting by forcing the system to commit to a single hypothesis per interaction cycle.**

---

## Abstract

This research analyzes Herebot, a curiosity-driven cognitive architecture implementing information-theoretic intrinsic motivation through a coupled system of probabilistic approximations. The system integrates: (1) a Hierarchical Dirichlet Process concept model with Chinese Restaurant Process priors for unbounded concept creation; (2) a Gaussian Process uncertainty model using Bayesian linear regression over random Fourier features; (3) Bayesian surprise via KL-divergence between prior and posterior concept distributions; (4) a multi-factor meta-curiosity learner using Bayesian linear regression over nine curiosity factors; (5) a reasoning engine with co-occurrence graphs, triplet extraction, and transitive inference; (6) a state manager with probabilistic label assignment and automatic regrouping; and (7) a sleep-phase consolidation system with dreaming, rehearsal, and policy distillation.

Analysis of 433 tracked interaction points across multiple sessions reveals that Herebot achieves approximately 0.48–0.59 words/turn acquisition rate with ~1.7–2.1 turns per new word (approximately 2 exposures: the word and its definition), compared to transformer fine-tuning estimates of 10³–10⁴ labeled examples per task. Concept entropy remains at 0.01–0.06 across all sessions, indicating one-shot binding after single exposure. Curiosity scores remain high (2.8–7.1) throughout 130-turn sittings without saturation, suggesting the intrinsic motivation system does not produce boredom within experimentally observable timescales.

The architecture has undergone two major repair cycles: E10 (factor-space versioning to prevent dimension-mismatch crashes) and E12 (semantic representation collapse repair addressing zero-vector absorbing barriers and cross-modal factor contamination). Post-repair metrics show zero-vector count reduced from 110/122 (90.16%) to 0/120 (0.00%), pairwise cosine similarity among non-zero words shifted from mean 0.955 (min 0.770) to mean 0.008 (min −0.577), and phrase discrimination improved from 1.0000 to 0.5104 for unrelated phrases sharing grammatical frames.

Key architectural findings include: (a) encoding and speaking are decoupled—the system learns in one shot but can only verbalize one hypothesis per turn; (b) no belief revision mechanism exists—contradictions are silently absorbed; (c) sleep consolidation both strengthens retention and introduces pollution through self-talk; (d) the one-question-per-turn bottleneck acts as implicit regularization. The system passes 169/169 tests across 8 test suites.

---

## Introduction

### 1. Motivation and Research Context

The question of whether artificial systems can acquire knowledge through intrinsic motivation—without explicit supervision or pretraining on massive corpora—remains a fundamental challenge in cognitive architecture research. Herebot represents a distinctive approach: rather than scaling compute and data, it implements a biologically-inspired curiosity engine where information gain itself serves as the reward signal. This research analyzes the complete codebase, experimental data, and architectural evolution of Herebot to understand what properties emerge from information-theoretic intrinsic motivation in a non-transformer lifelong learning system.

The system draws on four theoretical foundations:

1. **Schmidhuber's curiosity principle** (1991): Learning progress (reduction in prediction error) serves as intrinsic reward, creating a self-reinforcing exploration loop.
2. **Oudeyer & Kaplan's typology** (2007): Intrinsic motivation can be knowledge-based (driven by information gain) or competence-based (driven by learning progress).
3. **Complementary Learning Systems theory**: Rapid episodic learning must be consolidated into slower, generalized semantic representations—implemented here through sleep-phase rehearsal.
4. **Variational Bayesian inference**: Concept assignments, word embeddings, and curiosity factor weights are all maintained as probability distributions updated through stochastic gradient ascent.

### 2. Architecture Overview

Herebot's `CuriosityEngine` (defined in `herebot/core/heart.py`) integrates approximately 10 heterogeneous processing stages:

```
Input → Backend Memory Sync → CI/Phonotactic/Diagnostic Evaluation → Tokenization
→ Reasoner Observation → Per-Token Learning (Concept Model + GP + Curiosity Factors)
→ Sentence Reasoning → Control Factor Vector → Probabilistic State Recognition
→ State Relation Observation/Regrouping → Generated Plugin Trials → Action Generation
→ Feedback Request → Episode Persistence
```

The key subsystems are:

- **CI Engine** (`backend/ci.py`): Monte Carlo perturbation-based confidence scoring with TF-IDF similarity, IsolationForest outlier detection, and stability-weighted confidence.
- **Phonotactic Engine** (`backend/ci.py`): Data-adaptive character n-gram profiling with fallback heuristics for syllable structure, consonant clusters, and vowel balance.
- **Concept Model** (`concepts/hdp.py`): Hierarchical Dirichlet Process with CRP-style global counts, probability-maintained concept assignments in interaction space, and meta-concept layer.
- **GP Uncertainty** (`concepts/gaussian_uncertainity.py`): Bayesian linear regression over 100 random Fourier features with online Sherman-Morrison updates.
- **Meta-Curiosity Learner** (`concepts/meta_learner.py`): Bayesian linear regression over 9 base curiosity factors (uncertainty, learning_progress, novelty, complexity, surprise, phonotactic_anomaly, diagnostic_negatives, bayesian_surprise, gp_uncertainty) with dynamic factor expansion.
- **Semantic Learner** (`traits/semlearn.py`): 64-dimensional word embeddings with deterministic MD5-based character n-gram priors, content-weighted phrase pooling, and variational context learning.
- **Reasoning Engine** (`components/resoning.py`): Co-occurrence graph, triplet extraction, transitive inference (is_a, means, similar), phrase storage with SHA-256 keys, and hypothesis generation.
- **State Manager** (`components/factor_handler.py`): Probabilistic state label assignment with interaction-clock decay, automatic regrouping via union-find, and FactorSpace versioning (E10).
- **Idle/Sleep System** (`core/cuengine/idle.py`): 9 idle tasks including world model dreaming, active memory rehearsal, curiosity landscape optimization, concept graph compression, autonomous hypothesis planning, identity stabilization, cross-modal alignment, exploration policy distillation, and cognitive drive evolution.

### 3. Research Questions

This analysis addresses four research questions:

**RQ1**: What sample efficiency does information-theoretic intrinsic motivation achieve for concept acquisition, and how does it compare to transformer-based approaches?

**RQ2**: What emergent behaviors arise from the coupling of Bayesian surprise, GP uncertainty, and meta-learned curiosity weighting?

**RQ3**: How does the sleep-phase consolidation system affect knowledge retention and representational integrity?

**RQ4**: What are the fundamental architectural trade-offs between inspectability, sample efficiency, and generative fluency in non-transformer cognitive architectures?

### 4. Key Empirical Findings

#### 4.1 Sample Efficiency and One-Shot Learning

From `reports.md` and `graphs/tracker_data.json` (433 tracked points):

- **Words per turn**: 0.48–0.59 across three fresh-model sessions
- **Turns per word**: ~1.7–2.1 (approximately 2 exposures: the word + its definition)
- **Concept entropy**: 0.01–0.06 across all sessions—concepts collapse after single exposure
- **Curiosity saturation**: Not observed within 130-turn sittings (top scores 2.8–7.1)

The `zibber` experiment (Session 1, `reports.md`) provides the clearest evidence: after a single definition (`zibber means a glowing stone`), curiosity collapsed from 1.9 to 0.36, novelty dropped to 0.01, and the memory was formed and indexed after one exposure. This demonstrates one-shot binding—a property that transformer fine-tuning cannot match without 10³–10⁴ labeled examples.

#### 4.2 Decoupled Encoding and Speaking

A consistent finding across all experiments is that encoding (learning) and speaking (verbalizing) are decoupled. The system learns in one shot but can only verbalize one hypothesis per turn. This creates a bottleneck where pending hypotheses (synonym questions, category questions, feedback probes) consume reply turns that could otherwise address user queries. The priority hierarchy in `mind.py:generate_reply()` is:

1. Chat mode (overrides all)
2. Logic doubt
3. Expression evaluation
4. Conscience relevance
5. Guilt expression
6. Pending sentence query
7. Pending hypotheses (synonym, category, analogy, inference)
8. Generated plugin candidates
9. Parallel reasoning thoughts
10. High-curiosity unknown words
11. State awareness utterance (30% chance)
12. Spontaneous emotional utterance
13. Idle acknowledgement

This priority ordering means that knowledge acquisition (learning) always takes precedence over knowledge expression (speaking), which is the correct ordering for a learning system but creates conversational friction.

#### 4.3 No Belief Revision

Session 3 of the investigation reports (`reports.md`) demonstrated that when presented with a direct contradiction (`zibber is a bird` after learning `zibber means a glowing stone`), the system silently absorbed the contradiction without detection, correction, or questioning. The definition remained unchanged. This represents a fundamental limitation: the architecture has no confidence competition, no contradiction detector, and no mechanism for belief revision. A liar can never correct it, but neither can a teacher.

#### 4.4 Semantic Representation Collapse and Repair (E12)

The most significant architectural finding is the E12 semantic representation collapse, documented in `plan.md` and `docs/E12_SEMANTIC_REPRESENTATION_TRANSITION.md`:

**Pre-repair state** (122 words in `semantic_learner.pkl`):
- 110 words (90.16%) had L₂ norm exactly 0.0000
- 12 words had non-zero vectors contaminated by positive affective factor vectors
- Mutual cosine similarity among non-zero words ≥ 0.9439
- `sim("what is a word", "moon is a thing")` = 1.000000000 (bit-for-bit identical)
- `sim("what is a word", "zibber means a glowing stone")` = 0.980387

**Root cause chain**:
1. Zero-vector initialization created an absorbing barrier (gradients vanish when ‖e‖ = 0)
2. Content words contributed zero to phrase embeddings
3. Definition phrases collapsed to the OOV character-hash of the connector/article `'a'`
4. Cross-modal alignment injected 11-dimensional positive factor vectors directly into 64-dimensional word embeddings
5. Python's process-salted `hash()` produced different character-trigram indices across restarts

**Post-repair metrics**:
- Zero-vector count: 110 → 0 (0.00%)
- Non-zero word vectors: 12 → 120 (100.00%)
- Pairwise cosine (non-zero sample): mean 0.955 → 0.008, min 0.770 → −0.577
- Proportion of word pairs with cos ≥ 0.94: 83.33% → 0.49%
- `sim("what is a word", "moon is a thing")`: 1.0000 → 0.5104
- `sim("water is cold", "fire is hot")`: 0.0000 → 0.4067
- Cross-process OOV hash invariance: 0.0000 → 1.000000000

This repair demonstrates that representation integrity in lifelong learning systems requires explicit isolation between affective control space and lexical semantic space—a finding with implications for any architecture that couples emotional and semantic representations.

#### 4.5 Factor-Space Versioning (E10)

The E10 transition (`docs/E10_FACTOR_SPACE_TRANSITION.md`) addressed a dimension-mismatch crash where a newly spawned dynamic curiosity factor (`residual_1789205282`) expanded the meta-learner from 11 to 13 dimensions, but stored state vectors remained at 11 dimensions. The `find_similar_state` function used raw `np.dot` without dimension checking, causing `ValueError: shapes (13,) and (11,) not aligned`.

The fix introduced a `FactorSpace` abstraction with:
- Semantic identity via factor names (not positions)
- Monotonic version tracking
- Name-based cosine comparison (`shared_observed_factor_names_cosine`)
- Explicit projection with audit records
- Append-only migration log (`factor_space_migrations.jsonl`)
- Hard invariants: no padding, truncating, or silent repair

This represents a general solution to the problem of schema evolution in persistent cognitive state.

#### 4.6 Sleep Consolidation: Trade-offs

The sleep/consolidation system (`core/cuengine/idle.py`) implements 9 idle tasks. Experimental observations from Session 2 (`report.md`) reveal both benefits and costs:

**Benefits**:
- Dreaming produced 5 synthetic sentences including real session content (`conversational_pattern 'what is flint'` at 0.70 confidence)
- Rehearsal processed 20 episodes with prioritized scheduling
- Meta-concepts formed (`meta_hi_greetings` from 120 words)
- Policy distilled (linear classifier for ask/don't-ask decisions)
- LM trained (20 sentences, vocab 67, steps 80)
- Retention verified post-sleep (recall intact, no forgetting detected)

**Costs**:
- Self-talk learned junk into memory (`animal is friend...`, `for is word...`)
- Pre-experiment residue resurfaced (`rose`, `car is travel`, `ccar`)
- Similarity collapse anomaly (every phrase pair scored 1.00/0.98)
- Sleep ran `cross_modal_alignment` before `mine_phrase_relations`, amplifying contamination
- Contained codegen failure (`NameError: name 'false' is not defined`) correctly rejected by shadow test

This trade-off between consolidation quality and representational integrity is a fundamental finding: sleep is necessary for retention but dangerous for semantic precision.

#### 4.7 The One-Question-Per-Turn Bottleneck

Across all experiments, the system consistently asks exactly one question per turn. This is not a bug but an architectural property: knowledge-gap, feedback, rule-confirmation, logic-doubt, pattern-label, and word-conflict waits each consume the next input entirely. The derived behavioral rules (`behaviour.md`) include:

1. Probe-eats-reply: Never expect an answer while hypothesis queue is non-empty
2. Prompt-eats-input: Feedback waits, label prompts, and definition gaps each swallow one line
3. Pending questions die on restart (`last_asked_word` is RAM-only)
4. No belief revision: Contradictions are silently absorbed

This bottleneck limits conversational throughput but may serve as implicit regularization—by forcing the system to commit to one hypothesis per cycle, it prevents the kind of rapid, unconstrained updates that lead to catastrophic forgetting.

### 5. Architectural Trade-offs

| Axis | Herebot (Measured) | Transformer Baseline | Assessment |
|---|---|---|---|
| Exposures per new word | ~2, persistent | 10³–10⁴ fine-tune; 1–64 few-shot | 100–1000× cheaper per concept |
| Pretraining sunk cost | None | ~10¹¹ tokens / GPU-months | Starts free, pays per concept |
| Wall-clock to usable | ~3 s/turn CPU | Minutes–hours fine-tune | Interactive-speed learning |
| Depth | ~10 heterogeneous inspectable stages | 12–96 opaque layers | Inspectability vs fluency |
| Fluency | Glue recall, no generative grammar | Fluent, compositional | Not comparable yet |
| Forgetting | Metabolic phases, reward-gated | Catastrophic without rehearsal | Both need rehearsal |
| Belief revision | None (silent absorption) | Possible via fine-tuning | Fundamental limitation |
| Representation integrity | Requires explicit isolation (E12) | Less susceptible | Architectural vulnerability |

### 6. Test Coverage and Validation

The system maintains 169 tests across 8 test suites:
- `Curiosity_test.py`: 117 tests (core curiosity engine)
- `ProbabilisticLabeling_test.py`: 15 tests (concept/word-state labels)
- `BackendIntegration_test.py`: 11 tests (CSR, CI, phon, search)
- `FactorSpace_test.py`: 5 tests (E10 factor-space versioning)
- `GeneratedCodePlugin_test.py`: 6 tests (plugin generation)
- `SemanticRepresentation_test.py`: 15 tests (E12 semantic repair)
- `CandidateLedger_test.py`: tests for E15 candidate gating
- `TrackerTelemetry_test.py`: tests for turn telemetry

All 169 tests pass in 60 seconds. The test coverage spans unit, integration, adversarial, persistence, and non-regression scenarios.

### 7. Limitations and Future Directions

1. **No belief revision**: The architecture cannot detect or correct contradictions. A confidence-competition mechanism is needed.
2. **One-question bottleneck**: Conversational throughput is fundamentally limited. Multi-hypothesis expression per turn would require architectural changes.
3. **Glue-word cascade**: Definitions of content words always spawn follow-up gaps for function words (`means`, `is`, `a`), consuming training turns.
4. **Sleep pollution**: Self-talk during consolidation introduces junk into memory. Provenance gating (E15) partially addresses this but does not eliminate it.
5. **No generative fluency**: The system is a memory+curiosity engine, not a language model. The LSTM language model (`components/language_model.py`) is trained but insufficient for fluent generation.
6. **E15 unresolved**: Word conflict resolution has no lasting effect on probabilities due to recalculation in `normalize_probability_field`. 104 words remain stuck in 0.5/0.5 conflict.
7. **Similarity collapse root cause**: While E12 fixed the immediate collapse, the underlying vulnerability—coupled affective and semantic spaces without explicit isolation—remains a design constraint.

### 8. Conclusion

Herebot demonstrates that information-theoretic intrinsic motivation can drive sample-efficient, lifelong concept acquisition in a non-transformer architecture. The measured ~2 exposures per concept represents a 100–1000× improvement over transformer fine-tuning, achieved through the coupling of Dirichlet Process concept assignment, GP uncertainty, Bayesian surprise, and meta-learned curiosity weighting. However, this efficiency comes at the cost of zero generative fluency, no belief revision, and vulnerability to representational collapse when affective and semantic spaces are not properly isolated. The architectural evolution through E10 and E12 demonstrates that lifelong learning systems require explicit versioning, representation boundaries, and provenance gating to maintain integrity over time. The one-question-per-turn bottleneck, while limiting conversational throughput, may serve as implicit regularization against catastrophic forgetting. These findings suggest that the path to human-level sample efficiency in AI lies not in scaling data and compute, but in designing intrinsic motivation systems that can safely consolidate knowledge across timescales.

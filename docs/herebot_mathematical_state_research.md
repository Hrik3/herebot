# Herebot Mathematical State And Evaluation Research Notes

This document describes the mathematical state used by Herebot and the path an input follows through the current repository implementation. It is based on the live code in:

- `herebot/core/heart.py`
- `herebot/core/cuengine/processing.py`
- `herebot/core/cuengine/curiosity.py`
- `herebot/core/cuengine/idle.py`
- `herebot/core/cuengine/language.py`
- `herebot/backend/herebot_memory.py`
- `herebot/backend/incremental_csr_lib.py`
- `herebot/components/reward.py`
- `herebot/components/factor_handler.py`
- `herebot/components/probabilistic_labeling.py`
- `herebot/components/resoning.py`
- `herebot/components/learner.py`
- `herebot/components/feedback_process.py`
- `herebot/components/pattern_observer.py`
- `herebot/components/language_model.py`
- `herebot/concepts/hdp.py`
- `herebot/concepts/gaussian_uncertainity.py`
- `herebot/concepts/meta_learner.py`
- `herebot/concepts/controlspace.py`
- `herebot/traits/semlearn.py`
- `herebot/traits/identity_manager.py`
- `herebot/traits/code_generator.py`
- `herebot/backend/ci.py`

## 1. Scope And Assumptions

The implementation is not one pure mathematical model. It is a coupled system of probabilistic approximations, online counters, graph heuristics, symbolic inference, feedback tracking, state vectors, and persistent memory. The formulas below describe the actual code behavior where possible.

Assumptions used in this document:

1. "Input" means one call to `CuriosityEngine.process_input(text)`.
2. A Herebot "state" means a normalized vector built from control factors, not the complete Python object state.
3. A "token state" means the per-token feature, concept, uncertainty, curiosity, reward, and history entries.
4. Random choices in concept assignment, perturbation, dreaming, reply generation, and model sampling make execution nondeterministic.
5. Some components are named after stronger models, such as HDP or GP, but are implemented as practical approximations.
6. KaTeX formulas use implementation notation, not necessarily textbook notation.
7. Generated code files are derived through the `HerebotGeneratedPlugin` contract and are included as part of the mathematical runtime architecture.
8. Saved runtime artifacts are persistent state; this document describes their mathematical effect, not a fixed snapshot of their current serialized values.

## 2. Top-Level Engine Composition

`CuriosityEngine` is the surface class in `heart.py`. It inherits mixins that split the large engine into smaller files:

$$
\text{CuriosityEngine}
=
\text{Persistence}
+ \text{Curiosity}
+ \text{Processing}
+ \text{Idle}
+ \text{Language}.
$$

At initialization the engine constructs these main subsystems:

$$
\mathcal{E}
=
(
\text{MemoryBridge},
\text{CI},
\text{Phon},
\text{Diag},
\text{Concept},
\text{GP},
\text{Meta},
\text{Reward},
\text{Reasoner},
\text{Semantic},
\text{Syntax},
\text{Action},
\text{Control},
\text{State},
\text{Identity},
\text{CodeGen},
\text{PluginManager},
\text{Memory}
).
$$

Important persistent stores include CSR text memory, concept counts, probabilistic label stats, GP posterior, reward history, reasoning graph, word history, semantic embeddings, syntax learner, identity vector, feedback corpus, episodes, state files, state relation observations, state groups, generated plugin registry, hypotheses, and language model weights.

## 3. One Input Turn: Control Flow

For an input text $u$, `process_input(u)` first handles waiting states. These are early-return paths:

1. chat mode toggle,
2. shadow differentiation confirmation,
3. response feedback,
4. rule confirmation,
5. logic feedback,
6. pattern label,
7. trust feedback,
8. word conflict resolution,
9. definition teaching,
10. procedure teaching.

Only if no waiting handler consumes the input does the main path run.

The main path can be summarized as:

$$
u
\rightarrow
\text{memory ingest}
\rightarrow
\text{whole sentence CI/phon/diag}
\rightarrow
\text{tokenize}
\rightarrow
\text{reasoner observation}
\rightarrow
\text{per-token learning}
\rightarrow
\text{sentence reasoning}
\rightarrow
\text{control factor vector}
\rightarrow
\text{state recognition/learning}
\rightarrow
\text{action generation}
\rightarrow
\text{feedback request}
\rightarrow
\text{episode persistence}.
$$

In pseudocode:

```text
add_sentence_to_memory(u)
ci_report = CI.evaluate(u)
phon_score = Phon.score_text(u)
diag = Diagnostic.explain(ci_report, phon_score)
tokens = tokenize(u)
reasoner.observe_sentence(u, tokens)
reward.reset_episode()

for token in tokens:
    token_ci = CI.evaluate(token)
    token_phon = Phon.score_text(token)
    features = extract_features(token_ci, token_phon, diag)
    x = features_to_array(features)
    concept = concept_model.add_word_observation(token, features)
    factors = compute_curiosity_factors(token, token_ci, token_phon, diag, x)
    entropy = concept_model.entropy_word(token)
    igain = information_gain(token, entropy)
    if igain == 0:
        igain = 0.25 * bayesian_surprise + 0.25 * gp_uncertainty
    gp.update(x, token_ci.confidence)
    reward = reward_tracker.compute_reward(token_ci, igain, bayesian_surprise)
    update word_history
    structural = structural_score(token)
    meta_learner.update(factors, igain * (1 - structural))
    curiosity_score = meta_learner.predict(factors) * (1 - structural)

update token profiles
if all tokens known:
    process sentence reasoning
parse syntax / equations
maybe ask questions
factor_dict = factor_engine.compute(...)
factor_vector = ordered control factors
maybe state learn
generate reply
maybe ask feedback
store episode and save state
```

## 4. Tokenization

The tokenizer extracts alphabetic words, numbers, and math operators:

$$
\text{raw}(u)
=
\text{regex}_{[a-zA-Z']+|\text{number}|\text{operator}}(u).
$$

Tokens are lowercased. The minimum length is:

$$
\ell_{\min}
=
\begin{cases}
1, & |\mathcal{V}_{sem}| < 50,\\
2, & |\mathcal{V}_{sem}| \ge 50.
\end{cases}
$$

If no known semantic tokens are found but unknown tokens exist, the tokenizer relaxes back to length at least 1.

## 5. Whole-Text CI Evaluation

The CI engine perturbs the input many times, scores each perturbed form, and returns a stability-weighted confidence.

For one text $t$, base features are:

$$
\text{char\_valid}(t)
=
\frac{|\{c \in t : c \in \text{valid chars}\}|}{|t|}.
$$

Length score:

$$
\text{length}(t)
=
\begin{cases}
0.3, & |t| < 3,\\
0.6, & |t| > 300,\\
1.0, & \text{otherwise}.
\end{cases}
$$

Vowel structure:

$$
\text{structure}(t)
=
\operatorname{clip}_{[0,1]}
\left(
\frac{\#\text{vowels}(t)/|t|}{0.5}
\right).
$$

TF-IDF similarity is the best cosine similarity to the memory corpus:

$$
\text{similarity}(t)
=
\max_{c \in \mathcal{C}}
\cos(\operatorname{tfidf}(t), \operatorname{tfidf}(c)).
$$

Outlier score is:

$$
\text{outlier}(t)
=
\begin{cases}
1.0, & \text{IsolationForest marks length inlier},\\
0.4, & \text{otherwise}.
\end{cases}
$$

The CI scalar score is:

$$
s(t)
=
0.20\,\text{char\_valid}
+0.15\,\text{length}
+0.20\,\text{structure}
+0.30\,\text{similarity}
+0.15\,\text{outlier}.
$$

During evaluation, the Monte Carlo engine samples perturbed texts $t_i$. With $N=60$ by default:

$$
\mu
=
\frac{1}{N}\sum_{i=1}^{N}s(t_i),
\qquad
\sigma
=
\operatorname{std}(s(t_1),\ldots,s(t_N)) + 10^{-6}.
$$

Stability and confidence are:

$$
\text{stability}
=
\operatorname{clip}_{[0,1]}(1-\sigma),
$$

$$
\text{confidence}
=
\operatorname{clip}_{[0,1]}(\mu \cdot \text{stability}).
$$

## 6. Phonotactic Score

For each word $w$, the phonotactic engine computes five sub-scores:

$$
\text{phon}(w)
=
0.25\,s_{syllable}
+0.20\,s_{cluster}
+0.20\,s_{vowel}
+0.20\,s_{forbidden}
+0.15\,s_{morpheme}.
$$

For vowel balance:

$$
r_v
=
\frac{\#\text{vowels}(w)}{|w|},
$$

$$
H_v
=
-\sum_{a \in \text{vowels}} p(a)\log p(a),
$$

$$
s_{vowel}
=
\operatorname{clip}_{[0,1]}
\left(
0.5(1-|r_v-0.45|)
+0.5\,\operatorname{clip}_{[0,1]}\left(\frac{H_v}{2.5}\right)
\right).
$$

The sentence-level phonotactic score is the mean over words:

$$
\text{phon}(u)
=
\frac{1}{m}\sum_{j=1}^{m}\text{phon}(w_j).
$$

## 7. Diagnostic Explanation

Diagnostics repackage CI and phonotactic values into features:

$$
d
=
(
\text{char\_valid}=\mu,
\text{structure}=\text{phon}(u),
\text{similarity}=\mu,
\text{phonotactics}=\text{phon}(u),
\text{stability}
).
$$

Each diagnostic feature is classified by thresholds:

$$
\text{positive if } d_i \ge 0.7,
\qquad
\text{negative if } d_i \le 0.4,
\qquad
\text{neutral otherwise}.
$$

These negative diagnostics later affect token features and threat.

## 8. Token Feature Vector

For a token $w$, Herebot constructs:

$$
x_w
=
[
c_w,
s_w,
p_w,
m_w,
r_w
],
$$

where:

$$
c_w = \text{CI confidence}(w),
\qquad
s_w = \text{CI stability}(w),
\qquad
p_w = \text{phon}(w),
$$

$$
m_w = \text{CI mean}(w),
\qquad
r_w = \text{phon}(w).
$$

Diagnostic corrections are hard overrides:

$$
m_w =
0.1
\quad \text{if any negative diagnostic mentions characters},
$$

$$
r_w =
0.2
\quad \text{if any negative diagnostic mentions vowels}.
$$

This vector is used by the concept model and the GP uncertainty model.

## 9. Concept Model And Entropy

The concept model behaves like a simple Chinese Restaurant Process over latent concepts.

Let $n_c$ be count for concept $c$, $N=\sum_c n_c$, and $\alpha$ be concentration. Existing concept probability:

$$
P(z=c)
=
\frac{n_c}{N+\alpha}.
$$

New concept probability:

$$
P(z=\text{new})
=
\frac{\alpha}{N+\alpha}.
$$

After sampling a concept $z_w$, Herebot increments:

$$
n_{z_w} \leftarrow n_{z_w}+1,
\qquad
n_{w,z_w} \leftarrow n_{w,z_w}+1.
$$

The posterior concept distribution for a word is:

$$
P(c \mid w)
=
\frac{n_{w,c}}{\sum_{c'}n_{w,c'}}.
$$

Word entropy:

$$
H(w)
=
-\sum_c P(c \mid w)\log(P(c \mid w)+10^{-12}).
$$

Feature prediction for a word marginalizes over concepts:

$$
\hat{x}_{w,f}
=
\sum_c P(c \mid w)
\frac{n_{c,f}}{\sum_{f'}n_{c,f'}}.
$$

Bayesian surprise is a clipped KL divergence between the previous and current word-concept posteriors:

$$
B(w)
=
\operatorname{clip}_{[0,10]}
\left(
\sum_c P_{new}(c \mid w)
\log\frac{P_{new}(c \mid w)}
{P_{old}(c \mid w)+10^{-10}}
\right).
$$

## 10. Gaussian Process Uncertainty Approximation

The GP module is implemented as Bayesian linear regression over random Fourier features.

Random feature map:

$$
\phi(x)
=
\sqrt{\frac{2}{D}}
\cos(Wx+b),
$$

where $D$ is `n_features`, $W$ is sampled from a normal distribution scaled by length scale, and $b$ is uniform on $[0,2\pi]$.

Noise precision:

$$
\beta
=
\frac{1}{\sigma_{noise}^{2}}.
$$

Implementation update:

$$
P'
=
S^{-1}+\beta\,\phi\phi^\top,
$$

$$
S'
=
(P')^{-1},
$$

$$
m'
=
S'\left(P'm+\beta\,\phi y\right).
$$

The last equation is the exact implementation. A textbook Bayesian linear update would usually use the old precision in the prior term, but the current code uses the newly computed precision.

Prediction:

$$
\hat{y}
=
\phi(x)^\top m,
$$

$$
\operatorname{Var}(y \mid x)
=
\frac{1}{\beta}
+\phi(x)^\top S\phi(x).
$$

GP uncertainty:

$$
U_{gp}(x)
=
\sqrt{\max(\operatorname{Var}(y \mid x),10^{-12})}.
$$

## 11. Token Curiosity Factors

For each token, Herebot builds a dynamic dictionary of curiosity factors. Base factors are:

$$
\mathcal{F}_{token}
=
[
\text{uncertainty},
\text{learning\_progress},
\text{novelty},
\text{complexity},
\text{surprise},
\text{phonotactic\_anomaly},
\text{diagnostic\_negatives},
\text{bayesian\_surprise},
\text{gp\_uncertainty}
].
$$

Uncertainty:

$$
F_{unc}(w)=H(w).
$$

Learning progress uses stored entropy history:

$$
F_{lp}(w)
=
\max(0,H_{old}(w)-H_{new}(w)).
$$

In the current function, this is based on existing `word_history` entries, so it can lag the just-sampled concept update.

Novelty:

$$
F_{novel}(w)=1-c_w.
$$

Complexity:

$$
F_{complex}(w)
=
\operatorname{clip}_{[0,1]}
\left(
1-s_w+0.3\frac{\min(|w|,20)}{20}
\right).
$$

Feature surprise:

$$
F_{surprise}(w)
=
\operatorname{clip}_{[0,1]}
\left(
\sum_{f}
x_{w,f}\log\frac{x_{w,f}}{\hat{x}_{w,f}}
\right).
$$

Phonotactic anomaly:

$$
F_{phon}(w)=1-\text{phon}(w).
$$

Diagnostic negative count:

$$
F_{diag}(w)
=
\min\left(\frac{|\text{negative diagnostics}|}{5},1\right).
$$

Bayesian surprise:

$$
F_{bayes}(w)=B(w).
$$

GP uncertainty:

$$
F_{gp}(w)=U_{gp}(x_w).
$$

## 12. Knowledge Graph Modulation Of Curiosity

If a token has a knowledge graph node, graph degree and relation diversity modify factors.

Degree factor:

$$
d_w
=
\min\left(\frac{\deg(w)}{50},1\right).
$$

Then:

$$
F_{novel} \leftarrow
\min(1,F_{novel}+0.2d_w),
$$

$$
F_{unc} \leftarrow
\min(1,F_{unc}+0.15d_w),
$$

$$
F_{surprise} \leftarrow
\min(1,F_{surprise}+0.1d_w).
$$

Relation diversity:

$$
r_w
=
\frac{|\{\text{relation types adjacent to }w\}|}{5}.
$$

Complexity is boosted:

$$
F_{complex} \leftarrow
\min(1,F_{complex}+0.1r_w).
$$

If the word already has a stored definition, several curiosity terms are suppressed:

$$
F_i \leftarrow 0.1F_i
$$

for novelty, surprise, Bayesian surprise, GP uncertainty, uncertainty, and learning progress.

## 13. Top-Down Curiosity Modulation

If there is an active state embedding $e_s$ and token embedding $e_w$, the token novelty is reduced by identity alignment.

State/token cosine:

$$
\cos_s(w)
=
\frac{e_s^\top e_w}{\|e_s\|\|e_w\|+10^{-12}}.
$$

Identity stability bonus:

$$
I(w)
=
\max(0,\min(1,1-D(e_w,\text{identity}))),
$$

where drift distance is:

$$
D(v,i)
=
1-\frac{v^\top i}{\|v\|\|i\|+10^{-12}}.
$$

Novelty modulation:

$$
F_{novel}
\leftarrow
F_{novel}(1-0.5I(w)).
$$

Conscience beliefs can also reduce curiosity for matched ethical constraints:

$$
F_{curiosity}
\leftarrow
0.3F_{curiosity},
\qquad
F_{novel}
\leftarrow
0.5F_{novel}.
$$

## 14. Structural Token Score

Herebot keeps a behavioral profile for every token:

$$
P_w
=
(
\text{count},
\text{context set},
\overline{B},
\overline{IG},
\text{embedding centroid},
\text{embedding drift}
).
$$

Moving averages use $\eta=0.1$:

$$
\overline{B}
\leftarrow
(1-\eta)\overline{B}+\eta B_t,
$$

$$
\overline{IG}
\leftarrow
(1-\eta)\overline{IG}+\eta IG_t.
$$

Embedding centroid:

$$
\bar{e}_w
\leftarrow
(1-\eta)\bar{e}_w+\eta e_w.
$$

Drift:

$$
\delta_w
=
1-\frac{\bar{e}_w^\top e_w}{\|\bar{e}_w\|\|e_w\|+10^{-12}}.
$$

Running-stat normalization is sigmoid z-scoring:

$$
N(v;\mu,\sigma)
=
\frac{1}{1+\exp\left(-\frac{v-\mu}{\sigma+10^{-8}}\right)}.
$$

Structural score:

$$
S_{struct}(w)
=
\operatorname{clip}_{[0,1]}
\left(
0.35N(\text{count})
+0.35N(|\text{contexts}|)
+0.15(1-N(\overline{B}))
+0.15(1-N(\delta_w))
+b_{cold}
\right).
$$

Cold-start bias:

$$
b_{cold}
=
\begin{cases}
0.2, & |\mathcal{V}_{profile}| < 100 \text{ and } |w|\le 2,\\
0, & \text{otherwise}.
\end{cases}
$$

The structural score gates semantic curiosity:

$$
IG_{meta}(w)
=
IG(w)(1-S_{struct}(w)),
$$

$$
C_{token}(w)
=
(\mu^\top F_w)(1-S_{struct}(w)).
$$

## 15. Information Gain

Implementation information gain compares entropy against the previous stored entropy:

$$
IG(w)
=
\max(0,H_{old}(w)-H_{new}(w)).
$$

If this is zero in the main token loop, Herebot falls back to:

$$
IG(w)
\leftarrow
0.25B(w)+0.25U_{gp}(x_w).
$$

This means Bayesian surprise and GP uncertainty can substitute for entropy reduction when there is no measurable entropy drop.

## 16. Intrinsic Reward

The current reward implementation is feedback-signed and no longer a weighted average.

For a token:

$$
e_{pred}(w)
=
1-c_w.
$$

Each reward part is clipped finite and nonnegative:

$$
\operatorname{part}(z)
=
\begin{cases}
0, & z \text{ is invalid or NaN},\\
\min(1,\max(0,z)), & \text{otherwise}.
\end{cases}
$$

Intrinsic token reward:

$$
R_{intr}(w)
=
\operatorname{part}(e_{pred})
+\operatorname{part}(IG)
+\operatorname{part}(B).
$$

Episode and total reward update:

$$
R_{episode}
\leftarrow
R_{episode}+R_{intr}(w),
\qquad
R_{total}
\leftarrow
R_{total}+R_{intr}(w).
$$

Explicit feedback uses magnitude:

$$
M
=
|R_{episode}^{rated}|.
$$

Signed feedback reward:

$$
R_{fb}
=
\begin{cases}
+M, & \text{positive},\\
0, & \text{neutral},\\
-M, & \text{negative}.
\end{cases}
$$

Then:

$$
R_{episode}
\leftarrow
R_{episode}+R_{fb},
\qquad
R_{total}
\leftarrow
R_{total}+R_{fb}.
$$

Negative feedback also proposes a shadow differentiation link instead of immediately mutating the knowledge graph.

## 17. Meta-Curiosity Learner

The meta learner keeps weights $\mu_i$ over curiosity factors. Let:

$$
F_w
=
[F_1,\ldots,F_k].
$$

Prediction:

$$
\hat{C}(w)
=
\mu^\top F_w.
$$

The update target is information gain after structural gating:

$$
y_w
=
IG(w)(1-S_{struct}(w)).
$$

Exact implementation update per factor:

$$
\lambda_i'
=
\lambda_i+F_i^2,
$$

$$
\mu_i'
=
\frac{\lambda_i'\mu_i+F_i y_w}
{\lambda_i'+F_i^2}.
$$

The code initializes base factor weights near zero, so early curiosity predictions can be small until updates accumulate.

## 18. Semantic Learner

Each word has an embedding mean $e_w \in \mathbb{R}^{d}$ and log variance storage. Initialization:

$$
e_w = 0,
\qquad
\log\sigma_w = 0.
$$

Context learning pulls a word toward known context embeddings using a cosine-gradient approximation:

$$
\nabla_w
=
\sum_{c \in \text{context}}
\nabla_{e_w}\cos(e_w,e_c)
-\alpha e_w.
$$

Update:

$$
e_w
\leftarrow
e_w+\frac{1}{n_w+1}\nabla_w,
\qquad
n_w \leftarrow n_w+1.
$$

For definitions, known definition-token embeddings are averaged:

$$
t
=
\frac{1}{|\mathcal{K}|}\sum_{v \in \mathcal{K}}e_v,
\qquad
t \leftarrow \frac{t}{\|t\|+10^{-12}}.
$$

Then:

$$
e_w
\leftarrow
e_w
+0.1((t-e_w)-\alpha e_w).
$$

Semantic similarity:

$$
\operatorname{sim}(a,b)
=
\frac{e_a^\top e_b}{\|e_a\|\|e_b\|}.
$$

Unknown text can be embedded with hashed character trigrams:

$$
e_{char}(t)[h(g)]
\leftarrow
e_{char}(t)[h(g)]+1
\quad
\text{for each unique trigram }g.
$$

The vector is then normalized.

## 19. Sentence And Pattern Reasoning

The reasoner records co-occurrence counts:

$$
N(a,b)
\leftarrow
N(a,b)+1.
$$

Co-occurrence graph confidence saturates exponentially:

$$
\operatorname{conf}_{co}(a,b)
=
1-\exp\left(-\frac{N(a,b)}{5}\right).
$$

Structural word detection uses usage patterns. For word $w$:

$$
\operatorname{coverage}(w)
=
\frac{\text{sentences containing }w}
{\text{active sentences}},
$$

$$
\operatorname{bridge}(w)
=
\frac{\text{times }w\text{ has left and right neighbors}}
{\text{total occurrences of }w}.
$$

Neighbor entropy:

$$
H_N(w)
=
-\sum_j p_j\log(p_j+10^{-12}),
$$

normalized by $\log(k+1)$ where $k$ is number of neighbor count bins.

Pattern structural score:

$$
S_{pattern}(w)
=
0.4\,\operatorname{coverage}
+0.3\,\operatorname{bridge}
+0.3\,\operatorname{neighbor\_entropy}.
$$

A word becomes structural if:

$$
\operatorname{count}(w)\ge 3
\quad \text{and} \quad
S_{pattern}(w)\ge 0.25.
$$

Triplet extraction scans:

$$
\text{content run}
\quad
\text{structural bridge}
\quad
\text{content run}
$$

and emits:

$$
(\text{head phrase},\text{connector},\text{tail phrase}).
$$

If connector roles are learned, triplets become phrase relations such as `is_a`, `means`, or `similar`.

## 20. Knowledge Graph And Symbolic Inference

The knowledge graph stores typed nodes and directed relation edges:

$$
G=(V,E),
\qquad
e=(v_i,v_j,r,\gamma,\text{metadata}).
$$

Adding an edge sets:

$$
E[v_i][v_j][r]=\gamma.
$$

Some relations are symmetric in implementation, such as `synonym`, `similar`, and `co_occur`.

Transitive inference rules use multiplicative confidence. For:

$$
a \xrightarrow[r]{\gamma_1} b,
\qquad
b \xrightarrow[r]{\gamma_2} c,
$$

the inferred edge confidence is:

$$
\gamma_{a,c}
=
\gamma_1\gamma_2.
$$

Rules are bounded by max depth, for example depth 3 for `is_a` and `means`.

## 21. Shadow Differentiation Links

Negative feedback does not directly corrupt the graph. Instead, Herebot proposes a hypothesis:

$$
h
=
(
\text{subject},
\text{different\_context},
\text{object},
\text{condition},
\gamma=0.25
).
$$

The subject and object are selected from recent triplets, context tokens, response tokens, and one-hop graph neighbors. The proposal is stored as:

```text
type = shadow_differentiation_link
status = pending
predicate = different_context
```

No committed graph edge is added while pending.

On confirmation:

$$
G
\leftarrow
G
\cup
\{s \xrightarrow{\text{different\_context}} o,
o \xrightarrow{\text{different\_context}} s\}.
$$

Each edge carries metadata:

$$
\text{metadata}
=
(
\text{condition},
\text{source},
\text{rejected response},
\text{proposal key}
).
$$

On rejection, the hypothesis status becomes `rejected` and the graph remains unchanged.

## 22. Control Factors

After token analysis and possible questions, the factor engine computes the control state.

Base control factors:

$$
\mathcal{C}
=
[
\text{curiosity},
\text{threat},
\text{trust},
\text{confidence},
\text{uncertainty},
\text{energy},
\text{attention},
\text{comprehension},
\text{fluency},
\text{cognitive\_load}
].
$$

Curiosity:

$$
C
=
\max_{w \in u} C_{token}(w).
$$

Threat:

$$
T
=
\operatorname{clip}_{[0,1]}
\left(
0.3\frac{n_{neg}}{3}
+0.4(1-c_{sent})
+0.3(1-\text{phon}_{sent})
\right).
$$

Confidence:

$$
Q
=
c_{sent}.
$$

Global uncertainty:

$$
U
=
\begin{cases}
\operatorname{clip}_{[0,1]}
\left(
\frac{1}{5|\mathcal{W}|}\sum_{w \in \mathcal{W}}H(w)
\right),
& |\mathcal{W}|>0,\\
0.5,
& \text{otherwise}.
\end{cases}
$$

Energy:

$$
E
=
1-\operatorname{clip}_{[0,1]}(\text{CPU fraction}).
$$

Attention:

$$
A
=
\operatorname{clip}_{[0,1]}\left(\frac{|\text{token analyses}|}{20}\right).
$$

Comprehension is average pairwise semantic similarity where known embeddings exist:

$$
K
=
\operatorname{mean}_{w_i,w_j}
\operatorname{sim}(w_i,w_j),
$$

or $0.5$ if no usable similarities exist.

Fluency is currently fixed:

$$
L = 0.7.
$$

Cognitive load:

$$
CL
=
\operatorname{clip}_{[0,1]}(1-EA).
$$

Trust:

$$
\operatorname{Trust}
=
0.4Q
+0.3\,\text{stability}_{sent}
+0.3K.
$$

Conscience adds:

$$
\text{conscience\_active}
=
\min
\left(
1,
\frac{|\text{conscience beliefs}|}{\text{max beliefs}}
\right).
$$

Dynamic factors are appended through the adaptive control space.

## 23. State Vector And State Learning

The control factor vector is ordered by `control_space.all_factors`:

$$
v_t
=
[c_1,c_2,\ldots,c_n].
$$

State vectors are unit-normalized:

$$
\tilde{v}
=
\frac{v}{\|v\|}
\quad
\text{if } \|v\|>0.
$$

State similarity:

$$
\operatorname{sim}(a,b)
=
\frac{\tilde{a}^\top\tilde{b}}
{\|\tilde{a}\|\|\tilde{b}\|+10^{-12}}.
$$

State recognition checks labels whose similarity exceeds threshold, then applies a novelty distance gate:

$$
d(a,b)
=
\|\tilde{a}-\tilde{b}\|_2.
$$

Maturity:

$$
m
=
\operatorname{clip}_{[0.2,1]}
\left(
\frac{\log(1+n_{states})}{5}
\right).
$$

Novelty distance threshold:

$$
\tau_d
=
0.35(1.2-m).
$$

A matched state must satisfy:

$$
\operatorname{sim}(v_t,s)\ge \tau_s
\quad
\text{and}
\quad
d(v_t,s)\le \tau_d.
$$

If no close state exists and the vector variance is sufficient:

$$
\operatorname{Var}(v_t)\ge 0.05,
$$

the engine asks the user for a human label and saves the state.

## 24. State Conflict Handling

State conflicts are pairs of labels whose cosine similarity exceeds the conflict threshold:

$$
\operatorname{sim}(s_i,s_j)\ge \tau_{conflict}.
$$

A shadow merge vector is:

$$
s_{shadow}
=
\operatorname{normalize}
\left(
\frac{s_i+s_j}{2}
\right),
$$

after padding or truncating to aligned dimensions.

Conflict option 5 edits vectors manually. The current safe lookup checks canonical label, original label, aliases, reverse aliases, and shadow states:

$$
\operatorname{lookup}(\ell)
=
\text{first vector found in }
\{\ell,\operatorname{resolve}(\ell),\operatorname{alias}(\ell),\operatorname{reverse aliases}\}.
$$

If either vector is missing, the conflict is marked stale and processing continues without raising a `KeyError`.

## 25. Feedback Corpus

Feedback entries store:

$$
(
\text{timestamp},
\text{user input},
\text{system response},
\text{factor dict},
\text{rating},
\text{context tokens},
\text{response tokens},
\text{template id},
\text{metadata}
).
$$

Contexts are token keys, factor bucket keys, and response template keys:

$$
k \in
\{
\text{token:}w,
\text{factor:}f_{\text{low/mid/high}},
\text{template:}id
\}.
$$

Each context has Beta-like counters:

$$
\operatorname{Beta}(a,b)
$$

for positive, negative, and neutral labels. Posterior mean:

$$
P(s \mid k)
=
\frac{a_{k,s}}{a_{k,s}+b_{k,s}}.
$$

Posterior variance:

$$
\operatorname{Var}
=
\frac{ab}{(a+b)^2(a+b+1)}.
$$

Confidence:

$$
\operatorname{Conf}(k)
=
1-\operatorname{Var}_{best}.
$$

Feedback is requested while any relevant context has confidence below $0.85$, after at least 5 samples.

Implementation caveat: the current `_update_context_model` increments `a` only when the sentiment argument is positive; for negative or neutral sentiment arguments it increments `b` in that sentiment-specific model. This means the exact code behavior is not a conventional three-class Beta update.

## 26. Identity Vector

Identity is a persistent vector $i$ with confidence $q_i$. On reinforcement with state vector $v$ and reward $r$:

$$
\hat{v}
=
\frac{v}{\|v\|+10^{-12}}.
$$

Learning rate:

$$
\eta_i
=
\text{learning\_rate}\cdot r(1-q_i).
$$

Update:

$$
i
\leftarrow
(1-\eta_i)i+\eta_i\hat{v}.
$$

Then normalize:

$$
i
\leftarrow
\frac{i}{\|i\|}
\quad
\text{if } \|i\|>0.
$$

Confidence:

$$
q_i
\leftarrow
\min(1,q_i+0.01r).
$$

Drift:

$$
D(v,i)
=
1-\frac{v^\top i}{\|v\|\|i\|+10^{-12}}.
$$

## 27. Guilt, Conscience, Configuration, And Workspace

Guilt is a scalar $g \in [0,1]$ that reacts to negative feedback in an otherwise positive emotional state.

Positivity:

$$
P_{pos}
=
\frac{
\text{trust}
+(1-\text{threat})
+\text{energy}
+\text{comprehension}
}{4}.
$$

Time decay in hours:

$$
g
\leftarrow
g\cdot \rho^{\Delta h},
\qquad
\rho=0.99.
$$

Negative feedback update:

$$
g
\leftarrow
\min(1,g+b_gP_{pos}),
\qquad
b_g=0.2.
$$

Nonnegative feedback update:

$$
g
\leftarrow
0.95g.
$$

Guilt can modulate control signals:

$$
\text{confidence}
\leftarrow
\text{confidence}(1-0.3g),
$$

$$
\text{threat}
\leftarrow
\min(1,\text{threat}+0.2g),
$$

$$
\text{trust}
\leftarrow
\max(0,\text{trust}-0.2g).
$$

Conscience beliefs keep confidence and evidence. Confidence updates are clipped:

$$
b_{conf}
\leftarrow
\operatorname{clip}_{[0,1]}(b_{conf}+\Delta).
$$

A belief is promoted into active conscience when:

$$
b_{conf}\ge 0.85
\quad
\text{and}
\quad
\text{evidence}\ge 5.
$$

The global workspace model receives candidate coalitions $h_j$ and factors. Salience is:

$$
S(h_j)
=
\text{curiosity}\cdot r_{cur}(h_j)
+\text{threat}\cdot r_{thr}(h_j)
+\epsilon,
$$

where $\epsilon \sim U(0,0.1)$. The broadcast is:

$$
h^*
=
\arg\max_j S(h_j).
$$

Configuration modulation maps control factors into config parameters. For a modulation rule with transform `linear`:

$$
\theta'
=
\operatorname{clip}_{[\theta_{min},\theta_{max}]}
(a v_i+b).
$$

For `inverse`:

$$
\theta'
=
1-v_i.
$$

For `identity`:

$$
\theta'
=
v_i.
$$

After any transform, the implementation clamps the result when the rule provides `min` or `max`.

The default rules include:

$$
\alpha_{concept}
\leftarrow
\operatorname{clip}_{[0.1,2.0]}(0.5\,\text{curiosity}+0.5),
$$

$$
\sigma_{noise}
\leftarrow
\operatorname{clip}_{[0.01,0.5]}(0.2\,\text{uncertainty}+0.1).
$$

## 28. Action Generation

Action generation is priority-based rather than a single differentiable decision function. In non-chat mode the rough order is:

1. pending logic doubt,
2. simple math expression result,
3. conscience question,
4. guilt expression,
5. pending sentence query,
6. pending hypotheses,
7. parallel reasoning thought,
8. high-curiosity unknown word question,
9. state awareness,
10. trust/threat/comprehension/energy utterance,
11. default acknowledgement.

The normal fallback is:

$$
\text{reply}=\text{TPL.ACK.LISTEN}.
$$

In chat mode the generator first tries learned conversational rules, known phrase definitions, word-state labels, phrase similarity, word definitions, trigram continuation, structural patterns, and finally the LSTM language model.

State modulation can prepend technical state prefixes based on:

$$
\arg\max
\{
\text{curiosity},
\text{threat},
\text{trust},
\text{comprehension},
\text{energy}
\}.
$$

## 29. LSTM Language Model

The language model is a small numpy LSTM with expanding vocabulary.

For token id $x_t$, embedding:

$$
e_t=W_{emb}[x_t].
$$

Concatenate previous hidden state:

$$
z_t=[e_t;h_{t-1}].
$$

Gates:

$$
i_t=\sigma(z_tW_i+b_i),
$$

$$
f_t=\sigma(z_tW_f+b_f),
$$

$$
o_t=\sigma(z_tW_o+b_o),
$$

$$
g_t=\tanh(z_tW_c+b_c).
$$

Cell and hidden state:

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,
$$

$$
h_t=o_t\odot\tanh(c_t).
$$

Logits and probabilities:

$$
\ell_t=h_tW_{out}+b_{out},
$$

$$
P(x_{t+1}=j)
=
\frac{\exp(\ell_{t,j}/T)}
{\sum_k \exp(\ell_{t,k}/T)}.
$$

Training loss:

$$
\mathcal{L}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(y_t \mid x_{\le t}).
$$

Gradients are clipped to $[-5,5]$ and updated by SGD.

## 30. Episodic Memory

Each episode stores:

$$
e_t
=
(
\text{timestamp},
u_t,
a_t,
v_t,
R_t,
\text{metadata}
).
$$

Capacity is config-controlled and defaults to 10000. Episodes are saved as JSON files under state storage.

Episodic memory supports:

$$
\text{sample\_batch}(n)
\sim
\text{uniform sample of episodes},
$$

and:

$$
\text{replay\_sequence}()
=
\text{last 100 episodes}.
$$

## 31. Idle Learning And Dreaming

Idle introspection scans saved states for conflicts and creates shadow resolutions.

Offline consolidation replays recent episodes. For each token $w$ with context $C$:

$$
e_w
\leftarrow
\operatorname{observe\_context}(w,C).
$$

World model dreaming builds an $n$-gram model from known-token episodes:

$$
P(w_t \mid w_{t-n+1:t-1})
=
\frac{N(w_{t-n+1:t})}
{\sum_v N(w_{t-n+1:t-1},v)}.
$$

Active memory rehearsal prioritizes episodes by:

$$
\operatorname{priority}
=
0.4\,\text{surprise}
+0.3\,\text{rarity}
+0.2\,\text{identity drift}
+0.1\,\text{hypothesis bonus}.
$$

Chat reward multiplier:

$$
M_{chat}
=
\operatorname{clip}_{[0.5,2.0]}
\left(
1
+0.5\,\text{curiosity}
+0.3\,\text{trust}
-0.4\,\text{threat}
+0.2\,\text{comprehension}
\right).
$$

In chat mode:

$$
R_{episode}
\leftarrow
M_{chat}R_{episode}.
$$

## 32. Metabolic Maintenance

Rules, conscience beliefs, hypotheses, dynamic factors, states, and logic rules track:

$$
(
\text{energy cost},
\text{recent usage},
\text{predictive gain},
\text{phase},
\text{last used}
).
$$

Usage decay:

$$
u
\leftarrow
u\cdot 0.99^{\Delta t}.
$$

Effective gain for object items:

$$
g_{eff}
=
\frac{\text{predictive gain}\cdot \text{recent usage}}
{1+\Delta t}.
$$

For dictionary items, current code uses:

$$
g_{eff}
=
\max
\left(
0.5\,\text{predictive gain},
\frac{\text{predictive gain}\cdot \text{recent usage}}
{1+\Delta t}
\right).
$$

If effective gain is below cost, phase progresses:

$$
\text{active}
\rightarrow
\text{weak}
\rightarrow
\text{dormant}
\rightarrow
\text{archived}
\rightarrow
\text{deleted}.
$$

## 33. Mathematical View Of A Full Turn

For input $u_t$, the complete turn can be written compactly as:

$$
\text{Eval}(u_t)
=
(\text{CI}_t,\text{Phon}_t,\text{Diag}_t).
$$

Token analysis:

$$
\forall w \in \operatorname{Tok}(u_t):
\quad
x_w
\rightarrow
z_w
\rightarrow
H(w)
\rightarrow
F_w
\rightarrow
IG(w)
\rightarrow
R(w)
\rightarrow
C(w).
$$

Episode reward:

$$
R_t
=
\sum_{w \in \operatorname{Tok}(u_t)}
\left[
\operatorname{part}(1-c_w)
+\operatorname{part}(IG(w))
+\operatorname{part}(B(w))
\right].
$$

Control vector:

$$
v_t
=
\operatorname{FactorEngine}
(
\{C(w),F_w,H(w),B(w)\}_{w \in u_t},
\text{CI}_t,
\text{Phon}_t,
\text{Diag}_t,
R_{total}
).
$$

State update:

$$
s_t
=
\operatorname{StateManager}(v_t).
$$

Action:

$$
a_t
=
\operatorname{ActionGenerator}(u_t,v_t,\mathcal{H}_t,G_t,\mathcal{M}_t).
$$

Memory:

$$
\mathcal{M}_{t+1}
=
\mathcal{M}_t
\cup
\{(u_t,a_t,v_t,R_t)\}.
$$

Feedback, if requested later:

$$
R_{t+1}
=
R_t
+
\begin{cases}
+|R_t|, & \text{positive},\\
0, & \text{neutral},\\
-|R_t|, & \text{negative}.
\end{cases}
$$

Negative feedback additionally creates:

$$
h_{diff}
=
(\text{subject},\text{different\_context},\text{object},\text{condition})
$$

as a pending hypothesis.

## 34. Code-Derived Derivations

This section gives the derivation style that was missing from the first draft: each formula is traced from code assignments and branches.

### 34.1 CI Confidence Derivation

The implementation starts in `CIEngine.evaluate(text, samples=60)`.

Code-level steps:

```text
for each Monte Carlo sample:
    t = perturb(text)
    feats = extractor.extract(t)
    scores.append(model.score(feats))

mu, sigma = model.to_distribution(scores)
stability = clip01(1 - sigma)
confidence = clip01(mu * stability)
```

`model.score(feats)` loops over fixed weights:

```text
weights = {
    char_valid: 0.20,
    length: 0.15,
    structure: 0.20,
    similarity: 0.30,
    outlier: 0.15,
}
s += weight[k] * features[k]
```

Therefore the raw score for one perturbed sample is:

$$
s_i
=
0.20x_{char,i}
+0.15x_{len,i}
+0.20x_{struct,i}
+0.30x_{sim,i}
+0.15x_{outlier,i}.
$$

`to_distribution(scores)` computes:

$$
\mu
=
\frac{1}{N}\sum_i s_i,
\qquad
\sigma
=
\sqrt{\frac{1}{N}\sum_i(s_i-\mu)^2}+10^{-6}.
$$

Then `stability = clip01(1 - sigma)` gives:

$$
\operatorname{stability}
=
\operatorname{clip}_{[0,1]}(1-\sigma).
$$

Finally, `confidence = clip01(mu * stability)` gives:

$$
\operatorname{confidence}
=
\operatorname{clip}_{[0,1]}(\mu\operatorname{stability}).
$$

So CI confidence is not just a weighted feature score. It is a weighted feature score averaged over random perturbations, then multiplied by perturbation stability.

### 34.2 Phonotactic Derivation

`PhonotacticEngine.score_word(w)` returns five analyzer outputs:

```text
syllable, cluster, vowel, forbidden, morpheme
```

Then `score_text` appends:

```text
0.25 * syllable
+ 0.20 * cluster
+ 0.20 * vowel
+ 0.20 * forbidden
+ 0.15 * morpheme
```

Therefore:

$$
\operatorname{phon}(w)
=
0.25s_{syll}
+0.20s_{cluster}
+0.20s_{vowel}
+0.20s_{forbid}
+0.15s_{morph}.
$$

For a text with words $w_1,\ldots,w_m$, the code returns `float(np.mean(word_scores))`, so:

$$
\operatorname{phon}(u)
=
\frac{1}{m}\sum_{j=1}^{m}\operatorname{phon}(w_j).
$$

### 34.3 Token Feature Vector Derivation

`_extract_features` creates this dictionary:

```text
confidence = token_ci.get("confidence", 0.5)
stability = token_ci.get("stability", 0.5)
phon_score = token_phon
char_valid = token_ci.get("mean", 0.5)
structure = token_phon
```

Then diagnostic branches overwrite values:

```text
if "character" in negative:
    char_valid = 0.1
if "vowel" in negative:
    structure = 0.2
```

`_features_to_array` fixes the order:

```text
keys = ["confidence", "stability", "phon_score", "char_valid", "structure"]
return np.array([features.get(k, 0.5) for k in keys])
```

Therefore:

$$
x_w
=
[
\operatorname{confidence}_w,
\operatorname{stability}_w,
\operatorname{phon}_w,
\operatorname{meanCI}_w,
\operatorname{phon}_w
],
$$

except diagnostics can force:

$$
x_{w,4}=0.1
\quad
\text{for character negatives},
$$

and:

$$
x_{w,5}=0.2
\quad
\text{for vowel negatives}.
$$

### 34.4 Concept Entropy And Surprise Derivation

`add_word_observation(word, features)` first saves the previous posterior:

```text
prev_posterior[word] = {
    c: cnt / total for c, cnt in word_concept_counts[word].items()
}
```

Then it constructs CRP probabilities:

```text
for existing concept c:
    prob = concept_count[c] / (total + alpha)
new concept:
    prob = alpha / (total + alpha)
```

So:

$$
P(z=c)
=
\frac{n_c}{N+\alpha},
\qquad
P(z=new)
=
\frac{\alpha}{N+\alpha}.
$$

After sampling:

```text
concept_counts[concept] += 1
word_concept_counts[word][concept] += 1
```

This directly yields:

$$
P(c\mid w)
=
\frac{n_{w,c}}{\sum_{c'}n_{w,c'}}.
$$

`entropy_word` is:

```text
return -sum(p * math.log(p + 1e-12) for p in probs)
```

Therefore:

$$
H(w)
=
-\sum_cP(c\mid w)\log(P(c\mid w)+10^{-12}).
$$

`bayesian_surprise` loops over posterior concepts:

```text
kl += p_post * math.log(p_post / p_prior)
return np.clip(kl, 0, 10)
```

Therefore:

$$
B(w)
=
\operatorname{clip}_{[0,10]}
\sum_cP_{new}(c\mid w)
\log\frac{P_{new}(c\mid w)}{P_{old}(c\mid w)+10^{-10}}.
$$

### 34.5 Curiosity Factor Derivation

`_compute_curiosity_factors` derives each factor from local assignments:

```text
uncertainty = concept_model.entropy_word(token)
novelty = 1.0 - token_ci.get("confidence", 0.5)
stability = token_ci.get("stability", 0.5)
complexity = 1.0 - stability + (min(len(token), 20) / 20.0) * 0.3
complexity = clip(complexity, 0, 1)
```

Therefore:

$$
F_{unc}=H(w),
\qquad
F_{novel}=1-c_w,
$$

$$
F_{complex}
=
\operatorname{clip}_{[0,1]}
\left(1-s_w+0.3\frac{\min(|w|,20)}{20}\right).
$$

Feature surprise is derived from:

```text
predicted = concept_model.predict_features(token)
observed = _extract_features(...)
surprise = 0
for k in all_keys:
    p = predicted.get(k, 0.01)
    o = observed.get(k, 0.01)
    surprise += o * log(o / p)
surprise = clip(surprise, 0, 1)
```

Therefore:

$$
F_{surprise}
=
\operatorname{clip}_{[0,1]}
\sum_f o_f\log\frac{o_f}{p_f}.
$$

Diagnostic negatives are:

```text
diag_neg = min(len(diag["negatives"]) / 5.0, 1.0)
```

Therefore:

$$
F_{diag}
=
\min\left(\frac{n_{neg}}{5},1\right).
$$

Knowledge graph modulation is derived from:

```text
degree_factor = min(1.0, degree / 50.0)
novelty += 0.2 * degree_factor
uncertainty += 0.15 * degree_factor
surprise += 0.1 * degree_factor
complexity += 0.1 * rel_diversity
```

with clipping, giving the graph boost equations in Section 12.

### 34.6 Reward Derivation

`compute_reward` is:

```text
pred_error = 1.0 - token_ci.get("confidence", 0.5)
reward = clip(pred_error) + clip(igain) + clip(bayes_surprise)
episode_reward += reward
total_reward += reward
```

Therefore:

$$
R_{intr}
=
\operatorname{part}(1-c_w)
+\operatorname{part}(IG_w)
+\operatorname{part}(B_w).
$$

`apply_feedback_reward` normalizes by substring:

```text
magnitude = abs(float(magnitude or 0.0))
if "positive" in rating:
    reward = magnitude
elif "negative" in rating:
    reward = -magnitude
else:
    reward = 0.0
```

Therefore:

$$
R_{fb}
=
\begin{cases}
+M, & \text{positive},\\
-M, & \text{negative},\\
0, & \text{otherwise}.
\end{cases}
$$

This is why neutral is exactly zero and negative is exactly the negative of the positive magnitude.

### 34.7 Meta-Learner Derivation

The token loop computes:

```text
struct_score = structural_score(token)
igain_adjusted = igain * (1 - struct_score)
meta_learner.update(factor_array, igain_adjusted)
curiosity_score = meta_learner.predict_curiosity(factor_array)
curiosity_score *= (1 - struct_score)
```

Therefore:

$$
y_w=IG(w)(1-S_{struct}(w)),
$$

and:

$$
C(w)=(\mu^\top F_w)(1-S_{struct}(w)).
$$

Inside `MetaCuriosityLearner.update`:

```text
self.lambda_[i] += x ** 2
self.mu[i] = (
    (self.lambda_[i] * self.mu[i] + x * information_gain)
    / (self.lambda_[i] + x ** 2)
)
```

Because `lambda_[i]` is incremented before the assignment, the implemented rule is:

$$
\lambda_i'=\lambda_i+x_i^2,
$$

$$
\mu_i'
=
\frac{\lambda_i'\mu_i+x_iy}
{\lambda_i'+x_i^2}.
$$

That double appearance of $x_i^2$ is code-accurate.

### 34.8 Control Vector Derivation

`FactorEngine.compute` starts with a dictionary and assigns keys. For example:

```text
result["curiosity"] = max(token curiosity scores)
result["confidence"] = ci_report.get("confidence", 0.5)
result["attention"] = clip(token_count / 20.0, 0, 1)
result["fluency"] = 0.7
result["cognitive_load"] = clip(1.0 - energy * attention, 0, 1)
```

Therefore:

$$
C=\max_w C(w),
\qquad
Q=c_{sent},
\qquad
A=\operatorname{clip}_{[0,1]}\frac{n}{20},
\qquad
L=0.7,
$$

and:

$$
CL=\operatorname{clip}_{[0,1]}(1-EA).
$$

The final vector is not built by sorting keys. In `processing.py`:

```text
for k in control_space.all_factors:
    factor_vector_list.append(factor_dict.get(k, 0.5))
factor_vector = np.array(factor_vector_list)
```

Therefore:

$$
v_t[i]
=
\operatorname{factor\_dict}
[
\operatorname{control\_space.all\_factors}[i]
],
$$

with default $0.5$ for missing factors.

### 34.9 State Matching Derivation

The state manager normalizes vectors:

```text
norm = np.linalg.norm(vector)
if norm != 0:
    vector = vector / norm
```

Therefore:

$$
\tilde{v}
=
\frac{v}{\|v\|}
\quad
\text{if } \|v\|\ne 0.
$$

Similarity is implemented as:

```text
sim = dot(query_vec, stored_norm) / (norm(query_vec) * norm(stored_norm))
```

with zero-vector guards, giving:

$$
\operatorname{sim}(q,s)
=
\frac{q^\top s}{\|q\|\|s\|}.
$$

In `_handle_state_learning`, a state must pass both similarity and novelty distance:

```text
similar = find_similar_states(factor_vector, threshold=sim_threshold)
dist = np.linalg.norm(v1 - v2)
if dist <= novelty_dist_threshold:
    active_labels.append(label)
```

So recognition is:

$$
\operatorname{sim}(v,s)\ge \tau_s
\quad
\land
\quad
\|\tilde{v}-\tilde{s}\|_2\le \tau_d.
$$

### 34.10 Shadow Link Derivation

Negative feedback in `processing.py` calls:

```text
_propose_negative_feedback_link(...)
```

which calls:

```text
reasoner.propose_shadow_differentiation_link(...)
```

The proposal method chooses subject and object by priority:

1. last two observed triplets,
2. first four context content tokens,
3. first context token plus graph neighbor or second context token,
4. context token plus response token.

Then it stores:

```text
predicate = "different_context"
confidence = 0.25
status = "pending"
```

Therefore the pending hypothesis is:

$$
h
=
(s,\text{different\_context},o,\text{condition},0.25,\text{pending}).
$$

Only `commit_shadow_differentiation_link` calls `knowledge_graph.add_edge`, and it adds both directions. Therefore pending proposals have no graph edge until confirmation.

## 35. Additional Subsystems From The Repo Sweep

These parts were not detailed enough in the first version. They matter for a "whole repo" mathematical reading.

### 35.1 Incremental CSR Text Memory

`add_sentence_to_memory(text)` sends text into an `IncrementalCSR` backend. The vectorizer is a fixed-size `HashingVectorizer`:

$$
\phi_{csr}(u)
\in
\{0,1\}^{2^{20}}.
$$

Because `binary=True`, entries are presence indicators, not counts:

$$
\phi_j(u)
=
\begin{cases}
1, & \text{an n-gram hashes to bucket }j,\\
0, & \text{otherwise}.
\end{cases}
$$

The n-gram range is `(1,2)`, so both unigrams and bigrams contribute. Sentences accumulate in memory until:

$$
\text{pending rows}\ge \text{batch size}.
$$

Then a shard is written. A rebuild streams main matrix plus shards into a new CSR matrix:

$$
X_{final}
=
\begin{bmatrix}
X_{main}\\
X_{shard,1}\\
\vdots\\
X_{shard,k}
\end{bmatrix}.
$$

This memory backend supports the CI corpus used by TF-IDF matching.

### 35.2 Behavioral Rule Mining

Behavioral rules have confidence $q$, evidence $e$, and thresholds:

$$
q_{test}\ge 0.6,
\qquad
q_{confirm}\ge 0.8.
$$

Confidence update is:

$$
q
\leftarrow
\operatorname{clip}_{[0,1]}(q+\Delta).
$$

Evidence update:

$$
e
\leftarrow
e+
\begin{cases}
1, & \Delta>0,\\
-1, & \Delta\le 0.
\end{cases}
$$

Identity alias candidates are mined from recent episodes. If an alias appears after patterns such as `you are`, the candidate confidence is:

$$
q_{alias}
=
0.3+0.1\,n_{alias}.
$$

Greeting pattern confidence:

$$
q_{greet}
=
0.4+0.1\,n_{utterance}.
$$

Question-answer pattern confidence:

$$
q_{qa}
=
0.3+0.1\,n_{question}.
$$

Transition pattern confidence:

$$
q_{trans}
=
0.5+0.1\frac{n_{transition}}{n_{prev\ response}}.
$$

These are heuristic confidence scores, not probabilities calibrated from a likelihood model.

### 35.3 Type Hierarchy, Expression Trees, And Anti-Unification

The type hierarchy is a tree rooted at `Entity` with child types such as `Word`, `Phrase`, `Number`, `Operator`, and `State`.

Subtype relation is recursive:

$$
A\preceq B
\iff
A=B
\quad
\text{or}
\quad
\operatorname{parent}(A)\preceq B.
$$

`SyntaxTree.to_expression` converts parser output into expression trees. A single lowercase letter becomes a variable:

$$
\text{root}=x,\ |x|=1
\Rightarrow
\operatorname{Expression}(\operatorname{Variable}(x)).
$$

A numeric string becomes a numeric constant expression. Operators become binary expressions when arity is 2 and two children exist:

$$
\operatorname{Expression}(op,left,right).
$$

Anti-unification in `SchemaLearner` generalizes two expressions:

$$
\operatorname{anti}(a,a)=a,
$$

$$
\operatorname{anti}(a,b)=X_i
\quad
\text{if }a,b\text{ are different leaves},
$$

and recursively:

$$
\operatorname{anti}(op(l_1,r_1),op(l_2,r_2))
=
op(\operatorname{anti}(l_1,l_2),\operatorname{anti}(r_1,r_2)).
$$

If roots or tree structures differ, anti-unification returns no schema.

### 35.4 Generated Code And Shadow Execution

Confirmed logic can be converted into Python code. For generated state definitions, matching is:

$$
\operatorname{match}(v,p)
=
\left[
\frac{v^\top p}{\|v\|\|p\|+10^{-12}}
>
\tau
\right].
$$

Shadow execution imports the generated module, runs `run(...)`, captures stdout and stderr, and returns:

$$
(\text{success},\text{error message},\text{output}).
$$

If execution raises, the exception is appended to the generated file as comments and the shadow test is logged.

### 35.5 Tracker Metrics

`CuriosityTracker.record(engine)` snapshots metrics after interactions.

Words known:

$$
W_t=|\operatorname{word\_history}|.
$$

Concept count:

$$
K_t=|\operatorname{concept\_counts}|.
$$

Average entropy:

$$
\bar{H}_t
=
\frac{1}{W_t}\sum_{w\in \operatorname{word\_history}}H(w).
$$

Top curiosity score:

$$
C^*_t
=
\max_w \mu^\top F_w.
$$

The tracker also records factor weights, states learned, unique token profile count, and session metadata. Its plotting layer is visualization, not part of the engine decision math.

### 35.6 Search Backend And System Stats

`AdaptiveSearchBackend` wraps the same incremental CSR memory for search. Query text is hashed with the live CSR vectorizer:

$$
q=\phi_{csr}(u).
$$

Live memory scores are sparse dot products:

$$
s_i
=
X_{live,i}q^\top.
$$

The backend returns top rows only if:

$$
\max_i s_i \ge \tau_{search},
$$

where the default confidence threshold is loaded from `search_config.json` or falls back to $0.4$.

System stats are wrappers around `psutil`. CPU, RAM, swap, disk, and I/O readings feed resource gates elsewhere in the repo. For example, several idle tasks use:

$$
\text{cpu\_ok}
=
[\text{CPU percent}<80],
\qquad
\text{mem\_ok}
=
[\text{memory percent}<90].
$$

When either gate fails, idle work is skipped or throttled.

### 35.7 CLI And Idle Thread

`main.py` constructs the engine, starts `IdleReasoner`, and loops over user input. The idle thread checks:

$$
\Delta t
=
t_{now}-t_{last\ interaction}.
$$

Short idle fires when:

$$
\Delta t>\tau_{short}.
$$

Long idle fires once per idle period when:

$$
\Delta t>\tau_{long}
\quad
\text{and}
\quad
\text{long\_idle\_done}=\text{false}.
$$

Short idle records tracker snapshots with trigger `idle_short`. Long idle runs deep state introspection and all idle tasks, saves state, records `idle_long`, then blocks repeat long-idle execution until the next user interaction resets the flag.

### 35.8 Visualization Utilities

The `visualisation` directory is a support layer. `run.py` and `structure.py` convert code graph JSON into viewer JSON:

$$
N_{out}
=
\{(id,label,name,type,original\_id)\}_{nodes},
$$

$$
E_{out}
=
\{(id,source,target,original\_id)\}_{edges}.
$$

The HTML files render Markdown/KaTeX or dependency graphs. They do not change Herebot's mathematical state; they are inspection tools for humans.

## 36. Important Limitations

1. The concept model is CRP-like, but not a full hierarchical Bayesian nonparametric model with complete posterior inference.
2. The GP is an online Bayesian linear model over random Fourier features, not an exact Gaussian process.
3. Meta-curiosity update is a custom online rule, not full Bayesian linear regression despite class comments.
4. Structural scoring and pattern detection are usage heuristics and can be biased by repeated sessions.
5. Feedback sentiment confidence currently has a nonstandard Beta update for negative and neutral labels.
6. State vectors depend on dynamic factor ordering. Persisted vectors may need padding or truncation as factors evolve.
7. Some decisions are priority-ordered symbolic control flow, not optimized by a single objective function.
8. Text generation mostly emits technical templates (`TPL.*`) unless the LSTM is trained enough or chat-mode heuristics select learned text.
9. State learning may ask for interactive labels, so mathematical processing can pause for human naming.
10. Saved runtime state strongly affects behavior; two identical inputs may produce different outputs after different histories.

## 37. Practical Reading Map

To understand one behavior, follow these files in order:

1. `core/heart.py` for engine construction.
2. `core/cuengine/processing.py` for turn-level control flow.
3. `backend/ci.py` for text confidence and phonotactics.
4. `core/cuengine/curiosity.py` for token factors and structural gating.
5. `concepts/hdp.py` for concept entropy and Bayesian surprise.
6. `concepts/gaussian_uncertainity.py` for uncertainty.
7. `concepts/meta_learner.py` for curiosity factor weights.
8. `components/reward.py` for intrinsic and feedback reward.
9. `components/factor_handler.py` for control vectors and states.
10. `components/resoning.py` and `components/learner.py` for graph, syntax, symbolic inference, and shadow links.
11. `core/mind.py` for response selection.
12. `core/cuengine/persistence.py` for what becomes long-term state.
13. `components/rules.py` for behavioral rule mining.
14. `traits/symbol_tracer.py` and `traits/tree_formation.py` for schemas, variables, operators, and expression conversion.
15. `traits/code_generator.py` for generated logic and shadow execution.
16. `backend/incremental_csr_lib.py` for the text-memory matrix used by the CI corpus.
17. `core/tracker.py` for persisted metrics and plots.
18. `backend/backendSearch.py` and `backend/SystemStats.py` for search and resource telemetry.
19. `main.py` for CLI flow and idle-thread scheduling.
20. `visualisation/` for graph/Markdown viewing utilities.

# HereBot Investigation Reports

Genuine architecture interrogation: one-on-one sittings through the real
launch script (`herebot/main.py`), each input hand-picked from the previous
reply. No batches. Template vocabulary (`TPL.*`) is defined in
`abbreavations.md`; training log lives in `report.md`.

## Part I — Interrogation

### Session 1 — Can it form a one-shot memory? (`zibber`)

| Turn | I said | It did |
|---|---|---|
| 1 | `zibber` (novel word) | `TPL.ASK.DEF target='zibber'?` — curiosity ~1.9, textbook knowledge gap |
| 2 | `zibber means a glowing stone` | `📘 [Definition Learning] Learned about 'zibber'` — then spent its reply on a queued `TPL.ASK.SYN stone/thing` |
| 3 | `what is zibber` | Spoke `TPL.ASK.SYN river/thing` (hypothesis queue stole the reply), **but** artifacts show `zibber` curiosity collapsed 1.9 → 0.36, novelty 0.01, `TPL.HYP.REL zibber→means` — the memory is formed and indexed after a single exposure |

Finding: encoding and speaking are decoupled. It learns in one shot but can
only *say* one thing per turn, and pending hypotheses outrank answers.

### Session 2 — Cross-session recall + analogy (chat mode)

| Turn | I said | It replied |
|---|---|---|
| 2 | `what is zibber` (new process!) | `TPL.RECALL.PHRASE phrase='what is zibber'` — phrase memory survived a restart via `reasoner.pkl`, and the text (not a hash) displayed: phrase-storage fix verified live |
| 3 | `is fire like the sun` | `TPL.DEF.KNOWN word='is' value='is means means'` — no analogy machinery fired; chat answers analogy questions with nearest-token recall |

Finding: recall works across restarts (persistent one-shot memory — the
architecture's strongest trait). Analogy questions get no analogy treatment;
`generate_chat_response` has no analogy path.

### Session 3 — Contradiction, emotion, math (adversarial)

- `zibber is a bird` (direct contradiction of the learned `glowing stone`):
  reply consumed by `TPL.ASK.SYN river/thing`; post-session check shows the
  definition is **unchanged** (`zibber means a glowing stone`). First-write-wins;
  there is no contradiction detector, no confidence competition, no question
  asked. A liar can never correct it, but neither can a teacher.
- `i am sad and alone` was eaten as a feedback rating ("No clear rating,
  ignoring") because the previous turn printed a `TPL.ASK.FB` probe — the
  one-question-per-turn tax again.
- `3 * 4` → `TPL.MATH.RESULT value='12'`: math path re-verified end to end.
- **Bug found live**: the state-label `input()` prompt ate my `exit` line and
  saved a state literally named `exit`; then stdin exhaustion sent `main.py`
  into an infinite EOF traceback loop (6.4M lines until timeout kill).
  Fixed: `main.py` now breaks cleanly on `EOFError`; garbage state deleted.

### Casual session (6 chat turns, see `report.md` for table)

Graceful degradation confirmed: 3-unknown sentences
(`the bright sun knows nothing`) get nearest-token recall + state prefix
instead of gaps or crashes; `goodbye friend` recalled the true hypernym
(`goodbye is a greeting`).

### Incidents → behaviour rules derived

1. Probe-eats-reply (Sessions 1, 3): never expect an answer while its
   hypothesis queue is non-empty; drain with neutral turns or use chat mode.
2. Prompt-eats-input (Sessions 3, earlier rounds): feedback waits, label
   prompts, and definition gaps each swallow exactly one line — plan one line
   per prompt, and make filler lines label-safe.
3. Pending questions die on restart (`last_asked_word` is RAM-only):
   multi-sitting teaching must re-elicit the question first.
4. No belief revision: contradictions are silently absorbed (Session 3).

## Part II — Experimental data

### Measured learning curve (from `graphs/tracker_data.json`, 433 points)

Fresh-model sessions (load_state=False):

| Session | Interactions | Words known | Concepts | Lifetime reward | Rate |
|---|---|---|---|---|---|
| `55fb77` | 1 → 49 | 1 → 29 | 1 → ? | 3.1 → 101.0 | ~0.58 words/turn |
| `5cfeb3` | 1 → 47 | 1 → 28 | — | → 86.2 | ~0.59 words/turn |
| `9901f5` (loaded) | 6 → 133 | 33 → 94 | — | → 299.3 | ~0.48 words/turn |

Marginal cost per word: **≈1.7–2.1 turns (≈2 exposures: the word + its
definition)**. Avg concept entropy stays 0.01–0.06 across all sessions:
concepts collapse after a single exposure (one-shot binding, Session 1
`zibber` trace). Curiosity stays high throughout (top scores 2.8–7.1), i.e.
it does not saturate/bore within 130-turn sittings.

### Against transformer-based learning (estimates, order-of-magnitude)

| Axis | HereBot (measured) | Transformer baseline (public approx.) | Reading |
|---|---|---|---|
| Exposures per new word/concept | ~2, persistent after 1 definition | Fine-tune: ~10³–10⁴ labeled examples per task; few-shot: 1–64 in-context, transient | ~2–3 orders of magnitude cheaper per concept than fine-tuning; comparable to in-context but persistent |
| Pretraining sunk cost | none (starts blank) | ~10¹¹ tokens / GPU-months | HereBot starts free; pays per concept instead of upfront |
| Wall-clock to usable knowledge | seconds (one reply cycle, ~3 s/turn CPU) | minutes–hours fine-tune + days of data curation | interactive-speed learning vs batch-speed learning |
| Depth | ~10 heterogeneous inspectable stages (CI → phonotactic → diagnostic → GP/HDP/meta-learner → factor engine → state manager → hypotheses → behavioral rules → logic inheritance → generated plugins → guilt/conscience), every step printed as artifacts | 12–96 homogeneous attention layers, opaque activations | different depth axes: traceable-architectural vs representational; HereBot wins inspectability, loses fluency |
| Fluency / breadth | recalls glue (`is means means`), no generative grammar | fluent, compositional, broad | not comparable yet — HereBot is a memory+curiosity engine, not a language model |
| Forgetting | metabolic phases (active→weak→dormant→archived→deleted), reward-gated | catastrophic without rehearsal; in-context forgotten next session | both need rehearsal; HereBot's is built-in and usage-driven |

Estimate: for narrow vocabulary/knowledge acquisition with a human in the
loop, HereBot is roughly **100–1000× more sample-efficient per concept**
than fine-tuning a transformer, at the cost of zero generative fluency and
zero transfer breadth. Against few-shot prompting it ties on exposures but
wins on persistence (recall across restarts, Session 2) and loses on
expression quality. These are experimental bounds from 400+ tracked turns,
not benchmarks — the honest next step is a fixed 100-word curriculum timed
against a LoRA fine-tune on the same list.

### Resume checklist (unchanged)

`know` still undefined; SYN queue (`moon`, `sky`, …) unconfirmed;
`TPL.ASK.CONN.REL` confirmations pending (`rel='unknown X'` cleanup); glue
closures (`have`, `means`); longer chat sittings for trigram/LM growth.

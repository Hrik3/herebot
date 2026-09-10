# HereBot Live Behaviour Observations

Training approach: HereBot architecture-based thinking training — start with
single **words**, expand to **sentences** only on model feedback. All output
messages below use the template vocabulary from `abbreavations.md`
(`TPL.*` prefix = hardcoded template; anything else is learned/retrieved).

Launch: `python herebot/main.py` from repo root (`/home/hrirake/herebot`).
State persists in `./curiosity_state/` **relative to the CWD you launch from**,
so always launch from the same directory or the model starts blank.
Chat-mode toggle is the literal phrase `switch chat` (not `chat`).
Tests only run with the package on the path, e.g.
`PYTHONPATH=herebot python3 herebot/tests/cor/Curiosity_test.py`
(`unittest discover` finds 0 tests because files are named `*_test.py`).

## Fixes applied this session (`herebot/core/cuengine/processing.py`)

1. **Tokenizer dropped all single-char tokens** (`_tokenize` filtered
   `len(t) >= 2` before the min-length gate). `2 + 2` tokenized to `[]`,
   `a`/`I` vanished. Raw tokens now keep digits/operators/single letters;
   the vocab-size gate (`min_len = 1` below 50 words, else `2`) still filters
   noise. Verified: `2 + 2 -> ['2', '+', '2']`, `a -> ['a']`.
2. **Math path was dead**: after fix 1, pure-math input still hit the
   unknown-word early return. `2 + 2` now bypasses the knowledge-gap branch
   and reaches `generate_reply`, returning `TPL.MATH.RESULT value='4'`
   (previously `TPL.ASK.DEF target='2'`-style gap).
3. **Definition misattribution (real bug)**: any sentence containing
   `is`/`means`/`refers to` was consumed as the definition of whatever word
   was pending (`last_asked_word`), even if that word was absent. Observed:
   `road is a path for cars` was stored as the definition of `vehicle`,
   `fly means to move in air` was stored under `black`. The handler now only
   consumes the turn when the pending word/phrase actually occurs in the
   input.
4. **Silent pending-definition loss**: a new unknown word overwrote
   `last_asked_word` with no trace (e.g. `hello`'s definition lost when
   `water` arrived). Now logs `[Definition Dropped] Previous pending word
   'X' never got a definition; now asking about 'Y'.`
5. **Empty input burned a training cycle** (counter +1, episodic write,
   trust utterance). Now returns `TPL.NO.CTX` with no state change
   (`main.py` already skipped empties; `process_input` did not).

All 149 existing tests still pass after the fixes
(Curiosity 117, BackendIntegration 11, ProbabilisticLabeling 15,
GeneratedCodePlugin 6).

## Live training log (fresh engine, `CuriosityEngine(load_state=False)`)

### Stage 1 — single words (model asks, trainer answers immediately)

| # | Trainer input | Model output | Observation |
|---|---|---|---|
| 1 | `hello` | `TPL.ASK.DEF target='hello'?` | unknown_ratio 1.0 → knowledge-gap branch, `last_asked_word='hello'`, episode reward 0.0 |
| 2 | `hello means a greeting` | learned `hello`; next asks `TPL.ASK.DEF target='means'?` | definition stored, but glue words (`means`, `greeting`) become the next gaps — every definition spawns 1–2 follow-up gaps |
| 3 | `water` | `TPL.ASK.DEF target='water'?` (+ `[Definition Dropped]` warning for `means`) | pending word is overwritten, never learned — **protocol rule: answer the pending word before introducing new words** |
| 4 | `water is a liquid` | learned `water`; asks `TPL.ASK.DEF target='liquid'?` | same glue cascade (`is`, `liquid`) |
| 5–8 | `crow` / `crow is a black bird`, `fly` / `fly is to move in air` | same pattern; defs = `crow, fly, hello, water` | total intrinsic reward climbs 2.15 → 36.3 across stage 1; each definition turn carries curiosity artifacts with per-token `curiosity`/`entropy` |

### Stage 2 — two-word combos (known vocabulary → full pipeline)

- `crow fly` → no question; full `Meta-Curiosity Analysis` artifacts,
  token curiosities (`crow 0.54`, `fly 1.13`), factor snapshot
  (`curiosity 1.13, threat 0.21, trust 0.56, energy 0.3`), episode reward
  ~0.6 added to lifetime total.
- `hello crow` → artifacts + `TPL.ASK.FB rating=positive|negative|neutral?`
  feedback probe; answering it drives `reward_tracker` and (on `negative`)
  proposes a `TPL.ASK.DIFF.LINK` shadow-differentiation link for confirmation.
- Meta-learner weights adapt live, e.g.
  `complexity 3.5, uncertainty 2.3, gp_uncertainty 2.2, novelty 1.8` —
  structural/complexity factors dominate early training.

### Stage 3 — sentences, math, modes

- `crow fly over water` → full-pipeline response, no gap asked
  (all tokens known or structurally scored); hypotheses surface as
  `TPL.HYP.REL` / `TPL.HYP.SIM`.
- `2 + 2` → `TPL.MATH.RESULT value='4'` (post-fix; was a knowledge gap).
- `switch chat` → `TPL.MODE.CHAT.ON?`; in chat mode replies use learned
  content first (`TPL.DEF.KNOWN word='hello' ...`), modulated by state
  prefixes (`TPL.PFX.*`), with a 2.0× reward multiplier on high curiosity;
  second `switch chat` → `TPL.MODE.CHAT.OFF`.
- `idle_reflection()` with no history → `TPL.IDLE.EMPTY`-family;
  with history → `TPL.IDLE.STATS words=4 concepts=1 ... reward=9.05 ...`
  plus a `TPL.HYP.REL` candidate.
- `positive` / `good` typed spontaneously → treated as **vocabulary**
  (`TPL.ASK.DEF target='positive'?`), not as ratings. Ratings only count
  when a feedback wait-state is actually pending (`expecting_feedback` /
  `expecting_response_feedback`).

## Behavioural quirks to respect while training (not bugs, architecture)

1. **One question per turn.** Knowledge-gap, feedback, rule-confirmation,
   logic-doubt, pattern-label, and word-conflict waits each consume the next
   input entirely. A training word typed while a wait is pending is eaten by
   that wait (observed: second `hello` swallowed as trust feedback,
   interaction counter not incremented, `Feedback processed.`).
2. **Answer pending `TPL.ASK.DEF` first.** Send `<word> is/means ...`
   containing that exact word before any new word, or the pending word is
   dropped (now loudly).
3. **Chase or tolerate glue-word gaps.** Definitions of content words always
   surface function-word gaps next (`means`, `is`, `liquid`, `black`,
   `move`). Either define them too (`move is a thing` worked and stuck) or
   leave them pending and move on at the cost of the drop warning.
4. **Feedback vocabulary must be taught.** `positive/negative/good/bad` start
   as unknown words; only train them as ratings inside real feedback turns,
   otherwise they just become more definitions.
5. **State learning wakes late.** `_handle_state_learning` gates on
   `len(word_embeddings) > 20`; before that, `current_state_labels == []`
   and all `TPL.STATE.*` output is template-only with no learned states.
6. **Guilt/unease utterances** (`TPL.GUILT.*`, `TPL.STATE.MAYBE_WRONG`)
   appear spontaneously on low-confidence turns (e.g. first-ever empty turn
   returned `TPL.LEARN TPL.STATE.MAYBE_WRONG`); they set
   `expecting_feedback`, so the next input is consumed as a rating.

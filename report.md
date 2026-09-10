# HereBot Behaviour Report (training paused)

Model state at pause: vocab 94, definitions 80, episodic memory 124 turns,
lifetime intrinsic reward 360.5, learned states: `greetings`, `bright curiosity`.
All output templates (`TPL.*`) are defined in `abbreavations.md`.
Training method: thinking-architecture protocol — words first, sentences only
on model feedback, every prompt (`TPL.ASK.DEF`, state `Label:`, `TPL.ASK.FB`)
answered manually in live `main.py` sittings.

## Casual verification session (chat mode, 6 turns)

| # | I said | It replied | Trace |
|---|---|---|---|
| 1 | `switch chat` | `TPL.MODE.CHAT.ON?` | command toggle, chat history reset |
| 2 | `hello, how are you today` (`today` unknown) | `TPL.DEF.REL word='how' value='how is a word' rel='unknown word'` | unknown tolerated in chat (gap branch skipped); fell back to first known token's definition |
| 3 | `do you know what water is` (`know` undefined) | `TPL.PFX.CURIOUS \| TPL.DEF.KNOWN word='do' ...` | curiosity-state prefix modulation + recall of a known token; unknown `know` did not derail it |
| 4 | `the bright sun knows nothing` (3 unknowns) | `TPL.PFX.CURIOUS \| TPL.DEF.KNOWN word='the' ...` | same graceful degradation; no crash, no gap asked |
| 5 | `why is the sky blue` (all known) | `TPL.PFX.CURIOUS \| TPL.DEF.REL word='why' ...` | full-vocabulary sentence → definition recall with relation |
| 6 | `goodbye friend` | `TPL.DEF.REL word='goodbye' value='goodbye is a greeting' rel='unknown greeting'` | **best learning evidence**: recalled the true taught hypernym (`greeting`), not a glue word |

## Behaviour observed (with deriving incidents)

1. **One question per turn, always.** Every reply is a single move: a pending
   synonym hypothesis (`TPL.ASK.SYN water/thing, tree/thing, sun/thing,
   moon/thing, sky/thing, rain/thing, fire/thing, fish/thing, horse/thing`),
   a definition request, or a normal reply — never mixed. Incident: my
   `bird is an animal with wings` turn was spent uttering `TPL.ASK.SYN
   word='water'...` instead of replying; the sentence still trained silently
   (observed via embeddings) but produced no visible answer.
2. **Waiting states eat the next input.** A `🏷️ Label:` state prompt consumed
   my scripted `crow fly over water` line as a state name (renamed afterwards
   to `bright curiosity`); a `TPL.ASK.FB` probe consumed `quiet river` as a
   rating ("No clear rating, ignoring"). Derived rule: only one planned line
   per prompt; filler lines must be safe as labels *and* as inputs.
3. **Glue-word cascade.** Each definition spawns gaps for function words
   (`means`, `is`, `a`, `have`). The model now holds degenerate but stable
   glue entries (`is means means`, `how is a word`) and *reuses* them in
   conversation instead of crashing — a crutch, not understanding.
4. **State recognition works.** `hello` turns print `State recognized as:
   'greetings'` with factor snapshot (trust ~0.76, threat ~0.09); chat replies
   get modulated (`TPL.PFX.CURIOUS |`, `TPL.STATE.CURIOUS word=...`).
5. **Sentence hypotheses emerge.** `who are you` produced
   `TPL.ASK.CAT subject='who' object='you'?` unprompted — the analogy/
   category engine firing on pronouns.
6. **`rel='unknown X'` is a display artifact, not knowledge.** Triplet
   connector `is` is not yet a learned connector, so relations store as type
   `unknown` + object (`rel='unknown word'`, `rel='unknown nature'`,
   `rel='unknown greeting'`). The `TPL.ASK.CONN.REL` hypotheses exist to
   resolve exactly this — future training should confirm them.
7. **Pending questions do not persist across restarts**
   (`last_asked_word` is runtime-only). Incident: round 4 asked
   `TPL.ASK.DEF target='know'?`, but round 5's `know means to have learned`
   was treated as a fresh sentence (asked `have` instead) — `know` is still
   undefined at pause. Resume point: re-ask `know` first.
8. **Substring feedback bug (fixed).** `the sun is bright and hot` was
   recorded as *positive trust feedback* because `right ⊂ bright`. All five
   feedback classifiers now match whole words (`_feedback_words`). Verified:
   same sentence now trains normally; 128/128 tests pass.

## Fixes landed during training (code)

- `core/cuengine/processing.py`: tokenizer kept dropping single-char tokens
  (`2+2` → `[]`); pure-math inputs bypass the knowledge-gap branch
  (`TPL.MATH.RESULT value='4'`); definitions only attach when the pending
  word occurs in the input (was misattributing, e.g. `road...` → `vehicle`);
  `[Definition Dropped]` warning on overwrite; empty input → `TPL.NO.CTX`;
  whole-word feedback matching; state-label `input()` survives EOF with
  `auto_state_N` fallback.
- `backend/incremental_csr_lib.py` + `herebot_memory.py`: snapshot sync
  rewrote ~1MB `document_index.json` per document (954× ≈ 85 s/turn, found by
  profiler). Batched to one write: turn ≈ 3 s, snapshot ≈ 0.05 s.
- `components/resoning.py`: `store_phrase` always keeps phrase text;
  `find_similar_phrases` skips legacy hash-only keys (was printing
  `TPL.RECALL.PHRASE phrase='51085...'`).

## Emergent capabilities (early, fragile)

- **Recall-based conversation**: with ~80 definitions it answers novel
  sentences using nearest known token + state prefix, degrading gracefully
  instead of interrogating (chat mode) — incidents 2–6 of the casual session.
- **Self-directed questioning**: unprompted `TPL.ASK.SYN/CAT/CONN.REL`
  hypotheses queue up between turns; answering/confirming them is the
  highest-leverage resume work.
- **State naming**: it detects novel internal states and asks for a human
  label (`greetings` recognized reliably; `bright curiosity` freshly saved).

## Resume checklist

1. Teach `know` (ask, then define in the same sitting).
2. Drain/confirm the SYN queue (`moon`, `sky`, `rain`, `fire`, …) — say `yes`
   only to true synonyms or the mapping stays noisy.
3. Confirm `TPL.ASK.CONN.REL` hypotheses so `is` becomes a real connector and
   `rel='unknown X'` disappears.
4. Chase glue gaps (`have`, `means`, `is`) with `X is a word` closures.
5. Longer chat sittings for trigram/LM growth (LM: 45 words, 60 steps).

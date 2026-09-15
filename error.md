# HereBot Error Trace Log (read-only investigation — nothing edited)

Conventions: `path:line` pins the exact code. Status is FIXED (landed +
tests green), CONTAINED (system handled it, no fix needed), or OPEN
(recorded, awaiting human ruling). Evidence = live log excerpts from
`main.py` sittings.

## E1 — EOF infinite loop in launcher [FIXED]

- Symptom: piped stdin exhausted → 6.4M traceback lines until external kill.
- Trace: `herebot/main.py:105-128` — inner `except Exception` catches the
  `EOFError` from `input()` at line 106 and `continue`s onto dead stdin.
- Root cause: no `EOFError` branch; every exception treated as recoverable.
- Fix: `herebot/main.py:119-124` breaks cleanly on `EOFError`.
- Evidence: probe3 run (EXIT 124) vs all later runs (EXIT 0, `Shutting down.`).

## E2 — Backend snapshot rewrote index per document (~85 s/turn) [FIXED]

- Symptom: single turn took 85.5 s (profiler).
- Trace: `processing.py:149 process_input` → `_sync_backend_turn`
  → `backend/herebot_memory.py:139 sync_turn` → `sync_snapshot` (line ~164)
  → 954× `_sync_one` → `add_runtime_text` → `incremental_csr_lib.py:309
  add_document` → `_save_document_index` (line ~354) — full ~1 MB JSON
  rewrite per document (65M encoder calls in profile).
- Root cause: no batch path; even the `doc_exists` touch rewrote the file.
- Fix: `save=False` param + `save_document_index()` flush;
  `add_documents` saves once; `sync_snapshot` uses the batch API
  (`herebot_memory.py:164-208`).
- Evidence: turn 85.5 s → 3.3 s; snapshot 954 docs → 0.05 s.
  `BackendIntegration_test.py` 11/11 still green.

## E3 — Tokenizer dropped all single-char tokens [FIXED]

- Trace: `processing.py:1296` — `raw` filtered `len(t) >= 2` before the
  vocab-size gate, so `2 + 2` → `[]`, `a`/`I` vanished.
- Fix: keep digits/operators/single letters in `raw`; the `min_len` gate
  below still filters noise past 50 words.

## E4 — Math path unreachable [FIXED]

- Trace: unknown-word branch (`processing.py:858-864`) fired on any
  all-unknown input, so math never reached `mind.py` Priority 0.75
  (`eval` → `TPL.MATH.RESULT`).
- Fix: pure number+operator inputs bypass the gap branch (`is_pure_math`).
- Evidence: `2 + 2` → gap (before); `3 * 4` → `TPL.MATH.RESULT value='12'`
  live (Session 3 probe).

## E5 — Definition attached to the wrong word [FIXED]

- Symptom: `road is a path for cars` stored under pending `vehicle`;
  `fly means to move in air` stored under `black`.
- Trace: handler #8 (`processing.py:549-550`) consumed ANY `is/means`
  sentence for `last_asked_word` without checking membership.
- Fix: `pending_in_text` guard; mismatch falls through to normal processing.
- Follow-up warning: overwrite now logs `[Definition Dropped]`
  (`processing.py:869`).

## E6 — Empty input burned a training cycle [FIXED]

- Trace: `process_input` had no empty guard (`main.py` skipped empties, the
  engine did not) → counter+1, episodic write, trust utterance.
- Fix: early return `TPL.NO.CTX` (`processing.py:186`).

## E7 — State-label `input()` crashed piped sessions [FIXED]

- Trace: `curiosity.py:334` `input("🏷️ Label:")` → raw `EOFError` on dead stdin.
- Fix: `except EOFError` → `auto_state_N` fallback (`curiosity.py:334-338`).
  Interactive behavior unchanged.

## E8 — Phrase recall leaked internal hashes [FIXED]

- Symptom: `TPL.RECALL.PHRASE phrase='5108527926291681122'`.
- Trace: `curiosity.py:606` stores sentences via `store_phrase(..., metadata
  without 'phrase')`; `resoning.py:719` fell back to the hash key; old
  entries predate metadata entirely.
- Fix: `store_phrase` always sets `phrase` (`resoning.py:736`); lookup skips
  unresolvable legacy keys (`resoning.py:722`).
- Evidence: `what is zibber` → `TPL.RECALL.PHRASE phrase='what is zibber'`.

## E9 — Substring feedback classification [FIXED]

- Symptom: `the sun is bright and hot` recorded as *positive trust feedback*
  (`right ⊂ bright`); same class: `no ⊂ know`, `yes ⊂ eyes`.
- Trace: five `any(kw in feedback_lower ...)` sites in `processing.py`
  handlers (§1.5 shadow-diff, §2 response, §3 rule, §4 logic, §6 trust).
- Fix: `_feedback_words` whole-word matcher (`processing.py:23`) used at all
  five sites (lines 255, 282, 340, 376, 451). Word-conflict label matching
  intentionally left substring (needs flexible choice).
- Evidence: same sentence now trains normally; `Curiosity_test.py` 117/117
  (includes whole-word `negative` case).

## E10 — World-model transition dimension crash [OPEN — Level 2, training paused]

- Symptom (deep-idle session): `[ERROR in build_world_model_transitions]`
  `ValueError: shapes (13,) and (11,) not aligned` at
  `components/factor_handler.py:462 find_similar_state` ←
  `processing.py:143 build_world_model_transitions`.
- Trace of the drift (dual factor registries, no shared dimension contract):
  - `concepts/meta_learner.py:18-46` — `factor_names = base + dynamic`;
    `add_factor` (line 52) appends at runtime.
  - `concepts/controlspace.py:19-90` — separate `all_factors = base +
    dynamic`; new factors arrive via a different path (`add_factor`,
    line 92) plus a listener that adds `ctrl_`-prefixed copies into the
    meta-learner (`processing.py` `_on_new_control_factor`).
  - Sleep spawned `residual_1789205282` (idle log) → meta names grew to 13;
    stored state vectors frozen at 11.
  - `processing.py:143` aligns episode vectors to the 13-name space
    (`_align_factor_vector`), then `find_similar_state` runs a RAW
    `np.dot(query, stored)` (`factor_handler.py:462`) → crash.
- Blast radius (traced, not fixed): any caller passing current-dim vectors
  against old states can hit this — `_handle_state_learning`
  (`curiosity.py:261`, fires after 30-interaction cooldown),
  `_generate_artifacts` state check, and every future factor spawn
  re-opens it. The main conversational path was verified working post-sleep
  (Session 2 Interaction 12), so the failure is currently confined to the
  one idle subtask, caught by its own wrapper (`idle.py:1340`).
- Why not Level 1: repair means defining one dimension contract across
  control space ↔ meta-learner ↔ stored states ↔ persistence — non-local by
  the experiment protocol. No fix attempted; needs human design ruling.

## E11 — Generated-logic NameError, correctly contained [CONTAINED]

- Symptom (idle codegen): `logic_1789205287_507` failed shadow test with
  `NameError: name 'false' is not defined`.
- Trace: `traits/code_generator.py` generation → `mark_shadow_result`
  (line 340) records failure; plugin never promotes (stays out of trial).
- Root cause: generator emits Python with lowercase `false` (JS-style
  literal) — a generator-template bug, but the failure is evidence the
  sandbox works. Left for the human to decide (template fix vs prompt fix).

## E12 — Phrase-similarity collapse 1.00 (anomaly, no crash) [OPEN]

- Symptom (idle mining): every definition pair scores sim 1.00/0.98
  (`what is a word` ↔ everything, incl. `zibber means a glowing stone`).
- Trace: `traits/semlearn.py:95,134` — sentence embedding = mean of word
  embeddings; all `X is a Y` definitions share is/a/word → near-identical
  means; `resoning.py:710 find_similar_phrases` (threshold 0.85) then
  proposes everything to everything.
- Consequence (traced downstream): noisy SYN queue (`fish/thing`,
  `moon/thing`…) that consumes one full reply turn each
  (`mind.py` Priority 2). Likely structural, needs human review.

## E13 — Legacy interactive conflict prompts [DORMANT]

- Trace: seven `input()` calls in `processing.py:910-1003`, all guarded by
  `config_manager.get('state_manager.legacy_interactive_conflicts', False)`
  (default False, `config_manager.py:67`); archive path at line 1120.
- No action: unreachable in current config; enabling it in a piped session
  would hang exactly like E1/E7 did.

## E14 — Test discovery finds 0 tests [NOTED]

- Cause: files named `*_test.py`, `unittest discover` defaults to `test*.py`.
  Workaround: `PYTHONPATH=herebot python3 herebot/tests/cor/<Name>_test.py`.
  All 149 green (117 + 11 + 15 + 6). Docs-only, no code impact.

## E15 — WORD.CONFLICT resolution has no lasting effect on probabilities [OPEN]

- Symptom: user answers WORD.CONFLICT with a label (e.g. "bright curiosity"),
  system prints `✅ [Conflict Resolved] 'fire' is now primarily 'bright curiosity'`,
  but `word_state.json` still shows both labels at 0.5/0.5.
- Trace: `processing.py:564-597` handler #7 calls
  `reasoner.word_state.break_bridge(word, label1, label2, keep, ...)`.
  `break_bridge` (`learner.py:390-416`) does:
    1. `add_association(word, keep_label, confidence=1.0)` → calls
       `_normalize_word_labels` → `normalize_probability_field` → normalizes
       all labels to probability = activation/total.
    2. Manually halves rejected label: `rejected['usage'] *= 0.5`.
    3. Calls `_normalize_word_labels` again with `touched_label=reject_label,
       weight=0.0` → `normalize_probability_field` recalculates all probs.
  Root cause: step 3 recalculates activation for ALL labels (not just the
  rejected one). `update_probabilistic_stats` (`probabilistic_labeling.py:52`)
  decays `usage` by `exp(-lambda * delta)` then sets
  `activation = usage * reward_gate`. Since both labels started with equal
  usage (2.1) and the manual halving in step 2 is applied BEFORE step 3's
  decay, the decay can equalize them again. In practice both labels end up
  with identical activation (~1.05) and probability (0.5).
- Consequence: 104 words stuck in permanent 0.5/0.5 conflict between
  'greetings' and 'bright curiosity'. Every sentence containing any of these
  words triggers WORD.CONFLICT, consuming the response pipeline. The
  conflict cascade blocks normal training throughput.
- Also noted: `break_bridge` is called but `conflict_history` only shows
  entries from the PREVIOUS session (Day 1, word='hello'). The Day 2
  'fire' resolution was NOT recorded in conflict_history, suggesting the
  history append (`learner.py:410-416`) may not execute or the state is
  loaded from disk after the append.
- Evidence: live session Sep 15, word='fire', labels='greetings|bright curiosity',
  answer='bright curiosity', post-resolution `word_state.json` still 0.5/0.5.
  104 conflicting words verified via script. Conflict cascade observed in
  8+ consecutive turns blocking natural conversation.

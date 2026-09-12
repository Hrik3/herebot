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

---

# Herebot Training Report (formal experiment — appended, history preserved above)

Method: single live `main.py` process held open across turns (FIFO wire, no
restarts, no scripts feeding data). Every input below was hand-chosen after
reading the actual previous reply. No source/memory/config changes were made
during these sessions. No sleep phase yet (short sittings so far).

## Session 1 (live wire, non-chat mode)

### Interaction 1

**Input to Herebot:**
> hello

**Herebot response:**
> TPL.HYP.REL word='hello' candidate='how' (state recognized: greetings/bright curiosity; backend sync added 30, corpus 1325)

**External model response:**
> (none yet — first contact)

**Observed behavior:**
- Greeting recognized against two learned states; reply is a relation
  hypothesis linking hello→how, not a greeting back.

**Teaching event:**
- None.

**Evidence of retention:** None. **Evidence of generalization:** None.
**Errors or contradictions:** None.
**Follow-up decision:** Probe the hello→how link it just offered: `how are you`.

### Interaction 2

**Input to Herebot:**
> how are you

**Herebot response:**
> Sentence Hypothesis + spoken TPL.ASK.CAT subject='how' object='you'?,
> then TPL.ASK.FB probe (all known words → full pipeline, no gap asked)

**External model response:**
> (withheld — feedback requested next turn)

**Observed behavior:**
- Question words trigger the category engine unprompted; it asks back
  instead of answering. No definition recall attempted in non-chat mode.

**Teaching event:** None.
**Evidence of retention:** None. **Evidence of generalization:** Weak (it
related how/you compositionally, but only as a question).
**Errors or contradictions:** None.
**Follow-up decision:** Its CAT question is apt, and the FB probe is pending
(my next input will be eaten as a rating regardless), so rate honestly.

### Interaction 3

**Input to Herebot:**
> positive

**Herebot response:**
> 📊 [Feedback Recorded] User rated response as: positive

**Observed behavior:**
- Whole-word rating path works (post-fix); reward banked, no shadow link.

**Teaching event:** Positive reinforcement of relevant questioning.
**Evidence of retention:** None. **Evidence of generalization:** None.
**Follow-up decision:** Teach a self-referential word: `herebot learns words`.

### Interaction 4

**Input to Herebot:**
> herebot learns words

**Herebot response:**
> TPL.ASK.DEF target='learns'? (2/3 unknown → knowledge gap, most semantic first)

**Observed behavior:**
- Correct gap behavior; `herebot` not selected (structural scoring
  preferred `learns`). A genuine question to answer.

**Teaching event:** None yet.
**Follow-up decision:** Answer it now while pending (live wire): define learns.

### Interaction 5

**Input to Herebot:**
> learns means to gain knowledge

**Herebot response:**
> 📘 [Definition Learning] Learned about 'learns'; spoken
> TPL.HYP.REL word='learns' candidate='herebot', then TPL.ASK.FB probe

**Observed behavior:**
- In-turn definition attach works (pending present). It spontaneously linked
  learns↔herebot — the exact relation in my sentence, unprompted.

**Teaching event:** `learns` = to gain knowledge (new glue: gain, knowledge).
**Evidence of retention:** Moderate (stored + immediate relational use).
**Follow-up decision:** Rate the apt linkage, then examine recall with new wording.

### Interaction 6

**Input to Herebot:**
> positive

**Herebot response:**
> 📊 [Feedback Recorded] User rated response as: positive

**Follow-up decision:** Paraphrased recall exam: `what is learns`.

### Interaction 7

**Input to Herebot:**
> what is learns

**Herebot response:**
> TPL.STATE.CONTEXT labels='greetings' | TPL.DEF.MAYBE word='travel'
> value='ccar is transport' TPL.ASK.CLARIFY (spontaneous trust utterance;
> trust rise armed a feedback trap for the next turn)

**Observed behavior:**
- Did NOT answer about `learns`. Reply policy chose a spontaneous utterance
  built from a pre-experiment phantom entry (`travel`/`ccar`/`transport` —
  never taught by me; `ccar` looks like an old typo now in long-term memory).
- Recall-via-reply failed; storage was verified at Interaction 5, so this is
  a retrieval-policy bottleneck, not a memory failure. Ambiguity noted:
  cannot yet separate "can't retrieve" from "won't utter".

**Teaching event:** None.
**Evidence of retention:** Moderate (unchanged — storage intact, untested here).
**Evidence of generalization:** None.
**Errors or contradictions:** Phantom vocabulary surfacing as confabulation
  (wrong-word recall presented with TPL.DEF.MAYBE hedging — at least hedged).
**Follow-up decision:** Investigate the phantom: `what is travel`. Note the
  feedback trap will eat this turn.

### Interaction 8

**Input to Herebot:**
> what is travel

**Herebot response:**
> 🌑 [Shadow Attempt] No feedback given … Feedback processed. (turn eaten by
> the trust-feedback trap armed in Interaction 7; no new reply)

**Observed behavior:**
- Confirms the trap mechanic live: any input following a spontaneous
  utterance is consumed as a (non-)rating.

**Follow-up decision:** Ask again now the trap is clear.

### Interaction 9

**Input to Herebot:**
> what is travel

**Herebot response:**
> TPL.HYP.REL word='what' candidate='is', then TPL.ASK.FB probe

**Observed behavior:**
- Still no travel content. Non-chat replies never directly answer content
  questions (policy order: pending hypotheses → curiosity questions → state
  utterances → trust utterances → ACK.LISTEN). Direct answers live in chat
  mode only. Open question: is `travel` reachable via chat recall?

**Teaching event:** None.
**Evidence of retention:** None (travel recall: no evidence either way).
**Evidence of generalization:** None.
**Follow-up decision:** End sitting cleanly (`exit`); next session in chat
  mode to test travel/learns recall where answers are possible.

## Periodic Summary — Session 1

### Vocabulary / Concepts Observed
- `learns` newly stored (`learns means to gain knowledge`); `herebot`,
  `gain`, `knowledge` encountered, undefined. Phantom `travel`/`ccar`/
  `transport` cluster predates the experiment.

### Newly Observed Behaviors
- Live DEF Q&A works when pending is alive (Interaction 5).
- Spontaneous trust utterances arm a one-turn feedback trap (Interactions 7–8).
- HYP.REL linkage of same-sentence words without being asked (Interaction 5).

### Retained Knowledge
- `learns` definition stored in-turn (direct evidence: 📘 line + artifact trace).

### Failed Knowledge
- Recall-via-reply for `learns` and `travel` in non-chat mode (policy, not
  proven memory loss).

### Generalization Evidence
- Weak: hello→how link, how/you category question, learns↔herebot link.

### Contradictions
- None this session.

### Behavioral Changes
- N/A (first formal session).

### Interesting Anomalies
- Phantom `ccar` typo-entry confabulated as an answer with MAYBE hedging.
- `hello` backend sync added 30 docs (snapshot cadence), corpus 1325.

### Questions for Future Testing
1. Chat-mode recall of `learns`, `travel` (direct answers possible there).
2. Delayed recall of `learns` after unrelated interaction.
3. Error detection: `learns means to forget things` — accepted or resisted?
4. Contrast: `is learns like know` (after teaching `know`).

### Confidence
Moderate — storage mechanism verified live; retrieval policy mapped but
recall evidence is one-directional so far (phrase recall from earlier work,
no fresh word-recall yet).

### Failures / protocol notes
- No Level 1/2/3 failures this session; no code or state touched.
- Environmental note: one earlier background launch hung on input (stale
  process-group kill suspected); relaunched via detached pipeline, then clean.

## Session 2 (live wire held open; conversational pace, no bombardment)

Goal: teach a little small-talk so it converses normally; idle windows for sleep.

### Interaction 1 — `good morning` → `TPL.HYP.REL good→morning`. Linked the
greeting unprompted. No question, no probe. Follow-up: enter chat for real answers.

### Interaction 2 — `switch chat` → `TPL.MODE.CHAT.ON?`. Clean toggle.

### Interaction 3 — `how are you feeling today` (today unknown) →
`TPL.DEF.REL word='how' value='how is a word'`. Unknown tolerated in chat;
glue recall again. Teaching gap identified: no feeling words usable. Exited
chat to teach one.

### Interaction 4 — `switch chat` → OFF. Interaction 5 — `fine` →
`TPL.ASK.DEF target='fine'?`. Genuine question, answered live next turn.

### Interaction 6 — `fine means feeling good and calm` → 📘 learned; spoken
`TPL.HYP.REL fine→means` + FB probe. Teaching event recorded.

### Interaction 7 — `ok` → neutral recorded (first neutral rating observed;
neutral path works: no reward swing, no shadow link).

### Interaction 8 — `switch chat` → ON. Interaction 9 — `are you fine` →
`TPL.RECALL.PHRASE phrase='are you fine'` (echoic self-phrase, on-topic;
reads as reflective listening). Interaction 10 — `you are good` →
`TPL.TOPIC topic='are you'` (trigram continuation from episodic memory).
Coherent small-talk arc held across 3 turns. No contradictions offered, none
to correct.

### Interaction 11 — post-first-idle exam `what is fine` →
`TPL.PFX.LOW_COMP | TPL.RECALL.PHRASE phrase='are you fine'`. Fine-associated
phrase recalled after 5 min idle; low-comprehension flag honestly raised on
the novel question shape. Retention: Moderate-Strong.

### Interaction 12 (post-sleep integrity probe) — `are you fine` →
`TPL.RECALL.PHRASE phrase='what is zibber'`, no crash, no traceback. Main
pipeline verified intact after the failed idle subtask (see failure log).
Association is odd (fine→zibber) — recorded as ambiguous, possibly phrase
graph cross-talk after consolidation.

## Sleep / Consolidation Period (Session 2)

**Training interactions before sleep:** 10. **Idle windows:** 5 min, 5 min,
then 7 min (long-idle line at ~6.6 min crossed on the third).
**External intervention during sleep:** None (watched only).

**Short-idle behavior:** `TPL.IDLE.STATS` every ~60 s (16 total):
words=127, concepts 7→8, reward 419.25→420.08, states=2, persistently
curious about token `'3'` (math residue) with `action=ask`.

**Long-idle (deep) behavior — full consolidation observed:**
- Dreaming (5 synthetic sentences), rehearsal (20 episodes), introspection
  (state groups refreshed, 1 active), new curiosity factor spawned
  (`residual_1789205282`), meta-concepts formed (`meta_hi_greetings` from 120
  words), policy distilled, weak drive `state_relevance` removed,
  counterfactual updates over 20 episodes, 3-turn dialogue simulation
  (IG 1.7), LM trained (20 sentences, vocab 67, steps 80).
- Dreamed behavioral rules include `conversational_pattern 'what is flint'`
  (0.70) and `identity_alias 'herebot'` (0.50) — real session content
  promoted during sleep. Retention mechanism confirmed behaviorally.
- **Sleep also pollutes:** self-talk learned junk into memory
  (`animal is friend...`, `for is word...`); pre-experiment residue
  (`rose`, `car is travel`, `ccar`) resurfaced in mining results.
- **Similarity collapse anomaly:** every phrase pair scores sim 1.00/0.98
  (`what is a word` ↔ everything) — averaged embeddings cannot distinguish
  definitions. This is the likely root of the noisy SYN queue. Recorded for
  human review (no fix attempted mid-experiment).
- **Contained codegen failure:** generated `logic_1789205287_507` failed its
  shadow test (`NameError: name 'false' is not defined`) and was correctly
  rejected — containment working as designed, not a learning failure.
- Knowledge graph recentralized on glue words (is:184, means:70, word:56) —
  definition-heavy training skews graph centrality to function words.

**Post-sleep observations:** recall intact (Interaction 12), reply policy
unchanged, no forgetting detected in probed items (`fine`, greeting behavior).

## Failure log — Level 2 (protocol classification)

**Observed failure:** `[ERROR in build_world_model_transitions]` during the
deep idle session:
`ValueError: shapes (13,) and (11,) not aligned` in
`factor_handler.py:462 find_similar_state`, called from
`processing.py:143 build_world_model_transitions`.

**Evidence:** full traceback in the live log; idle-task wrapper caught it and
the remaining ~15 idle tasks completed; main conversational path verified
working after sleep (Interaction 12, no traceback).

**Hypothesis (speculation, labeled):** the sleep session spawned a new
dynamic curiosity factor (13 dims now vs 11-dim stored state vectors), and the
world-model transition builder passes unaligned episode vectors into a raw
`np.dot`. The normal path appears to align or avoid the mismatch, the idle
path does not.

**Why Level 2, not 1:** the fix touches the factor/state dimension contract
shared by the control space, state manager, and persistence — non-local by
the protocol's definition — and I will not redesign dimension handling
mid-experiment.

**Action per protocol:** no fix attempted; state preserved; failing subtask
identified (single idle task, contained by its own error wrapper);
**TRAINING PAUSED — HUMAN INVESTIGATION REQUIRED** before Session 3.

## Periodic Summary — Session 2

### Vocabulary / Concepts Observed
`fine` taught and used conversationally the same sitting (teach → recall →
topical use). `calm`, `gain`, `knowledge`, `herebot` encountered, undefined.

### Newly Observed Behaviors
- Echoic phrase recall as conversational style (`are you fine` → itself).
- Neutral ratings recorded without side effects.
- Honest low-comprehension flagging on novel shapes post-idle.

### Retained Knowledge
- `fine` (recall + use), greeting behavior, `flint`/`zibber` untouched but
  present (zibber surfaced in post-sleep recall).

### Failed Knowledge
- None new; glue-recall crutch persists (`how is a word` style answers).

### Generalization Evidence
- Weak-Moderate: fine→feeling-thread reuse across three turns; no novel
  composition tested this session (deliberately, per no-bombardment rule).

### Contradictions
- None offered by either side; nothing to correct.

### Behavioral Changes
- Post-sleep: none detected in reply policy or recall quality.

### Interesting Anomalies
- Similarity-1.00 collapse; sleep-learned junk sentences; `curious='3'`
  fixation across all 16 idle reports.

### Questions for Future Testing (pending pause ruling)
1. Chat recall of `travel` (still open from Session 1).
2. Error detection: `fine means feeling sick`.
3. Contrast: `is fine like happy` (both defined).

### Confidence
Moderate-High for Session 2 claims (all from direct replies); the pause
ruling concerns one idle subtask only — the conversational loop is verified
healthy.

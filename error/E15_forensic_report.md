# E15 — READ-ONLY FORENSIC INVESTIGATION REPORT
## WORD.CONFLICT Resolution Persistence & Conflict Cascade Analysis

---

## E15 Status

**FULLY TRACED**

The forensic investigation has mapped the complete causal execution path, traced the exact data flow and variable values, disproven the initial hypothesis in `error.md`, identified the primary persistence gap, discovered the upstream generator responsible for the 104 conflicting words, and proved that the in-memory mathematical resolution in `break_bridge()` is functioning correctly but was masked by deferred disk serialization and an unmanaged ambiguity cascade.

---

## Execution Trace

The complete call and data flow from conflict presentation through persistence:

```
[Interaction Turn: "Fire burns."]
  │
  ├─► processing.py:1278-1298 (Section 8: Conflict Detection)
  │     │
  │     ├─► token = 'fire'
  │     ├─► reasoner.word_state.detect_conflicts('fire')  [components/learner.py:375]
  │     │     ├─► get_label_probabilities('fire')          [components/learner.py:317]
  │     │     │     └─► _normalize_word_labels('fire')     [components/learner.py:295]
  │     │     │           └─► normalize_probability_field() [components/probabilistic_labeling.py:89]
  │     │     │                 Returns: {'greetings': 0.500, 'bright curiosity': 0.500}
  │     │     ├─► abs(0.500 - 0.500) = 0.000 < conflict_resolution_threshold (0.300)
  │     │     └─► Returns: [('greetings', 'bright curiosity')]
  │     │
  │     ├─► Prints: " [Word Conflict] TPL.ASK.WORD.CONFLICT token='fire' labels='greetings|bright curiosity'?"
  │     ├─► self.expecting_word_conflict = True
  │     ├─► self._pending_conflict = ('fire', 'greetings', 'bright curiosity')
  │     ├─► self._record_turn_telemetry("word_conflict_detect")
  │     └─► RETURN EARLY (line 1288)  ──► [self._save_state() at line 1339 NOT REACHED]
  │
[User Input Turn: "bright curiosity"]
  │
  ├─► processing.py:564-597 (Handler #7: Word Conflict Resolution)
  │     │
  │     ├─► Guard: not self.chat_mode_active and self.expecting_word_conflict == True
  │     ├─► Matches: 'bright curiosity' in feedback_lower
  │     ├─► keep = 'bright curiosity', reject = 'greetings'
  │     │
  │     ├─► reasoner.word_state.break_bridge()  [components/learner.py:390]
  │     │     │
  │     │     ├─► 1. add_association('fire', 'bright curiosity', confidence=1.0, source='conflict_keep')
  │     │     │        └─► _normalize_word_labels(touched_label='bright curiosity', weight=1.0)
  │     │     │              └─► update_probabilistic_stats('bright curiosity'):
  │     │     │                    usage += 1.0, reward_trace += 1.0 * tanh(reward)
  │     │     │
  │     │     ├─► 2. rejected = word_label_stats['fire']['greetings']
  │     │     │        rejected['usage'] *= 0.5        (Usage halved from ~1.88 to ~0.94)
  │     │     │        rejected['reward_trace'] -= abs(reward) (Reward driven to -2.53)
  │     │     │
  │     │     ├─► 3. _normalize_word_labels(touched_label='greetings', weight=0.0, reward=-abs(reward))
  │     │     │        └─► normalize_probability_field():
  │     │     │              'bright curiosity': activation = 3.765, probability = 0.99843
  │     │     │              'greetings':        activation = 0.0059, probability = 0.00157
  │     │     │
  │     │     └─► 4. conflict_history.append({
  │     │              'word': 'fire',
  │     │              'labels': ['greetings', 'bright curiosity'],
  │     │              'kept': 'bright curiosity',
  │     │              'interaction_index': 22,
  │     │              'timestamp': 1789468541.065
  │     │            })
  │     │
  │     ├─► Prints: "✅ [Conflict Resolved] 'fire' is now primarily 'bright curiosity'."
  │     ├─► del self._pending_conflict
  │     ├─► self.expecting_word_conflict = False
  │     ├─► self._record_turn_telemetry("word_conflict_resolve")
  │     └─► RETURN EARLY (line 587)  ──► [self._save_state() at line 1339 NOT REACHED]
  │
  ▼
[PERSISTENCE GAP OCCURS HERE]
  - IN-MEMORY STATE: 'fire' resolved to 0.9984 / 0.0016; conflict_history has 2 items.
  - DISK STATE (curiosity_state/word_state.json): UNMODIFIED (retains 0.500 / 0.500 snapshot).
  │
[Later Interaction Turn: "The sun is bright today."]
  │
  ├─► Runs through full pipeline (no early return)
  ├─► Reaches processing.py:1339: self._save_state()
  ├─► persistence.py:324: self.reasoner.save()
  ├─► resoning.py:820: self.word_state.save("curiosity_state/word_state.json")
  └─► DISK UPDATED: 'fire' written as 0.99843 / 0.00157; conflict_history written with 2 items.
```

---

## Actual Value Transition

Direct forensic measurements of `fire`'s exact values across its lifecycle:

### Stage A: Immediately Before Resolution (Pre-Turn Baseline)
*Source: Git commit `1bb315fd452de9f1647f4580e501719b3a01112f~1:herebot/curiosity_state/word_state.json`*
```json
{
  "greetings": {
    "usage": 2.100,
    "reward_trace": 0.000,
    "activation": 1.050,
    "probability": 0.500,
    "tier": 2,
    "phase": "active"
  },
  "bright curiosity": {
    "usage": 2.100,
    "reward_trace": 0.000,
    "activation": 1.050,
    "probability": 0.500,
    "tier": 2,
    "phase": "active"
  }
}
```
- `word_to_labels['fire']`: `{'greetings': 0.500, 'bright curiosity': 0.500}`
- `abs(0.500 - 0.500) = 0.000 < 0.300` → Conflict detected.

### Stage B: Inside `break_bridge()` (Step 1: Kept Label Boost)
`add_association('fire', 'bright curiosity', confidence=1.0)`:
- `bright curiosity` touched with `weight=1.0`, `reward=abs(reward)`:
  - `usage` increased: `2.100 + 1.0 = 3.100` (prior to interaction decay alignment)
  - `reward_trace` increased positively.
- `greetings` touched with `weight=0.0`:
  - `usage` decayed by exponential interaction factor.

### Stage C: Inside `break_bridge()` (Step 2 & 3: Rejected Label Suppression & Renormalization)
- `greetings['usage']` manually halved: `usage *= 0.5` → `0.9436`
- `greetings['reward_trace']` penalized: `reward_trace -= abs(reward)` → `-2.5327`
- `_normalize_word_labels(touched_label='greetings', weight=0.0, reward=-abs(reward))`:
  - For `greetings`:
    `reward_gate = (1.0 + tanh(-2.5327)) / 2.0 = (1.0 - 0.9874) / 2.0 = 0.00627`
    `activation = usage * reward_gate = 0.9436 * 0.00627 = 0.005918`
  - For `bright curiosity`:
    `reward_gate = (1.0 + tanh(+0.9875)) / 2.0 = (1.0 + 0.7562) / 2.0 = 0.8781`
    `activation = usage * reward_gate = 4.2872 * 0.8781 = 3.7647`
  - Total activation: `3.7647 + 0.0059 = 3.7706`
  - `probability('bright curiosity') = 3.76473 / 3.77065 = 0.998430` (Tier 4)
  - `probability('greetings') = 0.00592 / 3.77065 = 0.001570` (Tier 0)

### Stage D: In Memory Immediately After `break_bridge()` Finishes
- `word_to_labels['fire']`: `{'greetings': 0.00157, 'bright curiosity': 0.99843}`
- `conflict_history[-1]`: `{'word': 'fire', 'kept': 'bright curiosity', ...}`
- `detect_conflicts('fire')`: `[]` (Conflict resolved!)

### Stage E: On Disk Immediately After Handler Returns (Turn T)
- `curiosity_state/word_state.json`: **STALE (Unchanged from Stage A)**
- Reason: Handler #7 early-returned at `processing.py:587`; line 1339 was skipped.

### Stage F: On Disk After Next Full Interaction Runs Line 1339 (Turn T+N)
*Source: Live `curiosity_state/word_state.json` and Git commit `1bb315fd452de9f1647f4580e501719b3a01112f`*
- `word_to_labels['fire']`: `{'greetings': 0.00156958, 'bright curiosity': 0.9984304}`
- `word_label_stats['fire']['greetings']['probability']`: `0.00156958`
- `word_label_stats['fire']['bright curiosity']['probability']`: `0.9984304`
- `conflict_history`: Length 2 (`hello` and `fire`).

---

## Primary Root Cause

The suspected root cause previously hypothesized in `error.md`—that `normalize_probability_field()` mathematically equalizes the labels back to 0.5/0.5—is **DISPROVEN**. The normalization mathematics in `break_bridge()` correctly and permanently polarize the probabilities to `0.9984` vs `0.0016`.

The actual failure is composed of two primary mechanisms:

1. **The Waiting-State Persistence Gap (Observation Artifact & Crash Vulnerability)**:
   In `herebot/core/cuengine/processing.py:564-597`, Handler #7 (`expecting_word_conflict`) executes `break_bridge()` in RAM, but returns early at line 587 without invoking `self._save_state()` or `self.reasoner.word_state.save()`. 
   Consequently, inspecting the filesystem immediately after `✅ [Conflict Resolved]` reads the pre-turn disk snapshot, creating the empirical appearance that the resolution vanished or failed to take effect. If HereBot terminates before a subsequent turn reaches line 1339, all in-memory resolutions are permanently lost.

2. **The 104-Word Upstream Conflict Factory**:
   Resolving `fire` resolved *only* `fire`. Users experienced conflicts on ~60% of subsequent turns because an upstream background process had mass-manufactured identical `greetings = 0.5 / bright curiosity = 0.5` labels across 104 *other* vocabulary tokens. The observer perceived that conflict resolution "did not work" because the system immediately demanded clarification for `is`, `what`, `where`, `are`, `means`, or `water` on the very next input.

---

## Secondary Causes

1. **The Decline Deadlock (`🌑 No clear label chosen`)**:
   In `processing.py:579-581`, when an interrogator responds with anything other than the exact label names (e.g., "Neither fits", "None", or "I don't know"), the handler prints `Conflict remains` and clears `_pending_conflict` without modifying the underlying 0.5/0.5 probabilities, without applying negative evidence, and without suppressing future queries. The word remains in conflict, ensuring that the next turn containing that token immediately halts again.

2. **Sleep / Idle Indiscriminate Cross-Modal Alignment**:
   In `herebot/core/cuengine/idle.py:531-540`:
   ```python
   if hasattr(self.reasoner, 'word_state') and getattr(self, 'current_state_labels', None):
       for ep in self.episodic_memory.episodes[-100:]:
           tokens = self._tokenize(ep.get('user_input', ''))
           for t in tokens:
               for lbl in self.current_state_labels:
                   self.reasoner.word_state.add_association(
                       t, lbl, confidence=0.7 * throttle, source="cross_modal_alignment",
                       interaction_index=getattr(self, 'interaction_counter', 0)
                   )
   ```
   Because HereBot currently only recognizes two active interaction states (`greetings` and `bright curiosity`), this function iterates through every episode in memory and adds **both** state labels with the **identical** confidence (`0.7 * throttle`) to every single word in the conversation. This continually re-equalizes label usage and serves as an ongoing ambiguity generator.

3. **Semantic Similarity Propagation Amplification**:
   In `herebot/core/cuengine/idle.py:1097-1156`, `propagate_word_state_associations()` copies labels between words with semantic embedding similarity > 0.8. When words like `say` or `is` acquire the 0.5/0.5 state, this function spreads those identical labels to neighboring vocabulary words, multiplying the contaminated population.

4. **Early-Return Exit Path in All Waiting-State Handlers**:
   All 9 waiting-state handlers in `processing.py` (chat toggle, shadow diff, response feedback, rule confirmation, logic feedback, pattern label, trust feedback, word conflict, procedure step) return early and bypass line 1339. None of them invoke persistence.

---

## Conflict Trigger Mechanism

The exact conditions governing `WORD.CONFLICT` generation:

1. **Trigger Code Location**: `herebot/core/cuengine/processing.py:1278-1298`.
2. **Gating Conditions**:
   - `not self.chat_mode_active`
   - `not self.expecting_response_feedback`
   - `not self.expecting_feedback`
   - `not self.expecting_rule_confirmation`
   - `not self.expecting_logic_feedback`
   - `not getattr(self, 'expecting_word_conflict', False)`
3. **Detection Criterion (`components/learner.py:375-388`)**:
   - Token is evaluated against `word_to_labels`.
   - Probabilities are extracted via `get_label_probabilities(token)`.
   - Every pair of active labels `(l1, c1), (l2, c2)` is compared.
   - A conflict is declared if:
     $$\lvert c_1 - c_2 \rvert < \text{conflict\_resolution\_threshold} \quad (0.300)$$
4. **Why 0.5 / 0.5 Guaranteed a Halt**:
   $\lvert 0.500 - 0.500 \rvert = 0.000 < 0.300$.
   The moment an input sentence contains any token where two active labels have probabilities within 0.30 of each other, the pipeline aborts normal reasoning, displays `TPL.ASK.WORD.CONFLICT`, sets `expecting_word_conflict = True`, and returns early.
5. **Why Resolved Words Stop Triggering**:
   After `break_bridge()`, `fire` had `greetings = 0.0016` and `bright curiosity = 0.9984`.
   $\lvert 0.9984 - 0.0016 \rvert = 0.9968 \ge 0.300$.
   `detect_conflicts('fire')` returns `[]`. It will never prompt again unless background idle alignment re-contaminates it.

---

## `conflict_history` Trace

Tracing `conflict_history` across its full lifecycle:

| Operation | Method / Location | Implementation Detail |
|---|---|---|
| **Creation** | `WordStateAssociator.__init__()`<br>`components/learner.py:279` | `self.conflict_history = []` |
| **Loading** | `WordStateAssociator.load()`<br>`components/learner.py:438` | `self.conflict_history = data.get('conflict_history', [])` |
| **Appending** | `WordStateAssociator.break_bridge()`<br>`components/learner.py:410-416` | Appends `{'word': word, 'labels': [...], 'kept': keep_label, 'interaction_index': ..., 'timestamp': ...}` |
| **Saving** | `WordStateAssociator.save()`<br>`components/learner.py:424` | Serialized directly to JSON as root key `"conflict_history"` |
| **File Path** | `curiosity_state/word_state.json` | Saved via `Reasoner.save()` at `components/resoning.py:820` |

### Why Day 2 `fire` Appeared Missing
- **Direct Cause**: Category **B** (The append happened in memory, but was not persisted to disk on that turn).
- When `fire` was resolved, `self.conflict_history.append()` executed in RAM. `self.conflict_history` held 2 entries.
- Handler #7 exited early without invoking `_save_state()`.
- The investigator inspected `curiosity_state/word_state.json` on disk immediately after the turn, reading the pre-resolution file which contained only 1 entry (`hello`).
- When interaction `"The sun is bright today."` subsequently executed line 1339, `_save_state()` flushed memory to disk.
- Inspection of `word_state.json` on disk now confirms `conflict_history` contains both `hello` and `fire`.

---

## Duplicate State / Source-of-Truth Analysis

Archaeology of word-label representations across the repository:

| Representation | In-Memory Owner | File Path | Read Locations | Write Locations | Sync Mechanism |
|---|---|---|---|---|---|
| `word_to_labels` | `WordStateAssociator`<br>(`components/learner.py`) | `curiosity_state/word_state.json` | `detect_conflicts()`, `get_labels()`, `_ensure_word_stats()` | `_normalize_word_labels()`, `add_association()` | Single instance owned by `Reasoner.word_state`. |
| `word_label_stats` | `WordStateAssociator`<br>(`components/learner.py`) | `curiosity_state/word_state.json` | `_normalize_word_labels()`, `break_bridge()` | `update_probabilistic_stats()`, `normalize_probability_field()` | Primary mathematical truth for `usage`, `reward_trace`, and `activation`. |
| `label_to_words` | `WordStateAssociator`<br>(`components/learner.py`) | `curiosity_state/word_state.json` | `get_words()` | `_normalize_word_labels()`, `add_association()` | Inverse index; synchronized inside `_normalize_word_labels()`. |
| `conflict_history` | `WordStateAssociator`<br>(`components/learner.py`) | `curiosity_state/word_state.json` | External telemetry / reporting | `break_bridge()` (line 410) | Single in-memory list serialized to JSON. |
| `knowledge_graph` | `KnowledgeGraph`<br>(`components/learner.py`) | `curiosity_state/knowledge_graph.json` | Reasoner relation queries | `curiosity.py:574-585` | Graph edges connect `word` nodes to `state` nodes independently. |
| `state_metadata` | `StateManager`<br>(`components/factor_handler.py`) | `curiosity_state/state_metadata.json` | `find_similar_state()`, `_active_state_labels()` | `save_state()`, `update_state()` | Tracks states themselves (not individual words). |

**Source of Truth Verdict**:
There is **NO competing duplicate representation** for word-state associations. `WordStateAssociator` (`reasoner.word_state`) is the sole authority for word labels. The conflict detector reads the exact same `word_label_stats` and `word_to_labels` that `break_bridge()` writes.

---

## Persistence Analysis

1. **Persistence Boundary Type**: **Deferred / Conditional End-of-Turn**.
2. **Where `word_state.json` is Saved**:
   - `herebot/components/resoning.py:820`: `self.word_state.save(os.path.join(STATE_DIR, 'word_state.json'))` inside `Reasoner.save()`.
   - `herebot/core/cuengine/persistence.py:324`: `self.reasoner.save()` inside `CuriosityEngine._save_state()`.
3. **Execution Gating**:
   - Line 1339 of `herebot/core/cuengine/processing.py` is the only point in normal input processing where `_save_state()` is called.
   - Any turn that terminates via `return` in lines 233, 293, 321, 380, 416, 472, 494, 548, 587, 677, 881, 903, 941, or 1288 **bypasses line 1339 entirely**.
   - Handler #7 (`expecting_word_conflict`) terminates at line 587. It does not persist.
4. **Process Exit Vulnerability**:
   - In `herebot/main.py:136-140` (`finally:` block), only `engine.tracker.save_data()` and `engine.tracker.plot_all()` are called. `engine._save_state()` is **omitted**.
   - If HereBot is interrupted (Ctrl+C) or terminates while unpersisted waiting-state modifications exist in RAM, those modifications are lost.

---

## Git History

1. **`commit 6596ff5b8e` (Mar 24, 2026) — `rf14`**:
   - Initial introduction of `WordStateAssociator` and `break_bridge()`.
   - Original `break_bridge` was a scalar subtraction: `self.word_to_labels[word][label] -= 0.5`.
2. **`commit b698f42f0e` (May 19, 2026) — `28938`**:
   - Introduced `probabilistic_labeling.py`, `_normalize_word_labels()`, and `normalize_probability_field()`.
   - Rewrote `break_bridge()` to use `word_label_stats`, probabilistic halving of `usage`, negative `reward_trace`, and `self.conflict_history.append()`.
   - Added unit test `test_break_bridge_demotes_rejected_label()` in `tests/cor/ProbabilisticLabeling_test.py:76-86`.
3. **`commit 4ec7710` vs `commit f5e06ea` (Sep 12, 2026)**:
   - In `4ec7710`: Total words in `word_state.json` = 37. Zero words had `greetings=0.5 / bright curiosity=0.5`.
   - In `f5e06ea` (following Day 1 sleep cycle): Total words = 110. Exactly **105 words** were written with `greetings=0.500 / bright curiosity=0.500`.
   - Provenance: `cross_modal_alignment` ran during sleep and blanketed all vocabulary with the two known states.
4. **`commit 1bb315fd45` (Sep 15, 2026) — `error:E15`**:
   - Demonstrates that `fire` **was successfully persisted** to disk as `bright curiosity: 0.99843` and `conflict_history` was saved with 2 entries as soon as a non-waiting interaction reached line 1339.

---

## Minimal Reproduction

### Evidence Directly Obtained from Live System

1. **Conflict Presentation (Observed Live at Log Line 853)**:
   ```
   [Word Conflict] TPL.ASK.WORD.CONFLICT token='fire' labels='greetings|bright curiosity'?
   ```
2. **Conflict Resolution (Observed Live at Log Line 862)**:
   Input: `"bright curiosity"`
   ```
   ✅ [Conflict Resolved] 'fire' is now primarily 'bright curiosity'.
   ```
3. **Immediate Post-Turn File Check**:
   Executed immediately after line 862:
   ```python
   # curiosity_state/word_state.json on disk:
   fire: {'greetings': 0.5, 'bright curiosity': 0.5}
   conflict_history: len = 1 (only 'hello')
   ```
4. **Delayed Post-Turn File Check (After Turn "The sun is bright today." reached line 1339)**:
   ```python
   # curiosity_state/word_state.json on disk:
   fire: {'greetings': 0.00156958, 'bright curiosity': 0.9984304}
   conflict_history: len = 2 (includes 'fire')
   ```
5. **Conflict Re-Evaluation Test**:
   ```python
   wsa.detect_conflicts('fire')  # Returns: []
   ```
   `fire` never prompted again in the live session.

---

## Confidence Classification

| Finding / Conclusion | Confidence Tier | Basis |
|---|---|---|
| In-memory `break_bridge()` math works correctly and drives probabilities to 0.998 / 0.002 | **DIRECTLY OBSERVED** | Empirically verified from in-memory dict and committed JSON |
| Waiting-state handler #7 omits `_save_state()`, causing deferred persistence | **STATICALLY PROVEN** | Exact line inspection of `processing.py:564-597` |
| 104 words acquired 0.5/0.5 from `idle.py:531-540` `cross_modal_alignment` | **STATICALLY PROVEN** | Git archaeology between `4ec7710` and `f5e06ea` and math alignment ($N \times 0.7$) |
| `conflict_history` is appended in memory but was unpersisted at time of initial check | **DIRECTLY OBSERVED** | Confirmed by commit `1bb315f` containing the Day 2 `fire` entry |
| Declining with "Neither fits" leaves the 0.5/0.5 conflict unresolved, causing loops | **STATICALLY PROVEN** | Exact line inspection of `processing.py:579-581` |
| Idle `cross_modal_alignment` re-contaminates resolved words over repeated sleep cycles | **STRONGLY INFERRED** | Observed `greetings` usage rise to 0.9436 on `fire` after post-resolution idle |

---

## Repair Requirements

*(Informational only — no changes have been or will be implemented)*

To cleanly resolve E15 in a future maintenance phase while preserving architectural invariants:

1. **Immediate Waiting-State Persistence**:
   - Handlers in `processing.py` that mutate state (specifically Handler #7 for `break_bridge` and Handler #3 for `behavioral_rules`) must invoke `self.reasoner.word_state.save(...)` or `self._save_state()` prior to returning.
   - `herebot/main.py` should invoke `engine._save_state()` in its `finally:` block to guard against unpersisted state on exit.

2. **Dampen or Gate `cross_modal_alignment` When Interaction State Diversity is Low**:
   - `core/cuengine/idle.py:531-540` should not add associations for all active states simultaneously to the same token, or should scale confidence inversely with state count. When `current_state_labels` contains only generic/default states (`greetings`, `bright curiosity`), blanket cross-modal association should be gated off to prevent mass-manufacturing ambiguous 0.5/0.5 priors across the entire vocabulary.

3. **Explicit Handling for Unresolved Conflict Responses**:
   - When an interrogator indicates that neither label fits ("Neither fits", "none", "neither"), Handler #7 in `processing.py:579` should not merely leave the conflict intact. It should either:
     - Demote both labels' confidences below threshold,
     - Mark the conflict as dismissed/dormant in `conflict_history`, or
     - Record negative feedback for both candidate state links so the same token does not trigger an identical halt on the next turn.

4. **Invariants That Must Be Preserved**:
   - Probabilistic normalization invariants ($\sum p_i = 1.0$) in `normalize_probability_field()` must remain untouched.
   - Metabolic decay in `update_probabilistic_stats()` must remain untouched.
   - The conflict threshold contract ($\lvert c_1 - c_2 \rvert < 0.3$) must remain intact.

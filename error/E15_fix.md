E15 Persistence & Learning-Integrity Repair: Final Validation Report
The repair has been implemented according to the final approved specification without restarting or interfering with the live HereBot process (PID 9737, active uptime: >2h 40m).
1. Summary of Inspection Findings (A–D)
- Inspection A (_save_state): Defined at herebot/core/cuengine/persistence.py:311–379. Throttled by _min_save_interval = 20 seconds, which silently drops rapid back-to-back saves. Writes 26 pickled/JSON subsystems. Identified save_word_state_immediately() as the safest unthrottled targeted persistence method for word-state changes.
- Inspection B (idle_reasoner.* config): Found active_config.json and components/config_manager.py. Confirmed enable_semantic_self_talk was previously overloaded to gate propagate_word_state_associations. Added dedicated keys: idle_reasoner.enable_word_state_propagation and idle_reasoner.candidate_ledger.
- Inspection C (Global Workspace / Broadcast): ParameterRevision in traits/param_revision.py defines global_broadcast and update(), but self.consciousness.update() is never called anywhere in the active codebase. No runtime winner-take-all signal exists. As mandated, global_broadcast_state is set to None, was_broadcast_winner is False (metadata-only), and zero broadcast weight multiplier is applied.
- Inspection D (Episode IDs): Confirmed episodes lacked an id field. All episode creation funnels strictly through EpisodicMemory.add_episode(). Assigned ep_{uuid.uuid4().hex[:12]} centrally at creation time. Historical episodes loaded from disk receive an in-memory-only deterministic ep_legacy_<filename> identifier; legacy JSON files on disk are completely preserved without rewrites.
2. Files Changed
 1. herebot/components/episodic_mem.py:
- Central uuid-based stable ID assignment (ep_{uuid.uuid4().hex[:12]}) in add_episode().
- In-memory deterministic fallback for legacy episodes (ep_legacy_<filename>).
- Added active_states_at_time, provenance, and global_broadcast_state fields.
 2. herebot/components/candidate_ledger.py (New File):
- Contains mandatory docstring invariants 1, 2, and 3.
- CandidateAssociation dataclass (retaining token, label, episode_id, timestamp, provenance, state_diversity, pmi, raw_weight, was_broadcast_winner, cycles_survived).
- CandidateLedger with:
- Episode-level single-sample-space PMI computation (_build_stats, generate).
- Idempotent replay accounting via processed_episode_ids set.
- Soft diversity scaling (diversity_target).
- Provenance weighting (OBSERVED=1.0, DREAMED=0.1, COUNTERFACTUAL=0.0, etc.).
- Hard volume-independent grounded-evidence gate in consolidate() (min_grounded_distinct_episodes >= 2, min_grounded_evidence_fraction >= 0.15).
- Distinct episodes, margin, and belief-change simulation (_simulate_kl) gates.
- save() and load() with atomic temporary-file replacement.
 3. herebot/components_global.py:
- Exported candidate_ledger globally.
 4. herebot/components/learner.py:
- Added conflict_cooldown: Dict[str, int] in WordStateAssociator.__init__.
- Added mark_unresolved(word, labels, interaction_index, cooldown_turns).
- Updated detect_conflicts(word, interaction_index) to respect conflict_cooldown.
- Updated save() and load() to persist conflict_cooldown.
 5. herebot/core/cuengine/processing.py:
- Added save_word_state_immediately() on HeartProcessingMixin.
- Rewrote Handler #7: explicit NEITHER_PHRASES branch (mark_unresolved + 20-turn cooldown), unmatched branch (5-turn cooldown), and save_word_state_immediately() before return.
- Updated line ~1288 detect_conflicts call to pass interaction_index=self.interaction_counter.
- Added active_states_at_time and provenance="OBSERVED" to episodic_memory.add_episode().
- Added self._save_state() before early returns across waiting-state handlers.
 6. herebot/core/cuengine/idle.py:
- world_model_dreaming(): sets active_states_at_time and provenance="DREAMED" on synthetic episodes.
- cross_modal_alignment(): refactored to stage candidates in CandidateLedger via generate(); removed direct word_state.add_association() loop.
- propagate_word_state_associations(): gated by dedicated idle_reasoner.enable_word_state_propagation; stages candidates with provenance='INFERRED' and strict similarity threshold (> 0.85); passes interaction_index to detect_conflicts.
- Added consolidate_candidates() and wired it into perform_all_idle_tasks immediately after propagation.
 7. herebot/core/cuengine/persistence.py:
- Added self.candidate_ledger instantiation, load() from curiosity_state/candidate_associations.json, and save() in _save_state().
 8. herebot/main.py:
- Added engine._save_state() in finally: block before tracker save.
 9. herebot/curiosity_state/active_config.json & herebot/components/config_manager.py:
- Added enable_word_state_propagation: true and the candidate_ledger config block.
10. herebot/tests/cor/CandidateLedger_test.py (New Test Suite):
- 8 isolated unit and adversarial proof tests.
11. plan.md:
- Updated with E15 persistence & candidate-ledger repair plan.
3. Exact Architectural & Data-Flow Changes
Old Contaminating Flow:
Episode Ingestion ──► Idle Cross-Modal Alignment ──► Direct write into word_label_stats (both labels 0.5/0.5)
                                                ──► 105 words corrupted into perpetual conflict

New Evidence-Gated Flow:
Episode Ingestion (with stable ID + provenance)
       │
       ▼
Idle Cycle:
  1. cross_modal_alignment()
       │  (Computes episode-level PMI; skips already-processed episode IDs)
       ▼
     candidate_ledger.generate() ──► Staged in candidate_ledger.candidates (NOT in word_state)
       │
  2. propagate_word_state_associations() (Sim > 0.85, provenance='INFERRED')
       ▼
     Staged in candidate_ledger.candidates (NOT in word_state)
       │
  3. consolidate_candidates()
       │
       ├─► Gate 1: Hard Grounded Prerequisite (grounded_distinct_eps >= 2, grounded_fraction >= 0.15)
       │           [DREAMED-only / COUNTERFACTUAL-only BLOCKED HERE]
       ├─► Gate 2: Total Distinct Episodes (>= 3)
       ├─► Gate 3: Winner/Runner-up Competitive Margin (>= 0.25)
       ├─► Gate 4: Simulated Belief Change (KL Divergence >= 0.05)
       │
       ▼ (All gates cleared)
     word_state.add_association(confidence=evidence, source='sleep_consolidated_evidence_gated')
       │
       ▼
     save_word_state_immediately()  ──► Immediate unthrottled flush of word_state.json
4. New Configuration Keys
Under idle_reasoner:
"enable_word_state_propagation": true,
"candidate_ledger": {
  "diversity_target": 3,
  "candidate_min_pmi": 0.15,
  "min_distinct_episodes": 3,
  "min_commit_margin": 0.25,
  "min_kl_surprise": 0.05,
  "min_grounded_distinct_episodes": 2,
  "min_grounded_evidence_fraction": 0.15,
  "candidate_decay": 0.9,
  "candidate_prune_floor": 0.001,
  "candidate_max_age_cycles": 30
}
5. New Persistence Mechanisms
- curiosity_state/candidate_associations.json: Persisted by CandidateLedger.save() and loaded by CandidateLedger.load() via atomic .tmp replacement. Stores candidate entries, the processed_episode_ids set (guaranteeing replay idempotency across process restarts), and audit commit_history.
- save_word_state_immediately(): Direct immediate serialization of word_state.json inside Handler #7 (word_conflict_resolve), bypassing _min_save_interval without modifying global checkpoint rate limiting.
- engine._save_state() in main.py::finally:: Guards against RAM-only mutations on clean or interrupted process exit.
6. Test Results
A. Isolated Candidate Ledger & Adversarial Suite (tests/cor/CandidateLedger_test.py)
All 8/8 tests passed in 0.035s:
1. test_replay_does_not_inflate_evidence: Generating twice on identical episodes yields added_2 == 0 and bit-for-bit identical candidate weights (PASSED).
2. test_dreamed_cannot_commit_even_with_overwhelming_volume: 500 distinct DREAMED episodes with 0 OBSERVED episodes yields exactly 0 commits (PASSED).
3. test_counterfactual_only_cannot_commit: Counterfactual evidence (weight = 0.0) yields 0 candidates and 0 commits (PASSED).
4. test_grounded_evidence_with_dreamed_support_can_commit: 3 OBSERVED episodes + 5 DREAMED episodes with discriminating base-rates successfully clears the grounded gate, margin gate, and KL gate to commit (PASSED).
5. test_persistence_of_processed_episode_ids_and_candidates: save() and load() fully preserve processed_episode_ids and candidate structures (PASSED).
6. test_mark_unresolved_sets_cooldown_and_penalizes_reward_trace: NEITHER response sets cooldown backoff, applies symmetric penalty, records outcome='NEITHER' in conflict_history, and suppresses conflicts during cooldown turns (PASSED).
7. test_cooldown_persists_across_save_and_load: Cooldown dictionary serializes to word_state.json and suppresses conflicts after reload (PASSED).
8. test_e15_resolution_immediate_durability_and_restart: Resolving a conflict followed by immediate reload without normal pipeline execution proves the resolution survives and detect_conflicts returns [] (PASSED).
B. Complete Architectural Regression Suite
- tests/cor/FactorSpace_test.py: 5/5 PASSED (E10 factor space versioning non-regressed)
- tests/cor/ProbabilisticLabeling_test.py: 15/15 PASSED (probabilistic conventions non-regressed)
- tests/cor/BackendIntegration_test.py: 11/11 PASSED
- tests/cor/GeneratedCodePlugin_test.py: 6/6 PASSED
- tests/cor/TrackerTelemetry_test.py: 6/6 PASSED
- tests/cor/SemanticRepresentation_test.py: 15/15 PASSED
- tests/cor/Curiosity_test.py: 117/117 PASSED
- tests/cor/CandidateLedger_test.py: 8/8 PASSED
- Total Suite: 183 / 183 PASSED (0 failures, 0 errors).
7. Confirmation of Required Invariants
- Repeated replay does not inflate evidence: Watermark tracking via processed_episode_ids ensures each episode contributes evidence at most once.
- Dreamed-only evidence cannot commit: Enforced structurally by min_grounded_distinct_episodes = 2 and min_grounded_evidence_fraction = 0.15 in consolidate().
- Counterfactual-only evidence cannot commit: PROVENANCE_WEIGHT['COUNTERFACTUAL'] = 0.0.
- Grounded evidence is strictly required: Commit is blocked if ungrounded evidence is not anchored by observed real-world interaction.
- Broadcast winner is metadata only: was_broadcast_winner is a boolean audit field; no multiplier or bonus is added to raw_weight.
- E15 conflict mutations survive interruption: save_word_state_immediately() writes to disk before returning from Handler #7.
- E10 remains non-regressed: FactorSpace_test.py 5/5 green.
8. Unresolved Issues / Disclosures
- Live Process Hot-Reload: Live training process PID 9737 was preserved untouched. In Python, loaded in-memory modules do not pick up filesystem source code edits without a process restart. When the ongoing training session reaches a scheduled checkpoint/restart boundary, the new candidate ledger and persistence protections will take effect for subsequent sessions.
- Pre-existing 76 Conflicting Words: Existing 76 historical 0.5/0.5 entries in curiosity_state/word_state.json were preserved unchanged (per Constraint 5). With the new NEITHER handler and cooldown mechanism, responding "Neither fits" will now apply a symmetric penalty and a 20-turn cooldown rather than locking into an inescapable loop
from ..curioimport import *

class HeartProcessingMixin:
    def _feedback_reward_magnitude(self, entry=None) -> float:
        """Return the feedback reward magnitude saved for the rated turn."""
        magnitude = None
        if entry is not None:
            try:
                magnitude = entry.metadata.get('reward_magnitude')
            except Exception:
                magnitude = None
        if magnitude is None:
            magnitude = getattr(self.reward_tracker, 'episode_reward', 0.0)
        try:
            return abs(float(magnitude or 0.0))
        except Exception:
            return 0.0

    def _apply_feedback_rating(self, rating: str, magnitude: float, source: str) -> float:
        """Apply signed reward for explicit user feedback."""
        return self.reward_tracker.apply_feedback_reward(rating, magnitude, source=source)

    def _feedback_words(self, text: str):
        """Tokenize rater input into whole words for feedback classification.

        Substring matching ('right' in 'bright', 'no' in 'know') caused
        ordinary sentences to be misread as ratings, so all feedback
        handlers must match whole words through this helper.
        """
        import re
        return set(re.findall(r"[a-z']+", (text or '').lower()))

    def _backend_report_dict(self, report) -> Dict[str, Any]:
        if report is None:
            return {}
        if hasattr(report, "to_dict"):
            try:
                return report.to_dict()
            except Exception:
                return {}
        if isinstance(report, dict):
            return report
        return {}

    def _sync_backend_turn(self, text: str):
        bridge = getattr(self, "backend_memory_bridge", None)
        if bridge is None:
            add_sentence_to_memory(text)
            return
        try:
            self.backend_sync_report = bridge.sync_turn(text, self)
        except Exception as e:
            self.backend_sync_report = {"ok": False, "errors": [str(e)], "corpus_size": 0}
            print(f"[Backend Sync] skipped: {e}")

    def _verify_backend_alignment(self):
        bridge = getattr(self, "backend_memory_bridge", None)
        if bridge is None:
            return
        try:
            self.backend_verification_report = bridge.verify_engine_alignment(self)
        except Exception as e:
            self.backend_verification_report = {"ok": False, "errors": [str(e)], "corpus_size": 0}
            print(f"[Backend Verify] skipped: {e}")

    def _ask_shadow_differentiation_link(self, proposal):
        if not proposal:
            return
        proposal['asked'] = True
        self.expecting_shadow_diff_confirmation = True
        self.pending_shadow_diff_key = proposal['key']
        print(f" Herebot> {proposal['text']}")

    def _propose_negative_feedback_link(
        self,
        context_tokens=None,
        response_tokens=None,
        source: str = "feedback",
        rejected_response: str = None,
        metadata: dict = None,
    ):
        condition = None
        if self.current_state_labels:
            condition = self.current_state_labels[0]
        proposal = self.reasoner.propose_shadow_differentiation_link(
            context_tokens=context_tokens or [],
            response_tokens=response_tokens or [],
            condition=condition,
            source=source,
            rejected_response=rejected_response,
            metadata=metadata or {},
        )
        self._ask_shadow_differentiation_link(proposal)
        return proposal

    def _run_generated_plugin_trials(
        self,
        text: str,
        tokens: List[str],
        token_analyses: List[Dict[str, Any]],
        factor_dict: Dict[str, Any],
        ci_report: Dict[str, Any],
        phon_score: float,
        diagnostic: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        manager = getattr(self, "generated_plugin_manager", None)
        if manager is None:
            return []
        try:
            context = manager.build_runtime_context(
                self,
                user_input=text,
                tokens=tokens,
                token_analyses=token_analyses,
                factor_dict=factor_dict,
                ci_report=ci_report,
                phon_score=phon_score,
                diagnostic=diagnostic,
            )
            return manager.run_matching_plugins(context)
        except Exception as e:
            print(f"[Generated Plugin] trial skipped: {e}")
            return []

    def _record_generated_plugin_feedback(self, influence, rating: str, reward: float, factor_dict: Dict[str, Any]):
        manager = getattr(self, "generated_plugin_manager", None)
        if manager is None or not influence:
            return
        try:
            manager.record_feedback(influence, rating, reward=reward, factor_dict=factor_dict)
        except Exception as e:
            print(f"[Generated Plugin] feedback tracking skipped: {e}")

    def build_world_model_transitions(self):
        """Scan episodic memory to build transition counts for the structured world model."""
        episodes = self.episodic_memory.episodes[-500:]
        for i in range(len(episodes)-1):
            ep = episodes[i]
            next_ep = episodes[i+1]
            if 'factor_vector' in ep and 'factor_vector' in next_ep:
                # Episodes are dictionaries.  Their insertion order is never a schema.
                # The state system compares named observations across their own versions.
                vec, space = self._episode_control_vector(ep)
                next_vec, next_space = self._episode_control_vector(next_ep)
                state_label = self.state_manager.find_similar_state(
                    vec,
                    factor_space=space,
                    factor_statuses=ep.get('factor_statuses'),
                    threshold=0.7,
                    interaction_index=self.interaction_counter,
                    reward=float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0),
                )
                next_state_label = self.state_manager.find_similar_state(
                    next_vec,
                    factor_space=next_space,
                    factor_statuses=next_ep.get('factor_statuses'),
                    threshold=0.7,
                    interaction_index=self.interaction_counter,
                    reward=float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0),
                )
                if state_label and next_state_label:
                    action = ep.get('system_response', '')   # crude action representation
                    self.structured_world_model.observe_episode(state_label, action, next_state_label)

    def _episode_control_vector(self, episode):
        """Materialize an episode only against its declared control-name schema."""
        values = episode.get('factor_vector', {})
        names = episode.get('factor_names')
        if not names:
            # Legacy dicts are already name-addressed.  Its key order is only used
            # to record a compatibility schema, never to match today's positions.
            names = list(values.keys())
        vector = np.array([values.get(name, 0.0) for name in names], dtype=float)
        space = {'space_id': episode.get('factor_space_id', 'control'),
                 'version': episode.get('factor_space_version', 0),
                 'factor_names': names}
        return vector, space
                    
    def process_input(self, text: str) -> Dict[str, Any]:
            """
            Full pipeline with meta-learning, reasoning, AND factor-based state learning.
            Also increments interaction counter and triggers self‑reconfiguration.

            FIXES applied vs previous version:
            - factor_dict initialized to {} at top to avoid NameError in section 8
            - expecting_word_conflict handler moved to the top-of-function waiting-state
            block (was previously buried after normal response generation, causing the
            full pipeline to run before the handler was ever reached)
            - 'section 2.5' handler body removed from bottom of function (it is now
            correctly placed as handler #7 at the top)
            - Chat-mode reward multiplier guarded so it only runs when factor_dict has
            been properly computed (i.e. not in early-return paths)
            - expecting_word_conflict flag defensively reset even if _pending_conflict
            is missing, to prevent permanent lock-up
            """
            self.last_interaction_time = time.time()

            # Empty/whitespace input carries no signal — ignore it instead of
            # burning a training cycle and polluting episodic memory.
            if not text or not text.strip():
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["TPL.NO.CTX"],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # Pre-initialise so section 8 (behavioral rule testing) can safely reference it
            # before the normal-response block computes the real value.
            factor_dict: Dict[str, Any] = {}

            # ----------------- MAINTENANCE -----------------
            self._cycle_counter += 1
            if self._cycle_counter % 250 == 0:
                self._update_dynamic_limits()

            if self._cycle_counter > self._max_cycles_per_session:
                print("[WATCHDOG] Cycle limit reached — checkpointing.")
                self._save_state()
                self._cycle_counter = 0

            self._memory_pressure_backoff()

            # Reset idle reasoner's long-idle flag
            if hasattr(self, '_idle_reasoner'):
                self._idle_reasoner.reset_long_idle_flag()

            self.interaction_history.append(text)
            if len(self.interaction_history) > getattr(self, "_max_interaction_history", 200000):
                self.interaction_history = self.interaction_history[-self._max_interaction_history:]
            # ----------------- expectation from last turn, compute prediction  -----------------
            if self._expected_next_state is not None and self.current_state_labels:
                actual_state = self.current_state_labels[0]
                if actual_state != self._expected_next_state:
                    # Mismatch – boost curiosity for relevant words
                    prediction_error = 1.0
                    # Could add a factor to meta-learner
                    print(f"   [Prediction Error] Expected {self._expected_next_state}, got {actual_state}")
                self._expected_next_state = None   # reset
            
            
            # ----------------- RESPONSE HANDLING -----------------
            # If we are in any waiting state, handle the response and return early.
            # All waiting-state handlers live here so the main pipeline is never entered
            # while the system is awaiting a reply.

            # 1. Chat mode toggle (special command, not a waiting state)
            if text.lower().strip() == self.chat_mode_trigger:
                self.chat_mode_active = not self.chat_mode_active
                if self.chat_mode_active:
                    self.chat_history = []   # start fresh
                    system_reply = "TPL.MODE.CHAT.ON?"
                else:
                    system_reply = "TPL.MODE.CHAT.OFF"
                print(f" Herebot> {system_reply}")
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": [f"Chat mode: {'on' if self.chat_mode_active else 'off'}"],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
#done
            # 1.5 Shadow differentiation proposal confirmation
            if not self.chat_mode_active and getattr(self, "expecting_shadow_diff_confirmation", False):
                feedback_words = self._feedback_words(text)
                is_positive = bool(feedback_words & {'yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm'})
                is_negative = bool(feedback_words & {'no', 'wrong', 'incorrect', 'not', 'nope', 'never', 'reject'})
                if is_positive:
                    self.reasoner.commit_shadow_differentiation_link(self.pending_shadow_diff_key)
                    artifact = "Shadow differentiation link committed."
                elif is_negative:
                    self.reasoner.reject_shadow_differentiation_link(self.pending_shadow_diff_key)
                    artifact = "Shadow differentiation link rejected."
                else:
                    artifact = "Shadow differentiation link ignored."
                self.expecting_shadow_diff_confirmation = False
                self.pending_shadow_diff_key = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": [artifact],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 2. Response feedback (emergent sentiment learning)
            if not self.chat_mode_active and self.expecting_response_feedback:
                feedback_words = self._feedback_words(text)
                has_thumb_up = '👍' in text
                has_thumb_down = '👎' in text
                rating = None
                if feedback_words & {'positive', 'good', 'great', 'yes', 'correct', 'right'} or has_thumb_up:
                    rating = 'positive'
                elif feedback_words & {'negative', 'bad', 'wrong', 'no', 'incorrect'} or has_thumb_down:
                    rating = 'negative'
                elif feedback_words & {'neutral', 'ok', 'fine', 'average'}:
                    rating = 'neutral'

                if rating is not None and self.pending_feedback_context is not None:
                    entry = self.pending_feedback_context
                    entry.rating = rating
                    entry.metadata['feedback_timestamp'] = time.time()
                    entry.metadata['feedback_text'] = text
                    magnitude = self._feedback_reward_magnitude(entry)
                    feedback_reward = self._apply_feedback_rating(rating, magnitude, source='response_feedback')
                    entry.metadata['feedback_reward'] = feedback_reward
                    entry.metadata['reward_magnitude'] = magnitude
                    self.reasoner.add_feedback(entry)
                    self._record_generated_plugin_feedback(
                        entry.metadata.get('generated_plugin_influence'),
                        rating,
                        feedback_reward,
                        entry.factor_dict,
                    )
                    print(f"📊 [Feedback Recorded] User rated response as: {rating}")

                    if 'positive' in rating and hasattr(self, '_last_factor_vector'):
                        self._reinforce_identity(self._last_factor_vector, reward_signal=0.2)
                    elif 'negative' in rating:
                        self._propose_negative_feedback_link(
                            context_tokens=entry.context_tokens,
                            response_tokens=entry.response_tokens,
                            source='response_feedback',
                            rejected_response=entry.system_response,
                            metadata=entry.metadata.copy(),
                        )
                else:
                    print(f"🌑 [Feedback] No clear rating detected, ignoring.")

                self.expecting_response_feedback = False
                self.pending_feedback_context = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Feedback recorded."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 3. Rule confirmation response
            if not self.chat_mode_active and self.expecting_rule_confirmation:
                feedback_words = self._feedback_words(text)
                is_positive = bool(feedback_words & {'yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm'})
                is_negative = bool(feedback_words & {'no', 'wrong', 'incorrect', 'not', 'nope', 'never'})

                if is_positive or is_negative:
                    rule_id = self.pending_rule_id
                    if rule_id in self.behavioral_learner.rules:
                        if is_positive:
                            print(f"✅ [Rule Confirmed] User accepted rule: {rule_id}")
                            self.behavioral_learner.update_confidence(rule_id, +0.3, feedback=True)
                            if self.behavioral_learner.rules[rule_id].confidence >= self.behavioral_learner.confirmation_threshold:
                                self._integrate_behavioral_rule(rule_id)
                        else:
                            print(f"❌ [Rule Rejected] User denied rule: {rule_id}")
                            self.behavioral_learner.update_confidence(rule_id, -0.5, feedback=True)
                else:
                    print(f"🌑 [Rule Shadow] No clear confirmation for rule: {self.pending_rule_id}")
                    if self.pending_rule_id in self.behavioral_learner.rules:
                        self.behavioral_learner.update_confidence(self.pending_rule_id, -0.1, feedback=False)

                self.expecting_rule_confirmation = False
                self.pending_rule_id = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Rule confirmation processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 4. Logic doubt feedback
            if not self.chat_mode_active and self.expecting_logic_feedback:
                feedback_words = self._feedback_words(text)
                is_positive = bool(feedback_words & {'yes', 'correct', 'right', 'yeah', 'true', 'yep', 'ok', 'sure', 'confirm'})
                is_negative = bool(feedback_words & {'no', 'wrong', 'incorrect', 'not', 'nope', 'never'})
                is_neutral = bool(feedback_words & {'neutral', 'maybe', 'unsure', 'average'})

                if self.pending_logic_doubt:
                    logic_id = self.pending_logic_doubt['logic_id']
                    if is_positive:
                        self.logic_inheritance.confirm_logic(logic_id, is_correct=True)
                        print(f"✅ [Logic Confirmed] User accepted logic: {logic_id}")
                    elif is_negative:
                        self.logic_inheritance.confirm_logic(logic_id, is_correct=False)
                        print(f"❌ [Logic Rejected] User denied logic: {logic_id}")

                    if is_positive or is_negative or is_neutral:
                        rating = "positive" if is_positive else "negative" if is_negative else "neutral"
                        magnitude = self._feedback_reward_magnitude()
                        self._apply_feedback_rating(rating, magnitude, source='logic_feedback')
                        self.generated_code_feedback.add_feedback(
                            logic_id,
                            rating,
                            {"factor_dict": self._last_factor_dict.copy() if self._last_factor_dict else {}}
                        )
                        if self._last_factor_dict is not None:
                            self.guilt_tracker.update_guilt(rating, self._last_factor_dict)
                        if is_negative:
                            rule = self.pending_logic_doubt.get('rule', {})
                            context_tokens = self._tokenize(str(rule.get('description', logic_id)))
                            self._propose_negative_feedback_link(
                                context_tokens=context_tokens,
                                response_tokens=[],
                                source='logic_feedback',
                                rejected_response=str(rule),
                                metadata={'logic_id': logic_id, 'reward_magnitude': magnitude},
                            )
                    else:
                        print(f"🌑 [Logic Shadow] No clear confirmation for logic: {logic_id}")

                self.expecting_logic_feedback = False
                self.pending_logic_doubt = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Logic feedback processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 5. Pattern label response
            if not self.chat_mode_active and getattr(self, "expecting_pattern_label", False):
                label = text.strip().lower()
                self.reasoner.relations[self.last_asked_word].append(("category", label, 1.0))
                if self.last_asked_word not in self.semantic_learner.word_embeddings:
                    self.semantic_learner.initialize_word(self.last_asked_word)
                print(f"📘 [Pattern Learning] '{self.last_asked_word}' categorized as '{label}'")
                self.expecting_pattern_label = False
                self.last_asked_word = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Pattern label learned."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 6. Trust feedback (from earlier attempt)
            if not self.chat_mode_active and self.expecting_feedback:
                feedback_words = self._feedback_words(text)
                is_positive = bool(feedback_words & {'yes', 'correct', 'right', 'yeah', 'true', 'yep'})
                is_negative = bool(feedback_words & {'no', 'wrong', 'incorrect', 'not', 'nope'})
                is_neutral = bool(feedback_words & {'neutral', 'ok', 'fine', 'average', 'maybe'})

                if is_positive or is_negative or is_neutral:
                    rating = "positive" if is_positive else "negative" if is_negative else "neutral"
                    magnitude = self._feedback_reward_magnitude()
                    self._apply_feedback_rating(rating, magnitude, source='trust_feedback')
                    if is_positive:
                        print(f"✅ [Positive Feedback] User confirmed my attempt: '{self.feedback_attempt_response}'")
                        if hasattr(self, '_last_factor_vector'):
                            self._reinforce_identity(self._last_factor_vector, reward_signal=0.4)
                    elif is_negative:
                        print(f"❌ [Negative Feedback] User corrected my attempt: '{self.feedback_attempt_response}'")
                        prior_input = self.interaction_history[-2] if len(self.interaction_history) >= 2 else ""
                        self._propose_negative_feedback_link(
                            context_tokens=self._tokenize(prior_input),
                            response_tokens=self._tokenize(self.feedback_attempt_response or ""),
                            source='trust_feedback',
                            rejected_response=self.feedback_attempt_response,
                            metadata={'reward_magnitude': magnitude},
                        )
                else:
                    print(f"🌑 [Shadow Attempt] No feedback given for: '{self.feedback_attempt_response}'")
                    if self.feedback_attempt_response:
                        hyp = {
                            'key': f"shadow_trust_{int(time.time())}",
                            'type': 'shadow_trust',
                            'utterance': self.feedback_attempt_response,
                            'timestamp': self.feedback_timestamp,
                        }
                        if len(self.reasoner.hypotheses) < getattr(self, "_max_hypotheses", 5000):
                            self.reasoner.hypotheses.append(hyp)

                self.expecting_feedback = False
                self.feedback_attempt_word = None
                self.feedback_attempt_response = None
                self.feedback_timestamp = None
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Feedback processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }

            # 7. Word-label conflict resolution
            # FIX: this was previously buried after normal response generation, meaning
            # the entire pipeline ran before the handler was reached. It must live here
            # alongside all other waiting-state handlers.
            if not self.chat_mode_active and getattr(self, "expecting_word_conflict", False):
                feedback_lower = text.lower()
                if hasattr(self, '_pending_conflict'):
                    word, label1, label2 = self._pending_conflict
                    if label1.lower() in feedback_lower or label2.lower() in feedback_lower:
                        keep = label1 if label1.lower() in feedback_lower else label2
                        self.reasoner.word_state.break_bridge(
                            word,
                            label1,
                            label2,
                            keep,
                            interaction_index=self.interaction_counter,
                            reward=-abs(float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0)),
                        )
                        print(f"✅ [Conflict Resolved] '{word}' is now primarily '{keep}'.")
                    else:
                        print(f"🌑 No clear label chosen for '{word}'. Conflict remains.")
                    del self._pending_conflict
                else:
                    # Safety: flag was set but state was lost — reset cleanly
                    print("🌑 [Word Conflict] Expected conflict state missing; resetting flag.")
                self.expecting_word_conflict = False
                return {
                    "ci_report": {},
                    "phon_score": 0,
                    "diagnostic": {},
                    "token_analyses": [],
                    "curiosity_artifacts": ["Word conflict processed."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
#done2

            # 8. Definition response (when last_asked_word is set and user provides a definition)
            # Guard: only consume the turn as a definition when the pending
            # word/phrase actually occurs in the input. Otherwise the sentence
            # belongs to a different word and must flow into normal processing
            # (previously e.g. 'road is ...' was attached to pending 'vehicle').
            handled_definition = False
            if not self.chat_mode_active and self.last_asked_word is not None:
                pending = self.last_asked_word
                pending_in_text = pending.lower() in text.lower()
                if ("means" in text.lower() or "is" in text.lower() or "refers to" in text.lower()) and pending_in_text:
                    is_phrase = ' ' in self.last_asked_word
                    if is_phrase:
                        phrase = self.last_asked_word
                        definition = text
                        phrase_tokens = phrase.split()
                        known_embs = [self.semantic_learner.word_embeddings[t] for t in phrase_tokens if t in self.semantic_learner.word_embeddings]
                        phrase_emb = np.mean(known_embs, axis=0) if known_embs else None
                        self.reasoner.phrase_definitions[phrase] = definition
                        if phrase_emb is not None:
                            self.reasoner.phrase_embeddings[phrase] = phrase_emb
                        print(f"📘 [Phrase Definition] Learned about '{phrase}': {definition}")
                        self._derive_inferences_from_definition(phrase, definition)
                    else:
                        word = self.last_asked_word
                        if word not in self.semantic_learner.word_embeddings:
                            self.semantic_learner.initialize_word(word)
                        detector = getattr(getattr(self.reasoner, 'pattern_observer', None), 'detector', None)
                        self.semantic_learner.observe_definition(word, text, structural_detector=detector)
                        def_emb = self.semantic_learner.text_to_embedding(text, structural_detector=detector)
                        if def_emb is not None:
                            self.reasoner.store_definition(word, text, def_emb)
                            self.concept_model.register_definition_event(
                                word,
                                interaction_index=self.interaction_counter,
                                reward=max(1.0, float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0)),
                            )
                        if word in self.word_history:
                            for rec in self.word_history[word][-3:]:
                                rec['curiosity_factors'].gp_uncertainty *= 0.3
                                rec['curiosity_factors'].bayesian_surprise *= 0.3
                            if hasattr(self.concept_model, 'word_concept_counts') and word in self.concept_model.word_concept_counts:
                                counts = self.concept_model.word_concept_counts[word]
                                if counts:
                                    dominant = max(counts, key=counts.get)
                                    new_counts = Counter({dominant: 100})
                                    self.concept_model.word_concept_counts[word] = new_counts
                            hypothesis = self.reasoner.generate_synonym_hypothesis(word)
                            if hypothesis:
                                print(f"💡 [Semantic Hypothesis] {hypothesis}")
                        print(f"📘 [Definition Learning] Learned about '{word}': {text}")
                        self._derive_inferences_from_definition(word, text)

                    if hasattr(self, '_last_factor_vector'):
                        self._reinforce_identity(self._last_factor_vector, reward_signal=0.3)
                    self.last_asked_word = None
                    handled_definition = True
                    # Fall through to main processing so the sentence is still analyzed
            # 9. Procedure explanation (when user is teaching step by step)
            if not self.chat_mode_active and self.expecting_procedure_steps:
                # Record the step provided by user
                step_text = text.strip()
                self.recorded_steps.append(step_text)
                print(f"📝 [Step Recorded] {step_text}")
                # Simple heuristic: if step ends with '.' or we have a solution, maybe stop.
                # We'll just continue until user says "done" or "that's it".
                if "done" in step_text.lower() or "that's it" in step_text.lower() or "finish" in step_text.lower():
                    # End of teaching
                    if self.pending_problem_expr is not None:
                        # Store procedure
                        proc = self.reasoner.procedural_learner.learn_procedure(
                            self.pending_problem_expr, self.recorded_steps, None
                        )
                        print(f" [Procedure Learned] TPL.PROC.LEARN expr='{self.pending_problem_expr}'")
                    self.expecting_procedure_steps = False
                    self.pending_problem_expr = None
                    self.recorded_steps = []
                    return {
                        "ci_report": {},
                        "phon_score": 0,
                        "diagnostic": {},
                        "token_analyses": [],
                        "curiosity_artifacts": ["Procedure recorded."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }
                else:
                    # Continue expecting more steps
                    return {
                        "ci_report": {},
                        "phon_score": 0,
                        "diagnostic": {},
                        "token_analyses": [],
                        "curiosity_artifacts": ["Step recorded. Continue."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }
            # ----------------- MAIN PROCESSING -----------------
            # If we reach here, we are not in a waiting state (or we handled a definition and continue).
            self.interaction_counter += 1

            # 1. Memory ingestion
            self._sync_backend_turn(text)
            self._verify_backend_alignment()

            # 2. Whole-sentence evaluation
            ci_report = self.ci_engine.evaluate(text)
            phon_score, phon_diagnostics = self.phon_engine.score_text(text)
            diag_explanation = self.diag_engine.explain(ci_report, phon_score)

            # 3. Token analysis
            tokens = self._tokenize(text)
            self.reasoner.observe_sentence(text, tokens)
            self.reward_tracker.reset_episode()

            token_analyses = []
            for token in tokens:
                token_ci = self.ci_engine.evaluate(token)
                token_phon, _ = self.phon_engine.score_text(token)

                features = self._extract_features(token, token_ci, token_phon, diag_explanation)
                features_array = self._features_to_array(features)
                concept = self.concept_model.add_word_observation(
                    token,
                    features,
                    interaction_index=self.interaction_counter,
                    reward=0.0,
                )

                factors = self._compute_curiosity_factors(
                    token, token_ci, token_phon, diag_explanation, features_array,
                    global_factor_dict=self._last_factor_dict
                )

                new_entropy = self.concept_model.entropy_word(token)
                igain = self._information_gain(token, new_entropy)

                if igain == 0.0:
                    igain = 0.25 * factors.bayesian_surprise + 0.25 * factors.gp_uncertainty

                self.gp_model.update(features_array, token_ci['confidence'])

                reward = self.reward_tracker.compute_reward(token_ci, igain, factors.bayesian_surprise)
                if hasattr(self.concept_model, 'record_concept_reward'):
                    self.concept_model.record_concept_reward(
                        token,
                        concept,
                        interaction_index=self.interaction_counter,
                        reward=reward,
                    )

                self.word_history[token].append({
                    'ci_confidence': token_ci['confidence'],
                    'phon_score': token_phon,
                    'features': features,
                    'concept': concept,
                    'entropy': new_entropy,
                    'curiosity_factors': factors,
                    'information_gain': igain,
                    'intrinsic_reward': reward,
                })

                factor_array = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))

                struct_score = self.structural_score(token)

                # Adjust information gain for meta-learner to avoid structural contamination
                
                igain_adjusted = igain * (1 - struct_score)
                self.meta_learner.update(factor_array, igain_adjusted)

                curiosity_score = self.meta_learner.predict_curiosity(factor_array)
                curiosity_score *= (1 - struct_score)

                token_analyses.append({
                    'token': token,
                    'concept_id': concept,
                    'entropy': new_entropy,
                    'curiosity_score': curiosity_score,
                    'factors': factors,
                    'igain': igain,
                    'intrinsic_reward': reward,
                    'gp_uncertainty': factors.gp_uncertainty,
                    'bayesian_surprise': factors.bayesian_surprise,
                })
            # ---- Update token profiles for structural scoring ----
            for token in tokens:
                # Get embedding (if available)
                emb = self.semantic_learner.word_embeddings.get(token)
                if emb is None:
                    emb = np.zeros(self.semantic_learner.emb_dim)  # fallback
                # Find the corresponding token analysis entry for this token (to get bayes_surprise and igain)
                ta = next((t for t in token_analyses if t['token'] == token), None)
                if ta:
                    bayes = ta.get('bayesian_surprise', 0.0)
                    igain = ta.get('igain', 0.0)
                else:
                    bayes = 0.0
                    igain = 0.0
                # Context: all other tokens in this input
                context = [t for t in tokens if t != token]
                self._update_token_profile(token, context, bayes, igain, emb)
            # ---- Sentence-level reasoning (if all tokens known) ----
            if self._all_tokens_known(tokens):
                detector = getattr(getattr(self.reasoner, 'pattern_observer', None), 'detector', None)
                sentence_emb = self.semantic_learner.text_to_embedding(text, structural_detector=detector)
                if sentence_emb is not None:
                    self._process_sentence_reasoning(text, tokens, token_analyses, sentence_emb)

            # ---- Build expression from syntax tree ----
            syntax_tree = self.syntax_learner.parse(tokens)
            if syntax_tree is not None:
                expr = syntax_tree.to_expression(self.reasoner.type_hierarchy, self.reasoner.operators)
                if expr is not None and isinstance(expr.root, Operator):
                    self.reasoner.store_expression(text, expr)

                # ---- Detect equations (left = right) ----
                # Heuristic: if the top node is '=' and has two children, it's an equation.
                if expr is not None and expr.root == '=' and expr.left and expr.right:
                    left = expr.left
                    right = expr.right
                    # Feed to schema learner
                    self.reasoner.schema_learner.incorporate_equation(left, right)

                    # Try to solve the equation
                    subs = self.reasoner.solve_equation(left, right)
                    if subs is not None:
                        # Build a readable result
                        result_parts = []
                        for var_name, val_expr in subs.items():
                            # Convert value expression to string
                            if val_expr.is_leaf():
                                val_str = str(val_expr.root)
                            else:
                                val_str = str(val_expr)
                            result_parts.append(f"{var_name} = {val_str}")
                        if result_parts:
                            self.pending_solution = ", ".join(result_parts)
                        else:
                            self.pending_solution = None
                    else:
                        self.pending_solution = None
                        # If we couldn't solve and user asked a question, consider teaching
                        if '?' in text:
                            if expr is not None:
                                proc = self.reasoner.procedural_learner.find_procedure(expr)
                                if proc:
                                    # Apply procedure (placeholder)
                                    result_expr = proc.apply(expr, self.reasoner)
                                    if result_expr:
                                        self.pending_solution = f"Using learned procedure: {result_expr}"
                                else:
                                    # No procedure, ask user to teach us
                                    self.pending_teaching_request = True
                                    self.pending_teaching_expr = expr
            if syntax_tree is not None:
                expr = syntax_tree.to_expression(self.reasoner.type_hierarchy, self.reasoner.operators)
                if expr is not None and isinstance(expr.root, Operator):
                    self.reasoner.store_expression(text, expr)

            # ----------------- DECIDE WHETHER TO ASK A QUESTION -----------------
            # These checks are in priority order; the first that triggers will ask and return early.

            # 4. Logic doubt (ask about a generated logic rule)
            # 4.5 Teaching request (ask user to teach us)
            if not self.chat_mode_active and self.pending_teaching_request:
                self.pending_teaching_request = False
                expr = self.pending_teaching_expr
                self.pending_teaching_expr = None
                self.expecting_procedure_steps = True
                self.pending_problem_expr = expr
                self.recorded_steps = []
                teach_question = f"TPL.ASK.TEACH expr='{expr}'?"
                print(f"\n [Teaching Request] {teach_question}")
                print(f" Herebot> {teach_question}")
                return {
                    "ci_report": ci_report,
                    "phon_score": phon_score,
                    "diagnostic": diag_explanation,
                    "token_analyses": token_analyses,
                    "curiosity_artifacts": ["Teaching mode activated."],
                    "meta_weights": self.meta_learner.get_weights(),
                    "total_intrinsic_reward": self.reward_tracker.total_reward,
                    "episode_reward": 0.0,
                    "factor_vector": {},
                }
            if not self.chat_mode_active:
                doubt = self.logic_inheritance.get_next_doubt()
                if doubt:
                    rule = doubt['rule']
                    description = rule.get('description', f"rule={rule}")
                    query = f"TPL.ASK.LOGIC desc='{description}'?"
                    print(f"\n [Logic Doubt] {query}")
                    print(f" Herebot> {query}")
                    self.expecting_logic_feedback = True
                    self.pending_logic_doubt = doubt
                    return {
                        "ci_report": ci_report,
                        "phon_score": phon_score,
                        "diagnostic": diag_explanation,
                        "token_analyses": token_analyses,
                        "curiosity_artifacts": ["Logic doubt – awaiting confirmation."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": 0.0,
                        "factor_vector": {},
                    }

            # 5. Unknown word detection
            if not self.chat_mode_active:
                unknown_tokens = [t for t in tokens if t not in self.semantic_learner.word_embeddings]
                unknown_ratio = len(unknown_tokens) / max(len(tokens), 1)
                # Pure math expressions (numbers + operators) must reach the
                # math reply path instead of being treated as vocabulary gaps.
                op_symbols = set(getattr(getattr(self, 'reasoner', None), 'operators', {}) or {})
                non_op = [t for t in tokens if t not in op_symbols]
                is_pure_math = bool(op_symbols and any(t in op_symbols for t in tokens)
                                    and non_op and all(t.replace('.', '', 1).isdigit() for t in non_op))
                if not is_pure_math and unknown_ratio > 0.5 and unknown_tokens:
                    # Score each unknown token (structural score)
                    scored_unknowns = [(t, self.structural_score(t)) for t in unknown_tokens]
                    # Sort by score ascending (most semantic first)
                    scored_unknowns.sort(key=lambda x: x[1])
                    target, struct_score = scored_unknowns[0]
                    # Only ask if structural score is not too high (i.e., it's likely semantic)
                    if struct_score <= 0.65:
                        if self.last_asked_word is not None and self.last_asked_word != target:
                            print(f"   [Definition Dropped] Previous pending word '{self.last_asked_word}' "
                                  f"never got a definition; now asking about '{target}'.")
                        self.last_asked_word = target
                        system_reply = f"TPL.ASK.DEF target='{target}'?"
                        print(f"\n [Knowledge Gap] {system_reply}")
                        print(f" Herebot> {system_reply}")
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Knowledge gap detected — waiting for definition."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": 0.0,
                            "factor_vector": {},
                        }
            # 6. Pattern discovery (ask for category label)
            if not self.chat_mode_active:
                for token in set(tokens):
                    pattern_query = self.reasoner.detect_pattern_and_query(token)
                    if pattern_query:
                        self.last_asked_word = token
                        self.expecting_pattern_label = True
                        print(f"\n [Pattern Discovery] {pattern_query}")
                        print(f" Herebot> {pattern_query}")
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Pattern discovered — awaiting label."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": self.reward_tracker.episode_reward,
                            "factor_vector": {},
                        }

            # 7. Legacy conflict resolution (debug only; probabilistic regrouping is automatic)
            if (
                not self.chat_mode_active
                and self.config_manager.get('state_manager.legacy_interactive_conflicts', False)
            ):
                unresolved = self.state_manager.get_unresolved_conflicts()
                if unresolved:
                    print("\n [Conflict Resolution] TPL.STATE.CONFLICT.PENDING")
                    for conflict in unresolved[:2]:
                        l1, l2 = conflict["labels"]
                        l1 = self.state_manager.resolve_label(l1)
                        l2 = self.state_manager.resolve_label(l2)
                        sim = conflict["similarity"]
                        shadow_label = conflict.get("resolution_label")
                        print(f"    Conflict: '{l1}' ↔ '{l2}' (similarity={sim:.3f})")
                        print(f"    I created a temporary shadow state '{shadow_label}'.")
                        print("    TPL.STATE.CONFLICT.MENU")
                        print("      1. keep_both")
                        print("      2. keep_one")
                        print("      3. rename_both")
                        print("      4. promote_shadow")
                        print("      5. edit_vectors")
                        choice = input("    Your choice (1/2/3/4/5): ").strip()

                        if choice == "2":
                            winner = input(f"    Which label do you want to keep? ({l1}/{l2}): ").strip()
                            if winner in [l1, l2]:
                                self.state_manager.resolve_conflict_permanently(conflict, winner, delete_other=True)
                                print(f"    ✅ Resolved: keeping '{winner}', deleted the other.")
                        elif choice == "3":
                            new_label1 = input(f"    New label for '{l1}': ").strip() or l1
                            new_label2 = input(f"    New label for '{l2}': ").strip() or l2
                            l1 = self.state_manager.resolve_label(l1)
                            l2 = self.state_manager.resolve_label(l2)

                            l1_found, vec1 = self.state_manager.get_state_vector(l1)
                            l2_found, vec2 = self.state_manager.get_state_vector(l2)

                            if vec1 is None or vec2 is None:
                                reason = f"missing:{l1 if vec1 is None else l2}"
                                self.state_manager.mark_conflict_stale(conflict, reason)
                                self.state_manager.refresh_conflicts()
                                print(f"    TPL.STATE.CONFLICT.STALE reason='{reason}'")
                                continue
                            l1 = l1_found
                            l2 = l2_found
                            self.state_manager.save_state(new_label1, vec1)
                            self.state_manager.save_state(new_label2, vec2)
                            self.state_manager.delete_state(l1)
                            self.state_manager.delete_state(l2)
                            conflict["resolved"] = True
                            conflict["resolution_label"] = f"{new_label1},{new_label2}"
                            for c in self.state_manager.conflicts:
                                if l1 in c["labels"] or l2 in c["labels"]:
                                    c["resolved"] = True
                                    c["resolution_label"] = "renamed"
                            print(f"    ✅ States renamed and kept.")
                        elif choice == "4" and shadow_label:
                            vec = self.state_manager.shadow_states.get(shadow_label)
                            if vec is not None:
                                perm_label = input(f"    Permanent name for this merged state: ").strip() or f"merged_{l1}_{l2}"
                                self.state_manager.save_state(perm_label, vec, is_shadow=False)
                                self.state_manager.delete_state(shadow_label)
                                # Preserve both original identities
                                l1_res = self.state_manager.resolve_label(l1)
                                l2_res = self.state_manager.resolve_label(l2)

                                self.state_manager.aliases[l1_res] = perm_label
                                self.state_manager.aliases[l2_res] = perm_label

                                self.state_manager.resolve_conflict_permanently(conflict, perm_label, delete_other=True)
                                print(f"    ✅ Shadow state promoted to '{perm_label}', both originals deleted.")
                        elif choice == "5":
                            # Show factor vectors and allow editing
                            l1 = self.state_manager.resolve_label(l1)
                            l2 = self.state_manager.resolve_label(l2)

                            l1_found, vec1 = self.state_manager.get_state_vector(l1)
                            l2_found, vec2 = self.state_manager.get_state_vector(l2)

                            if vec1 is None or vec2 is None:
                                reason = f"missing:{l1 if vec1 is None else l2}"
                                self.state_manager.mark_conflict_stale(conflict, reason)
                                self.state_manager.refresh_conflicts()
                                print(f"    TPL.STATE.CONFLICT.STALE reason='{reason}'")
                                continue
                            l1 = l1_found
                            l2 = l2_found
                            # Align to same length (max of both)
                            max_len = max(len(vec1), len(vec2))
                            v1_aligned = np.pad(vec1, (0, max_len - len(vec1)), 'constant') if len(vec1) < max_len else vec1[:max_len]
                            v2_aligned = np.pad(vec2, (0, max_len - len(vec2)), 'constant') if len(vec2) < max_len else vec2[:max_len]

                            # Get factor names (from control_space)
                            factor_names = self.control_space.all_factors
                            # If lengths differ, we may need to pad factor names with placeholders
                            if len(factor_names) < max_len:
                                factor_names = factor_names + [f"factor_{i}" for i in range(len(factor_names), max_len)]
                            elif len(factor_names) > max_len:
                                factor_names = factor_names[:max_len]

                            print(f"\n    Factor vectors for '{l1}' and '{l2}':")
                            for i, name in enumerate(factor_names):
                                print(f"      {name:20s} : {l1} = {v1_aligned[i]:.3f} , {l2} = {v2_aligned[i]:.3f}")

                            print("\n    You can now edit the factor vectors. Enter new values for each state as comma‑separated lists.")
                            print("    (Press Enter to keep current values for that state.)")
                            new_vec1_str = input(f"    New vector for '{l1}' (comma‑separated, {max_len} numbers): ").strip()
                            new_vec2_str = input(f"    New vector for '{l2}' (comma‑separated, {max_len} numbers): ").strip()

                            def parse_vector(s, default_vec):
                                if not s:
                                    return default_vec
                                parts = s.split(',')
                                try:
                                    nums = [float(p.strip()) for p in parts]
                                    if len(nums) < max_len:
                                        nums += [default_vec[len(nums):].tolist()] if len(default_vec) > len(nums) else []
                                    elif len(nums) > max_len:
                                        nums = nums[:max_len]
                                    return np.array(nums)
                                except:
                                    print("    Invalid input, keeping original.")
                                    return default_vec

                            new_v1 = parse_vector(new_vec1_str, v1_aligned)
                            new_v2 = parse_vector(new_vec2_str, v2_aligned)

                            # Update states (they will be normalized by update_state)
                            self.state_manager.update_state(l1, new_v1, metadata=conflict.get('metadata', {}))
                            self.state_manager.update_state(l2, new_v2, metadata=conflict.get('metadata', {}))

                            # Mark conflict as resolved (we keep both)
                            conflict["resolved"] = True
                            conflict["resolution_label"] = f"edited:{l1},{l2}"
                            print(f"    ✅ States '{l1}' and '{l2}' updated and conflict resolved.")
                        else:
                            print("    No action taken; conflict remains unresolved.")
                    self.state_manager._save_conflicts()

            # 8. Behavioral rule testing (ask for confirmation)
            # so the 'factor_dict in locals()' workaround is no longer needed.
            if not self.chat_mode_active:
                context = {
                    'user_input': text,
                    'tokens': tokens,
                    'token_analyses': token_analyses,
                    'factor_dict': factor_dict,
                    'state': self.current_state_labels
                }
                testable = self.behavioral_learner.get_testable_rules(context)
                if testable:
                    rule = max(testable, key=lambda r: r.confidence)
                    if rule.rule_type == 'identity_alias':
                        alias = rule.condition['alias']
                        query = f"TPL.ASK.ALIAS alias='{alias}'?"
                    elif rule.rule_type == 'pronoun_mapping':
                        query = f"TPL.ASK.PRONOUN pronoun='{rule.condition['pronoun']}'?"
                    else:
                        query = f"TPL.ASK.RULE condition='{rule.condition}'?"
                    rule.asked = True
                    rule.last_tested = time.time()
                    self.behavioral_learner.save()
                    self.expecting_rule_confirmation = True
                    self.pending_rule_id = rule.rule_id
                    print(f"\n [Rule Hypothesis] {query}")
                    print(f" Herebot> {query}")
                    return {
                        "ci_report": ci_report,
                        "phon_score": phon_score,
                        "diagnostic": diag_explanation,
                        "token_analyses": token_analyses,
                        "curiosity_artifacts": ["Behavioral rule hypothesis — awaiting confirmation."],
                        "meta_weights": self.meta_learner.get_weights(),
                        "total_intrinsic_reward": self.reward_tracker.total_reward,
                        "episode_reward": self.reward_tracker.episode_reward,
                        "factor_vector": factor_dict,
                    }

            # ----------------- NORMAL RESPONSE GENERATION -----------------
            # If no question was asked, compute factors and generate a normal reply.

            # Factor computation & state learning
            factor_dict = self.factor_engine.compute(
                token_analyses, ci_report, phon_score, diag_explanation,
                self.concept_model, self.meta_learner, self.reward_tracker.total_reward,
                self.semantic_learner
            )
            if not isinstance(factor_dict, dict):
                factor_dict = {}

            # Conscience influence
            factor_dict = self.conscience_manager.apply_ethical_influence(factor_dict)
            if not isinstance(factor_dict, dict):
                factor_dict = {}

            # Build factor vector
            control_names = self.control_space.factor_space.factor_names
            factor_vector_list = [factor_dict.get(k, 0.5) for k in control_names]
            factor_vector = np.array(factor_vector_list)
            self.control_space.validate_control_vector(factor_vector, factor_names=control_names)
            self._last_factor_vector = factor_vector
            self._last_factor_dict = factor_dict

            metadata = {
                "input_text": text,
                "timestamp": time.time(),
                "ci_confidence": ci_report.get("confidence"),
                "phon_score": phon_score,
            }
            # State learning gate
            if not self.chat_mode_active and len(self.semantic_learner.word_embeddings) > 20:
                self._handle_state_learning(factor_vector, metadata, factor_space=self.control_space.factor_space)
            if not self.config_manager.get('state_manager.legacy_interactive_conflicts', False):
                self.state_manager.archive_legacy_conflicts(interaction_index=self.interaction_counter)
                self.state_manager.refresh_state_regrouping(
                    interaction_index=self.interaction_counter,
                    reasoner=getattr(self, 'reasoner', None),
                    force=False,
                )

            # Semantic relation discovery
            self._detect_semantic_relations(tokens, token_analyses)

            # Generate artifacts
            artifacts = self._generate_artifacts(
                tokens, token_analyses, ci_report, phon_score, diag_explanation,
                factor_dict
            )
            generated_plugin_proposals = self._run_generated_plugin_trials(
                text, tokens, token_analyses, factor_dict, ci_report, phon_score, diag_explanation
            )
            if generated_plugin_proposals:
                artifacts.append(
                    f"Generated plugin proposals: {len(generated_plugin_proposals)}"
                )
            # Generate system reply
            self._last_generated_plugin_influence = None
            system_reply = self.action_generator.generate_reply(text, {
                'token_analyses': token_analyses,
                'factor_vector': factor_dict,
                'generated_plugin_proposals': generated_plugin_proposals,
            })
            if self.pending_solution is not None:
                system_reply = f"TPL.MATH.SOLUTION value='{self.pending_solution}' | " + system_reply
                self.pending_solution = None   # reset
            self._last_action = system_reply
            # ---- Top‑down expectation: predict next state using world model ----
            if self.structured_world_model and self.current_state_labels:
                current_state = self.resolve_state_label(self.current_state_labels[0]) if self.current_state_labels else None
                if current_state:
                    action = system_reply  # the action we just took
                    predicted_next = self.structured_world_model.predict_next_state(current_state, action)
                    if predicted_next:
                        # Compare predicted vs actual next state (which we don't have yet – would need next turn)
                        # For now, we store the prediction to be checked later.
                        self._expected_next_state = predicted_next
            if system_reply.startswith(("TPL.ASK.DEF", "TPL.ASK.MEAN")) and "'" in system_reply:
                import re
                match = re.search(r"(?:target|word)='([^']+)'", system_reply)
                if match:
                    self.last_asked_word = match.group(1)
            print(f" Herebot> {system_reply}")

            # Possibly ask for feedback on the response we just generated
            if not self.chat_mode_active and not self.expecting_response_feedback and not self.expecting_feedback and not self.expecting_rule_confirmation and not self.expecting_logic_feedback:
                template_id = f"resp_{hash(system_reply) % 10000:04x}"
                self.reasoner.register_response_template(template_id, system_reply)
                if self._should_ask_feedback(tokens, factor_dict, template_id):
                    feedback_question = self._generate_feedback_question()
                    print(f" Herebot> {feedback_question}")
                    entry = FeedbackEntry(
                        timestamp=time.time(),
                        user_input=text,
                        system_response=system_reply,
                        factor_dict=factor_dict.copy(),
                        rating='pending',
                        context_tokens=tokens,
                        response_tokens=self._tokenize(system_reply),
                        response_template_id=template_id,
                        metadata={
                            'reward_magnitude': abs(float(getattr(self.reward_tracker, 'episode_reward', 0.0))),
                            'current_state_labels': list(self.current_state_labels),
                            'triplets': getattr(self.reasoner, 'last_observed_triplets', []),
                            'generated_plugin_influence': getattr(self, '_last_generated_plugin_influence', None),
                        }
                    )
                    self.pending_feedback_context = entry
                    self.expecting_response_feedback = True

            # Check for word-label conflicts and ask user to resolve if any found.
            # This runs after response generation so we already have factor_dict.
            # FIX: the old "2.5" block that also appeared here (as a handler) has been
            # removed — the handler now lives at the top with all other waiting-state
            # handlers (#7). This block only *raises* new word conflicts; it never
            # handles replies to them.
            if not self.chat_mode_active and not any([
                self.expecting_response_feedback,
                self.expecting_feedback,
                self.expecting_rule_confirmation,
                self.expecting_logic_feedback,
                getattr(self, 'expecting_word_conflict', False),
            ]):
                for token in tokens:
                    conflicts = self.reasoner.word_state.detect_conflicts(token)
                    if conflicts:
                        label1, label2 = conflicts[0]
                        question = f"TPL.ASK.WORD.CONFLICT token='{token}' labels='{label1}|{label2}'?"
                        print(f"\n [Word Conflict] {question}")
                        print(f" Herebot> {question}")
                        self.expecting_word_conflict = True
                        self._pending_conflict = (token, label1, label2)
                        return {
                            "ci_report": ci_report,
                            "phon_score": phon_score,
                            "diagnostic": diag_explanation,
                            "token_analyses": token_analyses,
                            "curiosity_artifacts": ["Word conflict – awaiting clarification."],
                            "meta_weights": self.meta_learner.get_weights(),
                            "total_intrinsic_reward": self.reward_tracker.total_reward,
                            "episode_reward": self.reward_tracker.episode_reward,
                            "factor_vector": factor_dict,
                        }

            # Chat mode reward multiplier — only applied after factor_dict is fully computed.
            # FIX: previously ran even on early-return paths where factor_dict was still {}.
            if self.chat_mode_active:
                multiplier = self._compute_chat_reward_multiplier(factor_dict)
                original_reward = self.reward_tracker.episode_reward
                self.reward_tracker.episode_reward *= multiplier
                print(f" [Chat Mode] Reward multiplier: {multiplier:.2f} (original: {original_reward:.3f} → boosted: {self.reward_tracker.episode_reward:.3f})")

            # Store episode
            safe_factor_dict = factor_dict if isinstance(factor_dict, dict) else {}
            self.episodic_memory.add_episode(
                user_input=text,
                system_response=system_reply,
                factor_vector=safe_factor_dict,
                intrinsic_reward=float(self.reward_tracker.episode_reward),
                metadata={
                    'ci_confidence': ci_report.get('confidence'),
                    'generated_plugin_influence': getattr(self, '_last_generated_plugin_influence', None),
                },
                factor_space=self.control_space.factor_space,
                factor_statuses={name: 'observed' if name in factor_dict else 'unavailable'
                                 for name in control_names},
            )
            if getattr(self, '_last_generated_plugin_influence', None):
                try:
                    self.generated_plugin_manager.record_trial_health(
                        self._last_generated_plugin_influence,
                        factor_dict=safe_factor_dict,
                        reward=float(self.reward_tracker.episode_reward),
                    )
                except Exception as e:
                    print(f"[Generated Plugin] health tracking skipped: {e}")
            self._verify_backend_alignment()

            if self.chat_mode_active:
                self.chat_history.append((text, system_reply))
                if len(self.chat_history) > 20:
                    self.chat_history.pop(0)

            self._save_state()

            if self.interaction_counter % self.reconfigure_interval == 0:
                self._maybe_reconfigure()

            self.tracker.record(self)

            return {
                'ci_report': ci_report,
                'phon_score': phon_score,
                'diagnostic': diag_explanation,
                'token_analyses': token_analyses,
                'curiosity_artifacts': artifacts,
                'meta_weights': self.meta_learner.get_weights(),
                'total_intrinsic_reward': self.reward_tracker.total_reward,
                'episode_reward': self.reward_tracker.episode_reward,
                'factor_vector': factor_dict,
                'generated_plugin_proposals': generated_plugin_proposals,
            }


    def _tokenize(self, text: str) -> List[str]:
            import re

            # --- Base tokenization ---
            # Keep single-char numbers/operators so math + single-letter
            # words survive; the min-length gate below still filters noise
            # once the vocabulary is large enough.
            raw = [t.lower() for t in re.findall(r"[a-zA-Z']+|\d+\.?\d*|[+\-*/=<>%^()]", text)
                   if len(t) >= 2 or t.isdigit() or re.fullmatch(r"[+\-*/=<>%^()]", t) or len(t) == 1]

            # --- State-based min-length gate (adapts as vocabulary grows) ---
            vocab_size = len(self.semantic_learner.word_embeddings) if hasattr(self, 'semantic_learner') else 0
            min_len = 1 if vocab_size < 50 else 2

            tokens = [t for t in raw if len(t) >= min_len]

            # --- Self-developing: learn which token patterns are useful ---
            # Boost tokens that already exist in semantic learner (known = useful)
            known = [t for t in tokens if t in getattr(self, 'semantic_learner', {}).word_embeddings if hasattr(self, 'semantic_learner')]
            unknown = [t for t in tokens if t not in known]

            # If we have very few known tokens, relax filter to include shorter unknowns
            if len(known) == 0 and unknown:
                tokens = [t for t in raw if len(t) >= 1]

            # --- Track tokenization stats for self-improvement ---
            if not hasattr(self, '_tokenizer_stats'):
                self._tokenizer_stats = {'total': 0, 'known_ratio_history': []}
            self._tokenizer_stats['total'] += 1
            if tokens:
                ratio = len(known) / len(tokens)
                self._tokenizer_stats['known_ratio_history'].append(ratio)
                if len(self._tokenizer_stats['known_ratio_history']) > 200:
                    self._tokenizer_stats['known_ratio_history'].pop(0)
            return tokens if tokens else raw
    
    def _generate_artifacts(
        self,
        tokens: List[str],
        token_analyses: List[Dict],
        ci_report: Dict,
        phon_score: float,
        diag: Dict,
        factor_dict: Dict,
    ) -> List[str]:
        """Generate rich artifacts including dynamic factor state."""
        artifacts = []
        artifacts.append(" [Meta-Curiosity Analysis v0.5 – Adaptive Control Space]")
        artifacts.append(f"   Input tokens: {tokens}")

        artifacts.append(f"   • CI confidence: {ci_report.get('confidence', 0.5):.3f} | Stability: {ci_report.get('stability', 0.5):.3f}")
        artifacts.append(f"   • Phonotactic score: {phon_score:.3f}")

        sync_report = self._backend_report_dict(getattr(self, "backend_sync_report", None))
        if sync_report:
            artifacts.append(
                "   • Backend sync: "
                f"ok={sync_report.get('ok')} "
                f"corpus={sync_report.get('corpus_size', 0)} "
                f"added={sync_report.get('added', 0)} "
                f"skipped={sync_report.get('skipped', 0)}"
            )
        verification_report = self._backend_report_dict(getattr(self, "backend_verification_report", None))
        if verification_report:
            artifacts.append(
                "   • Backend verification: "
                f"ok={verification_report.get('ok')} "
                f"aligned={verification_report.get('current_input_aligned', True)}"
            )

        # Diagnostic summary
        artifacts.append(f"   • Diagnostic: {diag.get('summary', 'N/A')}")

        # Meta-learner weights (dynamic)
        weights = self.meta_learner.get_weights()
        artifacts.append("   • Current curiosity priorities (learned):")
        # Show top 5 absolute weights
        sorted_weights = sorted(weights.items(), key=lambda x: -abs(x[1]))
        for name, w in sorted_weights[:5]:
            artifacts.append(f"     - {name}: {w:.3f}")

        # Intrinsic reward
        artifacts.append(f"   • Total intrinsic reward (lifetime): {self.reward_tracker.total_reward:.3f}")
        artifacts.append(f"   • Episode intrinsic reward: {self.reward_tracker.episode_reward:.3f}")

        # ---- Factor state snapshot (now includes dynamic factors) ----
        artifacts.append("   • Current control-signal state:")
        # Show base factors first
        for k in self.control_space.base_factors:
            if k in factor_dict:
                artifacts.append(f"     - {k}: {factor_dict[k]:.3f}")
        # Then show dynamic factors
        for k in self.control_space.dynamic_factors:
            if k in factor_dict:
                artifacts.append(f"     - {k}: {factor_dict[k]:.3f} (dynamic)")
        
        # Check if this state is already known
        control_names = self.control_space.factor_space.factor_names
        vec = np.array([factor_dict.get(k, 0.5) for k in control_names])
        self.control_space.validate_control_vector(vec, factor_names=control_names)
        threshold = self.config_manager.get('engine.state_similarity_threshold', 0.85)
        similar_label = self.state_manager.find_similar_state(
            vec,
            factor_space=self.control_space.factor_space,
            factor_statuses={name: 'observed' if name in factor_dict else 'unavailable'
                             for name in control_names},
            threshold=threshold,
            interaction_index=self.interaction_counter,
            reward=float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0),
        )
        if similar_label:
            artifacts.append(f"   • State recognized as: '{similar_label}'")
        # ------------------------------------

        # Per-token deep analysis (adapt to show all available curiosity factors)
        artifacts.append("   • Token-level curiosity & reasoning:")
        for ta in token_analyses:
            token = ta['token']
            score = ta['curiosity_score']
            entropy = ta['entropy']
            factors = ta['factors']  # CuriosityFactors object
            gp_unc = ta['gp_uncertainty']
            bayes_surp = ta['bayesian_surprise']

            artifacts.append(f"     🔹 '{token}': curiosity={score:.3f} (entropy={entropy:.3f})")
            
            # Show all curiosity factors that are present
            factor_line_parts = []
            for name, val in factors.factor_dict.items():
                if name in ["bayesian_surprise", "gp_uncertainty"]:
                    continue  # show separately
                factor_line_parts.append(f"{name}:{val:.2f}")
            artifacts.append(f"        " + " ".join(factor_line_parts[:6]))  # first 6
            artifacts.append(f"        Bayesian surprise:{bayes_surp:.3f} GP uncertainty:{gp_unc:.3f}")

            hypothesis = self.reasoner.generate_hypothesis(token, {})
            artifacts.append(f"        💭 {hypothesis}")

            if score > 0.4:
                query = self.reasoner.plan_query(token, uncertainty=gp_unc)
                artifacts.append(f"        ❓ {query}")

        total_concepts = len(self.concept_model.concept_counts)
        artifacts.append(f"   • Total latent concepts discovered: {total_concepts}")
        if self.word_history:
            most_uncertain_word = max(self.word_history.keys(),
                                      key=lambda w: self.concept_model.entropy_word(w),
                                      default=None)
            if most_uncertain_word:
                artifacts.append(f"   • Most globally uncertain word: '{most_uncertain_word}' "
                                 f"(entropy={self.concept_model.entropy_word(most_uncertain_word):.3f})")

        return artifacts
    
    def _handle_math_sentence(self, math_result: dict, tokens: list):
        """
        Called when the pattern observer detects a math sentence.
        If all three values (left, op, right, expected_result) are present,
        verify the equation and optionally teach the operator back to the
        pattern observer if the operator word was unknown.
        """
        left = math_result.get('left')
        right = math_result.get('right')
        op = math_result.get('operator')
        expected = math_result.get('expected_result')
        sentence = math_result.get('sentence', '')

        if None in (left, right, op):
            return

        # Attempt to evaluate with known operators
        known_ops = self.reasoner.operators
        if op in known_ops:
            try:
                computed = known_ops[op].evaluate(left, right)
                if expected is not None:
                    match = abs(computed - expected) < 1e-9
                    print(f"🔢 [Math] {sentence} → {left} {op} {right} = {computed}"
                          f" {'✓' if match else '✗ (expected ' + str(expected) + ')'}")
                else:
                    print(f"🔢 [Math] {sentence} → {left} {op} {right} = {computed}")
            except Exception:
                pass
        elif expected is not None:
            # Unknown operator word: infer from values
            inferred = self.reasoner.pattern_observer.infer_operator(left, right, expected)
            if inferred:
                # Teach this word as an operator going forward
                self.reasoner.pattern_observer.teach_math_operator(op, inferred)
                print(f"🔢 [Math] Learned: '{op}' means '{inferred}' "
                      f"(from {left} {op} {right} = {expected})")
    # ------------------------------------------------------------------
    # Idle reflection and introspection (unchanged logic)
    # ------------------------------------------------------------------

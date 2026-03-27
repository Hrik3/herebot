    def _integrate_behavioral_rule(self, rule_id: str):
        """Permanently integrate a confirmed behavioral rule."""
        rule = self.behavioral_learner.rules.get(rule_id)
        if not rule or not rule.confirmed:
            return

        if rule.rule_type == 'identity_alias':
            alias = rule.condition['alias']
            self.identity_aliases.add(alias)
            # Add to semantic learner with embedding similar to "I"
            if 'i' in self.semantic_learner.word_embeddings:
                i_emb = self.semantic_learner.word_embeddings['i']
                self.semantic_learner.initialize_word(alias)
                self.semantic_learner.word_embeddings[alias] = i_emb.copy()
                print(f" [Rule Integration] Learned self-alias: '{alias}'")
            if 'i' in self.semantic_learner.word_embeddings:
                i_emb = self.semantic_learner.word_embeddings['i']
                self.reasoner.store_definition(alias, f"a name for myself", i_emb)
            else:
                self.reasoner.store_definition(alias, f"a name for myself", None)
        self.behavioral_learner.integrate_rule(rule_id)

    def _should_ask_feedback(self, tokens: List[str], factor_dict: Dict, template_id: Optional[str] = None) -> bool:
        """Determine whether to ask for feedback, based on confidence and probability."""
        # Base probability
        base_prob = self.feedback_ask_probability

        # Reduce probability if we have high context confidence
        context_confidence = self.reasoner.get_context_confidence(f"template:{template_id}") if template_id else 0.0
        if context_confidence > 0.8:
            base_prob *= 0.2  # ask much less when confident

        # Also reduce if we asked recently (cooldown)
        if time.time() - self._last_feedback_ask_time < 60:  # at most once per minute
            return False

        if np.random.rand() > base_prob:
            return False

        # Check the corpus if it still needs feedback for this context
        needs = self.reasoner.should_ask_feedback(tokens, factor_dict, template_id)
        if needs:
            self._last_feedback_ask_time = time.time()
        return needs

    def _generate_feedback_question(self) -> str:
        """Generate a neutral question asking for user feedback."""
        templates = [
            "How would you rate my response? (positive/negative/neutral)",
            "Was my answer helpful? (positive/negative/neutral)",
            "Did that make sense? Please rate: positive, negative, or neutral.",
            "How do you feel about my reply? (positive/negative/neutral)",
        ]
        return np.random.choice(templates)

    def build_world_model_transitions(self):
        """Scan episodic memory to build transition counts for the structured world model."""
        episodes = self.episodic_memory.episodes[-500:]
        for i in range(len(episodes)-1):
            ep = episodes[i]
            next_ep = episodes[i+1]
            if 'factor_vector' in ep and 'factor_vector' in next_ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                next_vec = self._align_factor_vector(np.array(list(next_ep['factor_vector'].values())))
                state_label = self.state_manager.find_similar_state(vec, threshold=0.7)
                next_state_label = self.state_manager.find_similar_state(next_vec, threshold=0.7)
                if state_label and next_state_label:
                    action = ep.get('system_response', '')   # crude action representation
                    self.structured_world_model.observe_episode(state_label, action, next_state_label)

    def world_model_dreaming(self):
        """1. Multi‑term dreaming using n‑gram models from episodic memory."""
        if not self.config_manager.get('idle_reasoner.enable_world_model_dreaming', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping World Model Dreaming (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🌙 [World Model Dreaming] Simulating multi‑term synthetic conversations...")

        # Build n‑gram model from episodic memory user inputs (only known words)
        n = self.config_manager.get('idle_reasoner.dreaming_ngram_order', 3)
        ngram_model = defaultdict(Counter)

        episodes = self.episodic_memory.episodes[-200:]   # recent only
        for ep in episodes:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < n:
                continue
            # only keep sentences where all tokens are known (to ensure plausible dreaming)
            if not self._all_tokens_known(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                context = tuple(tokens[i:i+n-1])
                next_word = tokens[i+n-1]
                ngram_model[context][next_word] += 1

        if not ngram_model:
            print("   [Dreaming] Not enough n‑gram data yet.")
            return

        # Generate synthetic sentences
        base_num = self.config_manager.get('idle_reasoner.world_model_num_synthetic', 5)
        num_syn = max(1, int(base_num * throttle))

        vocab_known = list(self.semantic_learner.word_embeddings.keys())
        if not vocab_known:
            return

        for _ in range(num_syn):
            # start with a random known context (first n-1 words)
            contexts = list(ngram_model.keys())
            start_ctx = contexts[np.random.randint(len(contexts))]
            words = list(start_ctx)
            # generate up to 12 words
            for _ in range(12 - n):
                ctx = tuple(words[-(n-1):]) if len(words) >= n-1 else None
                if ctx in ngram_model:
                    next_words = list(ngram_model[ctx].keys())
                    probs = np.array(list(ngram_model[ctx].values()))
                    probs = probs / probs.sum()
                    next_w = np.random.choice(next_words, p=probs)
                    words.append(next_w)
                else:
                    break
            sentence = ' '.join(words)
            # Only dream about sentences where all words are known
            if self._all_tokens_known(words):
                self._simulate_interaction(sentence)

        print("   [Dreaming] Processed {} synthetic multi‑term sentences (throttle={:.2f}, n={}).".format(num_syn, throttle, n))

    def _simulate_interaction(self, text: str):
        """Internal: process a synthetic input without side effects (no console output)."""
        # Minimal version of process_input – only update models, no queries to user.
        tokens = self._tokenize(text)
        self.reasoner.observe_sentence(text, tokens)
        for token in tokens:
            token_ci = self.ci_engine.evaluate(token)
            token_phon, _ = self.phon_engine.score_text(token)
            diag_explanation = self.diag_engine.explain(token_ci, token_phon)
            features = self._extract_features(token, token_ci, token_phon, diag_explanation)
            features_array = self._features_to_array(features)
            concept = self.concept_model.add_word_observation(token, features)
            new_entropy = self.concept_model.entropy_word(token)
            igain = self._information_gain(token, new_entropy)
            self.gp_model.update(features_array, token_ci['confidence'])
            # record history
            self.word_history[token].append({
                'ci_confidence': token_ci['confidence'],
                'phon_score': token_phon,
                'features': features,
                'concept': concept,
                'entropy': new_entropy,
                'information_gain': igain,
            })
            # update semantic learner via context
            # ---- update semantic learner via context ----
            context = [w for w in tokens if w != token]
            if self.chat_mode_active:
                # Aggressive training: update 3 times with higher learning rate
                for _ in range(3):
                    # Temporarily increase learning rate by scaling word count denominator
                    orig_count = self.semantic_learner.word_counts[token]
                    self.semantic_learner.word_counts[token] = max(1, orig_count * 0.5)  # artificially reduce
                    self.semantic_learner.observe_context(token, context)
                    self.semantic_learner.word_counts[token] = orig_count  # restore
            else:
                self.semantic_learner.observe_context(token, context)
        # store a dummy episode (optional)
        self.episodic_memory.add_episode(
            user_input=text,
            system_response="[synthetic]",
            factor_vector={},
            intrinsic_reward=0.0,
            metadata={'synthetic': True}
        )


from ..curioimport import *

class HeartIdleMixin:
    def perform_state_introspection(self):
        """
        Long-idle task:
        1. Refresh probabilistic state relation groups.
        2. Offline learning from episodic memory.
        """
        if not self.config_manager.get('state_manager.legacy_interactive_conflicts', False):
            print("\n [Deep State Introspection] Refreshing probabilistic state groups...")
            self.state_manager.archive_legacy_conflicts(interaction_index=getattr(self, 'interaction_counter', 0))
            observations = self.state_manager.refresh_state_relations(
                interaction_index=getattr(self, 'interaction_counter', 0),
                reward=float(getattr(self.reward_tracker, 'episode_reward', 0.0) or 0.0),
                reasoner=getattr(self, 'reasoner', None),
            )
            groups = self.state_manager.refresh_state_regrouping(
                interaction_index=getattr(self, 'interaction_counter', 0),
                reasoner=getattr(self, 'reasoner', None),
                force=True,
            )
            print(f"    Relation observations refreshed: {len(observations)}.")
            print(f"    Active state groups: {len([g for g in groups.values() if g.get('phase') != 'archived'])}.")
        else:
            print("\n [Deep State Introspection] Examining saved states for conflicts...")
            threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            self.state_manager.refresh_conflicts(threshold=threshold)
            unresolved = self.state_manager.get_unresolved_conflicts()
            if not unresolved:
                print("    No unresolved conflicts found.")
            else:
                print(f"    Found {len(unresolved)} unresolved conflict(s).")
                for conflict in unresolved[:3]:
                    l1, l2 = conflict["labels"]
                    sim = conflict["similarity"]
                    words1 = conflict.get("words1", [])
                    words2 = conflict.get("words2", [])
                    print(f"    Conflict: '{l1}' ↔ '{l2}' (similarity={sim:.3f})")
                    if words1:
                        print(f"      Words that led to '{l1}': {', '.join(words1)}")
                    if words2:
                        print(f"      Words that led to '{l2}': {', '.join(words2)}")
                    shadow_label = self.state_manager.create_shadow_resolution(conflict)
                    self.state_manager.apply_shadow_to_conflict(conflict, shadow_label)
                    print(f"      → Created shadow state '{shadow_label}'. Will ask user at next interaction.")
                self.state_manager._save_conflicts()

        # ---- Offline learning from episodic memory ----
        print("    Consolidating knowledge from recent episodes...")
        episodes = self.episodic_memory.replay_sequence()
        for ep in episodes[-20:]:   # last 20
            user_input = ep['user_input']
            tokens = self._tokenize(user_input)
            for t in tokens:
                context = [w for w in tokens if w != t]
                self.semantic_learner.observe_context(t, context)
            self.syntax_learner.add_sentence(user_input)
        print("    Offline consolidation complete.\n")
#done3
    # =========================================================================
    # IDLE TASK METHODS (called by IdleReasoner)
    # =========================================================================
    def _get_resource_status(self):
        """
        Returns (cpu_ok, mem_ok, throttle_factor, cpu_percent, mem_percent).
        throttle_factor ∈ [0.3, 1.0] – used to reduce workload when resources are strained.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
        except Exception:
            print("psutil failed [throttle mode]")
            # if psutil fails, assume everything is fine (but throttle a little)
            return True, True, 0.8, 0.0, 0.0

        cpu_ok = cpu_percent < 80.0          # CPU below 80%
        mem_ok = mem_percent < 90.0          # memory below 90%

        if not cpu_ok or not mem_ok:
            throttle = 0.3                  # severely reduce workload
        else:
            throttle = 1.0                 # full workload

        return cpu_ok, mem_ok, throttle, cpu_percent, mem_percent

    def _compute_chat_reward_multiplier(self, factor_dict: Dict) -> float:
        """
        Returns a dynamic reward multiplier based on emotional state.
        Used only when chat mode is active.
        Formula: base 1.0 + (curiosity * 0.5) + (trust * 0.3) - (threat * 0.4) + (comprehension * 0.2)
        Clamped to [0.5, 2.0].
        """
        curiosity = factor_dict.get('curiosity', 0.5)
        trust = factor_dict.get('trust', 0.5)
        threat = factor_dict.get('threat', 0.0)
        comprehension = factor_dict.get('comprehension', 0.5)

        multiplier = 1.0 + (curiosity * 0.5) + (trust * 0.3) - (threat * 0.4) + (comprehension * 0.2)
        return np.clip(multiplier, 0.5, 2.0)
    
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
            concept = self.concept_model.add_word_observation(
                token,
                features,
                interaction_index=getattr(self, 'interaction_counter', 0),
                reward=0.0,
            )
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
            metadata={'synthetic': True},
            active_states_at_time=list(getattr(self, "current_state_labels", []) or []),
            provenance="DREAMED",
        )

    def active_memory_rehearsal(self):
        """2. Active Memory Rehearsal Scheduling (CLS theory)."""
        if not self.config_manager.get('idle_reasoner.enable_active_memory_rehearsal', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Active Memory Rehearsal (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Active Memory Rehearsal] Prioritizing episodes...")
        episodes = self.episodic_memory.episodes
        if len(episodes) < 5:
            return
        # Compute a priority score for each episode
        scores = []
        for idx, ep in enumerate(episodes):
            surprise = 0.0
            # Bayesian surprise for words in user_input
            tokens = self._tokenize(ep.get('user_input', ''))
            for t in tokens:
                surprise += self.concept_model.bayesian_surprise(t)
            # rarity: inverse of frequency of state vector (if available)
            rarity = 0.0
            if 'factor_vector' in ep and ep['factor_vector']:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                sims = [1.0 - cdist([vec], [s], 'cosine')[0,0]
                        for s in self.state_manager.states.values() if len(s) == len(vec)]
                if sims:
                    rarity = 1.0 - np.mean(sims)
            # identity drift
            drift = 0.0
            if hasattr(self, 'identity_manager') and 'factor_vector' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                drift = self.identity_manager.detect_drift(vec)
            # unresolved hypothesis bonus
            hyp_bonus = 0.2 if any(h.get('asked') is False for h in self.reasoner.hypotheses) else 0.0
            scores.append(surprise * 0.4 + rarity * 0.3 + drift * 0.2 + hyp_bonus * 0.1)
        # Select top k episodes – scale by throttle
        base_k = self.config_manager.get('idle_reasoner.rehearsal_batch_size', 20)
        k = max(1, int(base_k * throttle))
        top_idx = np.argsort(scores)[-k:]
        print(f"   Replaying {len(top_idx)} high‑priority episodes (throttle={throttle:.2f}).")
        for idx in top_idx:
            ep = episodes[idx]
            user_input = ep.get('user_input', '')
            tokens = self._tokenize(user_input)
            for t in tokens:
                context = [w for w in tokens if w != t]
                self.semantic_learner.observe_context(t, context)
            self.syntax_learner.add_sentence(user_input)

    def curiosity_landscape_optimization(self):
        """3. Curiosity Landscape Optimization (meta‑curiosity gradient analysis)."""
        if not self.config_manager.get('idle_reasoner.enable_curiosity_landscape_optimization', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Curiosity Landscape Optimization (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("📈 [Curiosity Landscape Optimization] Analyzing factor correlations...")
        # collect recent factor vectors and IGs from word_history
        factor_vals = []
        ig_vals = []
        for w, hist in self.word_history.items():
            for entry in hist[-20:]:  # recent
                if 'curiosity_factors' in entry and 'information_gain' in entry:
                    factors = entry['curiosity_factors'].as_array(self.meta_learner.factor_names)
                    factor_vals.append(factors)
                    ig_vals.append(entry['information_gain'])
        if len(factor_vals) < 10:
            return
        factor_matrix = np.array(factor_vals)

        # remove constant columns (zero variance → NaN corr)
        stds = factor_matrix.std(axis=0)
        valid_cols = stds > 1e-6
        factor_matrix = factor_matrix[:, valid_cols]

        if factor_matrix.shape[1] < 2:
            return

        # If throttled, skip the expensive correlation matrix and merging
        if throttle < 0.8:
            print(f"   [Resource Aware] Skipping correlation analysis (throttle={throttle:.2f}).")
            return

        corr = np.corrcoef(factor_matrix.T)

        # detect highly correlated pairs (redundancy)
        thresh = self.config_manager.get('idle_reasoner.curiosity_correlation_threshold', 0.9)
        redundant = []
        valid_indices = np.where(valid_cols)[0]

        for ii in range(len(valid_indices)):
            for jj in range(ii+1, len(valid_indices)):
                i = valid_indices[ii]
                j = valid_indices[jj]
                if corr[ii, jj] > thresh:
                    redundant.append((i, j, corr[ii, jj]))
        # merge weak factors (placeholder)
        for i, j, r in redundant[:2]:  # limit per run
            name_i = self.meta_learner.factor_names[i]
            name_j = self.meta_learner.factor_names[j]
            # merge weights: weighted average by variance
            w_i = self.meta_learner.mu[i]
            w_j = self.meta_learner.mu[j]
            new_w = (w_i + w_j) / 2.0
            # keep first factor, zero second
            self.meta_learner.mu[i] = new_w
            self.meta_learner.mu[j] = 0.0
            print(f"   Merged factor '{name_j}' into '{name_i}' (corr={r:.3f})")
        # spawn new factor from residual unexplained variance
        ig_array = np.array(ig_vals)
        mu = align_vector(self.meta_learner.mu, factor_matrix.shape[1])
        pred_ig = factor_matrix @ mu
        residual = ig_array - pred_ig
        if np.var(residual) > 0.01:
            # simple heuristic: new factor = residual itself (for next iteration)
            new_factor_name = f"residual_{int(time.time())}"
            def residual_compute(token, token_ci, token_phon, diag, features, global_factors):
                # placeholder: returns small random value
                return np.random.rand() * 0.1
            self.meta_learner.add_factor(new_factor_name, initial_mu=0.2, compute_function=residual_compute)
            print(f"   Spawned new curiosity factor '{new_factor_name}' from unexplained variance.")

    def concept_graph_compression(self):
        """4. Concept Graph Compression (Symbolic abstraction formation)."""
        if not self.config_manager.get('idle_reasoner.enable_concept_graph_compression', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Concept Graph Compression (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🗂️ [Concept Graph Compression] Creating higher‑order concepts...")
        # cluster concepts based on word membership
        concept_words = defaultdict(list)
        for word, cnts in self.concept_model.word_concept_counts.items():
            # assign word to its most probable concept
            if cnts:
                concept = max(cnts, key=cnts.get)
                concept_words[concept].append(word)
        # for concepts with many words, create a meta‑concept
        for concept, words in concept_words.items():
            if len(words) >= 3:
                meta_label = f"meta_{'_'.join(words[:2])}"
                # store in reasoning engine as a relation
                for w in words:
                    self.reasoner.relations[w].append(("category", meta_label, 0.8))
                # also add as a definition
                meta_def = f"a group of words: {', '.join(words)}"
                # create a dummy embedding (average of member embeddings)
                emb_list = [self.semantic_learner.word_embeddings.get(w, np.zeros(self.semantic_learner.emb_dim))
                            for w in words if w in self.semantic_learner.word_embeddings]
                if emb_list:
                    avg_emb = np.mean(emb_list, axis=0)
                    self.reasoner.store_definition(meta_label, meta_def, avg_emb)
                print(f"   Created meta‑concept '{meta_label}' from {len(words)} words.")

    def autonomous_hypothesis_planning(self):
        """5. Generate experiments for both words AND sentences."""
        if not self.config_manager.get('idle_reasoner.enable_autonomous_hypothesis_planning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Autonomous Hypothesis Planning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔬 [Autonomous Hypothesis Planning] Generating word and sentence experiments...")

        # --- Word-level candidates (unchanged) ---
        candidates = []
        for word in self.concept_model.word_concept_counts:
            entropy = self.concept_model.entropy_word(word)
            if entropy > 0.5 and word not in self.reasoner.definitions:
                ig_estimate = entropy - 0.1
                candidates.append((word, ig_estimate, "definition"))
        for hyp in self.reasoner.hypotheses[:getattr(self, "_max_hypotheses", 5000)]:
            if hyp.get('type') == 'synonym' and not hyp.get('asked', False):
                candidates.append((hyp['word'], 0.8, "synonym"))

        # --- Sentence-level candidates from episodic memory ---
        sentence_candidates = []
        for ep in self.episodic_memory.episodes[-100:]:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < 3:
                continue
            if not self._all_tokens_known(tokens):
                continue
            # compute a crude "uncertainty" for the sentence (average novelty)
            avg_novelty = 0.0
            count = 0
            for t in tokens:
                if t in self.word_history and self.word_history[t]:
                    last = self.word_history[t][-1]
                    if 'curiosity_factors' in last:
                        avg_novelty += last['curiosity_factors'].factor_dict.get('novelty', 0.5)
                        count += 1
            if count > 0:
                avg_novelty /= count
                if avg_novelty > 0.5:   # uncertain sentence
                    ig_estimate = avg_novelty * 0.5   # heuristic
                    sentence_candidates.append((sent, ig_estimate, "sentence_meaning"))

        # Combine and sort
        all_candidates = candidates + sentence_candidates
        all_candidates.sort(key=lambda x: -x[1])
        self.research_queue = []
        max_queue = max(1, int(10 * throttle))
        for item, ig, typ in all_candidates[:max_queue]:
            if typ == "definition":
                query = f"TPL.ASK.DEF target='{item}'?"
            elif typ == "synonym":
                hyp = next((h for h in self.reasoner.hypotheses if h.get('word')==item and h.get('type')=='synonym'), None)
                if hyp:
                    query = f"TPL.ASK.SYN word='{item}' candidate='{hyp['candidate']}'?"
                else:
                    continue
            elif typ == "sentence_meaning":
                query = f"TPL.ASK.SENT sentence='{item}'?"
            else:
                continue
            self.research_queue.append({
                'query': query,
                'target': item if typ != "sentence_meaning" else "sentence",
                'ig': ig,
                'type': typ,
                'timestamp': time.time()
            })
        print(f"   Research queue now has {len(self.research_queue)} high‑priority queries (throttle={throttle:.2f}).")

    def identity_stabilization(self):
        """6. Identity Stabilization & Self‑Model Consolidation."""
        if not self.config_manager.get('idle_reasoner.enable_identity_stabilization', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Identity Stabilization (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧘 [Identity Stabilization] Consolidating self‑model...")
        # collect recent identity vectors from episodes with positive reward
        for ep in self.episodic_memory.episodes[-50:]:
            if ep.get('intrinsic_reward', 0) > 0.1 and 'factor_vector' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                self.identity_history.append(
                    self._align_vector_to_emb_dim(self._align_factor_vector(vec))
                )
        if len(self.identity_history) < 5:
            return
        # keep last 100
        self.identity_history = self.identity_history[-100:]

        # If throttled heavily, skip k‑means (expensive)
        if throttle < 0.5:
            print(f"   [Resource Aware] Skipping clustering due to low throttle ({throttle:.2f}).")
            # still do a simple moving average
            attractor = np.mean(self.identity_history[-10:], axis=0)
        else:
            # cluster identity vectors
            identity_matrix = np.array(self.identity_history)
            k = min(self.config_manager.get('idle_reasoner.identity_cluster_k', 3), len(identity_matrix))
            try:
                centroids, _ = kmeans2(identity_matrix, k, minit='points')
            except:
                return
            # largest cluster centroid as attractor
            labels = []
            for vec in identity_matrix:
                dists = cdist([vec], centroids, 'euclidean')[0]
                labels.append(np.argmin(dists))
            largest_cluster = np.argmax(np.bincount(labels))
            attractor = centroids[largest_cluster]

        # move identity towards attractor
        alpha = 0.1
        self.identity_manager.identity = (1 - alpha) * self.identity_manager.identity + alpha * attractor
        norm = np.linalg.norm(self.identity_manager.identity)
        if norm > 0:
            self.identity_manager.identity /= norm
        self.identity_manager.confidence = min(1.0, self.identity_manager.confidence + 0.01)
        self.identity_manager.save()
        # detect phase transition
        drift = self.identity_manager.detect_drift(attractor)
        if drift > 0.4:
            print(f"   ⚠️ Identity phase transition detected (drift={drift:.3f}). New attractor formed.")

    def cross_modal_alignment(self):
        """7. Cross-modal representation alignment — now candidate-gated."""
        if not self.config_manager.get('idle_reasoner.enable_cross_modal_alignment', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Cross‑modal Alignment (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        # Contrastive learning is CPU intensive – skip entirely if throttle < 0.5
        if throttle < 0.5:
            print(f"   [Resource Aware] Skipping Cross‑modal Alignment due to low resources (throttle={throttle:.2f}).")
            return

        print("🔄 [Cross‑modal Alignment] Generating candidate associations (not committing)...")
        if not hasattr(self, 'candidate_ledger') or self.candidate_ledger is None:
            from components.candidate_ledger import CandidateLedger
            self.candidate_ledger = CandidateLedger(self.config_manager.get('idle_reasoner.candidate_ledger', {}))

        added = self.candidate_ledger.generate(self.episodic_memory, self._tokenize, window=100)
        print(f"   Generated {added} candidate associations (unconsolidated).")

    def exploration_policy_distillation(self):
        """8. Exploration Policy Distillation."""
        if not self.config_manager.get('idle_reasoner.enable_exploration_policy_distillation', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Exploration Policy Distillation (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        if throttle < 0.4:
            print(f"   [Resource Aware] Skipping policy training (throttle={throttle:.2f}).")
            return

        print("🎯 [Exploration Policy Distillation] Learning when to ask...")
        # We need to collect decision points: episodes where we asked a question vs. not.
        # For simplicity, we classify based on whether system response contains '?'
        X = []
        y = []
        for ep in self.episodic_memory.episodes[-200:]:
            if 'factor_vector' in ep and ep['factor_vector'] and 'system_response' in ep:
                vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                vec = self._normalize_vector(vec)  # use state_manager's normalizer
                X.append(vec)
                response = ep['system_response']
                y.append(1 if '?' in response else 0)
        if len(X) < 20 or len(set(y)) < 2:
            return
        X = np.array(X)
        y = np.array(y)
        # train a simple logistic regression classifier (no sklearn, use scipy.optimize)
        from scipy.optimize import minimize
        def sigmoid(z): return 1 / (1 + np.exp(-z))
        def loss(w, X, y):
            z = X @ w[:-1] + w[-1]
            p = sigmoid(z)
            eps = 1e-12
            return -np.mean(y * np.log(p+eps) + (1-y) * np.log(1-p+eps))
        w0 = np.zeros(X.shape[1] + 1)
        # Use a simpler optimizer when throttled
        if throttle < 0.7:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'
        res = minimize(loss, w0, args=(X, y), method=method)
        self.policy_model = (res.x[:-1], res.x[-1])
        print(f"   Policy distilled. Intercept: {res.x[-1]:.3f} (method={method})")

    def cognitive_drive_evolution(self):
        """9. Cognitive Drive Evolution (Automatic Drive Creation)."""
        if not self.config_manager.get('idle_reasoner.enable_cognitive_drive_evolution', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Cognitive Drive Evolution (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🧬 [Cognitive Drive Evolution] Spawning/removing intrinsic drives...")
        # Remove weak dynamic factors (|mu| < threshold)
        weak_thresh = 0.001
        to_remove = []
        for i, name in enumerate(self.meta_learner.dynamic_factor_names):
            idx = self.meta_learner.factor_names.index(name)
            if abs(self.meta_learner.mu[idx]) < weak_thresh:
                to_remove.append(name)
        # Remove each factor one by one, recomputing the index each time
        for name in to_remove:
            if name not in self.meta_learner.dynamic_factor_names:
                continue
            idx = self.meta_learner.factor_names.index(name)
            self.meta_learner.dynamic_factor_names.remove(name)
            # Remove from arrays
            keep = [i for i in range(len(self.meta_learner.factor_names)) if i != idx]
            self.meta_learner.mu = self.meta_learner.mu[keep]
            self.meta_learner.lambda_ = self.meta_learner.lambda_[keep]
            self.meta_learner.alpha = self.meta_learner.alpha[keep]
            self.meta_learner.beta = self.meta_learner.beta[keep]
            if name in self.meta_learner.dynamic_compute_functions:
                del self.meta_learner.dynamic_compute_functions[name]
            if name in self.meta_learner.factor_metadata:
                del self.meta_learner.factor_metadata[name]
            print(f"   Removed weak drive: '{name}'")
        # Spawn new drive from residual variance (simplified) – reduce probability when throttled
        spawn_prob = 0.2 * throttle
        if np.random.rand() < spawn_prob:
            new_name = f"drive_{int(time.time())}"
            def new_drive_compute(token, token_ci, token_phon, diag, features, global_factors):
                return np.random.rand() * 0.5
            self.meta_learner.add_factor(new_name, initial_mu=0.1, compute_function=new_drive_compute)
            print(f"   Spawned new cognitive drive: '{new_name}' (throttle={throttle:.2f})")    

    def counterfactual_reasoning(self):
        """10. Counterfactual Reasoning Engine (Offline causal imagination)."""
        if not self.config_manager.get('idle_reasoner.enable_counterfactual_reasoning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Counterfactual Reasoning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Counterfactual Reasoning] Simulating 'what if I asked X instead?'...")
        # For each recent episode, pick an alternative word and estimate IG
        recent = self.episodic_memory.episodes[-20:]
        max_eps = max(1, int(len(recent) * throttle))
        for ep in recent[:max_eps]:
            if 'user_input' not in ep:
                continue
            tokens = self._tokenize(ep['user_input'])
            if not tokens:
                continue
            # actual word we might have asked about? Not stored; we just simulate
            alt_word = np.random.choice(tokens)
            entropy = self.concept_model.entropy_word(alt_word)
            # crude IG estimate if we had asked about it
            alt_ig = entropy - 0.1  # assume answer reduces entropy to 0.1
            # compare to actual IG from episode (if available)
            actual_ig = ep.get('intrinsic_reward', 0)  # crude proxy
            if alt_ig > actual_ig + 0.1:
                # alternative would have been better; increase weight of factors active in this state
                if 'factor_vector' in ep and ep['factor_vector']:
                    vec = self._align_factor_vector(np.array(list(ep['factor_vector'].values())))
                    # normalize
                    if np.linalg.norm(vec) > 0:
                        vec = vec / np.linalg.norm(vec)
                    # boost meta‑learner weights for factors with high activation
                    n = len(self.meta_learner.factor_names)

                    vec = align_vector(vec, n)
                    self.meta_learner.mu = align_vector(self.meta_learner.mu, n)

                    for i in range(n):
                        self.meta_learner.mu[i] += 0.01 * vec[i]
        print(f"   Counterfactual updates applied (processed {max_eps} episodes, throttle={throttle:.2f}).")

    def semantic_self_talk(self):
        """11. Advanced self‑talk: trigram + crossover from episodic memory."""
        if not self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Semantic Self‑Talk (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🗣️ [Semantic Self‑Talk] Recombining learned sentences (trigram + crossover)...")

        # Build trigram model from episodic memory (only known words)
        trigram = defaultdict(Counter)
        episodes = self.episodic_memory.episodes[-200:]
        for ep in episodes:
            sent = ep.get('user_input', '')
            tokens = self._tokenize(sent)
            if len(tokens) < 3:
                continue
            if not self._all_tokens_known(tokens):
                continue
            for i in range(len(tokens)-2):
                ctx = tuple(tokens[i:i+2])
                nxt = tokens[i+2]
                trigram[ctx][nxt] += 1

        if not trigram:
            print("   Not enough trigram data yet.")
            return

        # generate sentences
        num_sentences = max(1, int(5 * throttle))
        for _ in range(num_sentences):
            # either pure trigram walk (70%) or crossover (30%)
            if np.random.rand() < 0.7:
                # trigram walk
                keys = list(trigram.keys())
                ctx = keys[np.random.randint(len(keys))]
                words = list(ctx)
                for _ in range(np.random.randint(5, 12)):
                    ctx_key = tuple(words[-2:])
                    if ctx_key in trigram:
                        next_words = list(trigram[ctx_key].keys())
                        probs = np.array(list(trigram[ctx_key].values()))
                        probs = probs / probs.sum()
                        words.append(np.random.choice(next_words, p=probs))
                    else:
                        break
            else:
                # crossover: take prefix of one episode, suffix of another
                if len(episodes) >= 2:
                    idx = np.random.choice(len(episodes), 2, replace=False)
                    ep1 = episodes[idx[0]]
                    ep2 = episodes[idx[1]]
                    sent1 = ep1.get('user_input', '')
                    sent2 = ep2.get('user_input', '')
                    tok1 = self._tokenize(sent1)
                    tok2 = self._tokenize(sent2)
                    if len(tok1) >= 2 and len(tok2) >= 2:
                        split = np.random.randint(1, min(len(tok1), len(tok2)))
                        words = tok1[:split] + tok2[split:]
                    else:
                        words = tok1[:2] + tok2[-2:]
                else:
                    continue

            sentence = ' '.join(words)
            # only learn if all words are known
            if self._all_tokens_known(words):
                self.syntax_learner.add_sentence(sentence)
                for token in words:
                    context = [w for w in words if w != token]
                    self.semantic_learner.observe_context(token, context)
                print(f"   Generated and learned: '{sentence[:70]}...'")
        print(f"   Generated {num_sentences} sentences (throttle={throttle:.2f}).")

    def state_space_topology_learning(self):
        """12. State‑space topology learning (cognitive manifold discovery)."""
        if not self.config_manager.get('idle_reasoner.enable_state_space_topology_learning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping State‑space Topology Learning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        if throttle < 0.6:
            print(f"   [Resource Aware] Skipping topology learning (throttle={throttle:.2f}).")
            return

        print("🌐 [State‑space Topology] Discovering attractors and macro‑states...")
        # collect all state vectors from state_manager
        all_vectors = []
        labels = []
        for label, vec in self.state_manager.states.items():
            all_vectors.append(self._normalize_vector(vec))
            labels.append(label)
        if len(all_vectors) < 4:
            return
        all_vectors = np.array(all_vectors)
        n_clusters = min(self.config_manager.get('idle_reasoner.topology_n_clusters', 5), len(all_vectors))
        try:
            centroids, assignments = kmeans2(all_vectors, n_clusters, minit='points')
        except:
            return
        # create macro‑state for each centroid
        for i, cent in enumerate(centroids):
            # find members
            members = [labels[j] for j, a in enumerate(assignments) if a == i]
            if len(members) >= 2:
                macro_name = f"macro_{i}_{int(time.time())}"
                # save as a shadow state (macro‑state)
                self.state_manager.save_state(macro_name, cent, metadata={'members': members}, is_shadow=True)
                print(f"   Created macro‑state '{macro_name}' from {len(members)} states.")
        self.state_manifold = centroids

    def knowledge_gap_frontier_detection(self):
        """13. Knowledge gap frontier detection."""
        if not self.config_manager.get('idle_reasoner.enable_knowledge_gap_frontier', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Knowledge Gap Frontier (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🌍 [Knowledge Gap Frontier] Updating research agenda...")
        # recompute IG for all candidate words (high entropy, no definition)
        frontiers = []
        for word in self.concept_model.word_concept_counts:
            if word not in self.reasoner.definitions:
                entropy = self.concept_model.entropy_word(word)
                if entropy > 0.3:
                    ig = entropy - 0.1
                    frontiers.append((word, ig, "definition"))
        # also include unresolved hypotheses
        for hyp in self.reasoner.hypotheses:
            if not hyp.get('asked', False):
                word = hyp.get('word')
                if word:
                    frontiers.append((word, 0.8, "hypothesis"))
        # sort and keep top N, scaled by throttle
        frontiers.sort(key=lambda x: -x[1])
        self.research_queue = []
        max_items = max(1, int(20 * throttle))
        for word, ig, typ in frontiers[:max_items]:
            if typ == "definition":
                query = f"TPL.ASK.DEF target='{word}'?"
            else:
                query = f"TPL.ASK.CLARIFY target='{word}'?"
            self.research_queue.append({
                'query': query,
                'target': word,
                'ig': ig,
                'type': typ,
                'timestamp': time.time()
            })
        print(f"   Research queue updated: {len(self.research_queue)} items (throttle={throttle:.2f}).")

    def social_cognition_simulation(self):
        """14. Social cognition simulation (theory of mind dreaming)."""
        if not self.config_manager.get('idle_reasoner.enable_social_cognition_simulation', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Social Cognition Simulation (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("[Social Cognition] Simulating user responses...")
        # collect feedback events (stored in reasoner.hypotheses with type 'shadow_trust')
        pos_features = []
        neg_features = []
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') == 'shadow_trust' and 'utterance' in hyp:
                # we don't have state vector here; we need to store it with the hypothesis
                # For simplicity, we just simulate
                pass
        # Instead, use the user_model attributes already defined
        # if we have positive and negative examples, train a naive mean classifier
        # This is a placeholder: we simply record the mean state for positive feedback.
        # In a full system, you would store state vectors when feedback is received.
        # Here we simulate: if we have episodes where feedback was positive (from earlier handling)
        # we update user_model.
        # Since we don't have explicit feedback storage, we skip detailed training.
        print("   (User model update placeholder – would need feedback storage)")

    def long_horizon_planning_rehearsal(self):
        """15. Long‑horizon cognitive planning rehearsal."""
        if not self.config_manager.get('idle_reasoner.enable_long_horizon_planning', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Long‑horizon Planning (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("[Long‑horizon Planning] Simulating multi‑turn dialogues...")
        # simple rollout: start from a random state, simulate turns, accumulate reward
        if not self.state_manager.states:
            return
        # pick a random start state
        start_label = np.random.choice(list(self.state_manager.states.keys()))
        state_vec = self.state_manager.states[start_label].copy()
        total_reward = 0.0
        # reduce number of turns when throttled
        num_turns = max(1, int(3 * throttle))
        for _ in range(num_turns):
            # decide action: ask about most uncertain word? (simplified)
            # find word with highest entropy
            if self.concept_model.word_concept_counts:
                word = max(self.concept_model.word_concept_counts.keys(),
                           key=lambda w: self.concept_model.entropy_word(w))
            else:
                break
            # simulate asking
            ig_estimate = self.concept_model.entropy_word(word) - 0.1
            total_reward += ig_estimate
            # update state (crude: move towards identity)
            aligned_state = self._align_vector_to_emb_dim(state_vec)
            aligned_identity = self._align_vector_to_emb_dim(self.identity_manager.identity)

            state_vec = 0.9 * aligned_state + 0.1 * aligned_identity
            # normalize
            if np.linalg.norm(state_vec) > 0:
                state_vec = state_vec / np.linalg.norm(state_vec)
        print(f"   Simulated {num_turns}‑turn conversation, total IG = {total_reward:.3f} (throttle={throttle:.2f})")

    def behavioral_rule_discovery(self):
        """16. Dream about behavioral rules and propose candidates."""
        if not self.config_manager.get('idle_reasoner.enable_behavioral_rule_discovery', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Behavioral Rule Discovery (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print(" [Behavioral Rule Discovery] Dreaming about new rules...")
        candidates = self.behavioral_learner.propose_new_rules(self)
        if candidates:
            for rule in candidates:
                print(f"   ➕ Proposed new rule: {rule.rule_type} - {rule.condition} (confidence {rule.confidence:.2f})")
        else:
            print("   No new rule candidates found.")
        self.behavioral_learner.decay_confidences()

    def generate_logic_from_experience(self):
        filepath = None
        """Create new logic rules based on episodic patterns and write to ./logic."""
        if not self.config_manager.get('idle_reasoner.enable_logic_inheritance', True):
            return
        # Generate from highly confident behavioral rules that are not yet conscience beliefs
        candidates = []
        for rule in self.behavioral_learner.rules.values():
            if rule.confirmed and rule.confidence > 0.9 and rule.rule_id not in [b.belief_id for b in self.conscience_manager.beliefs.values()]:
                candidates.append(rule)
        if candidates:
            rule = max(candidates, key=lambda r: r.confidence)
            logic_rule = {
                'description': f"Behavioral rule: {rule.condition} -> {rule.action}",
                'condition': rule.condition,
                'action': rule.action,
                'ethical': True if rule.rule_type == 'identity_alias' else False,
                'source': 'behavioral_learner'
            }
            filepath = self.logic_inheritance.generate_logic_file(logic_rule)
            self.logic_inheritance.generate_doubt(logic_rule['logic_id'])
        if filepath is not None:
            print(f"📜 [Logic Generation] New logic rule written to {filepath}, doubt queued.")
        else:
            print("📜 [Logic Generation] No new rule candidates found.")

    def contextual_sentiment_analysis(self):
        """17. Deep analysis of feedback corpus – emergent sentiment learning."""
        if not self.config_manager.get('idle_reasoner.enable_contextual_sentiment_analysis', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping Contextual Sentiment Analysis (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("📊 [Contextual Sentiment Analysis] Mining feedback corpus for emergent patterns...")
        # No explicit analysis needed; the corpus updates models incrementally.
        # We just print a summary of confident contexts.
        confident_contexts = 0
        for ctx, models in self.reasoner.feedback_corpus.context_models.items():
            confidence = self.reasoner.feedback_corpus.get_confidence(ctx)
            if confidence >= ContextualSentimentCorpus.CONFIDENCE_THRESHOLD:
                confident_contexts += 1
        total_entries = len(self.reasoner.feedback_corpus.entries)
        print(f"   Feedback entries: {total_entries}, confident contexts: {confident_contexts}.")

    def autonomous_code_generation_and_testing(self):
        """
        Idle task: select promising logic candidates (high‑confidence behavioral rules,
        unresolved hypotheses, new state definitions) and generate + shadow‑test code.
        Does NOT integrate; only creates doubts for user confirmation.
        """
        if not self.config_manager.get('idle_reasoner.enable_logic_inheritance', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping autonomous code generation (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print(" [Autonomous Code Generation] Selecting candidates...")
        candidates = []

        # 1. High‑confidence behavioral rules that are confirmed but not yet coded
        for rule in self.behavioral_learner.rules.values():
            if rule.confirmed and rule.confidence > 0.85:
                if rule.rule_id not in self.logic_inheritance.generated_code_paths:
                    candidates.append(("behavioral", rule))

        # 2. Synonym hypotheses with strong similarity (>0.9) not yet asked
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') == 'synonym' and hyp.get('similarity', 0) > 0.9:
                if not hyp.get('asked', False):
                    # Use a unique logic_id based on word and candidate
                    logic_id = f"synonym_{hyp['word']}_{hyp['candidate']}"
                    if logic_id not in self.logic_inheritance.generated_code_paths:
                        candidates.append(("semantic", hyp, logic_id))

        # 3. Recently labeled states (if they have a stored embedding)
        if self.current_state_labels and self.current_state_embedding is not None:
            state_id = f"state_{self.current_state_labels[0]}"
            if state_id not in self.logic_inheritance.generated_code_paths:
                state_rule = {
                    "logic_id": state_id,
                    "rule_type": "state_definition",
                    "label": self.current_state_labels[0],
                    "prototype": self.current_state_embedding.tolist(),
                    "description": f"Recognize the '{self.current_state_labels[0]}' state"
                }
                candidates.append(("state", state_rule, state_id))

        if not candidates:
            print("   No candidates for code generation.")
            return

        # Process up to 2 candidates per idle cycle
        for item in candidates[:2]:
            if len(item) == 3:
                cand_type, cand, logic_id = item
            else:
                cand_type, cand = item
                logic_id = cand.rule_id if cand_type == "behavioral" else None

            rule_dict = None
            if cand_type == "behavioral":
                rule_dict = {
                    "rule_type": cand.rule_type,
                    "condition": cand.condition,
                    "action": cand.action,
                    "description": f"Behavioral rule: {cand.condition} -> {cand.action}"
                }
            elif cand_type == "semantic":
                word = cand.get('word', 'unknown')
                candidate = cand.get('candidate', 'unknown')
                similarity = cand.get('similarity', 0.0)
                rule_dict = {
                    "type": "synonym",
                    "word": word,
                    "candidate": candidate,
                    "similarity": similarity,
                    "description": f"Synonym hypothesis: {word} ≈ {candidate}"
                }
            elif cand_type == "state":
                rule_dict = cand  # already a dict

            if not rule_dict:
                continue

            # Step 1: Create a logic JSON file and get the assigned logic_id
            filepath_json = self.logic_inheritance.generate_logic_file(rule_dict)
            if not filepath_json:
                continue
            # The rule_dict now has 'logic_id' added by generate_logic_file
            logic_id = rule_dict['logic_id']

            # Step 2: Create a doubt (adds to pending_doubts)
            doubt = self.logic_inheritance.generate_doubt(logic_id)
            if not doubt:
                continue
            print(f"   ❓ Doubt created for {logic_id}")

            # Step 3: Generate code file (will use the rule from pending_doubts)
            filepath_py = self.logic_inheritance.generate_code_for_logic(logic_id)
            if not filepath_py:
                continue

            # Step 4: Shadow test
            test_result = self.logic_inheritance.shadow_test_generated_code(logic_id)
            if test_result["success"]:
                print(f"   ✅ Generated & tested: {logic_id}")
            else:
                print(f"   ⚠️ Generated {logic_id} but shadow test failed: {test_result['error'][:50]}")
                # Error already appended as comment in the .py file

    def deep_behavioral_pattern_mining(self):
        """18. Deep offline mining of conversational patterns."""
        if not self.config_manager.get('idle_reasoner.enable_behavioral_rule_discovery', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping deep pattern mining (CPU:{cpu_p:.1f}%  MEM:{mem_p:.1f}%).")
            return

        print("🔍 [Deep Pattern Mining] Mining long‑range conversational patterns...")
        candidates = self.behavioral_learner.deep_pattern_mining(self)
        if candidates:
            for rule in candidates:
                self.behavioral_learner.add_candidate_rule(rule)
                print(f"   ➕ Discovered pattern: {rule.condition} -> {rule.action}")
        else:
            print("   No new deep patterns found.")

    def propagate_word_state_associations(self):
        """
        During idle, use semantic similarity to propagate word-state labels.
        If two words have high embedding similarity, they likely share labels.
        """
        if not self.config_manager.get('idle_reasoner.enable_word_state_propagation', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping word-state propagation (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🔄 [Word-State Propagation] Staging candidate labels via semantic similarity...")
        assoc = self.reasoner.word_state
        # Get all words that have embeddings
        all_words = list(self.semantic_learner.word_embeddings.keys())
        if len(all_words) < 5:
            return

        # Compute pairwise similarities for a sample of words (to avoid O(n^2))
        import random
        sample = random.sample(all_words, min(50, len(all_words)))
        similarities = {}
        for w1 in sample:
            for w2 in sample:
                if w1 >= w2:
                    continue
                sim = self.semantic_learner.similarity(w1, w2)
                if sim > 0.85:  # high similarity (stricter threshold for un-PMI-gated channel)
                    similarities[(w1, w2)] = sim

        if not hasattr(self, 'candidate_ledger') or self.candidate_ledger is None:
            from components.candidate_ledger import CandidateLedger
            self.candidate_ledger = CandidateLedger(self.config_manager.get('idle_reasoner.candidate_ledger', {}))

        # Propagate candidate labels between similar words
        staged = 0
        for (w1, w2), sim in similarities.items():
            labels1 = set(assoc.get_labels(w1, threshold=0.3))
            labels2 = set(assoc.get_labels(w2, threshold=0.3))
            ep_id = f"similarity_prop_{w1}_{w2}_{int(time.time())}"
            for src_word, src_labels, dst_labels, dst_word in (
                (w1, labels1, labels2, w2), (w2, labels2, labels1, w1)
            ):
                for lab in src_labels:
                    if lab in dst_labels:
                        continue
                    from components.candidate_ledger import CandidateAssociation
                    self.candidate_ledger.candidates[(dst_word, lab)].append(CandidateAssociation(
                        token=dst_word,
                        label=lab,
                        episode_id=ep_id,
                        timestamp=time.time(),
                        provenance='INFERRED',
                        state_diversity=2,
                        pmi=sim,
                        raw_weight=sim * 0.5,
                        was_broadcast_winner=False,
                    ))
                    staged += 1
        print(f"   Staged {staged} candidate propagations (unconsolidated).")

        # Check for conflicts (words with multiple labels of similar confidence)
        for word in all_words:
            conflicts = assoc.detect_conflicts(word, interaction_index=getattr(self, 'interaction_counter', 0))
            if conflicts:
                print(f"   ⚠️ Word '{word}' has ambiguous labels: {conflicts}. (Will ask user later)")
                # We could add a hypothesis to ask the user, but for now just note.

    def consolidate_candidates(self):
        """Consolidate staged candidate associations using grounded-evidence and belief-change gates."""
        if not hasattr(self, 'candidate_ledger') or self.candidate_ledger is None:
            return
        self.candidate_ledger.age_and_prune()
        committed = self.candidate_ledger.consolidate(
            self.reasoner.word_state, interaction_index=getattr(self, 'interaction_counter', 0)
        )
        for token, label, evidence, margin, kl in committed:
            print(f"   🧠 [Consolidated] '{token}' -> '{label}' "
                  f"(evidence={evidence:.3f}, margin={margin:.3f}, kl={kl:.3f})")
        if committed:
            self.save_word_state_immediately()
        self.candidate_ledger.save(os.path.join(STATE_DIR, 'candidate_associations.json'))

    def mine_phrase_relations(self):
        """
        During idle, find similar phrases in episodic memory and hypothesize relations.
        """
        if not self.config_manager.get('idle_reasoner.enable_semantic_self_talk', True):
            return
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping phrase relation mining (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🔍 [Phrase Relation Mining] Finding similar phrases...")
        # Collect recent user inputs from episodic memory
        episodes = self.episodic_memory.episodes[-200:]
        phrases = []
        for ep in episodes:
            user_input = ep.get('user_input', '')
            if user_input and len(user_input.split()) <= 10:  # phrase up to 10 words
                phrases.append(user_input)

        if len(phrases) < 5:
            return

        # Compute embeddings for these phrases (if possible)
        detector = getattr(getattr(self.reasoner, 'pattern_observer', None), 'detector', None)
        phrase_embs = {}
        for p in phrases:
            emb = self.semantic_learner.text_to_embedding(p, fallback_to_char=True, structural_detector=detector)
            if emb is not None:
                phrase_embs[p] = emb

        # Compare pairwise (sample to avoid O(n^2))
        import random
        items = list(phrase_embs.items())
        if len(items) > 30:
            items = random.sample(items, 30)
        for i, (p1, emb1) in enumerate(items):
            for p2, emb2 in items[i+1:]:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2)+1e-12)
                if sim > 0.85:  # high similarity
                    # Check if we already have a relation or hypothesis
                    key = f"phrase_similar:{p1}:{p2}"
                    if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                        hyp = {
                            'key': key,
                            'type': 'phrase_similarity',
                            'phrase1': p1,
                            'phrase2': p2,
                            'similarity': sim,
                            'text': f"I notice that '{p1}' and '{p2}' seem similar. Are they related?",
                            'asked': False,
                            'provenance': 'idle_mining',
                            'evidence_count': 1,
                        }
                        self.reasoner.hypotheses.append(hyp)
                        print(f"   ➕ Proposed phrase similarity: '{p1}' ↔ '{p2}' (sim={sim:.2f})")

    def converge_knowledge_graph(self):
        """
        Idle-time consolidation of the knowledge graph:
        - Infer transitive relations (is_a, means)
        - Merge nodes that are synonyms or highly similar
        - Prune weak edges
        - Detect central nodes for curiosity prioritization
        """
        if not hasattr(self.reasoner, 'knowledge_graph'):
            return
        kg = self.reasoner.knowledge_graph

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping knowledge graph convergence (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🕸️ [Knowledge Graph] Converging...")

        # 1. Infer transitive relations for is_a and means
        for rel in ['is_a', 'means']:
            inferred = kg.infer_transitive_relations(rel, threshold=0.6)
            for n1, n2, rt, conf in inferred:
                kg.add_edge(n1, n2, rt, conf)
                print(f"   Inferred {rt}: {kg.nodes[n1]['label']} -> {kg.nodes[n2]['label']} (conf={conf:.2f})")

        # 2. Merge synonym candidates (if two words have very high similarity in embeddings)
        #    This is a placeholder – you could use semantic similarity or co-occurrence.
        #    For now, we'll skip because merging requires careful handling.

        # 3. Prune weak edges (confidence < 0.1)
        kg.prune_weak_edges(threshold=0.1)
        print("   Pruned weak edges.")
        self.reasoner.run_constraint_propagation()
        # 4. Identify central nodes to influence curiosity (store in a separate structure)
        central = kg.get_central_nodes(top_k=10)
        self.central_graph_nodes = central
        print("   Top central nodes: " + ", ".join(f"{kg.nodes[nid]['label']}({deg})" for nid, deg in central))

        kg._save()

    def form_meta_concepts_from_concepts(self):
        """Call the concept model's meta‑concept formation."""
        if hasattr(self.concept_model, 'form_meta_concepts'):
            self.concept_model.form_meta_concepts()

    def perform_all_idle_tasks(self):
        """Orchestrate all idle‑time tasks, respecting configuration."""
        now = time.time()
        # avoid running all tasks every idle cycle; run a random subset
        if now - self._idle_tasks_last_run < 300:  # at most every 5 minutes
            return
        self._idle_tasks_last_run = now

        # Global resource check – if system is overloaded, skip everything
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"\n⚠️ [Idle Session] Skipped – resources too low (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%)")
            return

        print("\n" + "="*80)
        print(" [Herebot Idle Session] Running deep consolidation tasks...")
        print("="*80)
        tasks = [
            self.world_model_dreaming,
            self.active_memory_rehearsal,
            self.curiosity_landscape_optimization,
            self.concept_graph_compression,
            self.autonomous_hypothesis_planning,
            self.identity_stabilization,
            self.cross_modal_alignment,
            self.exploration_policy_distillation,
            self.cognitive_drive_evolution,
            self.counterfactual_reasoning,
            self.semantic_self_talk,
            self.state_space_topology_learning,
            self.knowledge_gap_frontier_detection,
            self.social_cognition_simulation,
            self.long_horizon_planning_rehearsal,
            self.behavioral_rule_discovery,
            self.generate_logic_from_experience,
            self.autonomous_code_generation_and_testing,
            self.contextual_sentiment_analysis,
            self.deep_behavioral_pattern_mining,
            self.propagate_word_state_associations,
            self.consolidate_candidates,
            self.mine_phrase_relations,
            self._metabolic_maintenance,
            self.converge_knowledge_graph,
            self.train_language_model,
            self.propose_lm_improvement,
            self.reasoner.run_symbolic_inference,
            self.form_meta_concepts_from_concepts,
            self.build_world_model_transitions,            
        ]
        for task in tasks:
            try:
                task()
            except Exception:
                print(f"   [ERROR in {task.__name__}]")
                print(traceback.format_exc())
            # Delay between tasks – longer when throttled
            time.sleep(max(0.05, 0.2 * throttle))

        print("="*80 + "\n")

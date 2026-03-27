    def _handle_state_learning(self, factor_vector: np.ndarray, metadata: dict):
        """
        Compare current factor vector with known states. Update active states list.
        """
        if factor_vector is None or len(factor_vector) == 0:
            print("   [State Learning] Empty factor vector – skipping.")
            return

        if time.time() - self._last_state_learning_time < 30:
            return

        sim_threshold = self.config_manager.get('engine.state_similarity_threshold', 0.97)
        n_states = max(1, len(self.state_manager.states))
        maturity = np.clip(np.log1p(n_states) / 5.0, 0.2, 1.0)
        novelty_dist_threshold = 0.35 * (1.2 - maturity)

        # Find all states above similarity threshold
        similar = self.state_manager.find_similar_states(factor_vector, threshold=sim_threshold)

        # Apply meta novelty gate per candidate – keep only those within distance threshold
        active_labels = []
        active_embeddings = []
        for label, sim in similar:
            stored_vec = self.state_manager.states.get(label)
            if stored_vec is not None:
                v1 = self.state_manager._normalize_vector(factor_vector)
                v2 = self.state_manager._normalize_vector(stored_vec)
                if v1.shape != v2.shape:
                    max_len = max(len(v1), len(v2))
                    v1 = np.pad(v1, (0, max_len - len(v1)))
                    v2 = np.pad(v2, (0, max_len - len(v2)))
                dist = np.linalg.norm(v1 - v2)
                if dist <= novelty_dist_threshold:
                    active_labels.append(label)
                    active_embeddings.append(self._align_vector_to_emb_dim(v2))

        if active_labels:
            print(f"\n🔍 [State Recognition] This state matches: {', '.join(active_labels)}.")
            # Combine embeddings (average)
            combined = np.mean(active_embeddings, axis=0) if active_embeddings else None
            self._update_state_embedding(active_labels, combined)
            return

        # No close match – create new state
        vec_variance = np.var(factor_vector)
        if vec_variance < 0.05:
            print("   [State Learning] Factor vector too uniform – skipping state creation.")
            return

        print("\n [State Learning] I encountered a new internal state.")
        print("    Factor vector: " + " ".join(f"{v:.2f}" for v in factor_vector))
        print("    How do humans call this state? (e.g., 'confident exploration', 'cautious inquiry')")
        label = input("    🏷️  Label: ").strip()
        if not label:
            label = "unnamed_state"

        metadata['tokens'] = self._tokenize(metadata.get('input_text', ''))
        self.state_manager.save_state(label, factor_vector, metadata)
        print(f"    ✅ State saved as '{label}'.")
        self._last_state_learning_time = time.time()
        self._reinforce_identity(factor_vector, reward_signal=0.2)
        self._update_state_embedding([label], self._align_vector_to_emb_dim(factor_vector))

    # ------------------------------------------------------------------
    def _reinforce_identity(self, factor_vector: np.ndarray, reward_signal: float):
        """
        Reinforce the self-identity when a positive interaction occurs.
        Called after successful definition learning, hypothesis confirmation,
        or when user provides a label for a state.
        """
        self.identity_manager.reinforce(factor_vector, reward_signal)
        
        # Optionally log identity drift
        drift = self.identity_manager.detect_drift(factor_vector)
        if drift > 0.5:
            print(f" [Identity Drift] Current state diverges from learned identity (drift={drift:.3f}).")
    # ------------------------------------------------------------------
    # State embedding helpers (persistent latent concept)
    # ------------------------------------------------------------------
    def _align_vector_to_emb_dim(self, vector: np.ndarray) -> np.ndarray:
        """Truncate or pad a vector to match the semantic learner's embedding dimension."""
        target_dim = self.semantic_learner.emb_dim
        if len(vector) == target_dim:
            return vector.copy()
        elif len(vector) > target_dim:
            return vector[:target_dim].copy()
        else:
            # pad with zeros
            padded = np.zeros(target_dim)
            padded[:len(vector)] = vector
            return padded

    def _compute_state_relevance(self,
                                 token: str,
                                 token_ci: Dict,
                                 token_phon: float,
                                 diag: Dict,
                                 features_array: np.ndarray,
                                 global_factor_dict: Optional[Dict] = None) -> float:
        """
        Dynamic curiosity factor: similarity between the token's embedding
        and the current state embedding.
        Returns 0.0 if no state embedding or token embedding is available.
        """
        if self.current_state_embedding is None:
            return 0.0
        token_emb = self.semantic_learner.word_embeddings.get(token)
        if token_emb is None:
            return 0.0

        # align both embeddings to the same dimension (emb_dim)
        state_emb = self._align_vector_to_emb_dim(self.current_state_embedding)
        token_emb = self._align_vector_to_emb_dim(token_emb)

        norm_state = np.linalg.norm(state_emb)
        norm_token = np.linalg.norm(token_emb)
        if norm_state == 0 or norm_token == 0:
            return 0.0
        sim = np.dot(state_emb, token_emb) / (norm_state * norm_token)
        # scale from [-1,1] to [0,1]
        # update state usage
        if self.current_state_labels:
            current_label = self.current_state_labels[0]  # use first active state for metadata
            meta = self.state_manager.state_metadata.get(current_label)
            if meta:
                now = time.time()
                if meta.get('last_used'):
                    dt = now - meta['last_used']
                    meta['recent_usage'] *= 0.99 ** dt
                meta['recent_usage'] += 0.05
                meta['last_used'] = now
                self.state_manager._save_state_metadata()
        return (sim + 1.0) * 0.5

    def _update_state_embedding(self, labels: List[str], combined_vector: Optional[np.ndarray] = None):
        """
        Set the current active state labels and combined embedding.
        If combined_vector is None, compute it as the average of each label's embedding.
        Also store each label's embedding in the semantic learner.
        """
        if combined_vector is None:
            # compute average of embeddings for all labels
            embs = []
            for lbl in labels:
                if lbl in self.semantic_learner.word_embeddings:
                    embs.append(self.semantic_learner.word_embeddings[lbl])
                else:
                    # label not yet in semantic learner – create it
                    self.semantic_learner.initialize_word(lbl)
                    embs.append(self.semantic_learner.word_embeddings[lbl])
            if embs:
                combined = np.mean(embs, axis=0)
            else:
                combined = np.zeros(self.semantic_learner.emb_dim)
        else:
            combined = combined_vector

        # Ensure each label has an embedding in semantic learner
        for lbl in labels:
            if lbl not in self.semantic_learner.word_embeddings:
                self.semantic_learner.initialize_word(lbl)
                # use the combined vector as a rough embedding (or could use a stored state vector)
                if lbl in self.state_manager.states:
                    vec = self.state_manager.states[lbl]
                    self.semantic_learner.word_embeddings[lbl] = self._align_vector_to_emb_dim(vec)

        self.current_state_labels = labels
        self.current_state_embedding = self._align_vector_to_emb_dim(combined)

    def perform_state_introspection(self):
        """
        Long-idle task:
        1. Detect conflicts among saved states, create shadow resolutions.
        2. Offline learning from episodic memory.
        """
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
            # if psutil fails, assume everything is fine (but throttle a little)
            return True, True, 0.8, 0.0, 0.0

        cpu_ok = cpu_percent < 80.0          # CPU below 80%
        mem_ok = mem_percent < 90.0          # memory below 90%

        if not cpu_ok or not mem_ok:
            throttle = 0.3                  # severely reduce workload
        else:
            throttle = 1.0                 # full workload

        return cpu_ok, mem_ok, throttle, cpu_percent, mem_percent

from curioimport import Dict,List,np,Callable,CuriosityFactors,Optional,math,traceback,align_vector

    def _align_factor_vector(self, vec: np.ndarray) -> np.ndarray:
        return align_vector(vec, len(self.meta_learner.factor_names))
    # ------------------------------------------------------------------
    # Feature extraction helpers (unchanged)
    # ------------------------------------------------------------------
    def _extract_features(self, token: str, ci_report: Dict, phon_score: float, diag: Dict) -> Dict[str, float]:
        """Convert token analysis into a feature vector for the concept model."""
        features = {
            'confidence': ci_report.get('confidence', 0.5),
            'stability': ci_report.get('stability', 0.5),
            'phon_score': phon_score,
            'char_valid': ci_report.get('mean', 0.5),
            'structure': phon_score,
        }
        for neg in diag.get('negatives', []):
            if 'character' in neg.lower():
                features['char_valid'] = 0.1
            if 'vowel' in neg.lower():
                features['structure'] = 0.2
        return features

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to fixed-length array for GP."""
        keys = ['confidence', 'stability', 'phon_score', 'char_valid', 'structure']
        return np.array([features.get(k, 0.5) for k in keys])

    def _apply_knowledge_graph_curiosity(self, token: str, base_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Modify curiosity factors based on knowledge graph structure.
        - If token has many connections (high degree), boost novelty and uncertainty.
        - If token is a bridge between clusters, boost curiosity further.
        """
        # Get node ID for token
        nid = self.reasoner.knowledge_graph.get_node_id(token, "word")
        if nid is None:
            return base_factors

        # Compute degree
        neighbors = self.reasoner.knowledge_graph.get_neighbors(nid)
        degree = len(neighbors)
        # Normalize degree (max expected degree maybe 100)
        degree_factor = min(1.0, degree / 50.0)

        # Boost novelty and uncertainty based on degree
        base_factors['novelty'] = min(1.0, base_factors.get('novelty', 0.5) + 0.2 * degree_factor)
        base_factors['uncertainty'] = min(1.0, base_factors.get('uncertainty', 0.5) + 0.15 * degree_factor)
        # Boost surprise a little
        base_factors['surprise'] = min(1.0, base_factors.get('surprise', 0.5) + 0.1 * degree_factor)

        # Optionally, if token appears in many different relation types, boost more
        rel_types = set()
        for _, rt, _ in neighbors:
            rel_types.add(rt)
        rel_diversity = len(rel_types) / 5.0  # max 5 types
        base_factors['complexity'] = min(1.0, base_factors.get('complexity', 0.5) + 0.1 * rel_diversity)

        return base_factors

    def _update_token_profile(self, token: str, context_tokens: List[str], bayes_surprise: float, info_gain: float, embedding: np.ndarray):
        """Update token behavioral statistics for structural scoring."""
        if token not in self.token_profile:
            self.token_profile[token] = {
                'count': 0,
                'context_set': set(),
                'avg_surprise': 0.0,
                'avg_info_gain': 0.0,
                'embedding_drift': 0.0,
                'embedding_centroid': None,
            }
        prof = self.token_profile[token]

        # Update count
        prof['count'] += 1

        # Update context set with cap (prevent explosion)
        for ct in context_tokens:
            if len(prof['context_set']) < 200:
                prof['context_set'].add(ct)

        # Update moving averages (exponential with alpha=0.1)
        alpha = 0.1
        prof['avg_surprise'] = (1 - alpha) * prof['avg_surprise'] + alpha * bayes_surprise
        prof['avg_info_gain'] = (1 - alpha) * prof['avg_info_gain'] + alpha * info_gain

        # Centroid-based embedding drift
        if embedding is not None:
            if prof['embedding_centroid'] is None:
                prof['embedding_centroid'] = embedding.copy()
            else:
                # Update centroid (exponential moving average)
                centroid = prof['embedding_centroid']
                centroid = (1 - alpha) * centroid + alpha * embedding
                prof['embedding_centroid'] = centroid

                # Compute drift as 1 - cosine similarity between current embedding and centroid
                norm_cent = np.linalg.norm(centroid)
                norm_emb = np.linalg.norm(embedding)
                if norm_cent > 0 and norm_emb > 0:
                    cos_sim = np.dot(centroid, embedding) / (norm_cent * norm_emb)
                    drift = 1 - cos_sim
                else:
                    drift = 0.0

                # Update moving average of drift
                prof['embedding_drift'] = (1 - alpha) * prof['embedding_drift'] + alpha * drift

        # Update running statistics (global)
        stats = self.token_profile_stats
        def _update_running_stat(old_mean, old_std, new_value):
            new_mean = (1 - alpha) * old_mean + alpha * new_value
            new_std = (1 - alpha) * old_std + alpha * abs(new_value - new_mean)
            return new_mean, new_std

        stats['count_mean'], stats['count_std'] = _update_running_stat(stats['count_mean'], stats['count_std'], prof['count'])
        stats['ctx_mean'], stats['ctx_std'] = _update_running_stat(stats['ctx_mean'], stats['ctx_std'], len(prof['context_set']))
        stats['surp_mean'], stats['surp_std'] = _update_running_stat(stats['surp_mean'], stats['surp_std'], prof['avg_surprise'])
        stats['drift_mean'], stats['drift_std'] = _update_running_stat(stats['drift_mean'], stats['drift_std'], prof['embedding_drift'])

    def _normalize_stat(self, value, mean, std):
        """Sigmoid normalization using running statistics."""
        z = (value - mean) / (std + 1e-8)
        return 1 / (1 + np.exp(-z))
    
    def structural_score(self, token: str) -> float:
        """Compute structural score for a token (high = glue-like)."""
        prof = self.token_profile.get(token)
        if not prof:
            return 0.5  # neutral for unseen

        stats = self.token_profile_stats

        freq = self._normalize_stat(prof['count'], stats['count_mean'], stats['count_std'])
        ctx = self._normalize_stat(len(prof['context_set']), stats['ctx_mean'], stats['ctx_std'])
        surp = 1 - self._normalize_stat(prof['avg_surprise'], stats['surp_mean'], stats['surp_std'])
        drift = 1 - self._normalize_stat(prof['embedding_drift'], stats['drift_mean'], stats['drift_std'])

        score = 0.35 * freq + 0.35 * ctx + 0.15 * surp + 0.15 * drift

        # Cold start bootstrap: if few unique tokens, boost score for short tokens
        unique_tokens = len(self.token_profile)
        if unique_tokens < self._cold_start_threshold and len(token) <= 2:
            score = min(1.0, score + 0.2)

        return float(np.clip(score, 0, 1))        

    def _compute_curiosity_factors(self, token: str, token_ci: Dict, token_phon: float,
                                   diag: Dict, features_array: np.ndarray,
                                   global_factor_dict: Optional[Dict] = None) -> CuriosityFactors:
        """Compute all curiosity factors for a single token and return a dynamic CuriosityFactors object."""
        try:
            uncertainty = self.concept_model.entropy_word(token)

            learning_progress = 0.0
            if token in self.word_history and len(self.word_history[token]) > 1:
                prev_entropy = self.word_history[token][-2].get('entropy', 0.0)
                curr_entropy = self.word_history[token][-1].get('entropy', uncertainty)
                learning_progress = max(0, prev_entropy - curr_entropy)

            novelty = 1.0 - token_ci.get('confidence', 0.5)

            stability = token_ci.get('stability', 0.5)
            complexity = 1.0 - stability + (min(len(token), 20) / 20.0) * 0.3
            complexity = np.clip(complexity, 0, 1)

            predicted = self.concept_model.predict_features(token)
            observed = self._extract_features(token, token_ci, token_phon, diag)
            surprise = 0.0
            all_keys = set(predicted.keys()) | set(observed.keys())
            for k in all_keys:
                p = predicted.get(k, 0.01)
                o = observed.get(k, 0.01)
                surprise += o * math.log(o / p) if o > 0 and p > 0 else 0
            surprise = np.clip(surprise, 0, 1)

            phon_anomaly = 1.0 - token_phon
            neg_count = len(diag.get('negatives', []))
            diag_neg = min(neg_count / 5.0, 1.0)

            bayes_surprise = self.concept_model.bayesian_surprise(token)
            gp_uncertainty = self.gp_model.uncertainty(features_array)

            factor_dict = {
                "uncertainty": uncertainty,
                "learning_progress": learning_progress,
                "novelty": novelty,
                "complexity": complexity,
                "surprise": surprise,
                "phonotactic_anomaly": phon_anomaly,
                "diagnostic_negatives": diag_neg,
                "bayesian_surprise": bayes_surprise,
                "gp_uncertainty": gp_uncertainty,
            }
            # Apply knowledge graph based curiosity boost
            factor_dict = self._apply_knowledge_graph_curiosity(token, factor_dict)
            # If word has a definition, reduce its curiosity significantly
            if self.reasoner and self.reasoner.definitions.get(token):
                factor_dict["novelty"] *= 0.1
                factor_dict["surprise"] *= 0.1
                factor_dict["bayesian_surprise"] *= 0.1
                factor_dict["gp_uncertainty"] *= 0.1
                factor_dict["uncertainty"] *= 0.1
                factor_dict["learning_progress"] *= 0.1

            for dyn_factor in self.meta_learner.dynamic_factor_names:
                val = self.meta_learner.compute_dynamic_factor(
                    dyn_factor, token, token_ci, token_phon, diag,
                    features_array, global_factor_dict
                )
                factor_dict[dyn_factor] = val

            # Create CuriosityFactors object
            factors = CuriosityFactors(factor_dict=factor_dict)

            # Apply top‑down modulation (expects a CuriosityFactors object)
            factors = self._apply_top_down_modulation(token, factors)

            return factors

        except Exception as e:
            print(f"Error computing curiosity factors for '{token}': {e}. Returning neutral factors.")
            print(traceback.format_exc())
            # Return neutral factors to avoid crash
            neutral = {name: 0.5 for name in self.meta_learner.factor_names}
            return CuriosityFactors(factor_dict=neutral)

    def _apply_top_down_modulation(self, token: str, factors: CuriosityFactors) -> CuriosityFactors:
        """
        Modulate curiosity factors using top‑down signals:
        - Current state embedding
        - Identity embedding
        - Conscience beliefs
        """
        if self.current_state_embedding is not None:
            # Compute similarity between token embedding and current state
            token_emb = self.semantic_learner.word_embeddings.get(token)
            if token_emb is not None:
                state_emb = self._align_vector_to_emb_dim(self.current_state_embedding)
                token_emb_aligned = self._align_vector_to_emb_dim(token_emb)
                sim = np.dot(state_emb, token_emb_aligned) / (np.linalg.norm(state_emb)*np.linalg.norm(token_emb_aligned)+1e-12)
                # Boost curiosity if token is state‑relevant (already handled by state_relevance factor)
                # Here we also modulate novelty based on identity consistency
                identity_sim = self.identity_manager.get_stability_bonus(token_emb_aligned)
                # If token aligns with identity, reduce novelty (it's familiar)
                factors.factor_dict['novelty'] *= (1.0 - 0.5 * identity_sim)

        # Modulate by conscience: if a belief forbids asking about certain topics, reduce curiosity
        for belief in self.conscience_manager.get_active_conscience():
            if belief.rule_type == 'ethical' and belief.action.get('effect') == 'reduce_curiosity':
                # Simple keyword match
                if belief.condition.get('keyword') in token.lower():
                    factors.factor_dict['curiosity'] *= 0.3
                    factors.factor_dict['novelty'] *= 0.5
        return factors
    
    def _information_gain(self, token: str, new_entropy: float) -> float:
        if token not in self.word_history or len(self.word_history[token]) < 2:
            return 0.0
        old_entropy = self.word_history[token][-2].get('entropy', 1.0)
        return max(0, old_entropy - new_entropy)

    def _on_new_control_factor(self, factor_name: str, compute_function: Callable, metadata: dict = None):
        """
        Callback when a new dynamic control factor is added.
        Creates a corresponding curiosity factor with a per‑token compute function.
        """
        def curiosity_compute_function(token: str, token_ci: Dict, token_phon: float,
                                       diag: Dict, features_array: np.ndarray,
                                       global_factor_dict: Optional[Dict] = None) -> float:
            """
            Compute the curiosity value for this token based on the control factor.
            Heuristic: absolute deviation of the control factor from neutral (0.5),
            multiplied by the token's own novelty/uncertainty (approximated by 1 - confidence).
            This creates a drive to explore tokens that are relevant to the current control state.
            """
            # Get current global value of this control factor (default 0.5)
            if global_factor_dict is not None:
                ctrl_val = global_factor_dict.get(factor_name, 0.5)
            else:
                ctrl_val = 0.5
            
            # Deviation from neutral (0.5) – how "active" this factor is
            deviation = abs(ctrl_val - 0.5) * 2.0  # scale to [0,1]
            
            # Token's own uncertainty (novelty)
            novelty = 1.0 - token_ci.get('confidence', 0.5)
            
            # Combine: curiosity is high when the control factor is active AND the token is novel
            curiosity = deviation * novelty
            return np.clip(curiosity, 0, 1)
        
        # Add this as a new curiosity factor with initial weight 0.0 (neutral prior)
        self.meta_learner.add_factor(
            factor_name=f"ctrl_{factor_name}",
            initial_mu=0.0,
            compute_function=curiosity_compute_function,
            metadata={'source': 'control_space', 'original_factor': factor_name}
        )
        print(f" [Drive Creation] New curiosity factor 'ctrl_{factor_name}' created from control dimension '{factor_name}'.")  

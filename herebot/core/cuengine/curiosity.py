from ..curioimport import *

class HeartCuriosityMixin:
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
    # ------------------------------------------------------------------
    # State learning (unchanged logic, but uses configurable threshold)
    # ------------------------------------------------------------------
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
            label = self.resolve_state_label(label)
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

    def _detect_semantic_relations(self, tokens: List[str], token_analyses: List[Dict]):
        """
        After processing tokens, check for words that have definitions and look for
        similar definitions among other words. Generate hypotheses for synonymy.
        """
        # Only process words that are in token_analyses (i.e., were in the input)
        for ta in token_analyses:
            word = ta['token']
            # If this word has a stored definition, check for similar definitions
            if word in self.reasoner.definitions:
                hypothesis = self.reasoner.generate_synonym_hypothesis(word)
                if hypothesis:
                    # Hypothesis is already added to reasoner.hypotheses; we may print it or log it.
                    print(f"💡 [Semantic Hypothesis] {hypothesis}")
    # ------------------------------------------------------------------
    # Identity reinforcement
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
            current_label = self.resolve_state_label(current_label)
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

    def _process_sentence_reasoning(self, sentence: str, tokens: List[str],
                                    token_analyses: List[Dict], sentence_emb: np.ndarray):
        """
        Called only when all tokens are known.
        Stores sentence, generates hypotheses, detects relational statements,
        and updates word-state associations.
        """
        # Store sentence with embedding
        self.reasoner.store_sentence(sentence, tokens, embedding=sentence_emb,
                                     metadata={'timestamp': time.time(), 'source': 'user'})

        # --- Pattern-based relation detection (no hardcoded verb lists) ---
        # 1. Feed tokens to pattern observer: updates structural word stats,
        #    returns (head, connector, tail) triplets for this sentence.
        triplets = self.reasoner.pattern_observer.observe(tokens)
        self.reasoner.last_observed_triplets = triplets

        # 2. Try to parse as a math sentence using learned number/operator words.
        math_result = self.reasoner.pattern_observer.try_parse_math(
            tokens, self.reasoner.operators)
        if math_result:
            self._handle_math_sentence(math_result, tokens)

        # 3. Surface any new connector-role questions to the hypothesis queue.
        for q in self.reasoner.pattern_observer.get_pending_questions():
            conn_key = f"connector_role:{q['connector']}"
            if not any(h.get('key') == conn_key for h in self.reasoner.hypotheses):
                self.reasoner.hypotheses.append({
                    'key': conn_key,
                    'type': 'connector_role',
                    'connector': q['connector'],
                    'text': q['text'],
                    'asked': False,
                    'energy_cost': 0.3,
                    'predictive_gain': 0.7,
                    'phase': 'active',
                    'last_used': None,
                    'recent_usage': 0.0,
                })

        # 4. Process each extracted triplet.
        for (subject, connector, object) in triplets:
            if not subject or not object:
                continue
            # Map connector to a relation type if already learned, else 'unknown'
            rel_type = self.reasoner.pattern_observer.connector_to_reltype(connector)
            if rel_type is None:
                rel_type = 'unknown'

            # Store relation in reasoner
            self.reasoner.store_phrase_relation(subject, object, rel_type, confidence=1.0)

            # Generate a confirmation hypothesis if new
            key = f"relation:{rel_type}:{subject}:{object}"
            if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                self.reasoner.hypotheses.append({
                    'key': key,
                    'type': 'relation',
                    'subject': subject,
                    'predicate': rel_type,
                    'object': object,
                    'text': f"TPL.ASK.CONN.REL subject='{subject}' connector='{connector}' object='{object}'?",
                    'asked': False,
                })

            # Word-state association
            if rel_type in ('means', 'is_a'):
                self.reasoner.word_state.add_association(
                    subject, object, confidence=1.0, source="explicit")
                print(f"📝 [Word-State] Associated '{subject}' with label '{object}'"
                      f" (via learned connector '{connector}')")
                object_labels = self.reasoner.word_state.get_labels(object, threshold=0.3)
                for lab in object_labels:
                    if lab != object:
                        self.reasoner.word_state.add_association(
                            subject, lab, confidence=0.9, source="inheritance")
                        print(f"   ↪ Propagated label '{lab}' from '{object}' to '{subject}'")
            elif rel_type == 'similar':
                subj_labels = self.reasoner.word_state.get_labels(subject)
                obj_labels  = self.reasoner.word_state.get_labels(object)
                for lab in obj_labels:
                    self.reasoner.word_state.add_association(
                        subject, lab, confidence=0.7, source="similarity")
                for lab in subj_labels:
                    self.reasoner.word_state.add_association(
                        object, lab, confidence=0.7, source="similarity")
                print(f"📝 [Word-State] Propagated labels between '{subject}' and '{object}'")

            # Add to knowledge graph
            subj_nid = self.reasoner.knowledge_graph.get_node_id(
                subject, "word" if len(subject.split()) == 1 else "phrase")
            if subj_nid is None:
                subj_nid = self.reasoner.knowledge_graph.add_node(
                    subject, "word" if len(subject.split()) == 1 else "phrase")
            obj_nid = self.reasoner.knowledge_graph.get_node_id(object, "state")
            if obj_nid is None:
                obj_nid = self.reasoner.knowledge_graph.add_node(object, "state")
            self.reasoner.knowledge_graph.add_edge(
                subj_nid, obj_nid, "associated", confidence=1.0)

        # Generate other hypotheses (existing logic)
        hypothesis = self.reasoner.generate_sentence_hypothesis(sentence, tokens, sentence_emb)
        if hypothesis:
            print(f"💡 [Sentence Hypothesis] {hypothesis}")
            # Hypothesis is already in self.reasoner.hypotheses
        # Store phrase embedding for future similarity
        if sentence_emb is not None:
            self.reasoner.store_phrase(sentence, sentence_emb, metadata={'source': 'user', 'timestamp': time.time()})

        # Check if we should ask about this sentence (low comprehension)
        if token_analyses:
            avg_novelty = np.mean([ta['factors'].factor_dict.get('novelty', 0.5) for ta in token_analyses])
            comprehension = 1.0 - avg_novelty
            if comprehension < 0.5:   # threshold
                query = self.reasoner.plan_sentence_query(sentence, uncertainty=1.0 - comprehension)
                if query:
                    self.pending_sentence_query = query
                    print(f"❓ [Pending Sentence Query] {query}")

    def _derive_inferences_from_definition(self, subject: str, definition_text: str):
        """
        Given a definition of subject (word or phrase), try to derive transitive inferences.
        For example, if we have "X means Y" and later "Y means Z", we can infer "X means Z".
        This method is called after a definition is stored.
        """
        # For now, we rely on the reasoner's relation graphs; the actual inference is
        # triggered when a relation is added via store_phrase_relation.
        # Here we just log that a definition was added (no direct action needed).
        pass
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Pad or truncate vector to match the current reference dimension (from StateManager)."""
        return self.state_manager._normalize_vector(vector)    

    def _all_tokens_known(self, tokens: List[str]) -> bool:
        """Return True iff every token in the list has a semantic embedding."""
        return all(t in self.semantic_learner.word_embeddings for t in tokens)

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
        return "TPL.ASK.FB rating=positive|negative|neutral?"

    # ------------------------------------------------------------------
    # Automatic drive creation: control factor → curiosity factor
    # ------------------------------------------------------------------
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
        ctrl_name = f"ctrl_{factor_name}"

        # Ensure metadata is tracked (for tests + consistency)
        if ctrl_name not in self.meta_learner.factor_metadata:
            self.meta_learner.factor_metadata[ctrl_name] = {
                'source': 'control_space',
                'original_factor': factor_name
            }
        print(f" [Drive Creation] New curiosity factor 'ctrl_{factor_name}' created from control dimension '{factor_name}'.")

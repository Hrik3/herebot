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

        # --- Detect categorical or definitional statement ---
        # Find main verb: look for 'is' or 'means' in tokens
        verb_pos = None
        verb = None
        for i, tok in enumerate(tokens):
            if tok in ['is', 'are', 'was', 'were', 'means', 'like']:
                verb_pos = i
                verb = tok
                break
        if verb_pos is not None:
            subject = ' '.join(tokens[:verb_pos]).strip()
            object = ' '.join(tokens[verb_pos+1:]).strip()
            if subject and object:
                if verb in ['is', 'are', 'was', 'were']:
                    rel_type = 'is_a'
                elif verb == 'means':
                    rel_type = 'means'
                elif verb == 'like':
                    rel_type = 'similar'
                else:
                    rel_type = 'unknown'
                # Store relation in reasoner
                self.reasoner.store_phrase_relation(subject, object, rel_type, confidence=1.0)
                # Also generate a direct hypothesis about this relation if it's new
                key = f"relation:{rel_type}:{subject}:{object}"
                if not any(h.get('key') == key for h in self.reasoner.hypotheses):
                    hyp = {
                        'key': key,
                        'type': 'relation',
                        'subject': subject,
                        'predicate': rel_type,
                        'object': object,
                        'text': f"Did I understand correctly: '{subject}' {verb} '{object}'?",
                        'asked': False,
                    }
                    self.reasoner.hypotheses.append(hyp)

                # ---- Word-state association ----
                # If subject is a word/phrase and object is a state label (single word), associate
                # Also handle "X like Y" as suggesting similarity
                if rel_type in ('means', 'is_a'):
                    # Treat object as a label for subject
                    self.reasoner.word_state.add_association(subject, object, confidence=1.0, source="explicit")
                    print(f"📝 [Word-State] Associated '{subject}' with label '{object}'")
                    
                    # ---- IMMEDIATE PROPAGATION: if object has labels, propagate them to subject ----
                    object_labels = self.reasoner.word_state.get_labels(object, threshold=0.3)
                    for lab in object_labels:
                        if lab != object:  # avoid self‑propagation
                            self.reasoner.word_state.add_association(subject, lab, confidence=0.9, source="inheritance")
                            print(f"   ↪ Propagated label '{lab}' from '{object}' to '{subject}'")
                            
                elif rel_type == 'similar':
                    # If subject and object are both words, they should share labels
                    subj_labels = self.reasoner.word_state.get_labels(subject)
                    obj_labels = self.reasoner.word_state.get_labels(object)
                    # Propagate labels from object to subject with lower confidence
                    for lab in obj_labels:
                        self.reasoner.word_state.add_association(subject, lab, confidence=0.7, source="similarity")
                    for lab in subj_labels:
                        self.reasoner.word_state.add_association(object, lab, confidence=0.7, source="similarity")
                    print(f"📝 [Word-State] Propagated labels between '{subject}' and '{object}'")
                # ---- Also add to knowledge graph ----
                # Get node for subject (word/phrase) and object (label)
                subj_nid = self.reasoner.knowledge_graph.get_node_id(subject, "word" if len(subject.split())==1 else "phrase")
                if subj_nid is None:
                    subj_nid = self.reasoner.knowledge_graph.add_node(subject, "word" if len(subject.split())==1 else "phrase")
                obj_nid = self.reasoner.knowledge_graph.get_node_id(object, "state")
                if obj_nid is None:
                    obj_nid = self.reasoner.knowledge_graph.add_node(object, "state")
                self.reasoner.knowledge_graph.add_edge(subj_nid, obj_nid, "associated", confidence=1.0)

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

from __future__ import annotations
from core.heart import *
# ------------------------------------------------------------
# Action Generator – produces system responsesself.propagate_word_state_associations,
# ------------------------------------------------------------
class ActionGenerator:
    """
    Generates natural language responses based on:
    - Current curiosity (which word to ask about)
    - Conflict resolution prompts
    - Idle thoughts
    - Semantic similarity for definitions
    - Synonym/category hypotheses from the reasoning engine
    - High-trust spontaneous language attempts

    Now config-aware and uses reasoner for semantic queries.
    """

    def __init__(self, semantic_learner: SemanticLearner,
                 syntax_learner: SyntaxLearner,
                 reasoner: ReasoningEngine,
                 engine: 'CuriosityEngine',  # added to access trust factor and feedback state
                 config_manager: Optional['ConfigManager'] = None):
        self.semantic = semantic_learner
        self.syntax = syntax_learner
        self.reasoner = reasoner
        self.engine = engine
        self.config_manager = config_manager
        self.semantic_learner = semantic_learner
        self.guilt_tracker = getattr(engine, 'guilt_tracker', None)

    def _matches_conscience_condition(self, belief: ConscienceBelief, user_input: str, state: Dict) -> bool:
        """Check if the current context matches the condition of a conscience belief."""
        cond = belief.condition
        if belief.rule_type == 'identity_alias':
            alias = cond.get('alias')
            return alias in user_input.lower()
        for k, v in cond.items():
            if k == 'keyword' and isinstance(v, str) and v in user_input.lower():
                return True
            if k == 'tokens' and isinstance(v, list):
                tokens = self.engine._tokenize(user_input)
                if any(t in v for t in tokens):
                    return True
        return False

    def generate_guilt_expression(self, guilt_level: float) -> Optional[str]:
        """Return a sentence expressing guilt/unease if guilt is significant."""
        if guilt_level < 0.2:
            return None
        templates = [
            "TPL.GUILT.UNEASY",
            "TPL.GUILT.CHECK",
            "TPL.GUILT.ERROR",
            "TPL.GUILT.ACTIVE",
            "TPL.GUILT.WRONG",
            "TPL.GUILT.CONFLICT",
        ]
        base = np.random.choice(templates)
        # Use identity alias if available
        if self.engine.identity_aliases and not base.startswith("TPL."):
            alias = next(iter(self.engine.identity_aliases))
            base = base.replace("I", alias.capitalize()).replace("I", alias)
        return base  

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def generate_question(self, target: str, uncertainty: float = None) -> str:
        """Produce a question about an unknown word."""
        base = f"TPL.ASK.DEF target='{target}'?"
        if uncertainty is not None:
            return f"{base} uc={uncertainty:.2f}"
        return base

    def generate_definition(self, word: str) -> str:
        """If we have learned something, generate a definition."""
        # find similar known words
        sim_words = []
        for w in self.semantic.word_embeddings:
            if w != word:
                sim = self.semantic.similarity(word, w)
                if sim > 0.6:
                    sim_words.append((w, sim))
        sim_words.sort(key=lambda x: -x[1])
        if sim_words:
            best = sim_words[0][0]
            return f"TPL.HYP.SIM word='{word}' candidate='{best}'"
        else:
            return f"TPL.NO.DEF word='{word}'"

    def generate_synonym_question(self, word: str, candidate: str) -> str:
        """Generate a question about a potential synonym relationship."""
        return f"TPL.ASK.SYN word='{word}' candidate='{candidate}'?"

    def generate_trust_utterance(self, trust_value: float) -> str:
        """
        Generate a spontaneous, possibly broken-English utterance when trust is high.
        Now also used when emotional state is negative – but with different templates.
        Templates are not pre-defined as positive/negative; the system learns
        appropriateness through feedback.
        """
        # Use threat level to decide tone
        threat = self.engine._last_factor_dict.get('threat', 0.0) if self.engine._last_factor_dict else 0.0
        energy = self.engine._last_factor_dict.get('energy', 0.8) if self.engine._last_factor_dict else 0.8
        comprehension = self.engine._last_factor_dict.get('comprehension', 0.5) if self.engine._last_factor_dict else 0.5

        # Base templates – all neutral; sentiment is learned via feedback
        if self.reasoner.definitions:
            word = np.random.choice(list(self.reasoner.definitions.keys()))
            def_text = self.reasoner.definitions[word][:20]
            templates = [
                f"TPL.DEF word='{word}' value='{def_text}'",
                f"TPL.DEF.LIKE word='{word}' value='{def_text}'",
                f"TPL.DEF.EQ word='{word}' value='{def_text}'",
                f"TPL.DEF.MAYBE word='{word}' value='{def_text}'",
            ]
        else:
            if self.reasoner.co_occurrence:
                word_counts = Counter()
                for w, cnts in self.reasoner.co_occurrence.items():
                    word_counts[w] = sum(cnts.values())
                if word_counts:
                    word = word_counts.most_common(1)[0][0]
                    templates = [
                        f"TPL.OBS.FREQ word='{word}'",
                        f"TPL.OBS.HIGH word='{word}'",
                        f"TPL.OBS.IMPORTANT word='{word}'",
                    ]
                else:
                    templates = ["TPL.LEARN"]
            else:
                templates = ["TPL.LEARN"]

        utterance = np.random.choice(templates)

        # Emotional modulation – add a brief state expression (learned from feedback)
        # This is purely emergent: the system will learn which expressions get positive feedback.
        if threat > 0.6:
            state_phrases = [
                " TPL.STATE.THREAT",
                " TPL.STATE.HARD",
                " TPL.STATE.UNSURE",
                " TPL.STATE.MAYBE_WRONG",
            ]
            utterance += np.random.choice(state_phrases)
        elif comprehension < 0.3:
            state_phrases = [
                " TPL.STATE.LOW_COMP",
                " TPL.STATE.NO_PARSE",
                " TPL.ASK.CLARIFY",
            ]
            utterance += np.random.choice(state_phrases)
        elif energy < 0.3:
            state_phrases = [
                " TPL.STATE.LOW_ENERGY",
                " TPL.STATE.DRAIN",
                " TPL.STATE.PAUSE",
            ]
            utterance += np.random.choice(state_phrases)
        elif trust_value > 0.7:
            state_phrases = [
                " TPL.STATE.CONF",
                " TPL.STATE.OK",
                " TPL.STATE.UNDERSTAND",
            ]
            utterance += np.random.choice(state_phrases)

        # Replace "I" with learned self-alias if available
        if self.engine.identity_aliases and not utterance.startswith("TPL."):
            alias = next(iter(self.engine.identity_aliases))
            utterance = utterance.replace("I", alias.capitalize()).replace("I", alias)
        return utterance

    def generate_chat_response(self, user_input: str, state: Dict) -> str:
        """
        Generate a conversational, non-interrogative response when chat mode is active.
        PRIORITY 1: generate a base reply addressing the user input (definitions, patterns, trigram).
        PRIORITY 2: modulate that base reply with the current internal state.
        PRIORITY 3: if the base reply is generic, inject a state-based comment.
        """
        user_input_lower = user_input.lower().strip()
        factor_dict = state.get('factor_vector', {})
        if not isinstance(factor_dict, dict):
            factor_dict = {}

        # -------------------------------------------------------------
        # 1. Generate a base reply that addresses the user input
        # -------------------------------------------------------------
        base_reply = None

        # 1.1 Use identity alias if available
        alias = None
        if self.engine.identity_aliases:
            alias = next(iter(self.engine.identity_aliases))

        # 1.2 Check learned behavioral patterns (conversational_pattern)
        for rule in self.engine.behavioral_learner.rules.values():
            if rule.rule_type == 'conversational_pattern' and rule.confidence > 0.6:
                cond = rule.condition
                if 'user_input_pattern' in cond and user_input_lower == cond['user_input_pattern']:
                    if 'response_template' in rule.action:
                        self.engine.behavioral_learner.record_usage(rule.rule_id)
                        base_reply = rule.action['response_template']
                        break
                elif 'question_pattern' in cond and user_input_lower.rstrip('?') == cond['question_pattern']:
                    if 'answer_template' in rule.action:
                        self.engine.behavioral_learner.record_usage(rule.rule_id)
                        base_reply = rule.action['answer_template']
                        break
                elif 'expected_user_input' in cond and user_input_lower == cond['expected_user_input']:
                    self.engine.behavioral_learner.record_usage(rule.rule_id)
                    base_reply = "TPL.CONV.CONT"
                    break

        # 1.3 Try to understand the user's input using known definitions and relations
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for phrase in self.reasoner.phrase_definitions:
                if phrase in user_input_lower:
                    def_text = self.reasoner.phrase_definitions[phrase]
                    base_reply = f"TPL.DEF.PHRASE phrase='{phrase}' value='{def_text}'"
                    break

        # 1.4 Use word-state labels to provide contextual info
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for word in tokens:
                labels = self.reasoner.word_state.get_labels(word, threshold=0.6)
                if labels:
                    label_str = ', '.join(labels)
                    base_reply = f"TPL.STATE.ASSOC word='{word}' labels='{label_str}'"
                    break

        # 1.5 Check if user input is similar to known phrases
        if base_reply is None:
            input_emb = self.semantic_learner.text_to_embedding(user_input, fallback_to_char=True)
            if input_emb is not None:
                similar = self.reasoner.find_similar_phrases(input_emb, threshold=0.8)
                if similar:
                    best_phrase, sim = similar[0]
                    base_reply = f"TPL.RECALL.PHRASE phrase='{best_phrase}'"

        # 1.6 Check for words with definitions
        if base_reply is None:
            tokens = self.engine._tokenize(user_input)
            for word in tokens:
                if word in self.reasoner.definitions:
                    def_text = self.reasoner.definitions[word][:50]
                    relations = self.reasoner.phrase_relations.get(word, [])
                    rel_strs = []
                    for rel_type, obj, conf in relations[:2]:
                        if conf > 0.7:
                            rel_strs.append(f"{rel_type} {obj}")
                    if rel_strs:
                        base_reply = f"TPL.DEF.REL word='{word}' value='{def_text}' rel='{', '.join(rel_strs)}'"
                    else:
                        base_reply = f"TPL.DEF.KNOWN word='{word}' value='{def_text}'"
                    break

        # 1.7 Fall back to trigram model from episodic memory
        if base_reply is None:
            trigram = defaultdict(Counter)
            episodes = self.engine.episodic_memory.episodes[-50:]
            for ep in episodes:
                sent = ep.get('user_input', '')
                tokens_ep = self.engine._tokenize(sent)
                if len(tokens_ep) < 3 or not self.engine._all_tokens_known(tokens_ep):
                    continue
                for i in range(len(tokens_ep)-2):
                    ctx = tuple(tokens_ep[i:i+2])
                    nxt = tokens_ep[i+2]
                    trigram[ctx][nxt] += 1

            known_user_tokens = [t for t in tokens if t in self.semantic.word_embeddings]
            if len(known_user_tokens) >= 2 and trigram:
                seed = tuple(known_user_tokens[-2:])
                if seed in trigram:
                    next_words = list(trigram[seed].keys())
                    probs = np.array(list(trigram[seed].values()))
                    probs = probs / probs.sum()
                    next_w = np.random.choice(next_words, p=probs)
                    base_reply = f"TPL.NGRAM seed='{' '.join(known_user_tokens)}' next='{next_w}'?"
                else:
                    recent_ep = episodes[-1] if episodes else None
                    if recent_ep:
                        topic = recent_ep.get('user_input', '').split()[:2]
                        topic = ' '.join(topic)
                        base_reply = f"TPL.TOPIC topic='{topic}'"

        # 1.8 Use learned structural patterns
        if base_reply is None:
            freq_patterns = self.reasoner.structural_patterns.get_frequent_patterns(min_freq=2)
            if freq_patterns and np.random.rand() < 0.3:
                pattern = np.random.choice(freq_patterns)
                generated = self.reasoner.structural_patterns.generate_from_pattern(pattern)
                if generated:
                    base_reply = f"TPL.PATTERN text='{generated}'"

        # 1.8.5 Use language model to generate a response (if available and trained)
        if base_reply is None and hasattr(self.engine, 'language_model'):
            lm = self.engine.language_model
            # Only use if sufficiently trained
            if lm.steps >= self.engine.lm_config.get('min_steps_for_use', 100):
                try:
                    generated = lm.generate(user_input, max_len=15, temperature=0.8)
                    if generated and len(generated.split()) > 1:
                        base_reply = generated
                except Exception as e:
                    print(f"   [LM] Generation error: {e}")

        # 1.9 Ultimate generic fallback
        if base_reply is None:
            if known_user_tokens:
                word = known_user_tokens[0]
                if word in self.reasoner.definitions:
                    base_reply = f"TPL.DEF.RECALL word='{word}' value='{self.reasoner.definitions[word][:50]}'"
                else:
                    base_reply = f"TPL.ASK.MEAN word='{word}'?"
            else:
                base_reply = "TPL.CHAT.MORE?"

        # -------------------------------------------------------------
        # 2. Modulate the base reply with the current internal state
        # -------------------------------------------------------------
        modulated = self._modulate_with_state(base_reply, factor_dict)

        # If the base reply is the generic template, we can optionally
        # inject a more state-specific comment instead.
        if base_reply.startswith("TPL.CHAT.MORE") and factor_dict:
            # Determine dominant emotion
            emotions = ['curiosity', 'threat', 'trust', 'comprehension', 'energy']
            dominant = max(emotions, key=lambda e: factor_dict.get(e, 0.5))
            val = factor_dict.get(dominant, 0.5)

            # Pick a word that is relevant to the current state
            candidate_word = None
            if self.engine.current_state_labels and self.engine.current_state_embedding is not None:
                state_emb = self.engine.current_state_embedding
                best_word = None
                best_sim = -1
                for w, emb in self.semantic_learner.word_embeddings.items():
                    if w in self.reasoner.definitions:
                        continue
                    sim = np.dot(state_emb, emb) / (np.linalg.norm(state_emb)*np.linalg.norm(emb)+1e-12)
                    if sim > best_sim:
                        best_sim = sim
                        best_word = w
                if best_word and best_sim > 0.6:
                    candidate_word = best_word
            if not candidate_word and self.engine.word_history:
                candidate_word = np.random.choice(list(self.engine.word_history.keys()))

            if candidate_word:
                if dominant == 'curiosity' and val > 0.6:
                    modulated = f"TPL.STATE.CURIOUS word='{candidate_word}' | {base_reply}"
                elif dominant == 'threat' and val > 0.5:
                    modulated = f"TPL.STATE.THREAT | {base_reply}"
                elif dominant == 'trust' and val > 0.7:
                    modulated = f"TPL.STATE.CONF | {base_reply}"
                elif dominant == 'comprehension' and val < 0.3:
                    modulated = f"TPL.STATE.LOW_COMP | {base_reply}"
                elif dominant == 'energy' and val < 0.3:
                    modulated = f"TPL.STATE.LOW_ENERGY | {base_reply}"
                else:
                    modulated = f"TPL.STATE.TOPIC word='{candidate_word}' | {base_reply}"
            else:
                modulated = self._modulate_with_state(base_reply, factor_dict)

        return modulated

    def _modulate_with_state(self, response: str, factor_dict: Dict) -> str:
        """
        Optionally prepend a brief emotional phrase based on the current state.
        This makes the response reflect the internal state without changing its core content.
        """
        # Only modulate with some probability, to avoid overdoing it
        if np.random.rand() > 0.4:
            return response

        # Extract emotional signals
        threat = factor_dict.get('threat', 0.0)
        trust = factor_dict.get('trust', 0.5)
        curiosity = factor_dict.get('curiosity', 0.5)
        comprehension = factor_dict.get('comprehension', 0.5)
        energy = factor_dict.get('energy', 0.8)

        prefixes = []
        if threat > 0.7:
            prefixes.append("TPL.PFX.THREAT.HIGH |")
        elif threat > 0.4:
            prefixes.append("TPL.PFX.THREAT |")
        elif trust > 0.7:
            prefixes.append("TPL.PFX.TRUST |")
        elif curiosity > 0.7:
            prefixes.append("TPL.PFX.CURIOUS |")
        elif comprehension < 0.3:
            prefixes.append("TPL.PFX.LOW_COMP |")
        elif energy < 0.3:
            prefixes.append("TPL.PFX.LOW_ENERGY |")

        if prefixes:
            prefix = np.random.choice(prefixes)
            return f"{prefix} {response}"
        return response
        
    def generate_reply(self, user_input: str, state: Dict) -> str:
        """
        Main entry point: decide what to say.
        Priority:
          0. Chat mode → conversational reply (if active)
          1. Unresolved conflicts (handled in process_input)
          1.5. Pending sentence query (from sentence reasoning)
          2. Pending synonym/category/analogy hypotheses
          3. High curiosity words (unknown words)
          4. State awareness utterance
          5. High trust spontaneous attempt
          6. Chitchat
        """
        # --- Priority 0: Chat mode (overrides most other behaviours) ---
        if self.engine.chat_mode_active:
            return self.generate_chat_response(user_input, state)

        # --- Priority 0.5: Logic doubt (already handled in process_input, but keep as safeguard) ---
        doubt = self.engine.logic_inheritance.get_next_doubt()
        if doubt:
            rule = doubt['rule']
            description = rule.get('description', f"a logical rule")
            return f"TPL.ASK.LOGIC desc='{description}'?"

        # --- Priority 0.75: Evaluate expressions in user input ---
        # If user input contains an operator and all symbols have known values,
        # try to evaluate it and return result.
        if self.engine.reasoner.operators:
            tokens = self.engine._tokenize(user_input)
            # crude check: if tokens contain an operator and all others are numbers or known symbols
            op_symbols = set(self.engine.reasoner.operators.keys())
            if any(t in op_symbols for t in tokens):
                # attempt to parse as expression – for now just simple arithmetic
                try:
                    # Very simple evaluation: if input looks like "2 + 2", evaluate
                    # (This is a placeholder; a real parser would use syntax trees)
                    result = eval(user_input, {"__builtins__": None}, {})
                    return f"TPL.MATH.RESULT value='{result}'"
                except:
                    pass

        # --- Priority 0.7: Conscience relevance ---
        for belief in self.engine.conscience_manager.get_active_conscience():
            if self._matches_conscience_condition(belief, user_input, state):
                return f"TPL.ASK.CONSCIENCE desc='{belief.description}'?"

        # ---- Express guilt if significant ----
        if self.guilt_tracker and self.guilt_tracker.guilt_level > 0.3 and np.random.rand() < 0.4:
            guilt_expr = self.generate_guilt_expression(self.guilt_tracker.guilt_level)
            if guilt_expr:
                return guilt_expr

        # --- Priority 1.5: Pending sentence query ---
        if self.engine.pending_sentence_query:
            query = self.engine.pending_sentence_query
            self.engine.pending_sentence_query = None   # consume
            return query

        # --- Priority 2: Pending hypotheses (synonym, category, analogy, inference) ---
        for hyp in self.reasoner.hypotheses:
            if hyp.get('type') in ('synonym', 'category', 'analogy', 'inference', 'shadow_differentiation_link') and not hyp.get('asked', False):
                hyp['asked'] = True
                hyp['last_used'] = time.time()
                hyp['recent_usage'] = hyp.get('recent_usage', 0.0) * 0.99 + 0.1
                if hyp.get('type') == 'shadow_differentiation_link':
                    self.engine.expecting_shadow_diff_confirmation = True
                    self.engine.pending_shadow_diff_key = hyp.get('key')
                return hyp['text']

        # --- Priority 2.5: Parallel reasoning thoughts (if multiple chains available) ---
        if 'token_analyses' in state and state['token_analyses']:
            # pick the most curious token
            token_analyses = state['token_analyses']
            high_curiosity = [ta for ta in token_analyses if ta.get('curiosity_score', 0) > 0.4]
            if high_curiosity:
                token = high_curiosity[0]['token']
                # generate parallel thoughts
                thoughts = self.engine.parallel_reasoner.generate_thoughts(token, state)
                # present the most confident one
                best = max(thoughts, key=lambda t: t['confidence'])
                return best['hypothesis']

        # --- Priority 3: High curiosity words (unknown words) ---
        if 'token_analyses' in state:
            high_curiosity = [ta for ta in state['token_analyses'] if ta.get('curiosity_score', 0) > 0.5]
            if high_curiosity:
                target = high_curiosity[0]['token']
                unc = high_curiosity[0].get('gp_uncertainty')
                return self.generate_question(target, unc)

        # --- Priority 4: State awareness utterance (30% chance) ---
        if self.engine.current_state_labels and np.random.rand() < 0.3:
            # pick one of the active states to mention
            chosen_state = np.random.choice(self.engine.current_state_labels)
            return f"TPL.STATE.CURRENT label='{chosen_state}'"

        # --- Priority 5: Spontaneous emotional utterance (based on state) ---
        factor_vec = state.get('factor_vector', {})
        if not isinstance(factor_vec, dict):
            factor_vec = {}
        threat = factor_vec.get('threat', 0.0)
        trust = factor_vec.get('trust', 0.5)
        comprehension = factor_vec.get('comprehension', 0.5)
        energy = factor_vec.get('energy', 0.8)

        # Trigger in both positive and negative states, with different probabilities
        trigger_prob = 0.0
        if trust > 0.7 and not self.engine.expecting_feedback:
            trigger_prob = 0.4   # high trust, express confidence
        elif threat > 0.6:
            trigger_prob = 0.5   # high threat, express uncertainty
        elif comprehension < 0.3:
            trigger_prob = 0.6   # low comprehension, express confusion
        elif energy < 0.3:
            trigger_prob = 0.4   # low energy, express tiredness

        if np.random.rand() < trigger_prob:
            utterance = self.generate_trust_utterance(trust)  # now handles all emotional tones
            if self.engine.current_state_labels:
                # Prepend state information
                state_str = ', '.join(self.engine.current_state_labels)
                utterance = f"TPL.STATE.CONTEXT labels='{state_str}' | {utterance}"
            self.engine.expecting_feedback = True
            self.engine.feedback_attempt_word = None
            self.engine.feedback_attempt_response = utterance
            self.engine.feedback_timestamp = time.time()
            return utterance

        # --- Priority 6: idle thought / chitchat ---
        return "TPL.ACK.LISTEN"

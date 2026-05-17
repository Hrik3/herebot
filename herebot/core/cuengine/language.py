from ..curioimport import *

class HeartLanguageMixin:
    def train_language_model(self):
        """Train the neural language model on all available textual data."""
        if not self.lm_config.get('enabled', True):
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping language model training (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        # Collect training sentences from episodic memory (user inputs and system responses)
        sentences = []
        for ep in self.episodic_memory.episodes[-500:]:
            user = ep.get('user_input', '').strip()
            resp = ep.get('system_response', '').strip()
            if user and len(user.split()) <= 30:
                sentences.append(user)
            if resp and len(resp.split()) <= 30 and '?' not in resp:
                sentences.append(resp)

        # Add synthetic sentences from dreaming (if any)
        if hasattr(self, 'world_model') and self.world_model:
            # we don't have a store, but could use last few from dreaming task
            pass

        # Add definitions
        for word, def_text in self.reasoner.definitions.items():
            sentences.append(def_text)

        if len(sentences) < 10:
            return

        # Shuffle and take a batch
        np.random.shuffle(sentences)
        batch = sentences[:min(self.lm_config['train_batch_size'], len(sentences))]

        total_loss = 0.0
        lr = self.language_model.learning_rate * throttle  # scale by throttle

        # Add new words to vocabulary
        for sent in batch:
            for word in sent.split():
                if word not in self.language_model.word_to_idx:
                    self.language_model.add_word_to_vocab(word)

        # Train
        for sent in batch:
            try:
                loss = self.language_model.train_on_sentence(sent, lr=lr)
                total_loss += loss
            except Exception as e:
                print(f"   [LM] Error training on '{sent}': {e}")

        avg_loss = total_loss / len(batch)
        print(f"   [Language Model] Trained on {len(batch)} sentences, avg loss {avg_loss:.4f}, vocab size {self.language_model.vocab_size}, total steps {self.language_model.steps}")
#done3
    def _build_lm_validation_set(self, size=50):
        """Create a validation set from recent episodes, excluding training data."""
        sentences = []
        for ep in self.episodic_memory.episodes[-500:]:
            user = ep.get('user_input', '').strip()
            resp = ep.get('system_response', '').strip()
            if user and len(user.split()) <= 30:
                sentences.append(user)
            if resp and len(resp.split()) <= 30 and '?' not in resp:
                sentences.append(resp)
        # Shuffle and take last `size` as validation (assuming recent ones not in train)
        np.random.shuffle(sentences)
        return sentences[-size:] if len(sentences) >= size else sentences

    def evaluate_language_model(self, model=None, validation_sentences=None):
        """Compute average loss of a language model on validation sentences."""
        if model is None:
            model = self.language_model
        if validation_sentences is None:
            validation_sentences = self._build_lm_validation_set(30)
        total_loss = 0.0
        count = 0
        for sent in validation_sentences:
            try:
                ids = model.sentence_to_ids(sent)
                if len(ids) < 2:
                    continue
                input_ids = ids[:-1]
                target_ids = ids[1:]
                loss = model.loss(input_ids, target_ids)
                total_loss += loss
                count += 1
            except Exception:
                continue
        return total_loss / count if count > 0 else float('inf')
    
    def propose_lm_improvement(self):
        """
        When the language model has accumulated enough training steps,
        propose several new architectures, train each briefly, compare validation loss,
        and replace the current model if a better one is found.
        """
        if not self.lm_config.get('enabled', True):
            return
        if self.language_model.steps < self.lm_config.get('improvement_interval', 5000):
            return

        if self.language_model.steps % self.lm_config['improvement_interval'] != 0:
            return

        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self._get_resource_status()
        if not cpu_ok or not mem_ok:
            print(f"   [Resource Aware] Skipping LM improvement (CPU:{cpu_p:.1f}% MEM:{mem_p:.1f}%).")
            return

        print("🧬 [LM Improvement] Exploring new architectures...")

        # 1. Build validation set and evaluate current model
        val_sentences = self._build_lm_validation_set(50)
        current_loss = self.evaluate_language_model(self.language_model, val_sentences)
        print(f"   Current model validation loss: {current_loss:.4f}")

        # 2. Define candidate architectures (variations)
        candidates = []
        base_hidden = self.language_model.hidden_dim
        base_layers = getattr(self.language_model, 'num_layers', 1)
        base_cell = getattr(self.language_model, 'cell_type', 'lstm')

        # Try doubling hidden size
        candidates.append({
            'hidden_dim': base_hidden * 2,
            'num_layers': base_layers,
            'cell_type': base_cell,
            'learning_rate': self.language_model.learning_rate,
            'description': f"double hidden ({base_hidden}→{base_hidden*2})"
        })
        # Try adding a layer
        candidates.append({
            'hidden_dim': base_hidden,
            'num_layers': base_layers + 1,
            'cell_type': base_cell,
            'learning_rate': self.language_model.learning_rate * 0.5,  # lower LR for deeper net
            'description': f"add layer ({base_layers}→{base_layers+1})"
        })
        # Try switching cell type (if LSTM -> GRU, else LSTM)
        other_cell = 'gru' if base_cell == 'lstm' else 'lstm'
        candidates.append({
            'hidden_dim': base_hidden,
            'num_layers': base_layers,
            'cell_type': other_cell,
            'learning_rate': self.language_model.learning_rate,
            'description': f"switch to {other_cell}"
        })
        # Try smaller hidden but more layers (e.g., half hidden, double layers)
        if base_hidden > 32:
            candidates.append({
                'hidden_dim': base_hidden // 2,
                'num_layers': base_layers * 2,
                'cell_type': base_cell,
                'learning_rate': self.language_model.learning_rate * 0.5,
                'description': f"deeper+smaller ({base_hidden//2} hidden, {base_layers*2} layers)"
            })

        best_model = None
        best_loss = current_loss
        best_logic_id = None
        best_rule = None

        # 3. For each candidate, generate, train briefly, evaluate
        for cand in candidates:
            print(f"   Trying: {cand['description']}")
            logic_id = f"lm_improvement_{int(time.time())}_{np.random.randint(1000)}"
            rule = {
                'logic_id': logic_id,
                'type': 'language_model',
                'description': cand['description'],
                'hidden_dim': cand['hidden_dim'],
                'num_layers': cand['num_layers'],
                'cell_type': cand['cell_type'],
                'emb_dim': self.language_model.emb_dim,
                'learning_rate': cand['learning_rate'],
                'max_vocab': self.language_model.max_vocab,
                'source': 'propose_lm_improvement'
            }
            filepath = self.logic_inheritance.generate_logic_file(rule)
            if not filepath:
                continue

            code = self.code_generator.generate(rule)
            if not code:
                continue
            code_path = self.code_generator.save_code_file(logic_id, code)

            # Dynamically import
            spec = importlib.util.spec_from_file_location(f"lm_cand_{logic_id}", code_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"      Error loading: {e}")
                continue

            new_model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, LanguageModel) and attr != LanguageModel:
                    new_model_class = attr
                    break
            if new_model_class is None:
                continue

            new_model = new_model_class()

            # Copy vocabulary
            for word in self.language_model.word_to_idx:
                if word not in new_model.word_to_idx and word not in ('<PAD>','<BOS>','<EOS>','<UNK>'):
                    new_model.add_word_to_vocab(word)

            # Quick training (few sentences)
            train_sentences = []
            for ep in self.episodic_memory.episodes[-200:]:
                user = ep.get('user_input', '').strip()
                if user and len(user.split()) <= 30:
                    train_sentences.append(user)
            np.random.shuffle(train_sentences)
            for sent in train_sentences[:20]:
                try:
                    new_model.train_on_sentence(sent, lr=new_model.learning_rate * 0.1)
                except Exception:
                    continue

            new_loss = self.evaluate_language_model(new_model, val_sentences)
            print(f"      Validation loss: {new_loss:.4f}")

            if new_loss < best_loss * 0.95:  # at least 5% better
                best_loss = new_loss
                best_model = new_model
                best_logic_id = logic_id
                best_rule = rule

        # 4. If a better model found, replace current
        if best_model is not None:
            print(f"   ✅ Best model ({best_rule['description']}) outperforms current. Replacing language model.")
            self.language_model = best_model
            self.language_model.save(os.path.join(STATE_DIR, 'language_model.pkl'))
            self.logic_inheritance.confirm_logic(best_logic_id, is_correct=True)
        else:
            print(f"   No better model found. Keeping current.")
            # Still create a doubt for each candidate? Maybe only for the closest one.
            # For simplicity, we'll just note.
                
    def idle_reflection(self) -> str:
        """Autonomous introspection – now includes current state label if known."""
        if not self.word_history:
            return "I have no knowledge yet. I wonder what the world is like?"

        total_words = len(self.word_history)
        avg_entropy = np.mean([self.concept_model.entropy_word(w) for w in self.word_history])
        total_concepts = len(self.concept_model.concept_counts)

        # Most curious word
        word_scores = {}
        for w in self.word_history:
            last = self.word_history[w][-1]
            factors = last.get('curiosity_factors')
            if factors:
                farr = self._align_factor_vector(factors.as_array(self.meta_learner.factor_names))
                word_scores[w] = self.meta_learner.predict_curiosity(farr)
        most_curious = max(word_scores, key=word_scores.get, default=None)

        hypothesis = ""
        if most_curious:
            hypothesis = self.reasoner.generate_hypothesis(most_curious, {})

        thought = (
            f"[Idle Meta-Thought] I know {total_words} words and have formed {total_concepts} latent concepts. "
            f"My average uncertainty is {avg_entropy:.3f}. "
            f"I have accumulated {self.reward_tracker.total_reward:.2f} intrinsic reward so far. "
        )

        # ---- Mention current state if we have a recent factor vector ----
        num_states = len(self.state_manager.states)
        thought += f"I have learned {num_states} distinct interaction states. "

        if most_curious:
            thought += (
                f"The word I'm most curious about is '{most_curious}' (score={word_scores[most_curious]:.3f}). "
                f"{hypothesis} "
                f"I should ask about it to reduce my GP uncertainty and Bayesian surprise."
            )
        else:
            thought += "I have no strong curiosities at the moment."

        if np.random.rand() < 0.3:
            weights = self.meta_learner.get_weights()
            top_factor = max(weights, key=lambda k: abs(weights[k]))
            thought += f" I've learned that '{top_factor}' is currently the most informative curiosity factor."

        return thought
# ------------------------------------------------------------

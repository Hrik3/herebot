from ..curioimport import *

class HeartPersistenceMixin:
    # ------------------------------------------------------------------
    # Persistence (now with config_manager passed to .load() methods)
    # ------------------------------------------------------------------
    def _update_dynamic_limits(self):
        """Adapt internal limits to available system RAM."""
        try:
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024**3)

            # scale factors
            scale = max(0.5, min(8.0, avail_gb / 4.0))  # normalize around ~4GB baseline

            self._max_cycles_per_session = int(100000 * scale)
            self._max_hypotheses = int(5000 * scale)
            self._max_logic_rules = int(2000 * scale)

        except Exception:
            # fallback defaults
            self._max_cycles_per_session = 100000
            self._max_hypotheses = 5000
            self._max_logic_rules = 2000

    def _metabolic_maintenance(self):
        """
        Apply metabolic cost and phase transitions to all persistent structures.
        Called during idle introspection.
        """
        now = time.time()

        # --- 1. Behavioral rules ---
        for rule in self.behavioral_learner.rules.values():
            self._apply_metabolic_to_item(rule, now)

        # --- 2. Conscience beliefs ---
        for belief in self.conscience_manager.beliefs.values():
            self._apply_metabolic_to_item(belief, now)

        # --- 3. Hypotheses in reasoner ---
        for hyp in self.reasoner.hypotheses:
            self._apply_metabolic_to_dict_item(hyp, now)

        # --- 4. Dynamic curiosity factors ---
        for factor_name, meta in self.meta_learner.factor_metadata.items():
            self._apply_metabolic_to_dict_item(meta, now, key_prefix=("factor", factor_name))

        # --- 5. States ---
        for label, meta in self.state_manager.state_metadata.items():
            self._apply_metabolic_to_dict_item(meta, now, key_prefix=("state", label))

        # --- 6. Logic rules ---
        for logic_id, rule in self.logic_inheritance.confirmed_logic.items():
            self._apply_metabolic_to_dict_item(rule, now, key_prefix=("logic", logic_id))
        for doubt in self.logic_inheritance.pending_doubts:
            rule = doubt['rule']
            self._apply_metabolic_to_dict_item(rule, now, key_prefix=("doubt", doubt['doubt_id']))

    def _apply_metabolic_to_item(self, item, now):
        """Apply metabolic rules to an object with attributes."""
        if item.phase == 'deleted':
            return
        dt = now - (item.last_used if item.last_used else item.created)
        # decay recent_usage exponentially
        decay_factor = 0.99 ** dt
        item.recent_usage *= decay_factor

        # compute effective predictive_gain (could be based on recent_usage and confidence)
        effective_gain = item.predictive_gain * item.recent_usage / (1 + dt)

        # maintenance cost (could be type-dependent)
        cost = item.energy_cost

        if effective_gain < cost:
            # move to next phase
            if item.phase == 'active':
                item.phase = 'weak'
                print(f"   [Metabolic] {item} moved to weak")
            elif item.phase == 'weak':
                item.phase = 'dormant'
                print(f"   [Metabolic] {item} moved to dormant")
            elif item.phase == 'dormant':
                item.phase = 'archived'
                print(f"   [Metabolic] {item} archived")
            elif item.phase == 'archived':
                item.phase = 'deleted'
                print(f"   [Metabolic] {item} deleted")
                # actually remove from its container? We'll handle separately.

    def _apply_metabolic_to_dict_item(self, d, now, key_prefix=None):
        """Apply metabolic rules to a dictionary item."""
        if d.get('phase') == 'deleted':
            return
        last_used = d.get('last_used')
        created = d.get('created', now - 86400)  # fallback
        dt = now - (last_used if last_used else created)
        recent_usage = d.get('recent_usage', 0.0)
        recent_usage *= (0.99 ** dt)
        d['recent_usage'] = recent_usage

        predictive_gain = d.get('predictive_gain', 0.0)
        cost = d.get('energy_cost', 0.5)

        # ---- HARD OVERRIDE: high-value items survive regardless of usage ----
        if predictive_gain * 2 > cost:
            d['phase'] = 'active'
            return

        # normal effective gain
        effective_gain = max(
            predictive_gain * 0.5,
            predictive_gain * recent_usage / (1 + dt)
        )

        if effective_gain < cost:
            phase = d.get('phase', 'active')
            if phase == 'active':
                d['phase'] = 'weak'
            elif phase == 'weak':
                d['phase'] = 'dormant'
            elif phase == 'dormant':
                d['phase'] = 'archived'
            elif phase == 'archived':
                d['phase'] = 'deleted'
                # Optionally remove from container – we'll leave that for later cleanup.

    def _align_factor_vector(self, vec: np.ndarray) -> np.ndarray:
        return align_vector(vec, len(self.meta_learner.factor_names))
    def resolve_state_label(self, label):
        if label is None:
            return None
        return self.state_manager.resolve_label(label)
    
    def _memory_pressure_backoff(self):
        """Slow cognition when memory is low."""
        try:
            mem = psutil.virtual_memory()
            if mem.available / mem.total < 0.12:   # <12% free
                time.sleep(0.15)
                return True
        except Exception:
            pass
        return False

    def _load_state(self):
        """Load persistent state from disk, including the latent state concept."""
        concept_path = os.path.join(STATE_DIR, 'concept_model.pkl')
        meta_path = os.path.join(STATE_DIR, 'meta_learner.npz')
        gp_path = os.path.join(STATE_DIR, 'gp_model.pkl')
        reward_path = os.path.join(STATE_DIR, 'reward_tracker.pkl')
        reasoner_path = os.path.join(STATE_DIR, 'reasoner.pkl')
        history_path = os.path.join(STATE_DIR, 'history.pkl')
        semantic_path = os.path.join(STATE_DIR, 'semantic_learner.pkl')
        syntax_path = os.path.join(STATE_DIR, 'syntax_learner.pkl')
        # ---- Load Conscience & Logic ----
        conscience_path = os.path.join(STATE_DIR, 'conscience_manager.pkl')
        if os.path.exists(conscience_path):
            self.conscience_manager._load()
        self.logic_inheritance._load()
        if os.path.exists(concept_path):
            self.concept_model = HierarchicalConceptModel.load(concept_path, config_manager=self.config_manager)
        if os.path.exists(meta_path):
            self.meta_learner.load(meta_path)
        if os.path.exists(gp_path):
            with open(gp_path, 'rb') as f:
                self.gp_model = pickle.load(f)
            self.gp_model.config_manager = self.config_manager
        if os.path.exists(reward_path):
            self.reward_tracker.load(reward_path)
            self.reward_tracker.config_manager = self.config_manager
        if os.path.exists(reasoner_path):
            self.reasoner.load(reasoner_path)
            self.reasoner.config_manager = self.config_manager
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.word_history = pickle.load(f)
        if os.path.exists(semantic_path):
            self.semantic_learner.load(semantic_path)
            self.semantic_learner.config_manager = self.config_manager
        if os.path.exists(syntax_path):
            self.syntax_learner.load(syntax_path)
            self.syntax_learner.config_manager = self.config_manager
        # Load new idle‑task related data
        world_model_path = os.path.join(STATE_DIR, 'world_model.pkl')
        if os.path.exists(world_model_path):
            with open(world_model_path, 'rb') as f:
                self.world_model = pickle.load(f)
        policy_path = os.path.join(STATE_DIR, 'policy_model.pkl')
        if os.path.exists(policy_path):
            with open(policy_path, 'rb') as f:
                self.policy_model = pickle.load(f)
        user_model_path = os.path.join(STATE_DIR, 'user_model.pkl')
        if os.path.exists(user_model_path):
            with open(user_model_path, 'rb') as f:
                self.user_model = pickle.load(f)
        research_queue_path = os.path.join(STATE_DIR, 'research_queue.pkl')
        if os.path.exists(research_queue_path):
            with open(research_queue_path, 'rb') as f:
                self.research_queue = pickle.load(f)
        identity_history_path = os.path.join(STATE_DIR, 'identity_history.pkl')
        if os.path.exists(identity_history_path):
            with open(identity_history_path, 'rb') as f:
                self.identity_history = pickle.load(f)
        state_manifold_path = os.path.join(STATE_DIR, 'state_manifold.pkl')
        if os.path.exists(state_manifold_path):
            with open(state_manifold_path, 'rb') as f:
                self.state_manifold = pickle.load(f)
        # ---- Load identity aliases ----
        aliases_path = os.path.join(STATE_DIR, 'identity_aliases.json')
        if os.path.exists(aliases_path):
            try:
                with open(aliases_path, 'r') as f:
                    self.identity_aliases = set(json.load(f))
            except Exception as e:
                print(f"Could not load identity aliases: {e}")
        guilt_path = os.path.join(STATE_DIR, 'guilt_tracker.pkl')
        alias_path = os.path.join(STATE_DIR, 'state_aliases.json')
        if os.path.exists(alias_path):
            with open(alias_path, 'r') as f:
                self.state_manager.aliases = json.load(f)
        if os.path.exists(guilt_path):
            self.guilt_tracker.load(guilt_path)
        # Ensure conscience_manager shares the same guilt_tracker instance
        if self.conscience_manager:
            self.conscience_manager.guilt_tracker = self.guilt_tracker
        # Load generated code feedback
        self.generated_code_feedback._load()
        if hasattr(self, 'generated_plugin_manager'):
            self.generated_plugin_manager._load()
        # ---- Load language model ----
        lm_path = os.path.join(STATE_DIR, 'language_model.pkl')
        if os.path.exists(lm_path):
            try:
                self.language_model.load(lm_path)
                print(f"📖 Loaded language model with {self.language_model.vocab_size} words, {self.language_model.steps} steps.")
            except Exception as e:
                print(f"Could not load language model: {e}")
        else:
            # Build initial vocabulary from existing word embeddings
            for word in self.semantic_learner.word_embeddings:
                self.language_model.add_word_to_vocab(word)
        # ---- Load current latent state concept (multi‑state) ----
        current_state_path = os.path.join(STATE_DIR, 'current_state.json')
        if os.path.exists(current_state_path):
            try:
                with open(current_state_path, 'r') as f:
                    data = json.load(f)
                labels = data.get('labels', [])
                if labels:
                    self.current_state_labels = [
                        self.resolve_state_label(l)
                        for l in self.current_state_labels
                        if self.resolve_state_label(l) is not None
                    ]
                    self.current_state_labels = labels
                else:
                    # backward compatibility with old single-label format
                    single = data.get('label')
                    if single:
                        self.current_state_labels = [
                            self.resolve_state_label(l)
                            for l in self.current_state_labels
                            if self.resolve_state_label(l) is not None
                        ]
                        self.current_state_labels = [single]
                emb = data.get('embedding')
                if emb is not None:
                    self.current_state_embedding = np.array(emb)
            except Exception as e:
                self.current_state_labels = []
                self.current_state_embedding = None
        else:
            self.current_state_labels = [
                self.resolve_state_label(l)
                for l in self.current_state_labels
                if self.resolve_state_label(l) is not None
            ]
            self.current_state_labels = []
            self.current_state_embedding = None
        token_profile_path = os.path.join(STATE_DIR, 'token_profile.pkl')
        if os.path.exists(token_profile_path):
            try:
                with open(token_profile_path, 'rb') as f:
                    data = pickle.load(f)
                    self.token_profile = data.get('token_profile', {})
                    loaded_stats = data.get('token_profile_stats', {})
                    # Ensure new stats exist (backward compatibility)
                    self.token_profile_stats = {
                        'count_mean': loaded_stats.get('count_mean', 0.0),
                        'count_std': loaded_stats.get('count_std', 1.0),
                        'ctx_mean': loaded_stats.get('ctx_mean', 0.0),
                        'ctx_std': loaded_stats.get('ctx_std', 1.0),
                        'surp_mean': loaded_stats.get('surp_mean', 0.0),
                        'surp_std': loaded_stats.get('surp_std', 1.0),
                        'drift_mean': loaded_stats.get('drift_mean', 0.0),
                        'drift_std': loaded_stats.get('drift_std', 1.0),
                    }
                    self._cold_start_threshold = data.get('_cold_start_threshold', 100)
            except Exception as e:
                print(f"Could not load token profiles: {e}")
                self.token_profile = {}
                self.token_profile_stats = {
                    'count_mean': 0.0, 'count_std': 1.0,
                    'ctx_mean': 0.0, 'ctx_std': 1.0,
                    'surp_mean': 0.0, 'surp_std': 1.0,
                    'drift_mean': 0.0, 'drift_std': 1.0,
                }
                self._cold_start_threshold = 100

    def _save_state(self):
        """Persist current state, including the persistent latent state concept."""
        now = time.time()
        if now - getattr(self, "_last_save_time", 0) < getattr(self, "_min_save_interval", 20):
            return
        self._last_save_time = now

        try:
            self.concept_model.save(os.path.join(STATE_DIR, 'concept_model.pkl'))
            self.meta_learner.save(os.path.join(STATE_DIR, 'meta_learner.npz'))
            with open(os.path.join(STATE_DIR, 'gp_model.pkl'), 'wb') as f:
                pickle.dump(self.gp_model, f)
            self.reward_tracker.save(os.path.join(STATE_DIR, 'reward_tracker.pkl'))
            self.reasoner.save(os.path.join(STATE_DIR, 'reasoner.pkl'))
            with open(os.path.join(STATE_DIR, 'history.pkl'), 'wb') as f:
                pickle.dump(self.word_history, f)
            self.semantic_learner.save(os.path.join(STATE_DIR, 'semantic_learner.pkl'))
            self.syntax_learner.save(os.path.join(STATE_DIR, 'syntax_learner.pkl'))
            self.identity_manager.save()
            # ---- Save behavioral rule learner ----
            self.behavioral_learner.save()
            # ---- Save identity aliases ----
            aliases_path = os.path.join(STATE_DIR, 'identity_aliases.json')
            with open(aliases_path, 'w') as f:
                json.dump(list(self.identity_aliases), f)
            # Save new idle‑task related data
            with open(os.path.join(STATE_DIR, 'world_model.pkl'), 'wb') as f:
                pickle.dump(self.world_model, f)
            with open(os.path.join(STATE_DIR, 'policy_model.pkl'), 'wb') as f:
                pickle.dump(self.policy_model, f)
            with open(os.path.join(STATE_DIR, 'user_model.pkl'), 'wb') as f:
                pickle.dump(self.user_model, f)
            with open(os.path.join(STATE_DIR, 'research_queue.pkl'), 'wb') as f:
                pickle.dump(self.research_queue, f)
            with open(os.path.join(STATE_DIR, 'identity_history.pkl'), 'wb') as f:
                pickle.dump(self.identity_history, f)
            with open(os.path.join(STATE_DIR, 'state_manifold.pkl'), 'wb') as f:
                pickle.dump(self.state_manifold, f)
            self.conscience_manager.save()
            self.logic_inheritance.save()
            self.guilt_tracker.save(os.path.join(STATE_DIR, 'guilt_tracker.pkl'))
            self.generated_code_feedback.save()
            if hasattr(self, 'generated_plugin_manager'):
                self.generated_plugin_manager.save()
            # ---- Save language model ----
            self.language_model.save(os.path.join(STATE_DIR, 'language_model.pkl'))
            # ---- Save current latent state concept (multi‑state) ----
            current_state_path = os.path.join(STATE_DIR, 'current_state.json')
            # ---- Save token profiles ----
            token_profile_path = os.path.join(STATE_DIR, 'token_profile.pkl')
            with open(token_profile_path, 'wb') as f:
                pickle.dump({
                    'token_profile': self.token_profile,
                    'token_profile_stats': self.token_profile_stats,
                    '_cold_start_threshold': self._cold_start_threshold,
                }, f)
            data = {
                'labels': self.current_state_labels,
                'embedding': self.current_state_embedding.tolist() if self.current_state_embedding is not None else None
            }
            alias_path = os.path.join(STATE_DIR, 'state_aliases.json')
            with open(alias_path, 'w') as f:
                json.dump(self.state_manager.aliases, f)
            with open(current_state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}. State may be incomplete.")
            print(traceback.format_exc())
    # ------------------------------------------------------------------
    # Hot‑reload: refresh all components from config file
    # ------------------------------------------------------------------
    def reload_config(self):
        """Reload the configuration file and update all component parameters."""
        self.config_manager.reload()
        # Update each component
        self.concept_model.update_parameters_from_config()
        self.gp_model.update_parameters_from_config()
        self.meta_learner.update_parameters_from_config()
        self.reward_tracker.update_parameters_from_config()
        self.semantic_learner.update_parameters_from_config()
        self.syntax_learner.update_parameters_from_config()
        self.consciousness.update_parameters_from_config()
        self.action_generator.update_parameters_from_config()
        self.reasoner.update_parameters_from_config()
        self.factor_engine.update_parameters_from_config()
        self.state_manager.update_parameters_from_config()
        self.episodic_memory.update_parameters_from_config()
        if hasattr(self, '_idle_reasoner'):
            self._idle_reasoner.update_parameters_from_config()
        # Update engine parameters
        self.reconfigure_interval = self.config_manager.get('engine.reconfigure_interval', 20)
    # ------------------------------------------------------------------
    # Emotional self‑reconfiguration
    # ------------------------------------------------------------------
    def _maybe_reconfigure(self):
        """Generate a new config based on the current factor vector and reload it."""
        # Obtain the latest factor vector (we don't store it, so recompute from last token analyses?)
        # Since this is called at the end of process_input, we have factor_dict.
        # We'll access it from the local variable in process_input – we need to pass it.
        # To keep it simple, we recompute factor vector from the latest state.
        # In practice, we store the last factor_vector in self.last_factor_vector.
        if hasattr(self, '_last_factor_vector'):
            factor_vector = self._last_factor_vector
            self.config_manager.generate_modulated_config(factor_vector)
            self.reload_config()
            # Allow conscience to adjust its own thresholds
            self.conscience_manager.evolve_parameters()
            print("🧬 [Emotional Reconfiguration] Configuration updated based on internal state.")
    # ------------------------------------------------------------------
    # Feature extraction helpers (unchanged)
    # ------------------------------------------------------------------

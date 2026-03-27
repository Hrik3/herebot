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
        effective_gain = predictive_gain * recent_usage / (1 + dt)

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

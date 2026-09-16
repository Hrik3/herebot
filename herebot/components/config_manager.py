from importglobal import os, json,copy,np,Optional,List
from .directory_handler import STATE_DIR
# ------------------------------------------------------------
# Configuration Manager – loads parameters from active_config.json
# ------------------------------------------------------------
class ConfigManager:
    """
    Manages external JSON configuration for the Herebot.
    Loads from curiosity_state/active_config.json, provides defaults,
    supports deep updates, saving, and generating new configs via emotional modulation.
    """
    DEFAULT_CONFIG_PATH = os.path.join(STATE_DIR, "active_config.json")
    
    # Default configuration values
    DEFAULTS = {
        "hierarchical_concept_model": {
            "alpha": 0.5,
            "beta": 1.0,
            "active_probability_threshold": 0.01,
        },
        "gaussian_process": {
            "input_dim": 5,
            "n_features": 100,
            "length_scale": 1.0,
            "sigma_noise": 0.1,
        },
        "meta_curiosity_learner": {
            "n_factors": 9,
        },
        "intrinsic_reward_tracker": {
            # no parameters currently, placeholder
        },
        "semantic_learner": {
            "emb_dim": 64,
            "alpha": 0.1,
        },
        "syntax_learner": {
            # no parameters currently, placeholder
        },
        "consciousness_model": {
            # no parameters currently, placeholder
        },
        "action_generator": {
            # no parameters currently, placeholder
        },
        "reasoning_engine": {
            "enable_type_hierarchy": True,
            "enable_operators": True,
            "enable_rule_engine": True,
            "enable_constraint_propagation": True,
            "enable_schema_learning": True,
        },
        "factor_engine": {
            # no parameters currently, placeholder
        },
        "state_manager": {
            "state_dir": "states",
            "conflict_threshold": 0.85,
            "active_probability_threshold": 0.01,
            "selection_tie_tolerance": 1e-9,
            "state_relation_similarity_threshold": 0.85,
            "state_regroup_similarity_threshold": 0.92,
            "state_regroup_min_observations": 3,
            "state_regroup_interval_interactions": 10,
            "state_group_token_evidence_floor": 2,
            "state_group_head_label_max_words": 3,
            "legacy_interactive_conflicts": False,
        },
        "probabilistic_labeling": {
            "horizon_base": 80.0,
            "horizon_log_scale": 0.25,
            "min_decay_scale": 0.25,
            "max_decay_scale": 2.0,
            "archive_pressure_threshold": 0.98,
            "epsilon": 1e-9,
            "tier_count": 5,
        },
        "episodic_memory": {
            "capacity": 10000,
        },
        "idle_reasoner": {
            "short_idle_threshold": 60.0,
            "long_idle_threshold": 400.0,
            "enable_world_model_dreaming": True,
            "enable_active_memory_rehearsal": True,
            "enable_curiosity_landscape_optimization": True,
            "enable_concept_graph_compression": True,
            "enable_autonomous_hypothesis_planning": True,
            "enable_identity_stabilization": True,
            "enable_cross_modal_alignment": True,
            "enable_exploration_policy_distillation": True,
            "enable_cognitive_drive_evolution": True,
            "enable_counterfactual_reasoning": True,
            "enable_semantic_self_talk": True,
            "enable_state_space_topology_learning": True,
            "enable_knowledge_gap_frontier": True,
            "enable_social_cognition_simulation": True,
            "enable_long_horizon_planning": True,
            "world_model_num_synthetic": 5,
            "dreaming_ngram_order": 3, 
            "rehearsal_batch_size": 20,
            "enable_behavioral_rule_discovery": True,
            "enable_logic_inheritance": True,
            "enable_contextual_sentiment_analysis": True,
            "curiosity_correlation_threshold": 0.9,
            "identity_cluster_k": 3,
            "topology_n_clusters": 5,
            "enable_word_state_propagation": True,
            "candidate_ledger": {
                "diversity_target": 3,
                "candidate_min_pmi": 0.15,
                "min_distinct_episodes": 3,
                "min_commit_margin": 0.25,
                "min_kl_surprise": 0.05,
                "min_grounded_distinct_episodes": 2,
                "min_grounded_evidence_fraction": 0.15,
                "candidate_decay": 0.9,
                "candidate_prune_floor": 0.001,
                "candidate_max_age_cycles": 30,
            },
        },
        "conscience_manager": {
            "trust_threshold_for_addition": 0.85,
            "min_evidence": 5,
            "decay_rate": 0.99,
            "max_beliefs": 50,
            "evolution_trigger": 10,
        },
        "engine": {
            "reconfigure_interval": 20,
            "state_similarity_threshold": 0.85,
            "state_learning_cooldown_interactions": 30,
        },
        "adaptive_control_space": {
            "variance_threshold": 0.05,
            "utility_threshold": 0.01,
            "max_factors": 50,
            "expansion_cooldown": 100,
        },
        # Emotional modulation rules: mapping from factor name to config path and transform
        "modulation_rules": {
            "curiosity": {
                "target": "hierarchical_concept_model.alpha",
                "transform": "linear",
                "scale": 0.5,
                "offset": 0.5,
                "min": 0.1,
                "max": 2.0,
            },
            "uncertainty": {
                "target": "gaussian_process.sigma_noise",
                "transform": "linear",
                "scale": 0.2,
                "offset": 0.1,
                "min": 0.01,
                "max": 0.5,
            },
            "threat": {
                "target": "meta_curiosity_learner.n_factors",
                "transform": "identity",
            },
            # more rules can be added later
        },
        "language_model": {
            "enabled": True,
            "min_steps_for_use": 100,
            "train_batch_size": 20,
            "improvement_interval": 5000,
        },
    }  
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_defaults()
        self.load()
    
    def _load_defaults(self) -> dict:
        """Return a deep copy of default configuration."""
        return copy.deepcopy(self.DEFAULTS)
    
    def load(self):
        """Load configuration from file, merging with defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    try:
                        user_config = json.load(f)
                    except Exception as e:
                        print(f"[WARN] Config corrupted, loading defaults: {e}")
                        user_config = {}
                self.config = self.deep_update(self.config, user_config)
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
        else:
            # No config file, save defaults
            self.save()
    
    def save(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        tmp_path = self.config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        os.replace(tmp_path, self.config_path)
    
    def reload(self):
        """Reload configuration from disk (hot reload)."""
        self.load()
    
    def get(self, key: str, default=None):
        """
        Get a configuration value using dot notation.
        e.g., config.get('hierarchical_concept_model.alpha')
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value):
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()
    
    def deep_update(self, base: dict, update: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self.deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def generate_modulated_config(self, factor_vector: np.ndarray, factor_names: List[str] = None):
        """
        Generate a new configuration based on the current factor vector
        and the modulation rules defined in DEFAULTS.
        Saves the new config to disk and returns the config dict.        """


        if factor_names is None:
            factor_names = ["curiosity","threat","trust","confidence",
                            "uncertainty","energy","attention",
                            "comprehension","fluency","cognitive_load"]
        
        # Start from current config
        new_config = copy.deepcopy(self.config)
        
        rules = self.config.get("modulation_rules", self.DEFAULTS["modulation_rules"])
        
        for factor_name, value in zip(factor_names, factor_vector):
            if factor_name not in rules:
                continue
            rule = rules[factor_name]
            target_path = rule.get("target")
            if not target_path:
                continue
            
            # Compute modulated value
            transform = rule.get("transform", "linear")
            if transform == "linear":
                scale = rule.get("scale", 1.0)
                offset = rule.get("offset", 0.0)
                modulated = scale * value + offset
            elif transform == "identity":
                modulated = value
            elif transform == "inverse":
                modulated = 1.0 - value
            else:
                modulated = value
            
            # Clamp
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val is not None:
                modulated = max(min_val, modulated)
            if max_val is not None:
                modulated = min(max_val, modulated)
            
            # Apply to nested dict
            keys = target_path.split('.')
            target = new_config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = modulated
        
        # Save the new config
        self.config = new_config
        self.save()
        return new_config

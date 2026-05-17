from .curioimport import *
from .cuengine import (
    HeartCuriosityMixin,
    HeartIdleMixin,
    HeartLanguageMixin,
    HeartPersistenceMixin,
    HeartProcessingMixin,
)


class CuriosityEngine(HeartPersistenceMixin, HeartCuriosityMixin, HeartProcessingMixin, HeartIdleMixin, HeartLanguageMixin):
    """ss
    Integrates all components: HDP, GP uncertainty, Bayesian surprise,
    intrinsic reward, meta-learner, reasoning, AND now factor-based
    state snapshots with interactive human labeling.

    Now config‑aware and self‑reconfiguring via emotional modulation.
    """
    def __init__(self, load_state: bool = True, config_manager: Optional[ConfigManager] = None):
        # Configuration
        if config_manager is None:
            self.config_manager = ConfigManager()  # loads from default path
        else:
            self.config_manager = config_manager

        # Real engines
        self.ci_engine = CIEngine()
        self.phon_engine = PhonotacticEngine()
        self.diag_engine = DiagnosticEngine()

        # Probabilistic concept model
        self.concept_model = HierarchicalConceptModel(config_manager=self.config_manager)

        # Gaussian Process uncertainty model
        self.gp_model = GaussianProcessUncertainty(config_manager=self.config_manager)

        # Curiosity meta-learner (9 factors)
        self.meta_learner = MetaCuriosityLearner(config_manager=self.config_manager)

        # Intrinsic reward tracker
        self.reward_tracker = IntrinsicRewardTracker(config_manager=self.config_manager)

        # Reasoning engine
        from components.resoning import ReasoningEngine
        self.reasoner = ReasoningEngine(config_manager=self.config_manager)
        self._last_state_learning_time = 0
        # ---- Language & consciousness modules ----
        self.semantic_learner = SemanticLearner(config_manager=self.config_manager)
        self.syntax_learner = SyntaxLearner(config_manager=self.config_manager)
        from .mind import ActionGenerator
        self.action_generator = ActionGenerator(
            self.semantic_learner, self.syntax_learner, self.reasoner, self, config_manager=self.config_manager
        )
        self.consciousness = ParameterRevision(config_manager=self.config_manager)
        self.episodic_memory = EpisodicMemory(config_manager=self.config_manager)
        # ---- Parallel reasoning ----
        from components.resoning import ParallelReasoner
        self.parallel_reasoner = ParallelReasoner(self.reasoner, self.meta_learner)
        # ------------------------------------------
        self._last_feedback_ask_time = 0
        # ---- Adaptive Control Space ----
        self.control_space = AdaptiveControlSpace(config_manager=self.config_manager)
        # Register listener for new control factors → become curiosity factors
        self.control_space.add_listener(self._on_new_control_factor)
        # ---------------------------------

        # ---- Factor engine & State manager ----
        self.factor_engine = FactorEngine(control_space=self.control_space,
                                          config_manager=self.config_manager)
        state_dir = self.config_manager.get('state_manager.state_dir', 'states')
        self.state_manager = StateManager(state_dir=state_dir, config_manager=self.config_manager)
        # ---------------------------------------
        
        # ---- Identity persistence module ----
        emb_dim = self.config_manager.get('semantic_learner.emb_dim', 64)
        self.identity_manager = IdentityManager(emb_dim=emb_dim)
        # ---------------------------------------

        # ---- Persistent latent state concept ----
        self.current_state_labels = []          # label of the currently recognized state
        self.current_state_embedding = None      # embedding of that state (aligned to emb_dim)
        # ---------------------------------------

        # Memory of past observations for each word
        self.word_history: Dict[str, List[Dict]] = defaultdict(list)

        # Interaction history
        self.interaction_history: List[str] = []
        self.last_interaction_time = time.time()
        self.last_asked_word = None      # for definition learning

        # Self‑reconfiguration counter
        self.interaction_counter = 0
        self.reconfigure_interval = self.config_manager.get('engine.reconfigure_interval', 20)

        # State persistence

        # --- Trust-based language feedback tracking ---
        self.expecting_feedback = False
        self.feedback_attempt_word = None
        self.feedback_attempt_response = None
        self.feedback_timestamp = None
        # --- Behavioral rule confirmation tracking ---
        self.expecting_rule_confirmation = False
        self.pending_rule_id = None
        # ---- Response feedback request tracking ----
        self.expecting_response_feedback = False
        self.pending_feedback_context = None   # store the FeedbackEntry being rated
        self.feedback_ask_probability = 0.3    # probability to ask for feedback (if not confident)
        # ----------------------------------------------
        # --- Last factor dict (for dynamic curiosity factors) ---
        self._last_factor_dict = None
        # ---- Add dynamic curiosity factor for state relevance ----
        if "state_relevance" not in self.meta_learner.factor_names:
            self.meta_learner.add_factor(
                "state_relevance",
                initial_mu=0.0,
                compute_function=self._compute_state_relevance
            )
        self._expected_next_state = None
        self._last_action = None        
        # ------------------------------------------------------------
        # New attributes for advanced idle‑time capabilities
        # ------------------------------------------------------------
        # World model for dreaming (simple bigram model)
        self.world_model = None          # will be built on first use
        # Rehearsal buffer for prioritized replay
        self.rehearsal_buffer = []      # list of episode indices / scores
        # Distilled exploration policy (linear classifier weights)
        self.policy_model = None        # (coeffs, intercept) or None
        # User model for social cognition (naive bayes parameters)
        self.user_model = {"pos_mean": None, "neg_mean": None, "pos_count": 0, "neg_count": 0}
        # Long‑term research agenda (priority queue of queries)
        self.research_queue = []        # list of dicts with 'query', 'ig', 'target'
        # History of identity vectors for stabilization
        self.identity_history = []      # list of np.ndarray
        # Episodic embeddings for cross‑modal alignment
        self.episodic_embeddings = {}  # timestamp -> {'state': np.array, 'text_emb': np.array}
        # State manifold centroids (macro‑states)
        self.state_manifold = None     # list of centroid vectors
        # Flag to avoid running all idle tasks at once
        # ---- Structural entropy gate: token behavioral profiles ----
        self.token_profile = {}  # token -> dict with count, context_set, avg_surprise, avg_info_gain, embedding_drift, last_embedding
        self.token_profile_stats = {
            'count_mean': 0.0,
            'count_std': 1.0,
            'ctx_mean': 0.0,
            'ctx_std': 1.0,
            'surp_mean': 0.0,
            'surp_std': 1.0,
            'drift_mean': 0.0,
            'drift_std': 1.0,
        }
        self._cold_start_threshold = 100  # number of unique tokens before cold start bias ends
        self._idle_tasks_last_run = 0
        self.pending_sentence_query = None   # used by ActionGenerator
        self.pending_solution = None   # for equation solving results
        # ---- Logic doubt feedback ----
        self.expecting_logic_feedback = False
        self.pending_logic_doubt = None

        # ---- Procedure learning ----
        self.expecting_procedure_steps = False
        self.pending_problem_expr = None
        self.recorded_steps = []
        self.pending_teaching_request = False
        self.pending_teaching_expr = None

        # ========== CHAT MODE ==========
        self.chat_mode_active = False
        self.chat_mode_trigger = "switch chat"   # exact phrase to enter/exit
        self.chat_history = []                  # recent turns for context
        # ================================
        # ========== BEHAVIORAL RULE LEARNER ==========
        from components.rules import BehavioralRuleLearner
        self.behavioral_learner = BehavioralRuleLearner(config_manager=self.config_manager)
        # ---- Conscience & Logic Inheritance ----
        self.conscience_manager = ConscienceManager(config_manager=self.config_manager)

        # ---- Code generation & guilt ----
        self.code_generator = CodeGenerator()
        self.shadow_executor = ShadowExecutor()
        # Only create a new GuiltTracker if one wasn't loaded from disk
        if not hasattr(self, 'guilt_tracker') or self.guilt_tracker is None:
            self.guilt_tracker = GuiltTracker()
        self.generated_code_feedback = GeneratedCodeFeedback()
        # Ensure conscience_manager also uses the same guilt_tracker
        if self.conscience_manager:
            self.conscience_manager.guilt_tracker = self.guilt_tracker
        # Add conscience_active to base control factors
        if "conscience_active" not in self.control_space.base_factors:
            self.control_space.base_factors.append("conscience_active")
        self.identity_aliases = set()   # store learned aliases for self-reference
        self.logic_inheritance = LogicInheritanceManager(
            config_manager=self.config_manager,
            conscience_manager=self.conscience_manager
        )
        # ---- Language model for fluent chat ----
        self.language_model = LSTMLanguageModel.create_default(self.config_manager)
        self.lm_config = {
            'enabled': self.config_manager.get('language_model.enabled', True),
            'min_steps_for_use': self.config_manager.get('language_model.min_steps_for_use', 100),
            'train_batch_size': self.config_manager.get('language_model.train_batch_size', 20),
            'improvement_interval': self.config_manager.get('language_model.improvement_interval', 5000),
        }
        # --- Cognition watchdog (RAM-adaptive) ---
        self._cycle_counter = 0
        self._update_dynamic_limits()

        try:
            mem = psutil.virtual_memory()
            scale = max(0.5, min(8.0, (mem.available/(1024**3))/4.0))
            self._max_interaction_history = int(200000 * scale)
        except Exception:
            self._max_interaction_history = 200000
        self._last_save_time = 0
        self._min_save_interval = 20

        if load_state:
            self._load_state()
        self.structured_world_model = StructuredWorldModel(self.reasoner.knowledge_graph, self.state_manager)
        from .tracker import CuriosityTracker
        self.tracker = CuriosityTracker(graphs_dir="graphs", plot_every=5)

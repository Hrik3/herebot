"""
test.py – Tests for CuriosityEngine in heart.py

Assumes heart.py lives inside its package (e.g. the repo root is on sys.path
so `from <package>.heart import CuriosityEngine` works, or heart.py is importable
directly).  All dependency packages are expected to be present in the environment.

Run:
    python -m pytest test.py -v
"""

import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Import the class under test.
# Adjust the import path to match where heart.py lives in your project.
# If the package is called e.g. `engine`, change to:
#   from engine.heart import CuriosityEngine
# ---------------------------------------------------------------------------
from core.heart import CuriosityEngine  


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_engine() -> CuriosityEngine:
    """Return a fresh CuriosityEngine without loading saved state from disk."""
    return CuriosityEngine(load_state=False)


# ===========================================================================
# __init__ / attribute wiring
# ===========================================================================

class TestInit(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_has_config_manager(self):
        self.assertIsNotNone(self.e.config_manager)

    def test_has_all_sub_engines(self):
        for attr in (
            'ci_engine', 'phon_engine', 'diag_engine',
            'concept_model', 'gp_model', 'meta_learner',
            'reward_tracker', 'reasoner', 'semantic_learner',
            'syntax_learner', 'consciousness', 'action_generator',
            'episodic_memory', 'parallel_reasoner',
            'control_space', 'factor_engine', 'state_manager',
            'identity_manager', 'language_model',
            'behavioral_learner', 'conscience_manager',
            'code_generator', 'shadow_executor', 'guilt_tracker',
            'logic_inheritance', 'structured_world_model', 'tracker',
        ):
            self.assertTrue(hasattr(self.e, attr), f"Missing: {attr}")

    def test_word_history_starts_empty(self):
        self.assertEqual(len(self.e.word_history), 0)

    def test_interaction_history_is_list(self):
        self.assertIsInstance(self.e.interaction_history, list)

    def test_interaction_counter_zero(self):
        self.assertEqual(self.e.interaction_counter, 0)

    def test_chat_mode_inactive(self):
        self.assertFalse(self.e.chat_mode_active)

    def test_feedback_flags_false(self):
        self.assertFalse(self.e.expecting_feedback)
        self.assertFalse(self.e.expecting_rule_confirmation)
        self.assertFalse(self.e.expecting_response_feedback)

    def test_token_profile_is_dict(self):
        self.assertIsInstance(self.e.token_profile, dict)

    def test_token_profile_stats_has_all_keys(self):
        expected = {'count_mean', 'count_std', 'ctx_mean', 'ctx_std',
                    'surp_mean', 'surp_std', 'drift_mean', 'drift_std'}
        self.assertEqual(expected, set(self.e.token_profile_stats))


# ===========================================================================
# _update_dynamic_limits
# ===========================================================================

class TestUpdateDynamicLimits(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_sets_positive_max_cycles(self):
        self.e._update_dynamic_limits()
        self.assertGreater(self.e._max_cycles_per_session, 0)

    def test_sets_positive_max_hypotheses(self):
        self.e._update_dynamic_limits()
        self.assertGreater(self.e._max_hypotheses, 0)

    def test_sets_positive_max_logic_rules(self):
        self.e._update_dynamic_limits()
        self.assertGreater(self.e._max_logic_rules, 0)

    def test_all_are_ints(self):
        self.e._update_dynamic_limits()
        self.assertIsInstance(self.e._max_cycles_per_session, int)
        self.assertIsInstance(self.e._max_hypotheses, int)
        self.assertIsInstance(self.e._max_logic_rules, int)


# ===========================================================================
# _memory_pressure_backoff
# ===========================================================================

class TestMemoryPressureBackoff(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_bool(self):
        result = self.e._memory_pressure_backoff()
        self.assertIsInstance(result, bool)


# ===========================================================================
# _normalize_stat
# ===========================================================================

class TestNormalizeStat(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_output_in_zero_one(self):
        for v, m, s in [(0, 0, 1), (5, 0, 2), (-10, 0, 1), (1000, 0, 1)]:
            r = self.e._normalize_stat(v, m, s)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)

    def test_center_gives_half(self):
        self.assertAlmostEqual(self.e._normalize_stat(0, 0, 1), 0.5, places=5)

    def test_zero_std_no_crash(self):
        r = self.e._normalize_stat(1, 0, 0)
        self.assertIsInstance(r, float)

    def test_returns_float(self):
        self.assertIsInstance(self.e._normalize_stat(2, 1, 1), float)


# ===========================================================================
# structural_score
# ===========================================================================

class TestStructuralScore(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_unknown_token_returns_half(self):
        self.assertAlmostEqual(self.e.structural_score("xyzunknown"), 0.5)

    def test_known_token_in_unit_interval(self):
        self.e.token_profile['hello'] = {
            'count': 10, 'context_set': {'a', 'b'},
            'avg_surprise': 0.3, 'embedding_drift': 0.1,
        }
        s = self.e.structural_score('hello')
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_result_clipped_at_one(self):
        self.e.token_profile['hi'] = {
            'count': 1e9, 'context_set': set(range(500)),
            'avg_surprise': 0.0, 'embedding_drift': 0.0,
        }
        self.assertLessEqual(self.e.structural_score('hi'), 1.0)

    def test_cold_start_boost_for_short_token(self):
        self.e._cold_start_threshold = 999999
        self.e.token_profile['go'] = {
            'count': 1, 'context_set': set(),
            'avg_surprise': 0.0, 'embedding_drift': 0.0,
        }
        self.assertGreater(self.e.structural_score('go'), 0.0)


# ===========================================================================
# _align_factor_vector
# ===========================================================================

class TestAlignFactorVector(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_truncates_longer_vector(self):
        n = len(self.e.meta_learner.factor_names)
        result = self.e._align_factor_vector(np.ones(n + 5))
        self.assertEqual(len(result), n)

    def test_pads_shorter_vector(self):
        n = len(self.e.meta_learner.factor_names)
        result = self.e._align_factor_vector(np.ones(max(1, n - 3)))
        self.assertEqual(len(result), n)

    def test_exact_length_unchanged(self):
        n = len(self.e.meta_learner.factor_names)
        vec = np.arange(n, dtype=float)
        result = self.e._align_factor_vector(vec)
        self.assertEqual(len(result), n)
        np.testing.assert_array_equal(result, vec)


# ===========================================================================
# _align_vector_to_emb_dim
# ===========================================================================

class TestAlignVectorToEmbDim(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()
        self.d = self.e.semantic_learner.emb_dim

    def test_truncates_to_emb_dim(self):
        result = self.e._align_vector_to_emb_dim(np.ones(self.d + 10))
        self.assertEqual(len(result), self.d)

    def test_pads_to_emb_dim(self):
        result = self.e._align_vector_to_emb_dim(np.ones(max(1, self.d - 4)))
        self.assertEqual(len(result), self.d)

    def test_padding_is_zeros(self):
        vec = np.ones(1)
        result = self.e._align_vector_to_emb_dim(vec)
        self.assertTrue(np.all(result[1:] == 0.0))

    def test_returns_copy_not_original(self):
        vec = np.arange(self.d, dtype=float)
        result = self.e._align_vector_to_emb_dim(vec)
        self.assertIsNot(result, vec)


# ===========================================================================
# _normalize_vector
# ===========================================================================

class TestNormalizeVector(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_unit_norm(self):
        result = self.e._normalize_vector(np.array([3.0, 4.0]))
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=5)

    def test_zero_vector_no_crash(self):
        result = self.e._normalize_vector(np.zeros(4))
        self.assertEqual(len(result), 4)

    def test_unit_vector_unchanged(self):
        vec = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(self.e._normalize_vector(vec), vec)


# ===========================================================================
# _information_gain
# ===========================================================================

class TestInformationGain(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_unknown_token_returns_zero(self):
        self.assertEqual(self.e._information_gain("brand_new", 0.5), 0.0)

    def test_single_history_returns_zero(self):
        self.e.word_history['cat'] = [{'entropy': 0.8}]
        self.assertEqual(self.e._information_gain('cat', 0.5), 0.0)

    def test_positive_when_entropy_decreases(self):
        self.e.word_history['dog'] = [{'entropy': 0.9}, {'entropy': 0.4}]
        self.assertGreater(self.e._information_gain('dog', 0.4), 0.0)

    def test_clamped_to_zero_when_entropy_rises(self):
        self.e.word_history['cat'] = [{'entropy': 0.1}, {'entropy': 0.9}]
        self.assertGreaterEqual(self.e._information_gain('cat', 0.9), 0.0)


# ===========================================================================
# resolve_state_label
# ===========================================================================

class TestResolveStateLabel(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_none_returns_none(self):
        self.assertIsNone(self.e.resolve_state_label(None))

    def test_non_none_label_returns_something(self):
        result = self.e.resolve_state_label('some_label')
        # Whatever the state_manager resolves to, it must not raise
        self.assertIsNotNone(result)


# ===========================================================================
# _tokenize
# ===========================================================================

class TestTokenize(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_list(self):
        self.assertIsInstance(self.e._tokenize("hello world"), list)

    def test_lowercases(self):
        for t in self.e._tokenize("Hello WORLD"):
            self.assertEqual(t, t.lower())

    def test_empty_string_no_crash(self):
        self.assertIsInstance(self.e._tokenize(""), list)

    def test_includes_numbers(self):
        self.assertIn("42", self.e._tokenize("value is 42"))

    def test_tracks_stats(self):
        self.e._tokenize("some text here")
        self.assertTrue(hasattr(self.e, '_tokenizer_stats'))
        self.assertIn('total', self.e._tokenizer_stats)

    def test_stats_total_increments(self):
        self.e._tokenize("abc def")
        before = self.e._tokenizer_stats['total']
        self.e._tokenize("ghi jkl")
        self.assertEqual(self.e._tokenizer_stats['total'], before + 1)


# ===========================================================================
# _extract_features
# ===========================================================================

class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_dict(self):
        self.assertIsInstance(self.e._extract_features('hello', {}, 0.5, {}), dict)

    def test_has_required_keys(self):
        result = self.e._extract_features('hello', {}, 0.5, {})
        for k in ('confidence', 'stability', 'phon_score', 'char_valid', 'structure'):
            self.assertIn(k, result)

    def test_phon_score_stored(self):
        result = self.e._extract_features('hi', {}, 0.77, {})
        self.assertAlmostEqual(result['phon_score'], 0.77)

    def test_char_valid_lowered_on_character_negative(self):
        result = self.e._extract_features('x', {}, 0.5, {'negatives': ['invalid character pattern']})
        self.assertAlmostEqual(result['char_valid'], 0.1)

    def test_structure_lowered_on_vowel_negative(self):
        result = self.e._extract_features('x', {}, 0.5, {'negatives': ['bad vowel sequence']})
        self.assertAlmostEqual(result['structure'], 0.2)

    def test_all_values_numeric(self):
        for v in self.e._extract_features('test', {'confidence': 0.8}, 0.6, {}).values():
            self.assertIsInstance(v, (int, float))


# ===========================================================================
# _features_to_array
# ===========================================================================

class TestFeaturesToArray(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_ndarray(self):
        self.assertIsInstance(self.e._features_to_array({}), np.ndarray)

    def test_always_length_five(self):
        self.assertEqual(len(self.e._features_to_array({})), 5)

    def test_defaults_to_half(self):
        np.testing.assert_array_almost_equal(self.e._features_to_array({}), np.full(5, 0.5))

    def test_uses_provided_values(self):
        arr = self.e._features_to_array({'confidence': 1.0, 'stability': 0.0})
        self.assertEqual(arr[0], 1.0)
        self.assertEqual(arr[1], 0.0)


# ===========================================================================
# _apply_metabolic_to_dict_item
# ===========================================================================

class TestApplyMetabolicToDictItem(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()
        import time
        self.now = time.time()

    def _item(self, phase='active', recent_usage=0.0, gain=0.0, cost=1.0):
        return {
            'phase': phase, 'recent_usage': recent_usage,
            'predictive_gain': gain, 'energy_cost': cost,
            'created': self.now - 86400,
        }

    def test_deleted_phase_skipped(self):
        d = self._item('deleted')
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'deleted')

    def test_active_moves_to_weak(self):
        d = self._item('active')
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'weak')

    def test_weak_moves_to_dormant(self):
        d = self._item('weak')
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'dormant')

    def test_dormant_moves_to_archived(self):
        d = self._item('dormant')
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'archived')

    def test_archived_moves_to_deleted(self):
        d = self._item('archived')
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'deleted')

    def test_recent_usage_decays(self):
        d = self._item('active', recent_usage=1.0, gain=99.0, cost=0.01)
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertLessEqual(d['recent_usage'], 1.0)

    def test_high_gain_stays_active(self):
        d = self._item('active', recent_usage=1.0, gain=1e6, cost=0.001)
        self.e._apply_metabolic_to_dict_item(d, self.now)
        self.assertEqual(d['phase'], 'active')


# ===========================================================================
# _reinforce_identity
# ===========================================================================

class TestReinforceIdentity(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_crash(self):
        self.e._reinforce_identity(np.ones(16), 0.5)

    def test_high_drift_no_crash(self):
        original = self.e.identity_manager.detect_drift
        self.e.identity_manager.detect_drift = lambda v: 0.99
        self.e._reinforce_identity(np.ones(16), 1.0)
        self.e.identity_manager.detect_drift = original


# ===========================================================================
# reload_config
# ===========================================================================

class TestReloadConfig(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_crash(self):
        self.e.reload_config()

    def test_reconfigure_interval_is_int(self):
        self.e.reload_config()
        self.assertIsInstance(self.e.reconfigure_interval, int)


# ===========================================================================
# _all_tokens_known
# ===========================================================================

class TestAllTokensKnown(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_empty_list_true(self):
        self.assertTrue(self.e._all_tokens_known([]))

    def test_unknown_token_false(self):
        self.e.semantic_learner.word_embeddings = {}
        self.assertFalse(self.e._all_tokens_known(['ghost']))

    def test_known_tokens_true(self):
        self.e.semantic_learner.word_embeddings = {'cat': np.ones(16), 'dog': np.ones(16)}
        self.assertTrue(self.e._all_tokens_known(['cat', 'dog']))

    def test_mixed_returns_false(self):
        self.e.semantic_learner.word_embeddings = {'cat': np.ones(16)}
        self.assertFalse(self.e._all_tokens_known(['cat', 'ghost']))


# ===========================================================================
# _should_ask_feedback
# ===========================================================================

class TestShouldAskFeedback(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_bool(self):
        self.assertIsInstance(self.e._should_ask_feedback(['hello'], {}), bool)


# ===========================================================================
# _generate_feedback_question
# ===========================================================================

class TestGenerateFeedbackQuestion(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_non_empty_string(self):
        result = self.e._generate_feedback_question()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ===========================================================================
# _on_new_control_factor
# ===========================================================================

class TestOnNewControlFactor(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_crash(self):
        self.e._on_new_control_factor('test_factor', lambda *a: 0.0, {})

    def test_adds_ctrl_prefixed_factor(self):
        self.e._on_new_control_factor('myfactor', lambda *a: 0.0, {})
        self.e._on_new_control_factor('myfactor', lambda *a: 0.0, {})

        after = set(self.e.meta_learner.factor_names)

        self.assertIn('ctrl_myfactor', after)


# ===========================================================================
# build_world_model_transitions
# ===========================================================================

class TestBuildWorldModelTransitions(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_crash_empty_episodes(self):
        self.e.episodic_memory.episodes = []
        self.e.build_world_model_transitions()

    def test_no_crash_with_episodes(self):
        self.e.episodic_memory.episodes = [
            {'user_input': 'hello world', 'system_response': 'hi there'},
            {'user_input': 'foo bar', 'system_response': 'baz'},
        ]
        self.e.build_world_model_transitions()


# ===========================================================================
# _get_resource_status
# ===========================================================================

class TestGetResourceStatus(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_five_tuple(self):
        result = self.e._get_resource_status()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    def test_first_two_are_bools(self):
        cpu_ok, mem_ok, throttle, cpu_p, mem_p = self.e._get_resource_status()
        self.assertIsInstance(cpu_ok, bool)
        self.assertIsInstance(mem_ok, bool)

    def test_throttle_in_valid_range(self):
        _, _, throttle, _, _ = self.e._get_resource_status()
        self.assertGreaterEqual(float(throttle), 0.3)
        self.assertLessEqual(float(throttle), 1.0)


# ===========================================================================
# _compute_chat_reward_multiplier
# ===========================================================================

class TestComputeChatRewardMultiplier(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_float(self):
        self.assertIsInstance(float(self.e._compute_chat_reward_multiplier({})), float)

    def test_clamped_between_half_and_two(self):
        for d in [{}, {'curiosity': 1.0}, {'threat': 1.0}, {'curiosity': 1.0, 'trust': 1.0}]:
            r = float(self.e._compute_chat_reward_multiplier(d))
            self.assertGreaterEqual(r, 0.5)
            self.assertLessEqual(r, 2.0)

    def test_high_curiosity_raises_multiplier(self):
        low  = float(self.e._compute_chat_reward_multiplier({'curiosity': 0.0}))
        high = float(self.e._compute_chat_reward_multiplier({'curiosity': 1.0}))
        self.assertGreater(high, low)

    def test_high_threat_lowers_multiplier(self):
        safe   = float(self.e._compute_chat_reward_multiplier({'threat': 0.0}))
        danger = float(self.e._compute_chat_reward_multiplier({'threat': 1.0}))
        self.assertLess(danger, safe)


# ===========================================================================
# perform_state_introspection
# ===========================================================================

class TestPerformStateIntrospection(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_none(self):
        self.assertIsNone(self.e.perform_state_introspection())

    def test_no_crash(self):
        self.e.perform_state_introspection()


# ===========================================================================
# idle_reflection
# ===========================================================================

class TestIdleReflection(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_history_returns_string(self):
        result = self.e.idle_reflection()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_with_word_history_returns_string(self):
        self.e.word_history['apple'] = [{'entropy': 0.5, 'curiosity_factors': None}]
        result = self.e.idle_reflection()
        self.assertIsInstance(result, str)

    def test_mentions_word_count(self):
        self.e.word_history['banana'] = [{'entropy': 0.5, 'curiosity_factors': None}]
        self.assertIn('1', self.e.idle_reflection())


# ===========================================================================
# evaluate_language_model
# ===========================================================================

class TestEvaluateLanguageModel(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_float(self):
        self.assertIsInstance(self.e.evaluate_language_model(validation_sentences=[]), float)

    def test_empty_sentences_returns_inf(self):
        self.assertEqual(self.e.evaluate_language_model(validation_sentences=[]), float('inf'))

    def test_valid_model_returns_finite(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.sentence_to_ids.return_value = [0, 1, 2]
        m.loss.return_value = 0.4
        result = self.e.evaluate_language_model(model=m, validation_sentences=['hello world', 'foo'])
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, 0.4)


# ===========================================================================
# _build_lm_validation_set
# ===========================================================================

class TestBuildLmValidationSet(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_list(self):
        self.assertIsInstance(self.e._build_lm_validation_set(), list)

    def test_bounded_by_size(self):
        self.e.episodic_memory.episodes = [
            {'user_input': f'sentence number {i}'} for i in range(200)
        ]
        self.assertLessEqual(len(self.e._build_lm_validation_set(size=10)), 10)


# ===========================================================================
# train_language_model
# ===========================================================================

class TestTrainLanguageModel(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()
        # Stub resource check so CPU/mem comparisons work with plain floats
        self.e._get_resource_status = lambda: (True, True, 1.0, 30.0, 50.0)

    def test_no_crash_with_no_episodes(self):
        self.e.episodic_memory.episodes = []
        self.e.train_language_model()

    def test_trains_when_enough_episodes(self):
        from unittest.mock import MagicMock
        self.e.episodic_memory.episodes = [
            {'user_input': f'hello world this is sentence {i}', 'system_response': ''}
            for i in range(15)
        ]
        self.e.language_model.train_on_sentence = MagicMock(return_value=0.5)
        self.e.language_model.word_to_idx = {}
        self.e.language_model.add_word_to_vocab = MagicMock()
        self.e.train_language_model()
        self.assertTrue(self.e.language_model.train_on_sentence.called)

    def test_skips_when_cpu_high(self):
        from unittest.mock import MagicMock
        self.e._get_resource_status = lambda: (False, True, 0.3, 95.0, 50.0)
        self.e.language_model.train_on_sentence = MagicMock()
        self.e.train_language_model()
        self.e.language_model.train_on_sentence.assert_not_called()


# ===========================================================================
# _detect_semantic_relations
# ===========================================================================

class TestDetectSemanticRelations(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_no_crash_empty(self):
        self.e._detect_semantic_relations([], [])

    def test_no_crash_unknown_words(self):
        self.e.reasoner.definitions = {}
        self.e._detect_semantic_relations(['foo'], [{'token': 'foo'}])

    def test_calls_hypothesis_for_known_word(self):
        from unittest.mock import MagicMock
        self.e.reasoner.definitions = {'cat': 'a small animal'}
        self.e.reasoner.generate_synonym_hypothesis = MagicMock(return_value=None)
        self.e._detect_semantic_relations(['cat'], [{'token': 'cat'}])
        self.e.reasoner.generate_synonym_hypothesis.assert_called_once_with('cat')


# ===========================================================================
# _handle_state_learning
# ===========================================================================

class TestHandleStateLearning(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()
        self.e._last_state_learning_time = 0

    def test_none_vector_no_crash(self):
        self.e._handle_state_learning(None, {})

    def test_empty_vector_no_crash(self):
        self.e._handle_state_learning(np.array([]), {})

    def test_time_guard_prevents_run(self):
        import time
        from unittest.mock import MagicMock
        self.e._last_state_learning_time = time.time()
        self.e.state_manager.find_similar_states = MagicMock()
        self.e._handle_state_learning(np.ones(9), {})
        self.e.state_manager.find_similar_states.assert_not_called()


# ===========================================================================
# process_input  (smoke / integration)
# ===========================================================================

class TestProcessInput(unittest.TestCase):

    def setUp(self):
        self.e = make_engine()

    def test_returns_dict(self):
        self.assertIsInstance(self.e.process_input("hello"), dict)

    def test_response_key_present(self):
        out = self.e.process_input("hello world")
        self.assertIsInstance(out, dict)
        self.assertIn('ci_report', out)  # or any stable key

    def test_empty_string_no_crash(self):
        self.assertIsInstance(self.e.process_input(""), dict)

    def test_counter_increments(self):
        before = self.e.interaction_counter
        self.e.process_input("test input")
        self.assertGreater(self.e.interaction_counter, before)

    def test_history_grows(self):
        before = len(self.e.interaction_history)
        self.e.process_input("something new")
        self.assertGreater(len(self.e.interaction_history), before)
    def test_response_generated(self):
        before = len(self.e.interaction_history)

        self.e.process_input("hello world")

        after = len(self.e.interaction_history)

        self.assertGreater(after, before)

if __name__ == '__main__':
    unittest.main(verbosity=2)
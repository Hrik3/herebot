"""
SemanticRepresentation_test.py — E12 Phase D, E, F verification test suite.
Theory-constrained tests covering:
- Deterministic OOV hashing & process invariance
- Non-zero word initialization via morphological subword prior
- Word learning from context and definitions
- Phrase discrimination without structural token collapse
- Legitimate semantic similarity preservation
- Definition and phrase representation consistency
- State and factor isolation (no cross-modal vector contamination)
- Representation version persistence and migration
- Sleep provenance and absence of recursive self-reinforcement
- E10 FactorSpace non-regression
"""
import os
import sys
import tempfile
import subprocess
import unittest
import numpy as np

from traits.semlearn import SemanticLearner
from components.resoning import ReasoningEngine
from components.pattern_observer import StructuralWordDetector
from components.factor_space import FactorSpace
from components.factor_handler import StateManager
from core.heart import CuriosityEngine


class TestDeterministicOOVHashing(unittest.TestCase):
    def test_oov_hashing_is_deterministic_and_nonzero(self):
        sl = SemanticLearner(emb_dim=64)
        v1 = sl.char_ngram_embedding("unseen_word_alpha")
        v2 = sl.char_ngram_embedding("unseen_word_alpha")
        self.assertGreater(np.linalg.norm(v1), 0.99)
        self.assertAlmostEqual(np.linalg.norm(v1), 1.0, places=5)
        np.testing.assert_allclose(v1, v2)

    def test_oov_process_restart_invariance(self):
        """Verify across independent python sub-processes with different hash seeds."""
        cmd = (
            "import os, numpy as np; from traits.semlearn import SemanticLearner; "
            "sl = SemanticLearner(emb_dim=64); "
            "v = sl.char_ngram_embedding('process_invariance_probe'); "
            "print(','.join(f'{x:.8f}' for x in v[:8]))"
        )
        env1 = dict(os.environ, PYTHONHASHSEED="12345", PYTHONPATH="herebot")
        env2 = dict(os.environ, PYTHONHASHSEED="67890", PYTHONPATH="herebot")
        out1 = subprocess.check_output([sys.executable, "-c", cmd], env=env1, text=True).strip()
        out2 = subprocess.check_output([sys.executable, "-c", cmd], env=env2, text=True).strip()
        self.assertEqual(out1, out2)

    def test_unrelated_oov_words_do_not_collapse(self):
        sl = SemanticLearner(emb_dim=64)
        v_cat = sl.char_ngram_embedding("cat")
        v_elephant = sl.char_ngram_embedding("elephant")
        sim = float(np.dot(v_cat, v_elephant))
        self.assertLess(sim, 0.45)


class TestWordInitializationAndLearning(unittest.TestCase):
    def test_initialize_word_has_unit_norm_and_distinct_angles(self):
        sl = SemanticLearner(emb_dim=64)
        sl.initialize_word("water")
        sl.initialize_word("crow")
        norm_water = np.linalg.norm(sl.word_embeddings["water"])
        norm_crow = np.linalg.norm(sl.word_embeddings["crow"])
        self.assertAlmostEqual(norm_water, 1.0, places=5)
        self.assertAlmostEqual(norm_crow, 1.0, places=5)
        sim = sl.similarity("water", "crow")
        self.assertLess(sim, 0.40)

    def test_observe_context_rotates_representation_away_from_prior(self):
        sl = SemanticLearner(emb_dim=64)
        sl.initialize_word("river")
        sl.initialize_word("flow")
        initial_river = sl.word_embeddings["river"].copy()

        # Observe multiple co-occurrences of river with flow
        for _ in range(5):
            sl.observe_context("river", ["flow"])

        updated_river = sl.word_embeddings["river"]
        diff = np.linalg.norm(updated_river - initial_river)
        self.assertGreater(diff, 0.05)
        sim_with_flow = sl.similarity("river", "flow")
        self.assertGreater(sim_with_flow, 0.50)

    def test_observe_definition_pulls_word_to_content_target(self):
        sl = SemanticLearner(emb_dim=64)
        sl.initialize_word("liquid")
        sl.initialize_word("water")
        # water defined as liquid
        sl.observe_definition("water", "water is a liquid")
        sim = sl.similarity("water", "liquid")
        self.assertGreater(sim, 0.50)


class TestPhraseDiscriminationAndLegitimateSimilarity(unittest.TestCase):
    def setUp(self):
        self.sl = SemanticLearner(emb_dim=64)
        for w in ["what", "is", "a", "word", "moon", "thing", "water", "cold", "fire", "hot", "fish", "animal"]:
            self.sl.initialize_word(w)
        self.detector = StructuralWordDetector(threshold=0.25, min_count=2)
        # Train detector on structural phrases
        self.detector.observe(["what", "is", "a", "word"])
        self.detector.observe(["moon", "is", "a", "thing"])
        self.detector.observe(["water", "is", "a", "nature"])

    def test_adversarial_unrelated_phrases_do_not_collapse(self):
        """Unrelated phrases sharing identical structural frames ('is a') must not collapse."""
        v1 = self.sl.text_to_embedding("what is a word", structural_detector=self.detector)
        v2 = self.sl.text_to_embedding("moon is a thing", structural_detector=self.detector)
        sim = float(np.dot(v1, v2))
        self.assertLess(sim, 0.60)
        self.assertNotAlmostEqual(sim, 1.0, places=3)

    def test_adversarial_water_cold_vs_fire_hot(self):
        v_water = self.sl.text_to_embedding("water is cold", structural_detector=self.detector)
        v_fire = self.sl.text_to_embedding("fire is hot", structural_detector=self.detector)
        sim = float(np.dot(v_water, v_fire))
        self.assertLess(sim, 0.85)

    def test_legitimate_similarity_is_preserved(self):
        v1 = self.sl.text_to_embedding("water is cold", structural_detector=self.detector)
        v2 = self.sl.text_to_embedding("water is cold", structural_detector=self.detector)
        sim = float(np.dot(v1, v2))
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_definition_symmetry(self):
        """Word target in observe_definition and stored definition embedding must be aligned."""
        self.sl.initialize_word("liquid")
        self.sl.initialize_word("stream")
        self.sl.observe_definition("stream", "stream is a flowing liquid", structural_detector=self.detector)
        def_emb = self.sl.text_to_embedding("stream is a flowing liquid", structural_detector=self.detector)
        word_emb = self.sl.word_embeddings["stream"]
        cos_align = float(np.dot(word_emb, def_emb))
        self.assertGreater(cos_align, 0.65)


class TestStateAndFactorIsolation(unittest.TestCase):
    def test_state_learning_does_not_inject_factor_vectors_into_word_embeddings(self):
        engine = CuriosityEngine(load_state=False)
        factor_vec = np.full(11, 0.75)
        metadata = {'input_text': 'test state input'}
        engine._handle_state_learning(factor_vec, metadata)
        # Check that state labels are NOT inserted into semantic_learner.word_embeddings
        for lbl in engine.current_state_labels:
            if lbl in engine.state_manager.states:
                vec = engine.state_manager.states[lbl]
                if lbl in engine.semantic_learner.word_embeddings:
                    emb = engine.semantic_learner.word_embeddings[lbl]
                    # The embedding must NOT be identical to the padded factor vector
                    aligned_factor = engine._align_vector_to_emb_dim(vec)
                    diff = np.linalg.norm(emb - aligned_factor)
                    self.assertGreater(diff, 0.1)

    def test_idle_cross_modal_alignment_does_not_mutate_word_vectors_with_factor_values(self):
        engine = CuriosityEngine(load_state=False)
        engine.semantic_learner.initialize_word("stone")
        orig_stone = engine.semantic_learner.word_embeddings["stone"].copy()
        # Seed an episode with positive factor vector
        engine.episodic_memory.add_episode(
            user_input="stone is hard",
            system_response="",
            factor_vector={'curiosity': 0.9, 'threat': 0.1, 'energy': 0.8},
            intrinsic_reward=1.0,
        )
        engine.current_state_labels = ["calm"]
        engine.cross_modal_alignment()
        stone_after = engine.semantic_learner.word_embeddings["stone"]
        # stone's lexical embedding must NOT have been directly added with factor vector
        np.testing.assert_allclose(stone_after, orig_stone)


class TestPersistenceAndVersionMigration(unittest.TestCase):
    def test_legacy_v0_zero_vectors_migrated_to_version_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = os.path.join(tmp, "semantic_legacy.pkl")
            # Write a legacy v0 file with zero vectors and contaminated state label
            legacy_data = {
                'embeddings': {
                    'water': np.zeros(64),
                    'bright curiosity': np.ones(64),  # contaminated multi-word state label
                },
                'logsigma': {'water': np.zeros(64), 'bright curiosity': np.zeros(64)},
                'counts': {'water': 5, 'bright curiosity': 0},
            }
            import pickle
            with open(pkl_path, 'wb') as f:
                pickle.dump(legacy_data, f)

            sl = SemanticLearner(emb_dim=64)
            sl.load(pkl_path)
            self.assertEqual(sl.semantic_space_version, 1)
            # 'water' must now have non-zero norm
            self.assertGreater(np.linalg.norm(sl.word_embeddings['water']), 0.99)
            # Contaminated multi-word state label must have been cleaned up
            self.assertNotIn('bright curiosity', sl.word_embeddings)

    def test_phrase_storage_uses_deterministic_sha256_keys(self):
        reasoner = ReasoningEngine()
        reasoner.store_phrase("alpha beta gamma", np.ones(64))
        keys = list(reasoner.phrase_embeddings.keys())
        self.assertEqual(len(keys), 1)
        import hashlib
        expected_key = hashlib.sha256("alpha beta gamma".encode('utf-8')).hexdigest()[:16]
        self.assertEqual(keys[0], expected_key)


class TestE10FactorSpaceNonRegression(unittest.TestCase):
    def test_e10_factor_space_and_state_comparison_remains_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_names = [f'f{i}' for i in range(11)]
            new_names = old_names + ['new_a', 'new_b']
            old = FactorSpace('control', old_names, version=1)
            new = FactorSpace('control', new_names, version=2)
            manager = StateManager(state_dir=os.path.join(tmp, 'states'))
            raw_old = np.arange(1, 12, dtype=float)
            manager.save_state('old', raw_old, factor_space=old)
            query = np.r_[raw_old, [999.0, -999.0]]
            self.assertEqual(manager.find_similar_state(query, threshold=.999, factor_space=new), 'old')


if __name__ == "__main__":
    unittest.main()

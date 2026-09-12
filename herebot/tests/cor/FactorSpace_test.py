"""E10 representation-space lifecycle regression tests."""
import json
import os
import tempfile
import unittest
import numpy as np

from components.factor_space import (FactorSpace, FactorSpaceInvariantError,
    observation_from_vector, compare_observations, OBSERVED, UNAVAILABLE)
from components.factor_handler import StateManager
from concepts.meta_learner import MetaCuriosityLearner
from concepts.controlspace import AdaptiveControlSpace
from components.episodic_mem import EpisodicMemory


class FactorSpaceTests(unittest.TestCase):
    def test_e10_11_to_13_and_reverse_are_name_based_and_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_names = [f'f{i}' for i in range(11)]
            new_names = old_names + ['new_a', 'new_b']
            old = FactorSpace('control', old_names, version=1)
            new = FactorSpace('control', new_names, version=2)
            manager = StateManager(state_dir=os.path.join(tmp, 'states'))
            raw_old = np.arange(1, 12, dtype=float)
            manager.save_state('old', raw_old, factor_space=old)
            raw_file = next(x for x in os.listdir(os.path.join(tmp, 'states')) if x.endswith('.json'))
            with open(os.path.join(tmp, 'states', raw_file)) as fh:
                before = json.load(fh)
            query = np.r_[raw_old, [999.0, -999.0]]
            self.assertEqual(manager.find_similar_state(query, threshold=.999, factor_space=new), 'old')
            audit = manager.comparison_audit[-1]
            self.assertEqual(manager.find_similar_state(raw_old, threshold=.999, factor_space=old), 'old')
            self.assertEqual(audit['shared_factor_names'], old_names)
            self.assertEqual(set(audit['excluded_factor_names']), {'new_a', 'new_b'})
            projection = manager.project_state('old', new)
            self.assertEqual(projection['newly_introduced_factors'], ['new_a', 'new_b'])
            self.assertEqual(projection['statuses']['new_a'], 'absent')
            with open(os.path.join(tmp, 'states', raw_file)) as fh:
                self.assertEqual(json.load(fh), before)

    def test_observed_zero_is_not_absence_and_unavailable_is_excluded(self):
        space = FactorSpace('control', ['a', 'b'])
        a = observation_from_vector([0.0, 1.0], space.snapshot(), {'a': OBSERVED, 'b': OBSERVED})
        b = observation_from_vector([0.0, 1.0], space.snapshot(), {'a': OBSERVED, 'b': UNAVAILABLE})
        result = compare_observations(a, b)
        self.assertEqual(result['shared_factor_names'], ['a'])
        self.assertEqual(result['excluded_factor_names'], ['b'])
        self.assertEqual(result['similarity'], 0.0)  # observed zero is numerical, not missing

    def test_versioning_ordering_and_invariants_fail_loudly(self):
        space = FactorSpace('control', ['a', 'b'])
        self.assertEqual(space.add('c').version, 1)
        with self.assertRaises(FactorSpaceInvariantError):
            FactorSpace('control', ['b', 'a', 'c'], version=1, history=space.as_dict()['history'])
        learner = MetaCuriosityLearner()
        learner.mu = learner.mu[:-1]
        with self.assertRaises(FactorSpaceInvariantError):
            learner.get_weights()
        control = AdaptiveControlSpace()
        with self.assertRaises(FactorSpaceInvariantError):
            control.validate_control_vector(np.zeros(len(control.all_factors) - 1))

    def test_factor_space_restart_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                first = AdaptiveControlSpace()
                first.add_factor('sleep_drive', lambda state: .2)
                version = first.factor_space.version
                second = AdaptiveControlSpace()
                self.assertEqual(second.factor_space.version, version)
                self.assertEqual(second.factor_space.factor_names, first.factor_space.factor_names)
            finally:
                os.chdir(previous)

    def test_episode_keeps_named_factor_schema_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                space = FactorSpace('control', ['a', 'b'], version=3)
                episodes = EpisodicMemory()
                episodes.add_episode('u', 'r', {'b': 0.0, 'a': 1.0}, 0.1, factor_space=space)
                loaded = EpisodicMemory().episodes[-1]
                self.assertEqual(loaded['factor_names'], ['a', 'b'])
                self.assertEqual(loaded['factor_space_version'], 3)
                self.assertEqual(loaded['factor_vector']['b'], 0.0)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

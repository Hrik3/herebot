import importlib.util
import os
import tempfile
import unittest

from core.heart import CuriosityEngine
from traits.code_generator import (
    CodeGenerator,
    GeneratedPluginManager,
    HerebotCapabilityFacade,
    ShadowExecutor,
)


class DummyBridge:
    def get_corpus(self, limit=None):
        corpus = ["hello memory", "crow fly transport", "car roadways transport"]
        return corpus[:limit] if limit else corpus


class DummyCI:
    def evaluate(self, text):
        return {"mode": "dummy", "confidence": 0.7, "matched_text": text}


class DummyPhon:
    def score_text(self, text):
        return 0.6, [{"mode": "dummy", "text": text}]


class DummySemantic:
    def __init__(self):
        self.word_embeddings = {}

    def similarity(self, word, candidate):
        return 0.8 if {word, candidate} == {"hello", "memory"} else 0.1


class DummyGraph:
    def __init__(self):
        self.nodes = {"n1": {"label": "hello", "type": "word", "metadata": {}}}

    def get_all_edges(self):
        return [("n1", "n2", "related", 0.5)]


class DummyReasoner:
    def __init__(self):
        self.definitions = {"hello": "a greeting"}
        self.phrase_definitions = {"hello memory": "stored greeting"}
        self.phrase_relations = {"hello": [("related", "memory", 0.9)]}
        self.hypotheses = [{"type": "synonym", "word": "hello", "candidate": "memory"}]
        self.knowledge_graph = DummyGraph()


class DummyStateManager:
    states = {"calm": [0.1, 0.9]}
    shadow_states = {"shadow_calm": [0.2, 0.8]}


class DummyFeedback:
    feedback = {"logic_1": [{"rating": "positive"}]}

    def get_feedback_summary(self, logic_id):
        return {"positive": 1, "negative": 0, "neutral": 0, "other": 0}


class DummyEngine:
    def __init__(self):
        self.backend_memory_bridge = DummyBridge()
        self.reasoner = DummyReasoner()
        self.semantic_learner = DummySemantic()
        self.ci_engine = DummyCI()
        self.phon_engine = DummyPhon()
        self.state_manager = DummyStateManager()
        self.generated_code_feedback = DummyFeedback()
        self.episodic_memory = type("Memory", (), {"episodes": []})()
        self.interaction_history = ["hello there"]
        self.current_state_labels = ["calm"]
        self._last_factor_dict = {"trust": 0.8, "confidence": 0.7, "threat": 0.1}


class TestGeneratedPluginContract(unittest.TestCase):
    def test_generated_behavioral_code_uses_plugin_contract(self):
        engine = DummyEngine()
        with tempfile.TemporaryDirectory() as tmp:
            generator = CodeGenerator(engine=engine, code_dir=tmp)
            rule = {
                "logic_id": "logic_test",
                "rule_type": "behavioral",
                "description": "reply when user says hello",
                "condition": {"keyword": "hello"},
                "action": {"say": "TPL.TEST.REPLY"},
            }
            code = generator.generate(rule)
            self.assertIn("class HerebotGeneratedPlugin", code)
            self.assertIn("GENERATED_PLUGIN_METADATA", code)
            self.assertNotIn("CuriosityEngine", code)
            self.assertIsNotNone(generator.last_generation_snapshot.get("snapshot_hash"))

            path = generator.save_code_file("logic_test", code)
            manager = GeneratedPluginManager(engine=engine, registry_path=os.path.join(tmp, "registry.json"))
            context = manager.build_runtime_context(
                engine,
                user_input="hello friend",
                tokens=["hello", "friend"],
                factor_dict={"trust": 0.8, "confidence": 0.7, "threat": 0.1},
            )
            success, error, _ = ShadowExecutor.test_code_file(path, test_input=context)
            self.assertTrue(success, error)

            spec = importlib.util.spec_from_file_location("generated_contract_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin = module.HerebotGeneratedPlugin()
            result = plugin.run(context)
            self.assertTrue(result["ok"])
            self.assertEqual(result["proposals"][0]["response_candidate"], "TPL.TEST.REPLY")

    def test_legacy_run_context_files_still_shadow_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("def run(context):\n    return 'legacy-ok'\n")
            success, error, _ = ShadowExecutor.test_code_file(path)
            self.assertTrue(success, error)


class TestHerebotCapabilityFacade(unittest.TestCase):
    def test_facade_exposes_read_summaries_and_proposals_only(self):
        engine = DummyEngine()
        facade = HerebotCapabilityFacade(engine)
        self.assertIn("hello memory", facade.get_memory_corpus())
        self.assertEqual(facade.get_reasoner_summary()["definitions"]["hello"], "a greeting")
        self.assertTrue(facade.get_graph_summary()["nodes"])
        self.assertTrue(facade.get_state_summary()["states"])
        self.assertAlmostEqual(facade.semantic_similarity("hello", "memory"), 0.8)
        self.assertEqual(facade.evaluate_ci("hello")["mode"], "dummy")
        self.assertEqual(facade.score_phon("hello")["score"], 0.6)
        self.assertFalse(hasattr(facade, "add_edge"))

        proposal = facade.propose_change("graph_proposal", {"from": "a", "to": "b"})
        self.assertEqual(proposal["type"], "graph_proposal")
        self.assertEqual(len(facade.proposals), 1)


class TestGeneratedPluginRuntime(unittest.TestCase):
    def test_curiosity_engine_shares_generator_instances(self):
        engine = CuriosityEngine(load_state=False)
        self.assertIs(engine.logic_inheritance.code_generator, engine.code_generator)
        self.assertIs(engine.logic_inheritance.shadow_executor, engine.shadow_executor)
        self.assertIs(engine.logic_inheritance.generated_plugin_manager, engine.generated_plugin_manager)

    def test_shadow_success_enters_trial_and_promotes_after_healthy_similar_contexts(self):
        engine = DummyEngine()
        with tempfile.TemporaryDirectory() as tmp:
            manager = GeneratedPluginManager(
                engine=engine,
                registry_path=os.path.join(tmp, "registry.json"),
                min_healthy_trials=3,
                similarity_threshold=0.1,
            )
            generator = CodeGenerator(engine=engine, code_dir=tmp)
            rule = {
                "logic_id": "logic_trial",
                "rule_type": "behavioral",
                "description": "hello trial response",
                "condition": {"keyword": "hello"},
                "action": {"say": "TPL.TRIAL.REPLY"},
            }
            path = generator.save_code_file("logic_trial", generator.generate(rule))
            manager.register("logic_trial", path, rule=rule, snapshot=generator.last_generation_snapshot)
            success, error, _ = ShadowExecutor.test_code_file(path)
            manager.mark_shadow_result("logic_trial", success, error=error)
            self.assertEqual(manager.records["logic_trial"]["phase"], "trial")

            context = manager.build_runtime_context(
                engine,
                user_input="hello again",
                tokens=["hello", "again"],
                factor_dict={"trust": 0.8, "confidence": 0.7, "threat": 0.1},
            )
            proposals = manager.run_matching_plugins(context)
            self.assertEqual(len(proposals), 1)
            influence = manager.select_response_candidate(proposals)
            self.assertEqual(influence["candidate"], "TPL.TRIAL.REPLY")

            for _ in range(3):
                manager.record_trial_health(
                    influence,
                    factor_dict={"trust": 0.8, "confidence": 0.7, "threat": 0.1},
                    reward=1.0,
                )
            self.assertEqual(manager.records["logic_trial"]["phase"], "active")

            manager.record_feedback(influence, "negative", reward=-1.0, factor_dict={"threat": 0.2})
            self.assertEqual(manager.records["logic_trial"]["phase"], "disabled")

    def test_unused_plugin_decays_to_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = GeneratedPluginManager(registry_path=os.path.join(tmp, "registry.json"))
            old = 0.0
            manager.records["old_logic"] = {
                "logic_id": "old_logic",
                "phase": "trial",
                "created": old,
                "last_used": old,
            }
            manager.decay_unused(now=200 * 86400)
            self.assertEqual(manager.records["old_logic"]["phase"], "archived")


if __name__ == "__main__":
    unittest.main()

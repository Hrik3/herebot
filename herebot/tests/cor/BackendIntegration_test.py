import json
import os
import tempfile
import unittest

from backend import ci as ci_module
from backend.backendSearch import AdaptiveSearchBackend
from backend.ci import CIEngine, PhonotacticEngine
from backend.herebot_memory import HerebotMemoryBridge
from backend.incremental_csr_lib import DEFAULT_CONFIG, IncrementalCSR
from core.heart import CuriosityEngine


def make_temp_csr(tmpdir, batch_size=2):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "n_features": 256,
        "batch_size": batch_size,
        "stats_update_interval": 3600,
        "config_reload_interval": 3600,
        "rebuild_threshold_shards": 0,
    })
    path = os.path.join(tmpdir, "csr_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return IncrementalCSR(path)


class TestIncrementalCSRDocuments(unittest.TestCase):
    def test_add_document_dedupes_stable_doc_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            csr = make_temp_csr(tmp)
            first = csr.add_document("alpha beta", doc_id="doc-1", source="unit")
            second = csr.add_document("alpha beta", doc_id="doc-1", source="unit")
            self.assertTrue(first["added"])
            self.assertFalse(second["added"])
            self.assertEqual(second["reason"], "doc_exists")
            self.assertEqual(csr.get_document_texts(), ["alpha beta"])

    def test_document_text_survives_flush_and_rebuild_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            csr = make_temp_csr(tmp, batch_size=1)
            csr.add_document("persistent corpus text", doc_id="doc-2", source="unit")
            csr.flush()
            self.assertIn("persistent corpus text", csr.get_all_texts())
            csr.rebuild(keep_shards=False)
            self.assertIn("persistent corpus text", csr.get_all_texts())

    def test_corrupt_metadata_is_skipped_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            csr = make_temp_csr(tmp)
            bad_meta = os.path.join(csr.shard_dir, "shard_bad.meta.json")
            with open(bad_meta, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertEqual(csr.get_shard_texts(), [])


class TestBackendMemoryBridge(unittest.TestCase):
    def test_ci_empty_corpus_returns_neutral_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = HerebotMemoryBridge(csr=make_temp_csr(tmp))
            ci = CIEngine(memory_bridge=bridge)
            report = ci.evaluate("unseen text", samples=3)
            self.assertEqual(report["mode"], "insufficient_corpus")
            self.assertEqual(report["confidence"], 0.5)
            self.assertEqual(report["corpus_size"], 0)
            self.assertEqual(report["matched_text"], "")

    def test_ci_engine_uses_injected_bridge_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = HerebotMemoryBridge(csr=make_temp_csr(tmp))
            bridge.add_runtime_text("zorbular memory phrase", source="unit")
            ci = CIEngine(memory_bridge=bridge)
            self.assertIn("zorbular memory phrase", ci.corpus)
            report = ci.evaluate("zorbular memory phrase", samples=3)
            self.assertEqual(report["backend_source"], "bridge")

    def test_ci_engine_keeps_legacy_positional_corpus(self):
        ci = CIEngine(["legacy positional corpus"])
        self.assertIn("legacy positional corpus", ci.corpus)
        self.assertIsNone(ci.memory_bridge)

    def test_ci_engine_refreshes_new_memory_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = HerebotMemoryBridge(csr=make_temp_csr(tmp))
            ci = CIEngine(memory_bridge=bridge)
            self.assertNotIn("fresh bridge sentence", ci.corpus)
            bridge.add_runtime_text("fresh bridge sentence", source="unit")
            self.assertTrue(ci.refresh_corpus(force=True))
            self.assertIn("fresh bridge sentence", ci.corpus)

    def test_phon_profile_uses_bridge_corpus_and_fallback_stays_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = HerebotMemoryBridge(csr=make_temp_csr(tmp))
            for idx in range(12):
                bridge.add_runtime_text(f"mala mana nara tara word{idx}", source="unit")
            phon = PhonotacticEngine(memory_bridge=bridge)
            score, diagnostics = phon.score_text("mala mana")
            self.assertGreaterEqual(score, 0.0)
            self.assertTrue(phon.profile_active)
            self.assertIsNotNone(diagnostics[0]["data_adaptive"])
            self.assertEqual(diagnostics[0]["mode"], "data_adaptive")

            fallback = PhonotacticEngine(memory_bridge=None)
            fallback_score, fallback_diag = fallback.score_text("hello")
            self.assertGreaterEqual(fallback_score, 0.0)
            self.assertIn("data_adaptive", fallback_diag[0])
            self.assertEqual(fallback_diag[0]["mode"], "fallback_heuristic")

    def test_no_static_reference_words_in_production_phon_module(self):
        self.assertFalse(hasattr(ci_module, "REFERENCE_WORDS"))

    def test_backend_search_reports_only_live_memory_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "search_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"confidence_threshold": 0.1}, f)
            csr = make_temp_csr(tmp)
            bridge = HerebotMemoryBridge(csr=csr)
            search = AdaptiveSearchBackend(config_path=config_path, memory_bridge=bridge)
            stats = search.get_performance_stats()
            self.assertEqual(stats["memory_source"], "bridge")
            self.assertNotIn("tier1_hits", stats)
            self.assertFalse(hasattr(search, "_search_tier1"))

    def test_curiosity_engine_process_input_syncs_current_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = HerebotMemoryBridge(
                csr=make_temp_csr(tmp),
                snapshot_interval_turns=0,
                snapshot_interval_seconds=0,
            )
            engine = CuriosityEngine(load_state=False)
            engine.backend_memory_bridge = bridge
            engine.ci_engine = CIEngine(memory_bridge=bridge)
            engine.phon_engine = PhonotacticEngine(memory_bridge=bridge)

            result = engine.process_input("hello world")

            self.assertIsInstance(result, dict)
            self.assertIn("hello world", bridge.get_corpus())
            self.assertIsNotNone(engine.backend_verification_report)
            self.assertTrue(hasattr(engine.backend_verification_report, "ok"))


if __name__ == "__main__":
    unittest.main()

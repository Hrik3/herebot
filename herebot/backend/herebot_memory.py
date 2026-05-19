"""
Bridge Herebot's live research state into backend corpus engines.

The backend engines need a stable view over unstable runtime data.  This
module performs source-aware dedupe through IncrementalCSR, keeps turn sync
cheap, and lets larger snapshots be refreshed periodically.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from backend.incremental_csr_lib import IncrementalCSR


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _stable_hash(*parts: Any) -> str:
    raw = "\0".join(_clean_text(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class SyncReport:
    ok: bool = True
    mode: str = "turn"
    added: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    corpus_size: int = 0
    snapshot_synced: bool = False
    documents_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "added": self.added,
            "skipped": self.skipped,
            "errors": list(self.errors),
            "source_counts": dict(self.source_counts),
            "corpus_size": self.corpus_size,
            "snapshot_synced": self.snapshot_synced,
            "documents_seen": self.documents_seen,
        }


@dataclass
class VerificationReport:
    ok: bool = True
    corpus_size: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    missing_sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    current_input_aligned: bool = True
    ci_corpus_size: Optional[int] = None
    phon_profile_active: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "corpus_size": self.corpus_size,
            "source_counts": dict(self.source_counts),
            "missing_sources": list(self.missing_sources),
            "errors": list(self.errors),
            "current_input_aligned": self.current_input_aligned,
            "ci_corpus_size": self.ci_corpus_size,
            "phon_profile_active": self.phon_profile_active,
        }


class HerebotMemoryBridge:
    """
    Unified memory facade for backend engines.

    Current-turn text is synced inline.  Episodic/reasoner/graph data is synced
    by snapshot, either explicitly or periodically from sync_turn.
    """

    def __init__(
        self,
        csr: Optional[IncrementalCSR] = None,
        csr_config_path: str = "config_listner.json",
        config_manager: Any = None,
        snapshot_interval_turns: Optional[int] = None,
        snapshot_interval_seconds: Optional[float] = None,
        corpus_limit: Optional[int] = None,
    ):
        self.config_manager = config_manager
        self.csr = csr or IncrementalCSR(
            self._config("backend.memory_config_path", csr_config_path)
        )
        self.snapshot_interval_turns = int(
            snapshot_interval_turns
            if snapshot_interval_turns is not None
            else self._config("backend.sync_snapshot_turn_interval", 25)
        )
        self.snapshot_interval_seconds = float(
            snapshot_interval_seconds
            if snapshot_interval_seconds is not None
            else self._config("backend.sync_snapshot_seconds", 300.0)
        )
        self.corpus_limit = corpus_limit
        self._lock = threading.RLock()
        self._turn_count = 0
        self._last_snapshot_turn = 0
        self._last_snapshot_time = 0.0
        self._last_sync_report: Optional[SyncReport] = None
        self._last_verification_report: Optional[VerificationReport] = None

    def _config(self, key: str, default: Any) -> Any:
        if self.config_manager is None:
            return default
        try:
            return self.config_manager.get(key, default)
        except Exception:
            return default

    def add_runtime_text(
        self,
        text: str,
        source: str = "runtime",
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean = _clean_text(text)
        if not clean:
            return {"added": False, "skipped": True, "reason": "empty", "doc_key": None}
        doc_id = doc_id or _stable_hash(source, clean)
        return self.csr.add_document(clean, doc_id=doc_id, source=source, metadata=metadata or {})

    def sync_turn(self, text: str, engine: Any = None) -> SyncReport:
        report = SyncReport(mode="turn")
        with self._lock:
            self._turn_count += 1
            turn = self._turn_count

        self._sync_one(
            report,
            text,
            source="turn:user_input",
            doc_id=_stable_hash("turn:user_input", text),
            metadata={"turn": turn, "timestamp": time.time()},
        )

        if engine is not None and self._should_snapshot(turn):
            snapshot = self.sync_snapshot(engine)
            report.added += snapshot.added
            report.skipped += snapshot.skipped
            report.errors.extend(snapshot.errors)
            report.snapshot_synced = True
            report.documents_seen += snapshot.documents_seen

        self._finish_report(report)
        return report

    def sync_snapshot(self, engine: Any) -> SyncReport:
        report = SyncReport(mode="snapshot")
        docs = self._collect_engine_documents(engine)
        report.documents_seen = len(docs)
        for doc in docs:
            self._sync_one(
                report,
                doc.get("text"),
                source=doc.get("source", "snapshot"),
                doc_id=doc.get("doc_id"),
                metadata=doc.get("metadata") or {},
            )
        with self._lock:
            self._last_snapshot_turn = self._turn_count
            self._last_snapshot_time = time.time()
        self._finish_report(report)
        return report

    def get_corpus(self, limit: Optional[int] = None) -> List[str]:
        try:
            use_limit = limit if limit is not None else self.corpus_limit
            if hasattr(self.csr, "get_all_texts"):
                return self.csr.get_all_texts(limit=use_limit)
            if hasattr(self.csr, "get_document_texts"):
                return self.csr.get_document_texts(limit=use_limit)
        except Exception as e:
            print(f"[Backend Memory] corpus read skipped: {e}")
        return []

    def verify_engine_alignment(self, engine: Any) -> VerificationReport:
        report = VerificationReport()
        try:
            corpus = self.get_corpus()
            report.corpus_size = len(corpus)
            report.source_counts = self._source_counts()
            last_input = None
            try:
                history = getattr(engine, "interaction_history", []) or []
                if history:
                    last_input = _clean_text(history[-1])
            except Exception as e:
                report.errors.append(f"interaction_history:{e}")
            if last_input:
                report.current_input_aligned = any(_clean_text(item) == last_input for item in corpus)
                if not report.current_input_aligned:
                    report.missing_sources.append("turn:user_input")
            if not corpus:
                report.missing_sources.append("backend:corpus")
            ci_engine = getattr(engine, "ci_engine", None)
            if ci_engine is not None:
                report.ci_corpus_size = len(getattr(ci_engine, "corpus", []) or [])
            phon_engine = getattr(engine, "phon_engine", None)
            if phon_engine is not None:
                report.phon_profile_active = bool(getattr(phon_engine, "profile_active", False))
        except Exception as e:
            report.errors.append(str(e))
        report.ok = not report.errors and not report.missing_sources
        self._last_verification_report = report
        return report

    def _should_snapshot(self, turn: int) -> bool:
        if self.snapshot_interval_turns <= 0 and self.snapshot_interval_seconds <= 0:
            return False
        with self._lock:
            due_turn = (
                self.snapshot_interval_turns > 0
                and turn - self._last_snapshot_turn >= self.snapshot_interval_turns
            )
            due_time = (
                self.snapshot_interval_seconds > 0
                and time.time() - self._last_snapshot_time >= self.snapshot_interval_seconds
            )
        return due_turn or due_time

    def _sync_one(
        self,
        report: SyncReport,
        text: Any,
        source: str,
        doc_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        clean = _clean_text(text)
        if not clean:
            return
        try:
            result = self.add_runtime_text(clean, source=source, doc_id=doc_id, metadata=metadata or {})
            if result.get("added"):
                report.added += 1
            else:
                report.skipped += 1
        except Exception as e:
            report.errors.append(f"{source}:{e}")

    def _finish_report(self, report: SyncReport):
        report.ok = not report.errors
        report.source_counts = self._source_counts()
        report.corpus_size = len(self.get_corpus())
        self._last_sync_report = report

    def _source_counts(self) -> Dict[str, int]:
        try:
            if hasattr(self.csr, "get_source_counts"):
                return self.csr.get_source_counts()
        except Exception as e:
            print(f"[Backend Memory] source count skipped: {e}")
        return {}

    def _collect_engine_documents(self, engine: Any) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        self._collect_interaction_history(engine, docs)
        self._collect_episodes(engine, docs)
        self._collect_reasoner_memory(engine, docs)
        self._collect_knowledge_graph(engine, docs)
        return self._dedupe_docs(docs)

    def _collect_interaction_history(self, engine: Any, docs: List[Dict[str, Any]]):
        try:
            history = list(getattr(engine, "interaction_history", []) or [])[-250:]
            for idx, text in enumerate(history):
                clean = _clean_text(text)
                if clean:
                    docs.append({
                        "text": clean,
                        "source": "history:user_input",
                        "doc_id": _stable_hash("history:user_input", idx, clean),
                        "metadata": {"history_index": idx},
                    })
        except Exception as e:
            docs.append(self._error_doc("history", e))

    def _collect_episodes(self, engine: Any, docs: List[Dict[str, Any]]):
        try:
            memory = getattr(engine, "episodic_memory", None)
            episodes = list(getattr(memory, "episodes", []) or [])[-500:]
            for idx, ep in enumerate(episodes):
                ts = ep.get("timestamp", idx) if isinstance(ep, dict) else idx
                if not isinstance(ep, dict):
                    continue
                user_input = _clean_text(ep.get("user_input"))
                system_response = _clean_text(ep.get("system_response"))
                if user_input:
                    docs.append({
                        "text": user_input,
                        "source": "episode:user_input",
                        "doc_id": _stable_hash("episode:user_input", ts, user_input),
                        "metadata": {"episode_timestamp": ts},
                    })
                if system_response:
                    docs.append({
                        "text": system_response,
                        "source": "episode:system_response",
                        "doc_id": _stable_hash("episode:system_response", ts, system_response),
                        "metadata": {"episode_timestamp": ts},
                    })
        except Exception as e:
            docs.append(self._error_doc("episodes", e))

    def _collect_reasoner_memory(self, engine: Any, docs: List[Dict[str, Any]]):
        try:
            reasoner = getattr(engine, "reasoner", None)
            if reasoner is None:
                return
            self._collect_dict_texts(
                docs,
                getattr(reasoner, "definitions", {}) or {},
                source="reasoner:definition",
                template="{key}: {value}",
            )
            self._collect_dict_texts(
                docs,
                getattr(reasoner, "phrase_definitions", {}) or {},
                source="reasoner:phrase_definition",
                template="{key}: {value}",
            )
            for phrase, relations in (getattr(reasoner, "phrase_relations", {}) or {}).items():
                for rel in relations or []:
                    try:
                        rel_type, target, confidence = rel
                    except Exception:
                        continue
                    text = f"{phrase} {rel_type} {target}"
                    docs.append({
                        "text": text,
                        "source": "reasoner:phrase_relation",
                        "doc_id": _stable_hash("reasoner:phrase_relation", text),
                        "metadata": {"confidence": confidence},
                    })
            sentence_metadata = getattr(reasoner, "sentence_metadata", {}) or {}
            sentence_keys = set(sentence_metadata.keys()) | set((getattr(reasoner, "sentence_embeddings", {}) or {}).keys())
            for sentence_key in sentence_keys:
                metadata = sentence_metadata.get(sentence_key, {}) if isinstance(sentence_metadata, dict) else {}
                sentence = metadata.get("sentence") or metadata.get("text") or sentence_key
                clean = _clean_text(sentence)
                if clean:
                    docs.append({
                        "text": clean,
                        "source": "reasoner:sentence",
                        "doc_id": _stable_hash("reasoner:sentence", sentence_key, clean),
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    })
        except Exception as e:
            docs.append(self._error_doc("reasoner", e))

    def _collect_knowledge_graph(self, engine: Any, docs: List[Dict[str, Any]]):
        try:
            reasoner = getattr(engine, "reasoner", None)
            kg = getattr(reasoner, "knowledge_graph", None)
            if kg is None:
                return
            for node_id, data in (getattr(kg, "nodes", {}) or {}).items():
                label = _clean_text(data.get("label") if isinstance(data, dict) else data)
                if label:
                    docs.append({
                        "text": label,
                        "source": "knowledge_graph:node",
                        "doc_id": _stable_hash("knowledge_graph:node", node_id, label),
                        "metadata": {"node_id": node_id, "node_type": data.get("type") if isinstance(data, dict) else None},
                    })
            for n1, n2, rel_type, confidence in self._iter_graph_edges(kg):
                label1 = _clean_text(getattr(kg, "nodes", {}).get(n1, {}).get("label"))
                label2 = _clean_text(getattr(kg, "nodes", {}).get(n2, {}).get("label"))
                if label1 and label2 and rel_type:
                    text = f"{label1} {rel_type} {label2}"
                    docs.append({
                        "text": text,
                        "source": "knowledge_graph:relation",
                        "doc_id": _stable_hash("knowledge_graph:relation", n1, n2, rel_type),
                        "metadata": {"confidence": confidence, "node1": n1, "node2": n2},
                    })
        except Exception as e:
            docs.append(self._error_doc("knowledge_graph", e))

    def _iter_graph_edges(self, kg: Any) -> Iterable:
        try:
            return kg.get_all_edges()
        except Exception:
            return []

    def _collect_dict_texts(
        self,
        docs: List[Dict[str, Any]],
        values: Dict[Any, Any],
        source: str,
        template: str,
    ):
        for key, value in values.items():
            text = _clean_text(template.format(key=key, value=value))
            if text:
                docs.append({
                    "text": text,
                    "source": source,
                    "doc_id": _stable_hash(source, key, value),
                    "metadata": {"key": _clean_text(key)},
                })

    def _dedupe_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for doc in docs:
            text = _clean_text(doc.get("text"))
            if not text or doc.get("source") == "bridge:error":
                continue
            key = (doc.get("source"), _stable_hash(text))
            if key in seen:
                continue
            seen.add(key)
            doc["text"] = text
            deduped.append(doc)
        return deduped

    def _error_doc(self, source: str, exc: Exception) -> Dict[str, Any]:
        return {
            "text": "",
            "source": "bridge:error",
            "doc_id": _stable_hash("bridge:error", source, str(exc)),
            "metadata": {"source": source, "error": str(exc)},
        }

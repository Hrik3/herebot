import hashlib
import re
from importglobal import json,time,os,traceback,io,importlib,redirect_stderr,redirect_stdout
from components.directory_handler import CODE_GEN_DIR, STATE_DIR


def _safe_json_value(value, depth=0, max_items=50):
    """Convert unstable Herebot runtime values into bounded JSON-safe data."""
    if depth > 4:
        return repr(value)[:200]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        try:
            return _safe_json_value(value.tolist(), depth + 1, max_items)
        except Exception:
            return repr(value)[:200]
    if isinstance(value, dict):
        out = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated"] = True
                break
            out[str(key)] = _safe_json_value(item, depth + 1, max_items)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item, depth + 1, max_items) for item in list(value)[:max_items]]
    return repr(value)[:200]


def _stable_hash(*parts) -> str:
    payload = json.dumps(_safe_json_value(parts), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _tokenize_for_codegen(text) -> list:
    return [t.lower() for t in re.findall(r"[a-zA-Z']+|\d+\.?\d*", str(text or "")) if len(t) >= 2]


class HerebotCapabilityFacade:
    """
    Read-oriented facade passed to generated plugins.

    Generated code receives this object instead of the raw CuriosityEngine.  It can
    inspect current Herebot memory and ask for proposals to be recorded, but it
    does not expose graph/state mutation methods.
    """

    def __init__(self, engine=None, max_items: int = 50):
        self._engine = engine
        self.max_items = max_items
        self.proposals = []

    def _bounded_dict(self, data, limit=None):
        limit = self.max_items if limit is None else limit
        if not isinstance(data, dict):
            return {}
        out = {}
        for idx, (key, value) in enumerate(data.items()):
            if idx >= limit:
                out["_truncated"] = True
                break
            out[str(key)] = _safe_json_value(value)
        return out

    def get_memory_corpus(self, limit=None):
        engine = self._engine
        limit = self.max_items if limit is None else limit
        texts = []
        try:
            bridge = getattr(engine, "backend_memory_bridge", None)
            if bridge is not None:
                texts.extend(bridge.get_corpus(limit=limit) or [])
        except Exception:
            pass
        try:
            for ep in getattr(getattr(engine, "episodic_memory", None), "episodes", [])[-limit:]:
                user = str(ep.get("user_input", "")).strip()
                reply = str(ep.get("system_response", "")).strip()
                if user:
                    texts.append(user)
                if reply:
                    texts.append(reply)
        except Exception:
            pass
        try:
            texts.extend(list(getattr(engine, "interaction_history", []) or [])[-limit:])
        except Exception:
            pass

        seen = set()
        deduped = []
        for text in texts:
            clean = " ".join(str(text).split())
            if not clean:
                continue
            key = _stable_hash(clean)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clean)
            if len(deduped) >= limit:
                break
        return deduped

    def get_reasoner_summary(self, limit=None):
        engine = self._engine
        limit = self.max_items if limit is None else limit
        reasoner = getattr(engine, "reasoner", None)
        if reasoner is None:
            return {"available": False}
        return {
            "available": True,
            "definitions": self._bounded_dict(getattr(reasoner, "definitions", {}) or {}, limit),
            "phrase_definitions": self._bounded_dict(getattr(reasoner, "phrase_definitions", {}) or {}, limit),
            "phrase_relations": self._bounded_dict(getattr(reasoner, "phrase_relations", {}) or {}, limit),
            "hypotheses": _safe_json_value((getattr(reasoner, "hypotheses", []) or [])[-limit:]),
        }

    def get_graph_summary(self, limit=None):
        engine = self._engine
        limit = self.max_items if limit is None else limit
        reasoner = getattr(engine, "reasoner", None)
        graph = getattr(reasoner, "knowledge_graph", None)
        if graph is None:
            return {"available": False, "nodes": [], "edges": []}
        nodes = []
        try:
            for nid, data in list(graph.nodes.items())[:limit]:
                nodes.append({
                    "id": nid,
                    "label": data.get("label"),
                    "type": data.get("type"),
                    "metadata": _safe_json_value(data.get("metadata", {})),
                })
        except Exception:
            nodes = []
        edges = []
        try:
            for n1, n2, rel, conf in graph.get_all_edges()[:limit]:
                edges.append({"from": n1, "to": n2, "relation": rel, "confidence": conf})
        except Exception:
            edges = []
        return {"available": True, "nodes": nodes, "edges": edges}

    def get_state_summary(self, limit=None):
        engine = self._engine
        limit = self.max_items if limit is None else limit
        manager = getattr(engine, "state_manager", None)
        if manager is None:
            return {"available": False, "states": [], "shadow_states": []}
        states = []
        for label, vec in list((getattr(manager, "states", {}) or {}).items())[:limit]:
            states.append({"label": label, "dim": len(vec) if hasattr(vec, "__len__") else None})
        shadows = []
        for label, vec in list((getattr(manager, "shadow_states", {}) or {}).items())[:limit]:
            shadows.append({"label": label, "dim": len(vec) if hasattr(vec, "__len__") else None})
        return {"available": True, "states": states, "shadow_states": shadows}

    def semantic_similarity(self, word, candidate):
        learner = getattr(self._engine, "semantic_learner", None)
        if learner is None:
            return 0.0
        try:
            return float(learner.similarity(str(word), str(candidate)))
        except Exception:
            try:
                emb1 = learner.word_embeddings.get(str(word))
                emb2 = learner.word_embeddings.get(str(candidate))
                if emb1 is None or emb2 is None:
                    return 0.0
                import numpy as np
                return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12))
            except Exception:
                return 0.0

    def evaluate_ci(self, text):
        ci = getattr(self._engine, "ci_engine", None)
        if ci is None:
            return {"mode": "unavailable", "confidence": 0.5}
        try:
            return _safe_json_value(ci.evaluate(str(text)))
        except Exception as e:
            return {"mode": "error", "confidence": 0.5, "error": str(e)}

    def score_phon(self, text):
        phon = getattr(self._engine, "phon_engine", None)
        if phon is None:
            return {"score": 0.0, "diagnostics": []}
        try:
            score, diagnostics = phon.score_text(str(text))
            return {"score": float(score), "diagnostics": _safe_json_value(diagnostics)}
        except Exception as e:
            return {"score": 0.0, "diagnostics": [{"mode": "error", "error": str(e)}]}

    def get_factor_snapshot(self):
        return _safe_json_value(getattr(self._engine, "_last_factor_dict", {}) or {})

    def get_feedback_summary(self, logic_id=None):
        feedback = getattr(self._engine, "generated_code_feedback", None)
        if feedback is None:
            return {"available": False}
        if logic_id:
            try:
                return feedback.get_feedback_summary(logic_id)
            except Exception:
                return {"available": True, "error": "summary_failed"}
        return {"available": True, "logic_ids": list(getattr(feedback, "feedback", {}).keys())[:self.max_items]}

    def propose_change(self, proposal_type, payload=None, source=None):
        proposal = {
            "type": str(proposal_type),
            "payload": _safe_json_value(payload or {}),
            "source": source or "generated_plugin",
            "timestamp": time.time(),
        }
        self.proposals.append(proposal)
        return proposal

    def capability_summary(self):
        corpus = self.get_memory_corpus(limit=10)
        return {
            "memory_items": len(corpus),
            "reasoner": self.get_reasoner_summary(limit=5).get("available", False),
            "graph": self.get_graph_summary(limit=5).get("available", False),
            "states": len(self.get_state_summary(limit=5).get("states", [])),
            "factors": self.get_factor_snapshot(),
        }


class GeneratedPluginManager:
    """Registry, trial runner, health tracker, and decay manager for generated plugins."""

    ACTIVE_PHASES = {"trial", "active", "weak"}
    INACTIVE_PHASES = {"disabled", "dormant", "archived"}

    def __init__(self, engine=None, registry_path: str = None, min_healthy_trials: int = 3, similarity_threshold: float = 0.2):
        self.engine = engine
        self.registry_path = registry_path or os.path.join(STATE_DIR, "generated_plugin_registry.json")
        self.min_healthy_trials = min_healthy_trials
        self.similarity_threshold = similarity_threshold
        self.records = {}
        self._module_cache = {}
        self._load()

    def bind_engine(self, engine):
        self.engine = engine

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}

    def save(self):
        directory = os.path.dirname(self.registry_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_safe_json_value(self.records), f, indent=2, sort_keys=True)
        os.replace(tmp, self.registry_path)

    def build_generation_snapshot(self, engine, rule, limit=50):
        facade = HerebotCapabilityFacade(engine, max_items=limit)
        corpus = facade.get_memory_corpus(limit=limit)
        corpus_hashes = [_stable_hash(text) for text in corpus]
        snapshot = {
            "logic_id": rule.get("logic_id"),
            "rule_type": rule.get("rule_type", rule.get("type", "unknown")),
            "rule_hash": _stable_hash(rule),
            "memory": {
                "count": len(corpus),
                "hashes": corpus_hashes,
                "sample": corpus[: min(10, len(corpus))],
            },
            "reasoner": facade.get_reasoner_summary(limit=limit),
            "graph": facade.get_graph_summary(limit=limit),
            "states": facade.get_state_summary(limit=limit),
            "factors": facade.get_factor_snapshot(),
            "feedback": facade.get_feedback_summary(rule.get("logic_id")),
            "capabilities": facade.capability_summary(),
        }
        snapshot["snapshot_hash"] = _stable_hash(snapshot)
        return snapshot

    def register(self, logic_id, filepath, rule=None, snapshot=None):
        now = time.time()
        rule = rule or {}
        snapshot = snapshot or {}
        existing = self.records.get(logic_id, {})
        trigger_tokens = sorted(set(existing.get("trigger_tokens") or self.extract_trigger_tokens(rule, snapshot)))
        record = {
            "logic_id": logic_id,
            "filepath": filepath,
            "rule_type": rule.get("rule_type", rule.get("type", "unknown")),
            "description": rule.get("description", ""),
            "trigger_tokens": trigger_tokens,
            "trigger_state_labels": list(rule.get("state_labels", []) or []),
            "phase": existing.get("phase", "generated"),
            "created": existing.get("created", now),
            "last_used": existing.get("last_used"),
            "recent_usage": existing.get("recent_usage", 0.0),
            "predictive_gain": existing.get("predictive_gain", 0.0),
            "trial_count": existing.get("trial_count", 0),
            "healthy_trials": existing.get("healthy_trials", 0),
            "unhealthy_trials": existing.get("unhealthy_trials", 0),
            "feedback_reward_total": existing.get("feedback_reward_total", 0.0),
            "shadow_success": existing.get("shadow_success", False),
            "last_error": existing.get("last_error", ""),
            "snapshot_hash": snapshot.get("snapshot_hash"),
            "snapshot": {
                "snapshot_hash": snapshot.get("snapshot_hash"),
                "memory_count": (snapshot.get("memory") or {}).get("count", 0),
                "memory_hashes": (snapshot.get("memory") or {}).get("hashes", [])[:20],
                "rule_hash": snapshot.get("rule_hash"),
            },
            "evidence": existing.get("evidence", []),
        }
        self.records[logic_id] = record
        self.save()
        return record

    def extract_trigger_tokens(self, rule, snapshot=None):
        texts = [
            rule.get("description", ""),
            rule.get("word", ""),
            rule.get("candidate", ""),
            rule.get("label", ""),
            json.dumps(_safe_json_value(rule.get("condition", {})), sort_keys=True),
            json.dumps(_safe_json_value(rule.get("action", {})), sort_keys=True),
        ]
        snapshot = snapshot or {}
        for item in (snapshot.get("memory") or {}).get("sample", [])[:5]:
            texts.append(item)
        return sorted(set(t for text in texts for t in _tokenize_for_codegen(text)))

    def mark_shadow_result(self, logic_id, success, error="", output=""):
        record = self.records.get(logic_id)
        if not record:
            return None
        record["shadow_success"] = bool(success)
        record["last_error"] = error or ""
        record["last_shadow_output"] = str(output or "")[:500]
        record["phase"] = "trial" if success else "disabled"
        record["last_shadow_test"] = time.time()
        self.save()
        return record

    def build_runtime_context(self, engine=None, **kwargs):
        engine = engine or self.engine
        facade = HerebotCapabilityFacade(engine)
        context = {
            "facade": facade,
            "capabilities": facade.capability_summary(),
            "timestamp": time.time(),
            "state": list(getattr(engine, "current_state_labels", []) or []),
        }
        context.update(kwargs)
        context.setdefault("tokens", _tokenize_for_codegen(context.get("user_input", "")))
        context.setdefault("factor_dict", getattr(engine, "_last_factor_dict", {}) or {})
        return context

    def _load_plugin(self, logic_id, filepath):
        if logic_id in self._module_cache:
            return self._module_cache[logic_id]
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(filepath or logic_id)
        module_name = f"generated_plugin_{logic_id}_{int(time.time()*1000)}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load generated plugin spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "HerebotGeneratedPlugin"):
            plugin = module.HerebotGeneratedPlugin()
        elif hasattr(module, "run"):
            class LegacyPlugin:
                def inspect(self, context):
                    return {"legacy": True, "matches": True}
                def propose(self, context):
                    result = module.run(context)
                    return {"type": "legacy_result", "result": result, "confidence": 0.5}
                def run(self, context):
                    proposal = self.propose(context)
                    return {"ok": True, "plugin_id": logic_id, "proposals": [proposal]}
            plugin = LegacyPlugin()
        else:
            raise RuntimeError("Generated file has no HerebotGeneratedPlugin or run(context)")
        self._module_cache[logic_id] = plugin
        return plugin

    def _context_similarity(self, record, context):
        trigger = set(record.get("trigger_tokens") or [])
        tokens = set(context.get("tokens") or _tokenize_for_codegen(context.get("user_input", "")))
        if not trigger:
            return 1.0
        if not tokens:
            return 0.0
        overlap = len(trigger & tokens)
        union = len(trigger | tokens)
        token_score = overlap / max(union, 1)
        coverage_score = overlap / max(min(len(trigger), len(tokens)), 1)
        current_states = set(context.get("state") or [])
        trigger_states = set(record.get("trigger_state_labels") or [])
        state_score = 0.0
        if trigger_states and current_states:
            state_score = len(trigger_states & current_states) / max(len(trigger_states | current_states), 1)
        return max(token_score, coverage_score, state_score)

    def run_matching_plugins(self, context):
        proposals = []
        now = time.time()
        for logic_id, record in list(self.records.items()):
            if record.get("phase") not in self.ACTIVE_PHASES:
                continue
            similarity = self._context_similarity(record, context)
            if similarity < self.similarity_threshold:
                continue
            try:
                plugin = self._load_plugin(logic_id, record.get("filepath"))
                facade = context.get("facade")
                facade_start = len(getattr(facade, "proposals", []) or []) if facade is not None else 0
                result = plugin.run(context)
                result = result if isinstance(result, dict) else {"ok": True, "result": result}
                plugin_proposals = result.get("proposals", [])
                if isinstance(plugin_proposals, dict):
                    plugin_proposals = [plugin_proposals]
                if facade is not None:
                    plugin_proposals = list(plugin_proposals) + list(getattr(facade, "proposals", [])[facade_start:])
                for proposal in plugin_proposals:
                    if not isinstance(proposal, dict):
                        continue
                    proposal = _safe_json_value(proposal)
                    envelope = {
                        "logic_id": logic_id,
                        "phase": record.get("phase"),
                        "similarity": similarity,
                        "proposal": proposal,
                        "inspect": _safe_json_value(result.get("inspect", {})),
                    }
                    proposals.append(envelope)
                record["last_used"] = now
                record["recent_usage"] = record.get("recent_usage", 0.0) * 0.99 + 0.1
                if record.get("phase") == "trial":
                    record["trial_count"] = record.get("trial_count", 0) + 1
            except Exception as e:
                record["last_error"] = f"{type(e).__name__}: {e}"
                record["unhealthy_trials"] = record.get("unhealthy_trials", 0) + 1
                if record["unhealthy_trials"] >= 2:
                    record["phase"] = "disabled"
        self.decay_unused(now=now)
        self.save()
        return proposals

    def select_response_candidate(self, proposals):
        best = None
        best_score = -1.0
        for envelope in proposals or []:
            proposal = envelope.get("proposal", {})
            if proposal.get("type") not in ("response_candidate", "behavior_proposal", "legacy_result"):
                continue
            candidate = proposal.get("response_candidate") or proposal.get("text") or proposal.get("result")
            if not candidate:
                continue
            confidence = float(proposal.get("confidence", 0.5) or 0.5)
            score = confidence * float(envelope.get("similarity", 0.0) or 0.0)
            if score > best_score:
                best = {
                    "logic_id": envelope.get("logic_id"),
                    "candidate": str(candidate),
                    "confidence": confidence,
                    "similarity": envelope.get("similarity", 0.0),
                    "phase": envelope.get("phase"),
                }
                best_score = score
        return best

    def record_trial_health(self, influence, factor_dict=None, reward=0.0):
        if not influence:
            return None
        logic_id = influence.get("logic_id")
        record = self.records.get(logic_id)
        if not record:
            return None
        factor_dict = factor_dict or {}
        try:
            reward = float(reward or 0.0)
        except Exception:
            reward = 0.0
        threat = float(factor_dict.get("threat", 0.0) or 0.0)
        trust = float(factor_dict.get("trust", 0.5) or 0.5)
        confidence = float(factor_dict.get("confidence", 0.5) or 0.5)
        guilt = float(factor_dict.get("guilt", 0.0) or 0.0)
        healthy = reward >= 0.0 and threat <= 0.75 and trust >= 0.25 and confidence >= 0.2 and guilt <= 0.45
        entry = {
            "timestamp": time.time(),
            "rating": "factor_health_positive" if healthy else "factor_health_negative",
            "reward": reward,
            "factors": _safe_json_value(factor_dict),
        }
        record.setdefault("evidence", []).append(entry)
        record["evidence"] = record["evidence"][-20:]
        if healthy:
            record["healthy_trials"] = record.get("healthy_trials", 0) + 1
            record["predictive_gain"] = record.get("predictive_gain", 0.0) * 0.9 + min(1.0, max(0.0, reward)) * 0.1
        else:
            record["unhealthy_trials"] = record.get("unhealthy_trials", 0) + 1
            if record.get("phase") == "active":
                record["phase"] = "weak"
            elif record.get("unhealthy_trials", 0) >= 2:
                record["phase"] = "disabled"
        self._maybe_promote(record)
        self.save()
        return record

    def record_feedback(self, influence, rating, reward=0.0, factor_dict=None):
        if not influence:
            return None
        logic_id = influence.get("logic_id")
        record = self.records.get(logic_id)
        if not record:
            return None
        try:
            reward = float(reward or 0.0)
        except Exception:
            reward = 0.0
        record["feedback_reward_total"] = record.get("feedback_reward_total", 0.0) + reward
        record.setdefault("evidence", []).append({
            "timestamp": time.time(),
            "rating": rating,
            "reward": reward,
            "factors": _safe_json_value(factor_dict or {}),
        })
        record["evidence"] = record["evidence"][-20:]
        if str(rating).lower().startswith("negative") or reward < 0:
            record["phase"] = "disabled"
            record["unhealthy_trials"] = record.get("unhealthy_trials", 0) + 1
        else:
            self.record_trial_health(influence, factor_dict=factor_dict, reward=max(0.0, reward))
        self._maybe_promote(record)
        self.save()
        return record

    def _maybe_promote(self, record):
        if record.get("phase") != "trial":
            return
        if record.get("healthy_trials", 0) >= self.min_healthy_trials and record.get("feedback_reward_total", 0.0) >= 0:
            if record.get("unhealthy_trials", 0) == 0:
                record["phase"] = "active"
                record["promoted_at"] = time.time()

    def decay_unused(self, now=None):
        now = now or time.time()
        for record in self.records.values():
            phase = record.get("phase")
            if phase in self.INACTIVE_PHASES:
                continue
            last_used = record.get("last_used")
            if last_used is None:
                last_used = record.get("created")
            if last_used is None:
                last_used = now
            age_days = max(0.0, (now - last_used) / 86400.0)
            if age_days > 180:
                record["phase"] = "archived"
            elif age_days > 90 and phase != "active":
                record["phase"] = "dormant"
            elif age_days > 30 and phase == "active":
                record["phase"] = "weak"
# ------------------------------------------------------------
# CodeGenerator , writes Python code from confirmed logic rules
# Enhanced with resource monitoring, error handling, and dynamic code generation
# ------------------------------------------------------------
class CodeGenerator:
    """
    Converts high-level logic rules into executable Python code.
    Uses safe templates; every generated file contains a `run()` function.
    Enhanced to be resource-aware and scalable, with built-in performance tracking.
    """

    def __init__(self, engine=None, code_dir: str = None, snapshot_limit: int = 50):
        self.engine = engine
        self.code_dir = code_dir or CODE_GEN_DIR
        self.snapshot_limit = snapshot_limit
        self.last_generation_snapshot = None

    def bind_engine(self, engine):
        self.engine = engine

    # Common imports added to every generated file
    BASE_IMPORTS = '''import time
import logging
import psutil  # for resource monitoring (optional, gracefully degrades)
from functools import wraps
import numpy as np
'''

    # Template for behavioral rules (identity aliases, pronoun mappings)
    BEHAVIORAL_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# Rule type: {rule_type}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}

# Setup module-level logger
logger = logging.getLogger(__name__)

def resource_monitor(func):
    """Decorator to track execution time and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            # Log if execution takes too long (>0.5s)
            if elapsed > 0.5:
                logger.warning(f"{{func.__name__}} took {{elapsed:.3f}}s")
            # Optionally check memory
            try:
                mem = psutil.Process().memory_percent()
                if mem > 10.0:  # >10% of system memory
                    logger.warning(f"{{func.__name__}} using {{mem:.1f}}% memory")
            except:
                pass
        return result
    return wrapper

def run(context: dict) -> str:
    """
    context: {{'user_input': str, 'tokens': list, 'state': dict, ...}}
    Returns a response or None.
    """
    user_input = context.get("user_input", "").lower()
    tokens = context.get("tokens", [])
    state = context.get("state", {{}})
    
    # Resource check: if system under heavy load, return early
    try:
        if psutil.cpu_percent(interval=None) > 80:
            logger.info("CPU load high, skipping expensive processing")
            return None
    except:
        pass

    # Condition: {condition}
    {condition_code}

    # Action: {action}
    {action_code}

    return response if 'response' in locals() else None
'''

    # Template for synonym / category hypotheses
    SEMANTIC_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# Semantic relation: {relation}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}

logger = logging.getLogger(__name__)

def run(word: str, semantic_learner) -> float:
    """
    Returns similarity confidence (0-1) between word and known concept.
    """
    {body}
    return 0.0
'''

    # Template for state definitions (persistent latent concepts)
    STATE_TEMPLATE = '''# AUTO-GENERATED CODE
# Logic ID: {logic_id}
# State label: {label}
# Created: {timestamp}
# Version: 1.0

{BASE_IMPORTS}
import numpy as np

logger = logging.getLogger(__name__)

def run(factor_vector: list) -> bool:
    """
    Returns True if the current factor vector matches this state.
    """
{body}
    return False
'''

    def _build_snapshot(self, rule: dict, engine=None) -> dict:
        engine = engine or self.engine
        if engine is None:
            snapshot = {
                "logic_id": rule.get("logic_id"),
                "rule_type": rule.get("rule_type", rule.get("type", "unknown")),
                "rule_hash": _stable_hash(rule),
                "memory": {"count": 0, "hashes": [], "sample": []},
                "snapshot_hash": _stable_hash(rule, "no_engine"),
            }
            self.last_generation_snapshot = snapshot
            return snapshot
        snapshot = GeneratedPluginManager(engine=engine).build_generation_snapshot(
            engine, rule, limit=self.snapshot_limit
        )
        self.last_generation_snapshot = snapshot
        return snapshot

    @staticmethod
    def _plugin_action_descriptor(action: dict) -> dict:
        if not isinstance(action, dict):
            return {"type": "response_candidate", "response_candidate": "TPL.ACK.ACTION", "confidence": 0.45}
        if "self_reference" in action:
            alias = str(action.get("self_reference", ""))
            return {
                "type": "response_candidate",
                "response_candidate": f"TPL.IDENTITY.ALIAS alias='{alias}'",
                "confidence": 0.65,
            }
        if "say" in action:
            return {
                "type": "response_candidate",
                "response_candidate": str(action.get("say", "")),
                "confidence": 0.65,
            }
        if "modify_state" in action:
            return {
                "type": "state_proposal",
                "payload": _safe_json_value(action.get("modify_state", {})),
                "response_candidate": "TPL.STATE.PROPOSED",
                "confidence": 0.55,
            }
        if "response_template" in action:
            return {
                "type": "response_candidate",
                "response_candidate": str(action.get("response_template")),
                "confidence": 0.65,
            }
        return {"type": "response_candidate", "response_candidate": "TPL.ACK.ACTION", "confidence": 0.45}

    def _generate_plugin(self, rule: dict, logic_id: str, timestamp: str, snapshot: dict) -> str:
        rule_type = rule.get("rule_type", rule.get("type", "unknown"))
        metadata = {
            "logic_id": logic_id,
            "rule_type": rule_type,
            "created": timestamp,
            "contract": "HerebotGeneratedPlugin.v1",
            "snapshot_hash": snapshot.get("snapshot_hash"),
            "rule_hash": snapshot.get("rule_hash"),
            "memory_count": (snapshot.get("memory") or {}).get("count", 0),
            "memory_hashes": (snapshot.get("memory") or {}).get("hashes", [])[:20],
            "trigger_tokens": GeneratedPluginManager().extract_trigger_tokens(rule, snapshot),
        }
        descriptor = self._plugin_action_descriptor(rule.get("action", {}))
        metadata_json = json.dumps(_safe_json_value(metadata), sort_keys=True, indent=2)
        rule_json = json.dumps(_safe_json_value(rule), sort_keys=True, indent=2)
        descriptor_json = json.dumps(_safe_json_value(descriptor), sort_keys=True, indent=2)

        return f'''# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: {logic_id}
# Rule type: {rule_type}
# Created: {timestamp}
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {metadata_json}
GENERATED_RULE = {rule_json}
ACTION_DESCRIPTOR = {descriptor_json}


class HerebotGeneratedPlugin:
    plugin_id = GENERATED_PLUGIN_METADATA.get("logic_id")
    rule_type = GENERATED_PLUGIN_METADATA.get("rule_type")
    snapshot_hash = GENERATED_PLUGIN_METADATA.get("snapshot_hash")
    trigger_tokens = GENERATED_PLUGIN_METADATA.get("trigger_tokens", [])

    def inspect(self, context):
        tokens = set(context.get("tokens") or [])
        trigger = set(self.trigger_tokens or [])
        overlap = len(tokens & trigger)
        similarity = overlap / max(len(tokens | trigger), 1) if trigger else 1.0
        capabilities = context.get("capabilities", {{}})
        return {{
            "plugin_id": self.plugin_id,
            "rule_type": self.rule_type,
            "snapshot_hash": self.snapshot_hash,
            "token_overlap": overlap,
            "similarity": similarity,
            "capabilities": capabilities,
        }}

    def _condition_matches(self, context):
        condition = GENERATED_RULE.get("condition") or {{}}
        if not isinstance(condition, dict) or not condition:
            return True
        user_input = str(context.get("user_input", "")).lower()
        tokens = [str(t).lower() for t in (context.get("tokens") or [])]
        state = context.get("factor_dict") or context.get("state") or {{}}
        if "alias" in condition and str(condition.get("alias", "")).lower() not in user_input:
            return False
        if "pronoun" in condition and str(condition.get("pronoun", "")).lower() not in tokens:
            return False
        if "keyword" in condition and str(condition.get("keyword", "")).lower() not in user_input:
            return False
        if "tokens" in condition:
            required = [str(t).lower() for t in condition.get("tokens") or []]
            if required and not any(t in tokens for t in required):
                return False
        if "regex" in condition:
            try:
                if re.search(str(condition.get("regex")), user_input) is None:
                    return False
            except Exception:
                return False
        if "state_key" in condition and "state_value" in condition and isinstance(state, dict):
            if state.get(condition.get("state_key")) != condition.get("state_value"):
                return False
        return True

    def _semantic_proposal(self, context):
        word = GENERATED_RULE.get("word")
        candidate = GENERATED_RULE.get("candidate")
        if not word or not candidate:
            return None
        facade = context.get("facade")
        similarity = 0.0
        if facade is not None and hasattr(facade, "semantic_similarity"):
            similarity = facade.semantic_similarity(word, candidate)
        if similarity <= 0.0:
            similarity = float(GENERATED_RULE.get("similarity", 0.0) or 0.0)
        return {{
            "type": "semantic_relation",
            "word": word,
            "candidate": candidate,
            "relation": GENERATED_RULE.get("type", "synonym"),
            "confidence": max(0.0, min(1.0, similarity or 0.5)),
            "source": "generated_plugin",
        }}

    def _state_proposal(self, context):
        prototype = GENERATED_RULE.get("prototype") or []
        if not prototype:
            return None
        factor_dict = context.get("factor_dict") or {{}}
        if isinstance(factor_dict, dict):
            values = list(factor_dict.values())[:len(prototype)]
        else:
            values = list(factor_dict or [])[:len(prototype)]
        if len(values) != len(prototype):
            return None
        dot = sum(float(a) * float(b) for a, b in zip(values, prototype))
        n1 = math.sqrt(sum(float(a) * float(a) for a in values))
        n2 = math.sqrt(sum(float(b) * float(b) for b in prototype))
        similarity = dot / (n1 * n2 + 1e-12)
        return {{
            "type": "state_match",
            "label": GENERATED_RULE.get("label", "unknown_state"),
            "confidence": max(0.0, min(1.0, similarity)),
            "source": "generated_plugin",
        }}

    def propose(self, context):
        if not self._condition_matches(context):
            return None
        if self.rule_type in ("synonym", "category", "analogy"):
            return self._semantic_proposal(context)
        if self.rule_type == "state_definition":
            return self._state_proposal(context)
        proposal = dict(ACTION_DESCRIPTOR)
        proposal["source"] = "generated_plugin"
        proposal["logic_id"] = self.plugin_id
        return proposal

    def run(self, context):
        inspected = self.inspect(context)
        proposal = self.propose(context)
        proposals = [proposal] if isinstance(proposal, dict) else []
        return {{
            "ok": True,
            "plugin_id": self.plugin_id,
            "inspect": inspected,
            "proposals": proposals,
            "metadata": GENERATED_PLUGIN_METADATA,
        }}


def run(context):
    return HerebotGeneratedPlugin().run(context)
'''
    @classmethod
    def _generate_language_model(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        hidden_dim = rule.get('hidden_dim', 128)
        emb_dim = rule.get('emb_dim', 64)
        lr = rule.get('learning_rate', 0.01)
        max_vocab = rule.get('max_vocab', 10000)
        cell_type = rule.get('cell_type', 'lstm').lower()
        num_layers = rule.get('num_layers', 1)

        # Generate code with full implementation
        code = f'''# AUTO-GENERATED LANGUAGE MODEL
# Logic ID: {logic_id}
# Created: {timestamp}
# Version: 1.0
# Architecture: {cell_type.upper()} with {num_layers} layer(s), hidden={hidden_dim}, emb={emb_dim}

{base_imports}
import numpy as np
import pickle
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.language_model import LanguageModel

class ImprovedLSTMLanguageModel(LanguageModel):
    """
    Self-improved language model with {cell_type.upper()} cell, {num_layers} layer(s), hidden_dim={hidden_dim}.
    """
    def __init__(self):
        super().__init__()
        self.emb_dim = {emb_dim}
        self.hidden_dim = {hidden_dim}
        self.num_layers = {num_layers}
        self.cell_type = "{cell_type}"
        self.learning_rate = {lr}
        self.max_vocab = {max_vocab}

        self.word_to_idx = {{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}}
        self.idx_to_word = {{0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}}
        self.vocab_size = 4

        self.W_emb = np.random.randn(self.vocab_size, self.emb_dim) * 0.1

        # Initialize parameters for each layer
        self.W_ih = []  # input to hidden
        self.W_hh = []  # hidden to hidden
        self.b_h = []   # bias

        for layer in range(self.num_layers):
            if self.cell_type == 'lstm':
                # LSTM has 4 gates: input, forget, cell, output
                self.W_ih.append(np.random.randn(self.emb_dim if layer==0 else self.hidden_dim, 4 * self.hidden_dim) * 0.1)
                self.W_hh.append(np.random.randn(self.hidden_dim, 4 * self.hidden_dim) * 0.1)
                self.b_h.append(np.zeros(4 * self.hidden_dim))
            else:  # gru
                # GRU has 3 gates: reset, update, new
                self.W_ih.append(np.random.randn(self.emb_dim if layer==0 else self.hidden_dim, 3 * self.hidden_dim) * 0.1)
                self.W_hh.append(np.random.randn(self.hidden_dim, 3 * self.hidden_dim) * 0.1)
                self.b_h.append(np.zeros(3 * self.hidden_dim))

        self.W_out = np.random.randn(self.hidden_dim, self.vocab_size) * 0.1
        self.b_out = np.zeros(self.vocab_size)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _one_hot(self, idx, size):
        vec = np.zeros(size)
        vec[idx] = 1
        return vec

    def forward(self, input_ids, prev_h=None, prev_c=None):
        T = len(input_ids)
        batch_size = 1  # simplified for single sequence

        # Initialize states
        if prev_h is None:
            h = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
            if self.cell_type == 'lstm':
                c = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        else:
            h = prev_h
            c = prev_c if self.cell_type == 'lstm' else None

        logits = []
        for t in range(T):
            x = self.W_emb[input_ids[t]].reshape(1, -1)  # (1, emb_dim)

            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    # LSTM
                    gates = x @ self.W_ih[layer] + h[layer] @ self.W_hh[layer] + self.b_h[layer]
                    i, f, g, o = np.split(gates, 4, axis=1)
                    i = self._sigmoid(i)
                    f = self._sigmoid(f)
                    g = self._tanh(g)
                    o = self._sigmoid(o)
                    c[layer] = f * c[layer] + i * g
                    h[layer] = o * self._tanh(c[layer])
                else:
                    # GRU
                    gates = x @ self.W_ih[layer] + h[layer] @ self.W_hh[layer] + self.b_h[layer]
                    r, z, n = np.split(gates, 3, axis=1)
                    r = self._sigmoid(r)
                    z = self._sigmoid(z)
                    n = self._tanh(n)
                    h[layer] = (1 - z) * n + z * h[layer]

                x = h[layer]  # output of this layer becomes input to next

            logit = h[-1] @ self.W_out + self.b_out
            logits.append(logit[0])  # remove batch dim

        final_h = [h_i[0] for h_i in h]
        final_c = [c_i[0] for c_i in c] if self.cell_type == 'lstm' else None
        return np.array(logits), (final_h, final_c)

    def loss(self, input_ids, target_ids):
        logits, _ = self.forward(input_ids)
        T = len(input_ids)
        loss = 0.0
        for t in range(T):
            probs = self._softmax(logits[t])
            loss -= np.log(probs[target_ids[t]] + 1e-12)
        return loss / T

    def backward(self, input_ids, target_ids, lr=None):
        if lr is None:
            lr = self.learning_rate
        T = len(input_ids)
        # Forward pass with storage – simplified; for production you'd need full BPTT.
        # Here we'll just do a placeholder update.
        # In a real system you'd implement proper backprop.
        # For demonstration, we'll use a simple SGD update on output layer only.
        logits, (h_final, _) = self.forward(input_ids)
        loss = 0.0
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        for t in range(T):
            probs = self._softmax(logits[t])
            dlogit = probs.copy()
            dlogit[target_ids[t]] -= 1.0
            dW_out += np.outer(h_final[-1], dlogit)
            db_out += dlogit
            loss -= np.log(probs[target_ids[t]] + 1e-12)
        # Update output layer
        self.W_out -= lr * dW_out / T
        self.b_out -= lr * db_out / T
        self.steps += 1
        self.loss_history.append(loss / T)
        return loss / T

    def add_word_to_vocab(self, word):
        if word in self.word_to_idx or self.vocab_size >= self.max_vocab:
            return self.word_to_idx.get(word, self.word_to_idx["<UNK>"])
        idx = self.vocab_size
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.vocab_size += 1
        # Expand embedding matrix
        new_row = np.random.randn(1, self.emb_dim) * 0.1
        self.W_emb = np.vstack([self.W_emb, new_row])
        # Expand output matrix (add a column)
        new_col = np.random.randn(self.hidden_dim, 1) * 0.1
        self.W_out = np.hstack([self.W_out, new_col])
        # Expand bias
        self.b_out = np.append(self.b_out, 0.0)
        return idx

    def sentence_to_ids(self, sentence, add_bos=True, add_eos=True):
        tokens = sentence.split()
        ids = []
        if add_bos:
            ids.append(self.word_to_idx["<BOS>"])
        for t in tokens:
            if t in self.word_to_idx:
                ids.append(self.word_to_idx[t])
            else:
                ids.append(self.word_to_idx["<UNK>"])
        if add_eos:
            ids.append(self.word_to_idx["<EOS>"])
        return ids

    def ids_to_sentence(self, ids, skip_special=True):
        words = []
        for i in ids:
            if i == self.word_to_idx["<EOS>"]:
                break
            if skip_special and i < 4:
                continue
            words.append(self.idx_to_word.get(i, "<UNK>"))
        return " ".join(words)

    def generate(self, prompt, max_len=20, temperature=0.8):
        prompt_ids = self.sentence_to_ids(prompt, add_bos=True, add_eos=False)
        if not prompt_ids:
            prompt_ids = [self.word_to_idx["<BOS>"]]
        input_ids = prompt_ids[:]
        h, c = None, None
        if input_ids:
            _, (h, c) = self.forward(input_ids)

        generated = input_ids[:]
        for _ in range(max_len):
            last_id = generated[-1]
            x = self.W_emb[last_id].reshape(1, -1)
            # Run through layers
            for layer in range(self.num_layers):
                if self.cell_type == 'lstm':
                    gates = x @ self.W_ih[layer] + (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim))) @ self.W_hh[layer] + self.b_h[layer]
                    i, f, g, o = np.split(gates, 4, axis=1)
                    i = self._sigmoid(i)
                    f = self._sigmoid(f)
                    g = self._tanh(g)
                    o = self._sigmoid(o)
                    c[layer] = f * (c[layer] if c is not None else np.zeros((1, self.hidden_dim))) + i * g
                    h[layer] = o * self._tanh(c[layer])
                else:
                    gates = x @ self.W_ih[layer] + (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim))) @ self.W_hh[layer] + self.b_h[layer]
                    r, z, n = np.split(gates, 3, axis=1)
                    r = self._sigmoid(r)
                    z = self._sigmoid(z)
                    n = self._tanh(n)
                    h[layer] = (1 - z) * n + z * (h[layer].reshape(1, -1) if h is not None else np.zeros((1, self.hidden_dim)))
                x = h[layer]
            logit = h[-1] @ self.W_out + self.b_out
            probs = self._softmax(logit[0] / temperature)
            next_id = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_id)
            if next_id == self.word_to_idx["<EOS>"]:
                break
        return self.ids_to_sentence(generated)

    def train_on_sentence(self, sentence, lr=None):
        ids = self.sentence_to_ids(sentence)
        if len(ids) < 2:
            return 0.0
        input_ids = ids[:-1]
        target_ids = ids[1:]
        return self.backward(input_ids, target_ids, lr)

    def save(self, path):
        data = {{
            'W_emb': self.W_emb,
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'steps': self.steps,
            'loss_history': self.loss_history,
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'cell_type': self.cell_type,
            'learning_rate': self.learning_rate,
            'max_vocab': self.max_vocab,
        }}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.W_emb = data['W_emb']
        self.W_ih = data['W_ih']
        self.W_hh = data['W_hh']
        self.b_h = data['b_h']
        self.W_out = data['W_out']
        self.b_out = data['b_out']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = data['vocab_size']
        self.steps = data['steps']
        self.loss_history = data['loss_history']
        self.emb_dim = data['emb_dim']
        self.hidden_dim = data['hidden_dim']
        self.num_layers = data['num_layers']
        self.cell_type = data['cell_type']
        self.learning_rate = data['learning_rate']
        self.max_vocab = data['max_vocab']
'''
        return code
    def generate(self, logic_rule: dict = None) -> str:
        """
        Dispatch to the appropriate template based on rule type.
        Generates Herebot plugin-class code for behavioral/semantic/state rules.
        Keeps the language-model generator on its original class contract because
        the language improvement path imports the generated model class directly.
        """
        if not isinstance(self, CodeGenerator):
            # Backward compatibility for CodeGenerator.generate(rule).
            return CodeGenerator().generate(self if logic_rule is None else logic_rule)
        if logic_rule is None:
            logic_rule = {}
        rule_type = logic_rule.get("rule_type", logic_rule.get("type", "unknown"))
        logic_id = logic_rule.get("logic_id", f"gen_{int(time.time())}")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Common imports will be inserted by the template formatter
        base_imports = self.BASE_IMPORTS

        if rule_type == "language_model":
            return self._generate_language_model(logic_rule, logic_id, timestamp, base_imports)

        snapshot = self._build_snapshot(logic_rule)
        return self._generate_plugin(logic_rule, logic_id, timestamp, snapshot)

    @classmethod
    def _generate_behavioral(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        condition = rule.get("condition", {})
        action = rule.get("action", {})

        # Convert condition dict to robust Python code
        condition_code = cls._parse_condition(condition)

        # Convert action dict to Python statements
        action_code = cls._parse_action(action)

        def _indent(block: str, spaces: int = 4) -> str:
            pad = " " * spaces
            lines = block.splitlines()
            # First line gets no extra indent (template already provides it),
            # subsequent lines each need the padding added.
            if len(lines) <= 1:
                return block
            return lines[0] + "\n" + "\n".join(pad + l for l in lines[1:])

        return cls.BEHAVIORAL_TEMPLATE.format(
            logic_id=logic_id,
            rule_type=rule.get("rule_type", "behavioral"),
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            condition=condition,
            condition_code=_indent(condition_code),
            action=action,
            action_code=_indent(action_code)
        )

    @classmethod
    def _generate_semantic(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        # Enhanced semantic: use embedding comparison if available, else fallback
        candidate = rule.get("candidate", "")
        word = rule.get("word", "")
        if candidate and word:
            body = f'''    # Learned similarity from rule
    if word == "{word}":
        # Check if semantic_learner provides embedding
        try:
            emb1 = semantic_learner.get_embedding(word)
            emb2 = semantic_learner.get_embedding("{candidate}")
            if emb1 is not None and emb2 is not None:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12)
                return float(sim)
        except (AttributeError, ImportError):
            pass
        return 0.95  # fallback confidence
    # TODO: implement more general similarity'''
        else:
            body = '''    # No specific word/candidate; return zero
    return 0.0'''

        return cls.SEMANTIC_TEMPLATE.format(
            logic_id=logic_id,
            relation=rule.get("type", "synonym"),
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            body=body
        )

    @classmethod
    def _generate_state(cls, rule: dict, logic_id: str, timestamp: str, base_imports: str) -> str:
        label = rule.get("label", "unknown_state")
        prototype = rule.get("prototype", [])
        if not prototype:
            prototype = [0.0] * 10  # default dimension
        proto_str = "[" + ", ".join(f"{x:.4f}" for x in prototype) + "]"
        threshold_val = rule.get("threshold", 0.85) if isinstance(rule, dict) else 0.85
        body = (
        '    try:\n'
        '        if __import__(\'psutil\').cpu_percent(interval=None) > 80:\n'
        '            logger.info("CPU load high, skipping state check")\n'
        '            return False\n'
        '    except Exception:\n'
        '        pass\n'
        '    try:\n'
        '        import numpy as np\n'
        '    except ImportError:\n'
        '        return False  # cannot compute without numpy\n'
        f'    prototype = np.array({proto_str})\n'
        '    if len(factor_vector) != len(prototype):\n'
        '        return False\n'
        '    vec = np.array(factor_vector)\n'
        '    sim = np.dot(vec, prototype) / (np.linalg.norm(vec) * np.linalg.norm(prototype) + 1e-12)\n'
        f'    threshold = {threshold_val}\n'
        '    return bool(sim > threshold)'
        )
        return cls.STATE_TEMPLATE.format(
            logic_id=logic_id,
            label=label,
            timestamp=timestamp,
            BASE_IMPORTS=base_imports,
            body=body
        )

    @staticmethod
    def _parse_condition(condition: dict) -> str:
        """
        Convert a condition dictionary into executable Python code.
        Supports multiple keys: 'alias', 'pronoun', 'regex', 'state', etc.
        Returns a string of Python statements that set a boolean variable `condition_met`.
        """
        if not condition:
            return "condition_met = False"

        lines = []
        parts = []

        # Alias condition
        if "alias" in condition:
            alias = condition["alias"].lower()
            parts.append(f"'{alias}' in user_input")

        # Pronoun condition
        if "pronoun" in condition:
            pronoun = condition["pronoun"].lower()
            parts.append(f"'{pronoun}' in tokens")

        # Regex condition
        if "regex" in condition:
            import re
            pattern = condition["regex"]
            # Safely compile regex; use raw string
            parts.append(f"re.search(r'{pattern}', user_input) is not None")
            lines.insert(0, "import re")

        # State condition (check a value in the state dict)
        if "state_key" in condition and "state_value" in condition:
            key = condition["state_key"]
            value = condition["state_value"]
            if isinstance(value, str):
                parts.append(f"state.get('{key}') == '{value}'")
            else:
                parts.append(f"state.get('{key}') == {value}")

        if parts:
            condition_code = "condition_met = " + " and ".join(parts)
        else:
            condition_code = "condition_met = False"

        if lines:
            condition_code = "\n".join(lines) + "\n" + condition_code
        return condition_code

    @staticmethod
    def _parse_action(action: dict) -> str:
        """
        Convert an action dictionary into Python statements.
        Returns a string of code to execute, which should set a `response` variable.
        """
        if not action:
            return '    response = "TPL.ACK.ACTION"'

        lines = []
        if "self_reference" in action:
            alias = action["self_reference"]
            lines.append(f'    response = "TPL.IDENTITY.ALIAS alias=\'{alias}\'"')
        elif "say" in action:
            text = action["say"]
            lines.append(f'    response = "{text}"')
        elif "modify_state" in action:
            # Example: modify_state: {"key": "value"}
            mod = action["modify_state"]
            for k, v in mod.items():
                lines.append(f'    state["{k}"] = "{v}"')
            lines.append('    response = "TPL.STATE.UPDATED"')
        else:
            lines.append('    response = "TPL.ACK.ACTION"')

        # Add optional logging
        lines.append('    logger.debug(f"Action executed: {{response}}")')
        return "\n".join(lines)

    def save_code_file(self, logic_id: str = None, code: str = None) -> str:
        """Write code to a .py file and return the full path."""
        if not isinstance(self, CodeGenerator):
            # Backward compatibility for CodeGenerator.save_code_file(logic_id, code).
            return CodeGenerator().save_code_file(self, logic_id)
        safe_id = "".join(c for c in logic_id if c.isalnum() or c == "_")
        filename = f"{safe_id}.py"
        os.makedirs(self.code_dir, exist_ok=True)
        filepath = os.path.join(self.code_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        return filepath
# ------------------------------------------------------------
# ShadowExecutor – imports and tests generated code in a sandbox
# ------------------------------------------------------------
class ShadowExecutor:
    """
    Dynamically imports a Python file, executes its `run()` function,
    captures output/errors, and appends errors as comments to the file.
    """

    @staticmethod
    def test_code_file(filepath: str, test_input: dict = None) -> tuple:
        """
        Returns (success: bool, error_message: str, output: str).
        If an exception occurs, it is appended as a comment to the file.
        """
        if not os.path.exists(filepath):
            return False, "File not found", ""

        # Prepare a dummy context for plugin or legacy behavioral rules.
        if test_input is None:
            test_input = {
                "user_input": "hello",
                "tokens": ["hello"],
                "state": ["neutral"],
                "factor_dict": {"trust": 0.5, "confidence": 0.5, "threat": 0.0},
                "capabilities": {},
            }

        module_name = f"shadow_{int(time.time()*1000)}_{os.path.basename(filepath)[:-3]}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return False, "Failed to load spec", ""

        module = importlib.util.module_from_spec(spec)
        # Capture stdout/stderr
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                start_time = time.time()
                spec.loader.exec_module(module)
                if time.time() - start_time > 12.0:
                    raise RuntimeError("Shadow execution timeout")
                if hasattr(module, "HerebotGeneratedPlugin"):
                    plugin = module.HerebotGeneratedPlugin()
                    missing = [
                        name for name in ("inspect", "propose", "run")
                        if not hasattr(plugin, name)
                    ]
                    if missing:
                        raise RuntimeError(f"Generated plugin missing methods: {missing}")
                    result = plugin.run(test_input)
                    if not isinstance(result, dict):
                        raise RuntimeError("Generated plugin run(context) must return a dict")
                elif hasattr(module, "run"):
                    result = module.run(test_input)
                else:
                    result = None
            success = True
            error_msg = ""
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
            # Append error as comment to the file
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(f"\n\n# SHADOW TEST ERROR [{time.strftime('%Y-%m-%d %H:%M:%S')}]:\n")
                    f.write(f"# {error_msg}\n")
                    f.write(f"# Traceback:\n")
                    for line in traceback.format_exc().split("\n"):
                        f.write(f"# {line}\n")
            except:
                pass

        output = f_stdout.getvalue() + f_stderr.getvalue()
        return success, error_msg, output
class GeneratedCodeFeedback:
    """
    Stores all feedback received for each generated code file.
    """
    FEEDBACK_FILE = os.path.join(STATE_DIR, "generated_code_feedback.json")

    def __init__(self):
        self.feedback = {}   # logic_id -> list of {rating, timestamp, context}
        self._load()

    def _load(self):
        if os.path.exists(self.FEEDBACK_FILE):
            try:
                with open(self.FEEDBACK_FILE, "r") as f:
                    self.feedback = json.load(f)
            except:
                self.feedback = {}

    def save(self):
        with open(self.FEEDBACK_FILE, "w") as f:
            json.dump(self.feedback, f, indent=2)

    def add_feedback(self, logic_id: str, rating: str, context: dict = None):
        """rating: 'positive', 'negative', 'neutral', 'context_positive', etc."""
        if logic_id not in self.feedback:
            self.feedback[logic_id] = []
        entry = {
            "rating": rating,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.feedback[logic_id].append(entry)
        self.save()

    def get_feedback_summary(self, logic_id: str) -> dict:
        """Return counts of positive/negative/neutral feedback."""
        entries = self.feedback.get(logic_id, [])
        counts = {"positive": 0, "negative": 0, "neutral": 0, "other": 0}
        for e in entries:
            rating = e["rating"].lower()
            if "positive" in rating:
                counts["positive"] += 1
            elif "negative" in rating:
                counts["negative"] += 1
            elif "neutral" in rating:
                counts["neutral"] += 1
            else:
                counts["other"] += 1
        return counts

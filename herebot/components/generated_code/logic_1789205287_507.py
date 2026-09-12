# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1789205287_507
# Rule type: state_definition
# Created: 2026-09-12 14:58:07
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-09-12 14:58:07",
  "logic_id": "logic_1789205287_507",
  "memory_count": 50,
  "memory_hashes": [
    "8cb3bda7b48f87d0",
    "1f8cdfb56859900f",
    "c02396fce315304f",
    "95ad0da0a74badce",
    "8e0ab6d4f17a5751",
    "1dd8542b1a48eecf",
    "991c3f54538db906",
    "3e4d3531d5b2d6c9",
    "6348721373658b5d",
    "47f8ec6e46154cfa",
    "6e2258a9ad1afd1b",
    "6f502899f60d701a",
    "8217f9e87edd69de",
    "bc55818971e4cbc3",
    "a311804b468889d7",
    "75304fe5d4bf4a6a",
    "0da3b8eaa16c92a2",
    "bca09d6abd73d54e",
    "80994c7baaca66f5",
    "53c34254fb134266"
  ],
  "rule_hash": "74090a32c977512c",
  "rule_type": "state_definition",
  "snapshot_hash": "a4a6b830a5321355",
  "trigger_tokens": [
    "'bright",
    "and",
    "bright",
    "calm",
    "co",
    "curiosity",
    "curiosity'",
    "feeling",
    "good",
    "means",
    "occur",
    "recognize",
    "state",
    "the",
    "you"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1789205287.355791,
  "description": "Recognize the 'bright curiosity' state",
  "energy_cost": 0.5,
  "label": "bright curiosity",
  "last_used": null,
  "logic_id": "logic_1789205287_507",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    0.5717807673578896,
    0.05535383714283025,
    0.3157752126658426,
    0.4279347112875223,
    0.0017420082911350553,
    0.1735071983716462,
    0.07543791233549835,
    0.0,
    0.3520435908989923,
    0.47689333581424204,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "recent_usage": 0.0,
  "rule_type": "state_definition"
}
ACTION_DESCRIPTOR = {
  "confidence": 0.45,
  "response_candidate": "TPL.ACK.ACTION",
  "type": "response_candidate"
}


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
        capabilities = context.get("capabilities", {})
        return {
            "plugin_id": self.plugin_id,
            "rule_type": self.rule_type,
            "snapshot_hash": self.snapshot_hash,
            "token_overlap": overlap,
            "similarity": similarity,
            "capabilities": capabilities,
        }

    def _condition_matches(self, context):
        condition = GENERATED_RULE.get("condition") or {}
        if not isinstance(condition, dict) or not condition:
            return True
        user_input = str(context.get("user_input", "")).lower()
        tokens = [str(t).lower() for t in (context.get("tokens") or [])]
        state = context.get("factor_dict") or context.get("state") or {}
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
        return {
            "type": "semantic_relation",
            "word": word,
            "candidate": candidate,
            "relation": GENERATED_RULE.get("type", "synonym"),
            "confidence": max(0.0, min(1.0, similarity or 0.5)),
            "source": "generated_plugin",
        }

    def _state_proposal(self, context):
        prototype = GENERATED_RULE.get("prototype") or []
        if not prototype:
            return None
        factor_dict = context.get("factor_dict") or {}
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
        return {
            "type": "state_match",
            "label": GENERATED_RULE.get("label", "unknown_state"),
            "confidence": max(0.0, min(1.0, similarity)),
            "source": "generated_plugin",
        }

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
        return {
            "ok": True,
            "plugin_id": self.plugin_id,
            "inspect": inspected,
            "proposals": proposals,
            "metadata": GENERATED_PLUGIN_METADATA,
        }


def run(context):
    return HerebotGeneratedPlugin().run(context)


# SHADOW TEST ERROR [2026-09-12 14:58:07]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1789205287_507.py", line 60, in <module>
#     "confirmed": false,
# NameError: name 'false' is not defined
# 

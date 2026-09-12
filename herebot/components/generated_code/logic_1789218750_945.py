# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1789218750_945
# Rule type: state_definition
# Created: 2026-09-12 18:42:30
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-09-12 18:42:30",
  "logic_id": "logic_1789218750_945",
  "memory_count": 50,
  "memory_hashes": [
    "fce173db5dd6272b",
    "5ebf190be1f1a09b",
    "9453386950a7af13",
    "507e2ddcbb648115",
    "3c2661ae5de979ae",
    "343d1021a3472eb6",
    "96a6d31d28f715d7",
    "84e09a1a1ed1799a",
    "a7bdd9985b505788",
    "c08a292ea6f26f72",
    "892027861f3844f5",
    "fdef54641e9ae9b4",
    "9a96d49fb287c1df",
    "cfa258e53a31b6eb",
    "6d2e485ca700aa1c",
    "3a1819b17a2378c6",
    "887efad296900871",
    "f0985b392d5ba589",
    "bd1e2eeb9db56bea",
    "f52ab5d578720970"
  ],
  "rule_hash": "6f340090fa20f927",
  "rule_type": "state_definition",
  "snapshot_hash": "dc86378122900a60",
  "trigger_tokens": [
    "'cold'",
    "'greetings'",
    "'is'",
    "'warm'",
    "'zibber'",
    "3019676126242902887",
    "5143736330148521941",
    "blue",
    "candidate",
    "cold",
    "color",
    "greetings",
    "hyp",
    "like",
    "means",
    "recognize",
    "rel",
    "sky",
    "state",
    "the",
    "tpl",
    "winter",
    "word"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1789218750.4203157,
  "description": "Recognize the 'greetings' state",
  "energy_cost": 0.5,
  "label": "greetings",
  "last_used": null,
  "logic_id": "logic_1789218750_945",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    0.0,
    0.0,
    0.0,
    0.0,
    -0.21628755676831377,
    0.0,
    0.0,
    0.16806985596554452,
    0.0,
    -0.4133669320613248,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.197079375293011,
    -0.197079375293011,
    0.21628755676831377,
    -0.16806985596554452,
    0.21628755676831377,
    -0.029009519327466492,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.3843574127338583,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.197079375293011,
    0.16806985596554452,
    -0.197079375293011,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.21628755676831377,
    -0.16806985596554452,
    0.0,
    -0.21628755676831377,
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


# SHADOW TEST ERROR [2026-09-12 18:42:30]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1789218750_945.py", line 68, in <module>
#     "confirmed": false,
# NameError: name 'false' is not defined
# 

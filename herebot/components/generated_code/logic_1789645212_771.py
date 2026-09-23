# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1789645212_771
# Rule type: state_definition
# Created: 2026-09-17 17:10:12
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-09-17 17:10:12",
  "logic_id": "logic_1789645212_771",
  "memory_count": 50,
  "memory_hashes": [
    "f0144ab5fdeb3c0c",
    "31963c8030d9f131",
    "cccac3ca8e7977e5",
    "0c0ee317335fd932",
    "4ad903e047476037",
    "001c10370466b336",
    "3563b9af785f50d9",
    "c4f3f9102aa71aa5",
    "aeb3f814002681f7",
    "63d7f7c56c7e57da",
    "a5f4e5292f7f8f3d",
    "bbb9010134b60979",
    "1553e3723cd4d1a3",
    "8f8287b9f7b9de79",
    "190f0e5902bfd5c6",
    "7ef7808a196187bb",
    "df50291568fdf16f",
    "fa757d7fab9e8c5d",
    "8e16802624c3fc08",
    "6bba5cfe85927061"
  ],
  "rule_hash": "934b4a538f7b9a4b",
  "rule_type": "state_definition",
  "snapshot_hash": "d1d89b197c458afa",
  "trigger_tokens": [
    "'greetings'",
    "and",
    "co",
    "deep",
    "frozen",
    "greetings",
    "ice",
    "it",
    "occur",
    "recognize",
    "state",
    "the",
    "water"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1789645212.804143,
  "description": "Recognize the 'greetings' state",
  "energy_cost": 0.5,
  "label": "greetings",
  "last_used": null,
  "logic_id": "logic_1789645212_771",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    -0.024654219288129088,
    -0.00010410729539563562,
    -0.31060395378797795,
    -0.009879088527779998,
    -0.04705039420856343,
    -0.0026257390935070794,
    -0.31088931325982283,
    0.1924408744736787,
    0.13209665396459289,
    -0.06940537177163231,
    0.0,
    -0.00645147758975244,
    0.07661083369409209,
    -0.014745234542364769,
    -0.3001972963349546,
    0.17186452096223598,
    -0.10513707010989357,
    0.019040101885261132,
    -0.07416820869053778,
    -0.3518459590105929,
    0.04211648780558818,
    0.16651653269492186,
    -0.03307068613525743,
    -0.028027401532736987,
    -0.08446739576061532,
    0.03061741808202913,
    0.0014631211001671757,
    -0.014323607127481423,
    0.01966444534014165,
    -0.0392030304182446,
    -0.24845870307736637,
    0.05659066427226556,
    0.0,
    -0.06349910212987105,
    -0.14163360858053803,
    0.027119423306887762,
    0.24079754386130803,
    -0.0006459538926889876,
    0.0,
    -0.0329508114842452,
    -0.030859576072819098,
    0.004270081098531718,
    -0.04488524701134855,
    0.03574966042589242,
    -0.0593459519797836,
    -0.1742961906157472,
    0.0018305798639849302,
    0.034836537756224355,
    0.01401939887343844,
    0.11473039882702359
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


# SHADOW TEST ERROR [2026-09-17 17:10:12]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1789645212_771.py", line 58, in <module>
#     "confirmed": false,
# NameError: name 'false' is not defined
# 

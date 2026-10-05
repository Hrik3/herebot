# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1791142634_167
# Rule type: state_definition
# Created: 2026-10-05 01:07:14
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-10-05 01:07:14",
  "logic_id": "logic_1791142634_167",
  "memory_count": 50,
  "memory_hashes": [
    "0873a7a21acabb2f",
    "bd28464421a87a27",
    "a741369fcbc6585b",
    "4829b25a8cdc283c",
    "dcc61d5bedc713ce",
    "69eeda641c1ea5ee",
    "fbc61129c46198c3",
    "127ad4a0d55c05a6",
    "a42be1f5713d18a5",
    "528918f285f5eb5c",
    "cf71041ca0cdde22",
    "95a322da25e4c8bc",
    "04c940115d2d4185",
    "cc225222a8087a55",
    "1a17c8ab0f2ebbd8",
    "ef5e12b3cdf67ba7",
    "3f16ef4bce54be66",
    "c05e044cd728e3ad",
    "12173337bae2953b",
    "443da8b3a66013fa"
  ],
  "rule_hash": "ae0f5481dbf8db0b",
  "rule_type": "state_definition",
  "snapshot_hash": "32edfe08b01e3e39",
  "trigger_tokens": [
    "'greetings'",
    "assistant",
    "co",
    "greetings",
    "helpful",
    "herebot",
    "is",
    "name",
    "occur",
    "recognize",
    "state",
    "the",
    "what",
    "your"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1791142634.0057042,
  "description": "Recognize the 'greetings' state",
  "energy_cost": 0.5,
  "label": "greetings",
  "last_used": null,
  "logic_id": "logic_1791142634_167",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    -0.019383998654828944,
    0.08581843313160767,
    -0.29867910535785247,
    0.0021293674011136853,
    -0.04528041108064696,
    -0.002319585996787047,
    -0.3032283970116397,
    0.20175961590193023,
    0.12737562373802644,
    -0.06785639086433964,
    0.0,
    -0.0047886407723942105,
    0.06622421065491713,
    -0.011914109116824828,
    -0.2949531285574897,
    0.1443190733706877,
    -0.0850684253662919,
    0.017819128750940102,
    -0.06923413464328636,
    -0.3363866680825271,
    0.043068232799121586,
    0.1820766807992879,
    -0.031315698618997266,
    -0.019367921817997256,
    -0.07860261854674325,
    0.02427127397479861,
    -0.006479422591519868,
    -0.012220145158339083,
    0.017416700587185113,
    -0.03914688124662468,
    -0.2609253157370453,
    0.0516288462489869,
    0.0,
    -0.054446477294379354,
    -0.13927757238615784,
    0.02395519904063022,
    0.24028366934124126,
    0.0015808983525378904,
    -3.829317250050985e-05,
    -0.03169671636748339,
    -0.02845023410286662,
    0.005497902219973718,
    -0.042385267738476715,
    0.027066983175181757,
    -0.06059746067140146,
    -0.18685022407359328,
    0.004305830071937451,
    0.02072935023347195,
    0.015528101540844537,
    0.1098632259322625
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


# SHADOW TEST ERROR [2026-10-05 01:07:14]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 995, in exec_module
#   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1791142634_167.py", line 59, in <module>
#     "confirmed": false,
#                  ^^^^^
# NameError: name 'false' is not defined. Did you mean: 'False'?
# 

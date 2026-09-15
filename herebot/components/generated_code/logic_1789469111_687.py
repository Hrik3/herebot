# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1789469111_687
# Rule type: state_definition
# Created: 2026-09-15 16:15:11
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-09-15 16:15:11",
  "logic_id": "logic_1789469111_687",
  "memory_count": 50,
  "memory_hashes": [
    "e1bf20d33ccf41bf",
    "2c62e11a88a38cac",
    "3223ea4984b475f1",
    "142f2770334dc5f0",
    "a111763ffa99d7ee",
    "2f55fd6f2cbfa438",
    "443fc22364b7b4c8",
    "41f62078b472339a",
    "b6f8fcfb9e3068de",
    "35d16fc975dd23f5",
    "57c4432d22537755",
    "d229895eaa4db24e",
    "39a62172dbbe8e9c",
    "15c299967c2566c9",
    "417a1c148a33eb94",
    "554ae613e9ae9542",
    "75772b2c4a1149c0",
    "3c54d61042a5c7db",
    "fc8d9b813ccdde0f",
    "d92a00cd7951ec59"
  ],
  "rule_hash": "b3d5df58ea13e042",
  "rule_type": "state_definition",
  "snapshot_hash": "1595fd631ae41517",
  "trigger_tokens": [
    "'bright",
    "bright",
    "co",
    "curiosity",
    "curiosity'",
    "know",
    "me",
    "occur",
    "recognize",
    "state",
    "tell",
    "the",
    "what",
    "you",
    "zibbers"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1789469111.5185723,
  "description": "Recognize the 'bright curiosity' state",
  "energy_cost": 0.5,
  "label": "bright curiosity",
  "last_used": null,
  "logic_id": "logic_1789469111_687",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    0.0,
    0.004719216944400328,
    0.0011570415953884193,
    -0.021982670647672013,
    0.07406972898636485,
    0.0022006201612579206,
    -0.0685221551919385,
    0.24246990286190445,
    0.008613635712540674,
    -0.12337008137215262,
    -0.010683329983387906,
    -0.06974380119883611,
    0.029910852872367354,
    -0.0015123826953429647,
    -0.03556144231128513,
    0.14776401636182415,
    -0.038713345348561135,
    -0.09586994361863363,
    -0.022696483644071667,
    -0.3996557588884113,
    0.047806592650359477,
    0.10682568991380567,
    -0.05724424078509477,
    -0.09929541649999248,
    -0.011631022840961645,
    -0.0422879747598449,
    0.0029814772809970756,
    0.0,
    0.009222198811190062,
    -0.15362996618821267,
    -0.1571897819288567,
    0.022577079539069258,
    0.0,
    -0.13043448377978353,
    -0.01247147490189922,
    -0.10509214242982372,
    0.5148316156471995,
    -0.060231941729154435,
    -0.0045105830045976065,
    -0.10297774815911484,
    -0.011488161960095673,
    0.0,
    -0.021463695913829496,
    0.13057411250055367,
    -0.008483760588149501,
    -0.1876257529244503,
    0.004235415825510043,
    0.25790874257598045,
    0.053570215465199605,
    -0.11209353242564822
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


# SHADOW TEST ERROR [2026-09-15 16:15:11]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 995, in exec_module
#   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1789469111_687.py", line 60, in <module>
#     "confirmed": false,
#                  ^^^^^
# NameError: name 'false' is not defined. Did you mean: 'False'?
# 

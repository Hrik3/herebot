# AUTO-GENERATED HEREBOT PLUGIN
# Logic ID: logic_1789643348_114
# Rule type: state_definition
# Created: 2026-09-17 16:42:10
# Contract: HerebotGeneratedPlugin.v1

import math
import re
import time

GENERATED_PLUGIN_METADATA = {
  "contract": "HerebotGeneratedPlugin.v1",
  "created": "2026-09-17 16:42:10",
  "logic_id": "logic_1789643348_114",
  "memory_count": 50,
  "memory_hashes": [
    "33e2c6d667c55fae",
    "d53a198de4141b9b",
    "d5aa869238edf197",
    "d780bef8b08cf955",
    "f810c27e39d62df0",
    "5624da33693209ab",
    "c94fdc83295c7a98",
    "ddf3d50fe2d2bc7e",
    "787e808a3ca1ea53",
    "8f37831ac1bbd925",
    "d45f6c5798a492a6",
    "4c831604a7416410",
    "2ed41dded37e40b0",
    "ec60147bba307ee2",
    "f930caaf108785fe",
    "90f2faa33c46bbad",
    "d623591e3c5c2a24",
    "5044de6ccb8767dc",
    "c43b8992f2c00853",
    "a8a1ea2d9079dc3c"
  ],
  "rule_hash": "8832d6dd2d117230",
  "rule_type": "state_definition",
  "snapshot_hash": "dbc4f495b720bf21",
  "trigger_tokens": [
    "'greetings'",
    "bright",
    "burns",
    "co",
    "dark",
    "greetings",
    "in",
    "occur",
    "recognize",
    "state",
    "the"
  ]
}
GENERATED_RULE = {
  "confirmed": false,
  "created": 1789643348.8047366,
  "description": "Recognize the 'greetings' state",
  "energy_cost": 0.5,
  "label": "greetings",
  "last_used": 1789643530.4141037,
  "logic_id": "logic_1789643348_114",
  "phase": "active",
  "predictive_gain": 0.5,
  "prototype": [
    -0.029626631730831635,
    -7.461072026259466e-05,
    -0.3006731224854705,
    -0.01256883094188772,
    -0.0970631194066378,
    -0.002260559206298721,
    -0.302802879206942,
    0.1830371084781378,
    0.09611568189210505,
    -0.11991006787911024,
    0.0,
    -0.00833193663724302,
    0.08421965016685494,
    -0.017953632659343265,
    -0.2917026587445975,
    0.1975723973604729,
    -0.12337398912871432,
    0.01791995044698687,
    0.011103805197529099,
    -0.3665924468674419,
    0.0939458484099954,
    0.149679227817984,
    -0.03365012631770514,
    -0.03342929325821122,
    -0.08611799699935986,
    0.0347459700070417,
    0.006090669547808044,
    -0.015355728985920254,
    0.020756017453148697,
    -0.04241334211284754,
    -0.28671102175742313,
    0.054280172342589465,
    0.0,
    -0.07089696460066347,
    -0.10387730646023365,
    0.027589899352118943,
    0.24049187699355373,
    0.00034224339198167265,
    0.0,
    -0.03068458784332047,
    -0.031943099974572974,
    0.002294386429952052,
    -0.044163797936502186,
    0.045581122470741246,
    -0.10766974421292991,
    -0.16163922142785736,
    -0.0019296820109801168,
    -0.005717587872667424,
    0.01409388615328587,
    0.07491794902716202
  ],
  "recent_usage": 0.1,
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


# SHADOW TEST ERROR [2026-09-17 16:42:10]:
# NameError: name 'false' is not defined
# Traceback:
# Traceback (most recent call last):
#   File "/home/hrirake/herebot/herebot/traits/code_generator.py", line 1446, in test_code_file
#     spec.loader.exec_module(module)
#   File "<frozen importlib._bootstrap_external>", line 883, in exec_module
#   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#   File "/home/hrirake/herebot/herebot/components/generated_code/logic_1789643348_114.py", line 56, in <module>
#     "confirmed": false,
# NameError: name 'false' is not defined
# 

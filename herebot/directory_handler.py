from .importglobal import os
# ------------------------------------------------------------
# Persistent state directory
# ------------------------------------------------------------
STATE_DIR = "curiosity_state"
try:
    os.makedirs(STATE_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create STATE_DIR: {e}")
# ------------------------------------------------------------
# Logic directory for inheritance – AGI writes own logic here
# ------------------------------------------------------------
LOGIC_DIR = os.path.join(os.path.dirname(__file__), "logic")
try:
    os.makedirs(LOGIC_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create LOGIC_DIR: {e}")
# ------------------------------------------------------------
# Generated code directory – AGI writes its own executable logic here
# ------------------------------------------------------------
CODE_GEN_DIR = os.path.join(os.path.dirname(__file__), "generated_code")
try:
    os.makedirs(CODE_GEN_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create CODE_GEN_DIR: {e}")
# Shadow test logs – store execution traces of generated code
SHADOW_LOG_DIR = os.path.join(STATE_DIR, "shadow_test_logs")
try:
    os.makedirs(SHADOW_LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create SHADOW_LOG_DIR: {e}")
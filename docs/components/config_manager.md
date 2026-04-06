# config_manager.py — Deep Overview

## What the system does

`config_manager.py` is responsible for managing and providing access to configuration parameters used by a complex AI system (e.g., Herebot). It reads configuration data from an external JSON file (`curiosity_state/active_config.json`) and fills in default values where keys are missing. It also supports deep (nested) updates and persistence of configuration changes back to disk.

## Why it exists

Hard-coded parameter settings are inflexible and require a full application restart to change. The configuration manager solves this by providing a centralized, file-backed config object that can be updated and saved at runtime — allowing real-time reconfiguration as conditions or requirements change.

## Conceptual model

The system revolves around a single `_config` dict loaded from a JSON file. Key responsibilities:

- **Configuration loading:** Reads settings from `curiosity_state/active_config.json` on startup.
- **Default values:** Fills missing top-level and nested keys from a `default_config` dict.
- **Deep updates:** Recursively merges new settings into nested sub-dicts without clobbering unrelated keys.
- **Saving:** Writes the current `_config` back to the JSON file on demand.

---

## Full Mechanism Breakdown

### 1. Initialization

```python
def __init__(self, config_path: str = os.path.join(STATE_DIR, "active_config.json")):
    self.config_path = config_path

    try:
        with open(self.config_path) as f:
            self._config = json.load(f)
    except FileNotFoundError:
        print("⚠️ [ConfigManager] No existing configuration found.")

    # Load defaults if needed
    for k, v in default_config.items():
        if k not in self._config or (isinstance(v, dict) and any(kk not in self._config[k] for kk in v)):
            self._config.update({k: copy.deepcopy(v)})
```

- If the config file exists, its contents are loaded into `_config`.
- Default values are applied for any top-level key that is absent, and for any nested dict key that is missing from an existing top-level entry.

### 2. Deep update

```python
def deep_update(self, other):
    for k, v in other.items():
        if isinstance(v, dict) and self._config.get(k):
            self.deep_update(self._config[k], v)
        else:
            self._config[k] = copy.deepcopy(v)
```

- For each key in `other`: if the value is a dict and the key already exists in `_config`, recurse into that sub-dict.
- Otherwise, overwrite `_config[k]` with a deep copy of the new value.

### 3. Saving

```python
def save(self):
    with open(self.config_path, "w") as f:
        json.dump(self._config, f)
```

- Serialises the current `_config` dict to the config file, overwriting its previous contents.

---

## Attribute-Level Deep Dive

### `self._config`

- **What it represents:** A nested dictionary holding all configuration settings for the system.
- **Why it exists:** Centralises parameter management so all components read from a single source of truth.
- **How it influences behavior over time:** Any runtime update to `_config` (via `deep_update`) immediately affects all code that reads from it; calling `save()` persists those changes across restarts.

---

## Method-Level Deep Analysis

### `__init__(self, config_path: str)`

**Purpose:** Initialise the configuration manager by loading settings from disk and applying defaults.

**Internal logic:**
1. Store the config file path.
2. Attempt to open and parse the JSON file into `_config`.
3. For each key in `default_config`, fill in the value if missing from `_config` or if any nested sub-key is absent.

### `deep_update(self, other)`

**Purpose:** Merge a new dictionary into the current configuration without flattening nested structures.

**Internal logic:**
1. Iterate over each key-value pair in `other`.
2. If the value is a dict and the key already exists in `_config`, recurse to update only the differing sub-keys.
3. Otherwise, replace `_config[k]` outright with a deep copy of the incoming value.

### `save(self)`

**Purpose:** Persist the current in-memory configuration to disk.

**Internal logic:**
1. Open the config file in write mode (truncates existing content).
2. Serialise `_config` as JSON.

---

## System Behavior Over Time

The configuration manager does not learn — its state is driven entirely by external JSON files and explicit `deep_update` calls. What changes over time is the content of `_config` as the application updates specific parameters and calls `save()`. Deep updates allow incremental changes to nested settings without requiring a full config reload or risking accidental overwrites of unrelated keys.

---

## Limitations and Failure Modes

- **Corrupted or inaccessible file.** If `active_config.json` is malformed or the path is wrong, `json.load` will raise and `_config` may be left uninitialised. There is no fallback beyond printing a warning.
- **Silent default promotion.** When a key is missing, the entire default sub-dict is deep-copied in. If `default_config` is stale, this silently introduces outdated values without any warning.
- **Deep update signature mismatch.** The implementation calls `self.deep_update(self._config[k], v)` with two arguments, but the method signature only accepts `other` — this is a bug that would raise a `TypeError` at runtime on any nested update.
- **No atomic writes.** `save()` opens the file in `"w"` mode and writes directly; a crash mid-write will corrupt the file with no recovery path.

---

## Confidence Tagging

**[CONFIDENT]** Directly from code:
- Initialization, deep update, and save logic.

**[UNCERTAIN]** Insufficient info:
- No additional methods or edge cases are provided in the given context.
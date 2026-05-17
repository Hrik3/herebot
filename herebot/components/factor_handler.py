from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from concepts.concepts_global import AdaptiveControlSpace
from importglobal import List,Dict,np,dataclass,field,json,os,Optional,time,Tuple,psutil
from components.config_manager import ConfigManager
from components.directory_handler import STATE_DIR

# ------------------------------------------------------------
# Multi-Factor Curiosity with Meta-Learned Weights (extended)
# ------------------------------------------------------------
@dataclass
class CuriosityFactors:
    """
    Container for all curiosity scores for a given word/token.
    This class is now dynamic – factors are stored in a dictionary.
    """
    factor_dict: Dict[str, float] = field(default_factory=dict)
    
    def __getattr__(self, name):
        # During pickle loading __dict__ may not yet contain factor_dict,
        # so access it safely without triggering recursion
        factor_dict = self.__dict__.get("factor_dict", None)

        if factor_dict is not None and name in factor_dict:
            return factor_dict[name]

        raise AttributeError(f"'CuriosityFactors' has no attribute '{name}'")

    
    def __setattr__(self, name, value):
        if name == 'factor_dict':
            super().__setattr__(name, value)
        else:
            self.factor_dict[name] = value
    
    def as_array(self, factor_names: List[str]) -> np.ndarray:
        """Return values for the given factor names in order."""
        return np.array([self.factor_dict.get(name, 0.0) for name in factor_names])
    
# ------------------------------------------------------------
# State Manager – persistent storage of factor vectors
# Now with conflict detection, shadow resolution, and persistent conflict storage.
# ------------------------------------------------------------
class StateManager:
    CONFLICT_FILE = os.path.join(STATE_DIR, "conflicts.json")
    SHADOW_PREFIX = "shadow_merged_"

    def __init__(self, state_dir: str = "states", config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.state_dir = state_dir
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create STATE_DIR: {e}")
        self.states = {}          # label -> factor_vector (np.array)
        self.aliases = {}  # old_label -> new_label
        self._load_all_states()

        # Conflict management
        self.conflicts = []       # list of dicts: {"labels": [a,b], "similarity": s, "resolved": bool, "resolution_label": str or None}
        self.shadow_states = {}   # label -> factor_vector (temporary shadow states)
        self._load_conflicts()
        self._load_shadows()
        self.state_metadata: Dict[str, dict] = {}   # label -> {energy_cost, recent_usage, predictive_gain, phase, last_used, confidence}
        # Dimensionality: all vectors are aligned to the current reference dimension.
        self.ref_dim = None
        self._update_ref_dim()
        self._load_state_metadata()

    def _load_state_metadata(self):
        meta_path = os.path.join(STATE_DIR, 'state_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    self.state_metadata = json.load(f)
            except Exception as e:
                print(f"Could not load state metadata: {e}")

    def _save_state_metadata(self):
        meta_path = os.path.join(STATE_DIR, 'state_metadata.json')
        tmp = meta_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state_metadata, f, indent=2)
        os.replace(tmp, meta_path)
    def resolve_label(self, label: str) -> str:
        """Follow alias chain to get the final canonical label."""
        visited = set()
        while label in self.aliases:
            if label in visited:
                break  # safety against loops
            visited.add(label)
            label = self.aliases[label]
        return label

    def get_state_vector(self, label: str, include_shadows: bool = True):
        """Safely resolve a state vector across labels, aliases, and shadows."""
        candidates = []

        def add_candidate(value):
            if value and value not in candidates:
                candidates.append(value)

        add_candidate(label)
        add_candidate(self.resolve_label(label))
        add_candidate(self.aliases.get(label))
        for old_label, new_label in self.aliases.items():
            if new_label == label:
                add_candidate(old_label)
                add_candidate(self.resolve_label(old_label))

        stores = [self.states]
        if include_shadows:
            stores.append(self.shadow_states)

        for candidate in candidates:
            for store in stores:
                if candidate in store:
                    return candidate, store[candidate]
        return None, None

    def mark_conflict_stale(self, conflict: Dict, reason: str):
        """Mark a conflict stale when its labels no longer resolve to vectors."""
        conflict["resolved"] = True
        conflict["stale"] = True
        conflict["resolution_type"] = "stale"
        conflict["resolution_label"] = reason
        conflict["stale_reason"] = reason
        self._save_conflicts()
    def _update_ref_dim(self):
        """Set reference dimension to the mode of all stored vectors."""
        dims = [len(vec) for vec in self.states.values()]
        if dims:
            self.ref_dim = max(set(dims), key=dims.count)  # most common dimension
        else:
            self.ref_dim = 10  # default base dimension

    def _normalize_vector(self, vector: np.ndarray, *, align: bool = False) -> np.ndarray:
        """
        Normalize vector (unit norm). Optionally align to reference dimension.
        
        - Default: pure normalization (mathematically correct)
        - align=True: normalize + pad/truncate to ref_dim
        """
        import numpy as np

        if vector is None or len(vector) == 0:
            return np.array([])

        vector = np.array(vector, dtype=float)

        # --- normalize FIRST (always correct math) ---
        norm = np.linalg.norm(vector)
        if norm != 0:
            vector = vector / norm

        # --- alignment ONLY if explicitly requested ---
        if align:
            if not hasattr(self, "ref_dim") or self.ref_dim is None:
                self.ref_dim = len(vector)

            if len(vector) < self.ref_dim:
                vector = np.pad(vector, (0, self.ref_dim - len(vector)))
            elif len(vector) > self.ref_dim:
                vector = vector[:self.ref_dim]

        return vector
    
    def update_parameters_from_config(self):
        """Reload parameters from config (hot reload)."""
        pass
    # ------------------------------------------------------------------
    # Core state I/O
    # ------------------------------------------------------------------
    def _load_all_states(self):
        """Load all existing state files from the directory."""
        self.states.clear()
        if not os.path.exists(self.state_dir):
            return
        for fname in os.listdir(self.state_dir):
            if fname.endswith(".json") and not fname.startswith(self.SHADOW_PREFIX):
                path = os.path.join(self.state_dir, fname)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    label = data.get("label", fname[:-5])
                    vector = data.get("vector", [])
                    metadata = data.get("metadata", {})
                    self.states[label] = self._normalize_vector(np.array(vector))
                    # Store metadata in a separate dictionary for later use
                    if not hasattr(self, 'state_metadata'):
                        self.state_metadata = {}
                    self.state_metadata[label] = metadata
                except Exception as e:
                    print(f"Could not load state {fname}: {e}")
        self._update_ref_dim()

    def save_state(self, label: str, vector: np.ndarray, metadata: dict = None, is_shadow: bool = False, response: str = None):
        """Save a factor vector under the given label."""
        os.makedirs(self.state_dir, exist_ok=True)
        safe_label = "".join(c for c in label if c.isalnum() or c in " _-").strip()
        if not safe_label:
            safe_label = "unnamed_state"
        if is_shadow:
            fname = f"{self.SHADOW_PREFIX}{safe_label}_{int(time.time())}.json"
        else:
            fname = f"{safe_label}_{int(time.time())}.json"
        path = os.path.join(self.state_dir, fname)
        
        # Store the vector as-is (original length)
        data = {
            "label": safe_label,
            "vector": vector.tolist(),
            "dimension": len(vector),
            "timestamp": time.time(),
            "metadata": metadata or {},
            "is_shadow": is_shadow,
        }
        # ---- metabolic metadata for new state ----
        self.state_metadata[safe_label] = {
            'energy_cost': 1.0,
            'recent_usage': 0.0,
            'predictive_gain': 1.0,  # placeholder
            'phase': 'active',
            'last_used': time.time(),
            'confidence': 1.0,        # how well the state represents the underlying pattern
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        vector = self._normalize_vector(vector)

        if is_shadow:
            self.shadow_states[safe_label] = vector
        else:
            self.states[safe_label] = vector

        self._update_ref_dim()
        self._save_state_metadata()   # <-- save metadata after change
        return safe_label

    def update_state(self, label: str, new_vector: np.ndarray, metadata: dict = None):
        """
        Update an existing state with a new vector.
        Deletes old file(s) for this label and saves a new one.
        """
        if label not in self.states:
            return
        # Delete old files associated with this label
        for fname in os.listdir(self.state_dir):
            if fname.startswith(label) and fname.endswith(".json"):
                try:
                    os.remove(os.path.join(self.state_dir, fname))
                except:
                    pass
        # Save new state
        self.save_state(label, new_vector, metadata=metadata, is_shadow=False)
        # Update in-memory dict
        self.states[label] = self._normalize_vector(new_vector)

    def delete_state(self, label: str):
        """Remove a state file and from memory."""
        if label in self.states:
            del self.states[label]
        if label in self.shadow_states:
            del self.shadow_states[label]
        for fname in os.listdir(self.state_dir):
            if fname.startswith(label) and fname.endswith(".json"):
                os.remove(os.path.join(self.state_dir, fname))
        if label in self.state_metadata:
            del self.state_metadata[label]
            self._save_state_metadata()
        self._update_ref_dim()

    def find_similar_state(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> Optional[str]:
        """
        Return the label of the most similar known state if cosine similarity > threshold.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        query_vec = self._normalize_vector(vector)
        all_states = dict(self.states)
        if include_shadows:
            all_states.update(self.shadow_states)

        if not all_states:
            return None
        best_label = None
        best_sim = -1.0
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        if best_sim >= threshold and best_label:
            # update usage
            meta = self.state_metadata.get(best_label)
            if meta:
                now = time.time()
                if meta.get('last_used'):
                    dt = now - meta['last_used']
                    meta['recent_usage'] *= 0.99 ** dt
                meta['recent_usage'] += 0.1
                meta['last_used'] = now
                self._save_state_metadata()
            return best_label
        return None

    def find_similar_states(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False) -> List[Tuple[str, float]]:
        """
        Return a list of (label, similarity) for all states with similarity > threshold.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        query_vec = self._normalize_vector(vector)
        all_states = dict(self.states)
        if include_shadows:
            all_states.update(self.shadow_states)

        results = []
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim >= threshold:
                results.append((label, sim))
                # update usage for this state
                meta = self.state_metadata.get(label)
                if meta:
                    now = time.time()
                    if meta.get('last_used'):
                        dt = now - meta['last_used']
                        meta['recent_usage'] *= 0.99 ** dt
                    meta['recent_usage'] += 0.05   # lighter update
                    meta['last_used'] = now
                    self._save_state_metadata()
        return results
    # ------------------------------------------------------------------
    # Conflict detection & resolution (adapt to handle different lengths)
    # ------------------------------------------------------------------
    def detect_conflicts(self, threshold: float = None) -> List[Dict]:
        """
        Scan all states and find pairs with similarity > threshold.
        Both vectors are normalised to the reference dimension.
        Returns conflicts including associated words if available in metadata.
        """
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        labels = list(self.states.keys())
        conflicts = []
        processed_pairs = set()
        for i, l1 in enumerate(labels):
            v1 = self._normalize_vector(self.states[l1])
            # Retrieve associated words from metadata (if any)
            meta1 = self.state_metadata.get(l1, {})
            words1 = meta1.get('tokens', [])  # stored during state creation
            for l2 in labels[i+1:]:
                pair = tuple(sorted([l1, l2]))
                if pair in processed_pairs:
                    continue
                v2 = self._normalize_vector(self.states[l2])
                meta2 = self.state_metadata.get(l2, {})
                words2 = meta2.get('tokens', [])
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                if sim >= threshold:
                    conflict = {
                        "labels": [l1, l2],
                        "similarity": float(sim),
                        "words1": words1,          # list of tokens that led to state l1
                        "words2": words2,          # list of tokens that led to state l2
                        "resolved": False,
                        "resolution_label": None,
                        "timestamp": time.time(),
                    }
                    conflicts.append(conflict)
                    processed_pairs.add(pair)
        return conflicts

    def refresh_conflicts(self, threshold: float = None):
        """Re-detect conflicts and update self.conflicts, keeping existing resolution status if possible."""
        new_conflicts = self.detect_conflicts(threshold)

        existing_map = {}
        for c in self.conflicts:
            key = tuple(sorted(c["labels"]))
            existing_map[key] = c

        merged = []
        for c in new_conflicts:
            key = tuple(sorted(c["labels"]))
            if key in existing_map and existing_map[key].get("resolved", False):
                c["resolved"] = True
                c["resolution_label"] = existing_map[key].get("resolution_label")
                c["resolution_type"] = existing_map[key].get("resolution_type")
                if existing_map[key].get("stale", False):
                    c["stale"] = True
                    c["stale_reason"] = existing_map[key].get("stale_reason")
            merged.append(c)

        self.conflicts = merged
        self._save_conflicts()

    def _save_conflicts(self):
        """Persist conflicts to JSON."""
        tmp = self.CONFLICT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.conflicts, f, indent=2)
        os.replace(tmp, self.CONFLICT_FILE)

    def _load_conflicts(self):
        """Load conflicts from disk."""
        if os.path.exists(self.CONFLICT_FILE):
            try:
                with open(self.CONFLICT_FILE, "r") as f:
                    self.conflicts = json.load(f)
            except Exception as e:
                print(f"Could not load conflicts: {e}")
                self.conflicts = []

    def _load_shadows(self):
        """Load all shadow states from the state directory."""
        self.shadow_states.clear()
        if not os.path.exists(self.state_dir):
            return
        for fname in os.listdir(self.state_dir):
            if fname.startswith(self.SHADOW_PREFIX) and fname.endswith(".json"):
                path = os.path.join(self.state_dir, fname)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    label = data.get("label", fname[:-5])
                    vector = data.get("vector", [])
                    self.shadow_states[label] = self._normalize_vector(np.array(vector))
                except Exception as e:
                    print(f"Could not load shadow state {fname}: {e}")

    def create_shadow_resolution(self, conflict: Dict) -> str:
        """Merge the two conflicting vectors (average) and create a temporary shadow state."""
        try:
            l1, l2 = conflict["labels"][0], conflict["labels"][1]
            v1 = self.states[l1]
            v2 = self.states[l2]
            # Align to the longer length before averaging
            max_len = max(len(v1), len(v2))
            v1_aligned = np.pad(v1, (0, max_len - len(v1)), 'constant') if len(v1) < max_len else v1[:max_len]
            v2_aligned = np.pad(v2, (0, max_len - len(v2)), 'constant') if len(v2) < max_len else v2[:max_len]
            merged_vector = self._normalize_vector((v1_aligned + v2_aligned) / 2.0)

            shadow_label = f"shadow_{l1}_{l2}_{int(time.time())}"
            metadata = {
                "merged_from": [l1, l2],
                "similarity": conflict["similarity"],
                "timestamp": time.time(),
            }
            self.save_state(shadow_label, merged_vector, metadata, is_shadow=True)
            return shadow_label
        except Exception as e:
            print(f"   [Error creating shadow resolution: {e}]")
            # Return a fallback label
            return f"shadow_error_{int(time.time())}"

    def apply_shadow_to_conflict(self, conflict: Dict, shadow_label: str):
        """Mark conflict as resolved with the given shadow label."""
        conflict["resolved"] = True
        conflict["resolution_label"] = shadow_label
        conflict["resolution_type"] = "shadow"
        self._save_conflicts()

    def resolve_conflict_permanently(self, conflict: Dict, winning_label: str, delete_other: bool = True):
        """User chooses one label as the canonical state."""
        l1, l2 = conflict["labels"]
        if winning_label == l1:
            to_delete = l2
        else:
            to_delete = l1

        # Preserve identity before deletion
        self.aliases[to_delete] = winning_label

        if delete_other:
            self.delete_state(to_delete)

        conflict["resolved"] = True
        conflict["resolution_label"] = winning_label
        conflict["resolution_type"] = "permanent"
        self._save_conflicts()

    def get_unresolved_conflicts(self) -> List[Dict]:
        """Return conflicts that are not yet resolved."""
        return [c for c in self.conflicts if not c.get("resolved", False)]
# ------------------------------------------------------------
# Factor Engine – computes the base control signals
# ------------------------------------------------------------
class FactorEngine:
    """
    Derives the base control signals from the current system state.
    All outputs are in [0,1].

    Now uses AdaptiveControlSpace to allow dynamic expansion.
    """

    def __init__(self, control_space: AdaptiveControlSpace,
                 config_manager: Optional['ConfigManager'] = None):
        self.control_space = control_space
        self.config_manager = config_manager

    def update_parameters_from_config(self):
        """Placeholder for hot reload."""
        pass

    def compute(self,
                token_analyses: List[Dict],
                ci_report: Dict,
                phon_score: float,
                diag: Dict,
                concept_model,
                meta_learner,
                reward_total: float,
                semantic_learner) -> Dict[str, float]:
        """
        Returns a dict with all control signals (base + dynamic).
        """
        result = {}

        # 1. curiosity – max curiosity score among tokens
        result['curiosity'] = max([ta.get("curiosity_score", 0.0) for ta in token_analyses], default=0.0)

        # 2. threat – based on negative diagnostics, low confidence, low phon_score
        neg_count = len(diag.get("negatives", []))
        result['threat'] = np.clip(0.3 * (neg_count / 3.0) +
                                   0.4 * (1.0 - ci_report.get("confidence", 0.5)) +
                                   0.3 * (1.0 - phon_score), 0, 1)

        # 3. trust – from CI confidence and stability (computed after comprehension)
        result['trust'] = 0.0

        # 4. confidence – CI confidence directly
        result['confidence'] = ci_report.get("confidence", 0.5)

        # 5. uncertainty – average entropy of all known words (global)
        if concept_model.word_concept_counts:
            uncertainties = [concept_model.entropy_word(w) for w in concept_model.word_concept_counts]
            result['uncertainty'] = np.clip(np.mean(uncertainties) / 5.0, 0, 1)
        else:
            result['uncertainty'] = 0.5

        # 6. energy – compute budget proxy (inverse of recent CPU usage)
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            result['energy'] = 1.0 - np.clip(cpu_percent, 0, 1)
        except:
            result['energy'] = 0.8

        # 7. attention – based on input length and number of tokens
        token_count = len(token_analyses)
        result['attention'] = np.clip(token_count / 20.0, 0, 1)

        # ---- Language factors ----
        # comprehension: average semantic similarity among words in input
        if token_analyses and semantic_learner:
            sims = []
            for ta in token_analyses:
                w = ta['token']
                others = [t['token'] for t in token_analyses if t['token'] != w]
                if others:
                    oth_sims = [semantic_learner.similarity(w, o) for o in others
                                if o in semantic_learner.word_embeddings]
                    if oth_sims:
                        sims.append(np.mean(oth_sims))
            result['comprehension'] = np.mean(sims) if sims else 0.5
        else:
            result['comprehension'] = 0.5

        # fluency: placeholder
        result['fluency'] = 0.7

        # cognitive load: inverse of energy * attention
        result['cognitive_load'] = np.clip(1.0 - (result['energy'] * result['attention']), 0, 1)

        # ---- Recompute trust using comprehension ----
        result['trust'] = (
            0.4 * result['confidence'] +
            0.3 * ci_report.get("stability", 0.5) +
            0.3 * result['comprehension']
        )

        # ---- Compute dynamic factors (if any) ----
        state = {
            'token_analyses': token_analyses,
            'ci_report': ci_report,
            'phon_score': phon_score,
            'diag': diag,
            'concept_model': concept_model,
            'meta_learner': meta_learner,
            'reward_total': reward_total,
            'semantic_learner': semantic_learner,
            **result
        }
        for factor_name in self.control_space.dynamic_factors:
            result[factor_name] = self.control_space.compute_factor(factor_name, state)
            self.control_space.record_value(factor_name, result[factor_name])
        # ensure numeric stability
        for k, v in result.items():
            if not isinstance(v, (int, float)) or np.isnan(v) or np.isinf(v):
                result[k] = 0.0
        return result

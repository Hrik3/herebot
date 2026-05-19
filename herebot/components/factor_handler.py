from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from concepts.concepts_global import AdaptiveControlSpace
from collections import Counter
from importglobal import List,Dict,np,dataclass,field,json,os,Optional,time,Tuple,psutil
from components.config_manager import ConfigManager
from components.directory_handler import STATE_DIR
from components.probabilistic_labeling import (
    ensure_probabilistic_stats,
    normalize_probability_field,
)

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
        self.relation_file = os.path.join(STATE_DIR, "state_relations.json")
        self.group_file = os.path.join(STATE_DIR, "state_groups.json")
        self.states = {}          # label -> factor_vector (np.array)
        self.aliases = {}  # old_label -> new_label
        self.state_metadata: Dict[str, dict] = {}   # label -> probabilistic and metabolic metadata
        self.state_relation_observations: Dict[str, dict] = {}
        self.state_groups: Dict[str, dict] = {}
        self.state_group_aliases: Dict[str, str] = {}
        self._load_all_states()

        # Conflict management
        self.conflicts = []       # list of dicts: {"labels": [a,b], "similarity": s, "resolved": bool, "resolution_label": str or None}
        self.shadow_states = {}   # label -> factor_vector (temporary shadow states)
        self._load_conflicts()
        self._load_shadows()
        self._load_state_relations()
        self._load_state_groups()
        # Dimensionality: all vectors are aligned to the current reference dimension.
        self.ref_dim = None
        self._update_ref_dim()
        self._load_state_metadata()
        for label in self.states:
            self._ensure_state_probability_metadata(label, interaction_index=0)

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

    def _prob_config(self) -> dict:
        if self.config_manager:
            return self.config_manager.config.get('probabilistic_labeling', {})
        return {}

    def _state_config(self, key: str, default):
        if self.config_manager:
            return self.config_manager.get(f'state_manager.{key}', default)
        return default

    def _legacy_interactive_conflicts_enabled(self) -> bool:
        return bool(self._state_config('legacy_interactive_conflicts', False))

    def _ensure_state_probability_metadata(self, label: str, interaction_index: int = 0):
        meta = self.state_metadata.setdefault(label, {})
        initial_usage = meta.get('recent_usage', meta.get('usage', 0.0))
        initial_probability = meta.get('probability', meta.get('confidence', 0.0))
        ensure_probabilistic_stats(
            meta,
            interaction_index=interaction_index,
            initial_usage=initial_usage,
            initial_probability=initial_probability,
        )
        return meta

    def _normalize_state_probabilities(self, labels=None, interaction_index: int = 0, reward: float = 0.0, touched_label: str = None, weight: float = 0.0):
        if labels is None:
            labels = list(self.states.keys())
        stats_map = {}
        for label in labels:
            if label in self.state_metadata:
                stats_map[label] = self._ensure_state_probability_metadata(label, interaction_index)
        normalize_probability_field(
            stats_map,
            interaction_index=interaction_index,
            reward=reward,
            touched_key=touched_label,
            touched_weight=weight,
            config=self._prob_config(),
            archive=True,
        )
        for label, meta in stats_map.items():
            self.state_metadata[label] = meta
        return stats_map

    def get_state_probabilities(self, interaction_index: int = 0, include_archived: bool = False) -> Dict[str, float]:
        labels = list(self.states.keys())
        stats = self._normalize_state_probabilities(labels, interaction_index=interaction_index)
        return {
            label: float(meta.get('probability', 0.0))
            for label, meta in stats.items()
            if include_archived or meta.get('phase', 'active') != 'archived'
        }

    def _state_relation_key(self, labels: List[str]) -> str:
        return "||".join(sorted(str(label) for label in labels))

    def _align_pair(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v1 = self._normalize_vector(v1)
        v2 = self._normalize_vector(v2)
        max_len = max(len(v1), len(v2))
        if len(v1) < max_len:
            v1 = np.pad(v1, (0, max_len - len(v1)), 'constant')
        elif len(v1) > max_len:
            v1 = v1[:max_len]
        if len(v2) < max_len:
            v2 = np.pad(v2, (0, max_len - len(v2)), 'constant')
        elif len(v2) > max_len:
            v2 = v2[:max_len]
        return v1, v2

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        v1, v2 = self._align_pair(v1, v2)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _factor_delta(self, v1: np.ndarray, v2: np.ndarray) -> dict:
        v1, v2 = self._align_pair(v1, v2)
        delta = np.abs(v1 - v2)
        return {
            'mean': float(np.mean(delta)) if len(delta) else 0.0,
            'max': float(np.max(delta)) if len(delta) else 0.0,
            'vector': delta.tolist(),
        }

    def _load_state_relations(self):
        if os.path.exists(self.relation_file):
            try:
                with open(self.relation_file, 'r') as f:
                    data = json.load(f)
                self.state_relation_observations = data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"Could not load state relations: {e}")
                self.state_relation_observations = {}

    def _save_state_relations(self):
        os.makedirs(os.path.dirname(self.relation_file) or ".", exist_ok=True)
        tmp = self.relation_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state_relation_observations, f, indent=2)
        os.replace(tmp, self.relation_file)

    def _load_state_groups(self):
        if os.path.exists(self.group_file):
            try:
                with open(self.group_file, 'r') as f:
                    data = json.load(f)
                self.state_groups = data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"Could not load state groups: {e}")
                self.state_groups = {}
        self.state_group_aliases = {}
        for group in self.state_groups.values():
            if group.get('phase', 'active') == 'archived':
                continue
            head = group.get('head_label')
            for child in group.get('child_labels', []):
                if head:
                    self.state_group_aliases[child] = head

    def _save_state_groups(self):
        os.makedirs(os.path.dirname(self.group_file) or ".", exist_ok=True)
        tmp = self.group_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state_groups, f, indent=2)
        os.replace(tmp, self.group_file)

    def get_group_heads_for_labels(self, labels: List[str]) -> List[str]:
        heads = []
        for label in labels or []:
            head = self.state_group_aliases.get(label)
            if head and head not in heads:
                heads.append(head)
        return heads

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

    def save_state(self, label: str, vector: np.ndarray, metadata: dict = None, is_shadow: bool = False, response: str = None, interaction_index: int = 0):
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
            'usage': 1.0,
            'reward_trace': 0.0,
            'last_interaction': int(interaction_index or 0),
            'last_decay_interaction': int(interaction_index or 0),
            'probability': 1.0,
            'tier': 4,
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

    def find_similar_state(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False, interaction_index: int = 0, reward: float = 0.0) -> Optional[str]:
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
        all_states = {
            label: vec for label, vec in all_states.items()
            if self.state_metadata.get(label, {}).get('phase', 'active') != 'archived'
        }
        probabilities = self.get_state_probabilities(interaction_index=interaction_index)

        if not all_states:
            return None
        candidates = []
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim >= threshold:
                prob = probabilities.get(label, 0.0)
                candidates.append((label, float(sim), float(prob), float(sim) * max(prob, 1e-9)))
        if candidates:
            tie_tol = self.config_manager.get('state_manager.selection_tie_tolerance', 1e-9) if self.config_manager else 1e-9
            best_score = max(c[-1] for c in candidates)
            best = [c for c in candidates if abs(c[-1] - best_score) <= tie_tol]
            if len(best) > 1:
                best_sim = max(c[1] for c in best)
                best_prob = max(c[2] for c in best)
                exact = [c for c in best if abs(c[1] - best_sim) <= tie_tol and abs(c[2] - best_prob) <= tie_tol]
                chosen = exact[int(np.random.choice(len(exact)))] if len(exact) > 1 else best[0]
            else:
                chosen = best[0]
            best_label = chosen[0]
            self._normalize_state_probabilities(
                list(all_states.keys()),
                interaction_index=interaction_index,
                reward=reward,
                touched_label=best_label,
                weight=0.1,
            )
            self._save_state_metadata()
            return best_label
        return None

    def find_similar_states(self, vector: np.ndarray, threshold: float = None, include_shadows: bool = False, interaction_index: int = 0, reward: float = 0.0) -> List[Tuple[str, float]]:
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
        all_states = {
            label: vec for label, vec in all_states.items()
            if self.state_metadata.get(label, {}).get('phase', 'active') != 'archived'
        }
        probabilities = self.get_state_probabilities(interaction_index=interaction_index)
        probability_threshold = self.config_manager.get('state_manager.active_probability_threshold', 0.01) if self.config_manager else 0.01

        results = []
        for label, stored_vec in all_states.items():
            stored_norm = self._normalize_vector(stored_vec)
            if np.linalg.norm(query_vec) == 0 or np.linalg.norm(stored_norm) == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, stored_norm) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_norm))
            if sim >= threshold and probabilities.get(label, 0.0) >= probability_threshold:
                results.append((label, sim))
                self._normalize_state_probabilities(
                    list(all_states.keys()),
                    interaction_index=interaction_index,
                    reward=reward,
                    touched_label=label,
                    weight=0.05,
                )
        if results:
            probabilities = self.get_state_probabilities(interaction_index=interaction_index)
            tie_tol = self.config_manager.get('state_manager.selection_tie_tolerance', 1e-9) if self.config_manager else 1e-9
            def sort_key(item):
                label, sim = item
                return (sim * max(probabilities.get(label, 0.0), 1e-9), sim, probabilities.get(label, 0.0))
            results.sort(key=sort_key, reverse=True)
            if len(results) > 1:
                first = results[0]
                tied = [
                    item for item in results
                    if abs(item[1] - first[1]) <= tie_tol and abs(probabilities.get(item[0], 0.0) - probabilities.get(first[0], 0.0)) <= tie_tol
                ]
                if len(tied) > 1:
                    chosen = tied[int(np.random.choice(len(tied)))]
                    rest = [item for item in results if item != chosen]
                    results = [chosen] + rest
            self._save_state_metadata()
        return results

    def _active_state_labels(self, interaction_index: int = 0) -> List[str]:
        self.get_state_probabilities(interaction_index=interaction_index)
        labels = []
        for label in self.states.keys():
            meta = self.state_metadata.get(label, {})
            if meta.get('phase', 'active') != 'archived':
                labels.append(label)
        return labels

    def record_state_relation_observation(
        self,
        labels: List[str],
        tokens: List[str] = None,
        factor_vector: np.ndarray = None,
        reward: float = 0.0,
        interaction_index: int = 0,
        reasoner=None,
    ) -> List[Dict]:
        """
        Record high-similarity state pairs as relation observations.
        This is the probabilistic replacement for automatic user-facing conflicts.
        """
        threshold = float(self._state_config('state_relation_similarity_threshold', 0.85))
        clean_labels = []
        for label in labels or []:
            resolved = self.resolve_label(label)
            if resolved in self.states and resolved not in clean_labels:
                if self.state_metadata.get(resolved, {}).get('phase', 'active') != 'archived':
                    clean_labels.append(resolved)
        if len(clean_labels) < 2:
            return []

        probabilities = self.get_state_probabilities(interaction_index=interaction_index)
        observations = []
        for i, l1 in enumerate(clean_labels):
            for l2 in clean_labels[i + 1:]:
                sim = self._cosine_similarity(self.states[l1], self.states[l2])
                if sim < threshold:
                    continue
                obs = self._record_state_relation_pair(
                    l1, l2, sim, probabilities, tokens=tokens or [],
                    reward=reward, interaction_index=interaction_index,
                )
                observations.append(obs)
        if observations:
            self._save_state_relations()
            self.refresh_state_regrouping(
                interaction_index=interaction_index,
                reasoner=reasoner,
                force=False,
            )
        return observations

    def refresh_state_relations(
        self,
        interaction_index: int = 0,
        tokens: List[str] = None,
        reward: float = 0.0,
        reasoner=None,
    ) -> List[Dict]:
        """Scan all active states and record high-similarity relation observations."""
        labels = self._active_state_labels(interaction_index=interaction_index)
        threshold = float(self._state_config('state_relation_similarity_threshold', 0.85))
        probabilities = self.get_state_probabilities(interaction_index=interaction_index)
        observations = []
        for i, l1 in enumerate(labels):
            for l2 in labels[i + 1:]:
                sim = self._cosine_similarity(self.states[l1], self.states[l2])
                if sim >= threshold:
                    observations.append(self._record_state_relation_pair(
                        l1, l2, sim, probabilities, tokens=tokens or [],
                        reward=reward, interaction_index=interaction_index,
                    ))
        if observations:
            self._save_state_relations()
            self.refresh_state_regrouping(
                interaction_index=interaction_index,
                reasoner=reasoner,
                force=True,
            )
        return observations

    def _record_state_relation_pair(
        self,
        l1: str,
        l2: str,
        similarity: float,
        probabilities: Dict[str, float],
        tokens: List[str],
        reward: float,
        interaction_index: int,
    ) -> Dict:
        key = self._state_relation_key([l1, l2])
        relation = self.state_relation_observations.get(key)
        if not relation:
            relation = {
                'labels': sorted([l1, l2]),
                'first_interaction': int(interaction_index or 0),
                'last_interaction': int(interaction_index or 0),
                'observation_count': 0,
                'similarity_history': [],
                'factor_delta_history': [],
                'probability_delta_history': [],
                'shared_tokens': {},
                'reward_trace': 0.0,
                'positive_reward_trace': 0.0,
                'phase': 'observed',
            }
        p1 = float(probabilities.get(l1, 0.0))
        p2 = float(probabilities.get(l2, 0.0))
        delta = self._factor_delta(self.states[l1], self.states[l2])
        relation['last_interaction'] = int(interaction_index or 0)
        relation['observation_count'] = int(relation.get('observation_count', 0)) + 1
        relation['similarity_history'].append({
            'interaction': int(interaction_index or 0),
            'value': float(similarity),
        })
        relation['factor_delta_history'].append({
            'interaction': int(interaction_index or 0),
            **delta,
        })
        relation['probability_delta_history'].append({
            'interaction': int(interaction_index or 0),
            'value': float(abs(p1 - p2)),
        })
        token_weight = 1.0 + max(0.0, float(reward or 0.0))
        for token in tokens or []:
            token = str(token).strip().lower()
            if token:
                relation['shared_tokens'][token] = float(relation['shared_tokens'].get(token, 0.0)) + token_weight
        relation['reward_trace'] = float(relation.get('reward_trace', 0.0)) + float(reward or 0.0)
        relation['positive_reward_trace'] = float(relation.get('positive_reward_trace', 0.0)) + max(0.0, float(reward or 0.0))
        relation['regroup_score'] = self._relation_regroup_score(relation)
        self.state_relation_observations[key] = relation
        return relation

    def _relation_regroup_score(self, relation: Dict) -> float:
        sims = [float(x.get('value', 0.0)) for x in relation.get('similarity_history', [])]
        prob_deltas = [float(x.get('value', 0.0)) for x in relation.get('probability_delta_history', [])]
        if not sims:
            return 0.0
        recent_n = max(1, int(self._state_config('state_regroup_min_observations', 3)))
        avg_sim = float(np.mean(sims[-recent_n:]))
        avg_prob_delta = float(np.mean(prob_deltas[-recent_n:])) if prob_deltas else 0.0
        reward_evidence = max(0.0, float(np.tanh(float(relation.get('positive_reward_trace', 0.0)))))
        return float(avg_sim * max(0.0, 1.0 - avg_prob_delta) * reward_evidence)

    def refresh_state_regrouping(self, interaction_index: int = 0, reasoner=None, force: bool = False) -> Dict[str, Dict]:
        """Create/update head-label groups from repeated state relation observations."""
        interval = int(self._state_config('state_regroup_interval_interactions', 10))
        if not force and interval > 0 and int(interaction_index or 0) % interval != 0:
            return self.state_groups

        min_obs = int(self._state_config('state_regroup_min_observations', 3))
        sim_threshold = float(self._state_config('state_regroup_similarity_threshold', 0.92))
        eligible_pairs = []
        for relation in self.state_relation_observations.values():
            if relation.get('phase') == 'archived':
                continue
            if int(relation.get('observation_count', 0)) < min_obs:
                continue
            sims = [float(x.get('value', 0.0)) for x in relation.get('similarity_history', [])]
            if not sims:
                continue
            recent = sims[-min_obs:]
            if float(np.mean(recent)) < sim_threshold:
                continue
            score = self._relation_regroup_score(relation)
            if score <= 0.0:
                continue
            labels = [label for label in relation.get('labels', []) if label in self.states]
            if len(labels) == 2 and all(self.state_metadata.get(label, {}).get('phase', 'active') != 'archived' for label in labels):
                eligible_pairs.append((labels[0], labels[1], score))

        if not eligible_pairs:
            return self.state_groups

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for l1, l2, _score in eligible_pairs:
            union(l1, l2)

        components = {}
        for l1, l2, _score in eligible_pairs:
            root = find(l1)
            components.setdefault(root, set()).update([l1, l2])

        for labels in components.values():
            self._upsert_state_group(sorted(labels), interaction_index=interaction_index, reasoner=reasoner)

        self._save_state_groups()
        return self.state_groups

    def _upsert_state_group(self, labels: List[str], interaction_index: int = 0, reasoner=None) -> Dict:
        key = "group::" + self._state_relation_key(labels)
        relation_keys = [
            self._state_relation_key([l1, l2])
            for i, l1 in enumerate(labels)
            for l2 in labels[i + 1:]
        ]
        token_scores = Counter()
        for rel_key in relation_keys:
            relation = self.state_relation_observations.get(rel_key, {})
            for token, score in relation.get('shared_tokens', {}).items():
                token_scores[token] += float(score)

        dominant_tokens = [token for token, _ in token_scores.most_common()]
        head_label = self._select_group_head_label(labels, token_scores, reasoner=reasoner)
        centroid = self._compute_group_centroid(labels, interaction_index=interaction_index)
        factor_signature = self._group_factor_signature(relation_keys)
        probabilities = self.get_state_probabilities(interaction_index=interaction_index, include_archived=True)
        metas = [self.state_metadata.get(label, {}) for label in labels]
        probability = float(sum(probabilities.get(label, 0.0) for label in labels))
        reward_trace = float(sum(float(meta.get('reward_trace', 0.0)) for meta in metas))
        label_set = set(labels)
        for existing_key, existing_group in self.state_groups.items():
            if existing_key == key or existing_group.get('phase') == 'archived':
                continue
            existing_children = set(existing_group.get('child_labels', []))
            if existing_children and existing_children.issubset(label_set):
                existing_group['phase'] = 'archived'
                existing_group['archived_interaction'] = int(interaction_index or 0)
        group = self.state_groups.get(key, {})
        group.update({
            'head_label': head_label,
            'child_labels': labels,
            'centroid_vector': centroid.tolist(),
            'dominant_tokens': dominant_tokens[:10],
            'dominant_phrases': self._dominant_phrases_for_tokens(dominant_tokens, reasoner=reasoner),
            'factor_signature': factor_signature,
            'probability': probability,
            'reward_trace': reward_trace,
            'last_interaction': int(interaction_index or 0),
            'phase': 'active',
        })
        group.setdefault('first_interaction', int(interaction_index or 0))
        self.state_groups[key] = group
        for child in labels:
            self.state_group_aliases[child] = head_label
            meta = self.state_metadata.setdefault(child, {})
            groups = meta.setdefault('state_groups', [])
            if head_label not in groups:
                groups.append(head_label)
        self._save_state_metadata()
        return group

    def _select_group_head_label(self, labels: List[str], token_scores: Counter, reasoner=None) -> str:
        floor = int(self._state_config('state_group_token_evidence_floor', 2))
        max_words = int(self._state_config('state_group_head_label_max_words', 3))
        all_tokens = [token for token, _score in token_scores.most_common()]
        dominant_tokens = [token for token, score in token_scores.most_common() if float(score) >= floor]
        if dominant_tokens:
            usable = dominant_tokens[:max_words]
            if len(usable) >= 1:
                return " ".join(usable)

        if reasoner is not None:
            phrase_defs = getattr(reasoner, 'phrase_definitions', {}) or {}
            for phrase in phrase_defs.keys():
                phrase_tokens = str(phrase).lower().split()
                if len(phrase_tokens) >= floor and any(token in all_tokens for token in phrase_tokens):
                    return " ".join(phrase_tokens[:max_words])
            definitions = getattr(reasoner, 'definitions', {}) or {}
            for word in definitions.keys():
                if str(word).lower() in all_tokens:
                    return str(word).lower()
            kg = getattr(reasoner, 'knowledge_graph', None)
            if kg is not None:
                best_label = None
                best_degree = -1
                for node_id, node in getattr(kg, 'nodes', {}).items():
                    label = str(node.get('label', '')).lower()
                    if label and label in all_tokens:
                        degree = len(getattr(kg, 'edges', {}).get(node_id, []))
                        if degree > best_degree:
                            best_label = label
                            best_degree = degree
                if best_label:
                    return best_label

        top_child = labels[0] if labels else "state"
        return f"group_{top_child}"

    def _dominant_phrases_for_tokens(self, dominant_tokens: List[str], reasoner=None) -> List[str]:
        if reasoner is None or not dominant_tokens:
            return []
        token_set = set(dominant_tokens)
        phrases = []
        for phrase in (getattr(reasoner, 'phrase_definitions', {}) or {}).keys():
            phrase_token_set = set(str(phrase).lower().split())
            if phrase_token_set & token_set:
                phrases.append(str(phrase))
        return phrases[:5]

    def _compute_group_centroid(self, labels: List[str], interaction_index: int = 0) -> np.ndarray:
        probabilities = self.get_state_probabilities(interaction_index=interaction_index, include_archived=True)
        vectors = []
        weights = []
        for label in labels:
            if label not in self.states:
                continue
            meta = self._ensure_state_probability_metadata(label, interaction_index)
            q = float(meta.get('reward_trace', 0.0))
            p = float(probabilities.get(label, 0.0))
            weight = p * (1.0 + max(0.0, float(np.tanh(q))))
            vectors.append(self._normalize_vector(self.states[label]))
            weights.append(weight)
        if not vectors:
            return np.array([])
        max_len = max(len(v) for v in vectors)
        aligned = []
        for vec in vectors:
            if len(vec) < max_len:
                vec = np.pad(vec, (0, max_len - len(vec)), 'constant')
            elif len(vec) > max_len:
                vec = vec[:max_len]
            aligned.append(vec)
        total = float(sum(weights))
        if total <= 1e-12:
            weights = [1.0 for _ in aligned]
            total = float(len(aligned))
        centroid = sum(weight * vec for weight, vec in zip(weights, aligned)) / total
        return self._normalize_vector(centroid)

    def _group_factor_signature(self, relation_keys: List[str]) -> List[Dict]:
        deltas = []
        for key in relation_keys:
            relation = self.state_relation_observations.get(key, {})
            for entry in relation.get('factor_delta_history', []):
                vector = entry.get('vector', [])
                if vector:
                    deltas.append(np.array(vector, dtype=float))
        if not deltas:
            return []
        max_len = max(len(v) for v in deltas)
        aligned = []
        for vec in deltas:
            if len(vec) < max_len:
                vec = np.pad(vec, (0, max_len - len(vec)), 'constant')
            aligned.append(vec[:max_len])
        avg_delta = np.mean(aligned, axis=0)
        top_indices = np.argsort(avg_delta)[::-1][:5]
        return [
            {'factor_index': int(idx), 'mean_delta': float(avg_delta[idx])}
            for idx in top_indices
            if float(avg_delta[idx]) > 0.0
        ]

    def archive_legacy_conflicts(self, interaction_index: int = 0):
        """Preserve old conflict records as historical data when automatic grouping is active."""
        changed = False
        for conflict in self.conflicts:
            if not conflict.get('resolved', False):
                conflict['resolved'] = True
                conflict['phase'] = 'archived'
                conflict['resolution_type'] = 'legacy_archived'
                conflict['archived_interaction'] = int(interaction_index or 0)
                changed = True
        if changed:
            self._save_conflicts()

    # ------------------------------------------------------------------
    # Conflict detection & resolution (adapt to handle different lengths)
    # ------------------------------------------------------------------
    def detect_conflicts(self, threshold: float = None) -> List[Dict]:
        """
        Scan all states and find pairs with similarity > threshold.
        Both vectors are normalised to the reference dimension.
        Returns conflicts including associated words if available in metadata.
        """
        if not self._legacy_interactive_conflicts_enabled():
            return []
        if threshold is None:
            if self.config_manager:
                threshold = self.config_manager.get('state_manager.conflict_threshold', 0.85)
            else:
                threshold = 0.85

        labels = self._active_state_labels()
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
                meta2 = self.state_metadata.get(l2, {})
                words2 = meta2.get('tokens', [])
                sim = self._cosine_similarity(v1, self.states[l2])
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
        if not self._legacy_interactive_conflicts_enabled():
            self.archive_legacy_conflicts()
            return []
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
        return self.conflicts

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
        """Create a temporary shadow state from the probability-weighted group centroid."""
        try:
            l1, l2 = conflict["labels"][0], conflict["labels"][1]
            merged_vector = self._compute_group_centroid([l1, l2], interaction_index=int(conflict.get('interaction_index', 0) or 0))

            shadow_label = f"shadow_{l1}_{l2}_{int(time.time())}"
            metadata = {
                "merged_from": [l1, l2],
                "similarity": conflict["similarity"],
                "merge_type": "probability_reward_centroid",
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
        if not self._legacy_interactive_conflicts_enabled():
            return []
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

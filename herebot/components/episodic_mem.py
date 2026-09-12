from importglobal import Optional,np,os,json,Dict,time,List
from .directory_handler import STATE_DIR
from .config_manager import ConfigManager
# ------------------------------------------------------------
# Episodic Memory – stores complete interaction episodes
# ------------------------------------------------------------
class EpisodicMemory:
    """
    Stores (state, action, reward, next_state) for reinforcement learning
    and offline consolidation. Episodes are saved to ./episodes/.

    Now config-aware: reads capacity from config.
    """

    EPISODE_DIR = os.path.join(STATE_DIR, "episodes")

    def __init__(self, capacity: int = None, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        if config_manager:
            self.capacity = config_manager.get('episodic_memory.capacity', 10000)
        else:
            self.capacity = capacity if capacity is not None else 10000

        try:
            os.makedirs(self.EPISODE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create EPISODE_DIR: {e}")
        self.episodes = []          # list of dicts
        self._load_episodes()

    def update_parameters_from_config(self):
        """Reload capacity from config (hot reload)."""
        if self.config_manager:
            new_capacity = self.config_manager.get('episodic_memory.capacity', 10000)
            self.capacity = new_capacity
            # Trim if needed
            while len(self.episodes) > self.capacity:
                removed = self.episodes.pop(0)
                # Optionally delete from disk

    def _load_episodes(self):
        """Load all episode files from disk."""
        for fname in sorted(os.listdir(self.EPISODE_DIR)):
            if fname.endswith(".json"):
                path = os.path.join(self.EPISODE_DIR, fname)
                try:
                    with open(path, 'r') as f:
                        ep = json.load(f)
                    self.episodes.append(ep)
                except Exception as e:
                    print(f"Could not load episode {fname}: {e}")
        # keep only last `capacity` episodes
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def add_episode(self, user_input: str, system_response: str,
                    factor_vector: Dict, intrinsic_reward: float,
                    metadata: dict = None, factor_space=None, factor_statuses: dict = None):

        if not isinstance(factor_vector, dict):
            factor_vector = {}

        episode = {
            "timestamp": time.time(),
            "user_input": user_input,
            "system_response": system_response,
            "factor_vector": dict(factor_vector),
            "factor_dim": len(factor_vector),
            "intrinsic_reward": float(intrinsic_reward),
            "metadata": metadata or {},
        }
        if factor_space is not None:
            snapshot = factor_space.snapshot() if hasattr(factor_space, 'snapshot') else factor_space
            episode.update({
                "factor_space_id": snapshot.space_id,
                "factor_space_version": snapshot.version,
                "factor_names": list(snapshot.factor_names),
                "factor_statuses": dict(factor_statuses or {
                    name: "observed" for name in factor_vector
                }),
            })
        self.episodes.append(episode)
        # save to disk
        fname = f"episode_{int(episode['timestamp'])}_{np.random.randint(100000)}.json"
        path = os.path.join(self.EPISODE_DIR, fname)
        with open(path, 'w') as f:
            json.dump(episode, f, indent=2)
        # trim
        if len(self.episodes) > self.capacity:
            removed = self.episodes.pop(0)
            # also remove from disk (optional)

    def sample_batch(self, n: int = 32) -> List[Dict]:
        """Random sample of episodes for offline learning."""
        indices = np.random.choice(len(self.episodes), size=min(n, len(self.episodes)), replace=False)
        return [self.episodes[i] for i in indices]

    def replay_sequence(self) -> List[Dict]:
        """Return episodes in chronological order (for offline consolidation)."""
        return self.episodes[-100:]  # last 100

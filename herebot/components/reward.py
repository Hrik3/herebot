from .config_manager import ConfigManager
from importglobal import Dict,Optional, pickle, os  
# ------------------------------------------------------------
# Intrinsic Motivation: Reward based on Prediction Error / Information Gain
# ------------------------------------------------------------
class IntrinsicRewardTracker:
    """
    Tracks cumulative intrinsic reward and provides reinforcement signal.
    Reward = prediction error (1 - confidence) + information gain + Bayesian surprise.

    Now config‑aware (placeholder for future tunable parameters).
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.reward_history = []
        # No tunable parameters currently, but we keep the config manager for consistency

    def update_parameters_from_config(self):
        """Placeholder for hot reload (no parameters currently)."""
        pass

    def compute_reward(self, token_ci: Dict, igain: float, bayes_surprise: float) -> float:
        """Compute intrinsic reward for a single token observation."""
        # Prediction error: 1 - confidence
        pred_error = 1.0 - token_ci.get('confidence', 0.5)
        # Reward = weighted combination
        reward = pred_error * 0.5 + igain * 0.3 + bayes_surprise * 0.2
        self.episode_reward += reward
        self.total_reward += reward
        return reward

    def reset_episode(self):
        self.episode_reward = 0.0

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({'total_reward': self.total_reward, 'reward_history': self.reward_history}, f)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.total_reward = data['total_reward']
            self.reward_history = data['reward_history']

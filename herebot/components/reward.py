from components.config_manager import ConfigManager
from importglobal import Dict,Optional, pickle, os  
# ------------------------------------------------------------
# Intrinsic Motivation: Reward based on Prediction Error / Information Gain
# ------------------------------------------------------------
class IntrinsicRewardTracker:
    """
    Tracks cumulative intrinsic reward and provides reinforcement signal.
    Reward = prediction error (1 - confidence) + information gain + Bayesian surprise.

    Now config-aware (placeholder for future tunable parameters).
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

    def _clip_reward_part(self, value) -> float:
        """Keep reward components finite and nonnegative."""
        try:
            value = float(value)
        except Exception:
            return 0.0
        if value != value:
            return 0.0
        return max(0.0, min(1.0, value))

    def compute_reward(self, token_ci: Dict, igain: float, bayes_surprise: float) -> float:
        """Compute intrinsic reward for a single token observation."""
        pred_error = 1.0 - token_ci.get('confidence', 0.5)
        reward = (
            self._clip_reward_part(pred_error)
            + self._clip_reward_part(igain)
            + self._clip_reward_part(bayes_surprise)
        )
        self.episode_reward += reward
        self.total_reward += reward
        self.reward_history.append({
            'source': 'intrinsic',
            'rating': 'intrinsic',
            'magnitude': reward,
            'reward': reward,
        })
        return reward

    def apply_feedback_reward(self, rating: str, magnitude: float, source: str = 'feedback') -> float:
        """Apply signed user-feedback reward."""
        magnitude = abs(float(magnitude or 0.0))
        rating = (rating or 'neutral').lower()
        if 'positive' in rating:
            reward = magnitude
            normalized = 'positive'
        elif 'negative' in rating:
            reward = -magnitude
            normalized = 'negative'
        else:
            reward = 0.0
            normalized = 'neutral'

        self.episode_reward += reward
        self.total_reward += reward
        self.reward_history.append({
            'source': source,
            'rating': normalized,
            'magnitude': magnitude,
            'reward': reward,
        })
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
            self.total_reward = data.get('total_reward', 0.0)
            self.reward_history = data.get('reward_history', [])

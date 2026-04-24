from importglobal import Optional,Tuple,np
from components.config_manager import ConfigManager
# ------------------------------------------------------------
# Gaussian Process Uncertainty Model (Bayesian Linear Regression with RBF features)
# ------------------------------------------------------------
class GaussianProcessUncertainty:
    """
    Simplified Gaussian Process using Bayesian linear regression with random Fourier features.
    Approximates an RBF kernel for online uncertainty estimation.
    Maintains posterior over weights and predicts mean and variance.

    Now config-aware: reads input_dim, n_features, length_scale, sigma_noise.
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None,
                 input_dim: int = None, n_features: int = None,
                 length_scale: float = None, sigma_noise: float = None):
        self.config_manager = config_manager
        if config_manager:
            self.input_dim = config_manager.get('gaussian_process.input_dim', 5)
            self.n_features = config_manager.get('gaussian_process.n_features', 100)
            self.length_scale = config_manager.get('gaussian_process.length_scale', 1.0)
            self.sigma_noise = config_manager.get('gaussian_process.sigma_noise', 0.1)
        else:
            self.input_dim = input_dim if input_dim is not None else 5
            self.n_features = n_features if n_features is not None else 100
            self.length_scale = length_scale if length_scale is not None else 1.0
            self.sigma_noise = sigma_noise if sigma_noise is not None else 0.1

        # Random Fourier features: generate random weights and biases
        self.W = np.random.randn(self.n_features, self.input_dim) / self.length_scale
        self.b = np.random.uniform(0, 2 * np.pi, self.n_features)

        # Bayesian linear regression posterior parameters
        self.m = np.zeros(self.n_features)          # mean of weights
        self.S = np.eye(self.n_features) * 1.0      # covariance of weights
        self.beta = 1.0 / self.sigma_noise**2       # precision of noise

    def update_parameters_from_config(self):
        """Reload parameters from the config manager (for hot reload)."""
        if self.config_manager:
            self.input_dim = self.config_manager.get('gaussian_process.input_dim', 5)
            self.n_features = self.config_manager.get('gaussian_process.n_features', 100)
            self.length_scale = self.config_manager.get('gaussian_process.length_scale', 1.0)
            self.sigma_noise = self.config_manager.get('gaussian_process.sigma_noise', 0.1)
            self.beta = 1.0 / self.sigma_noise**2
            # Note: Random Fourier features are not regenerated on hot reload
            # because that would change the feature map.
            # In a full implementation you might want to handle that, but we keep it simple.

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Compute random Fourier features for input x (1D or 2D)."""
        x = np.atleast_2d(x)
        features = np.cos(x @ self.W.T + self.b) * np.sqrt(2.0 / self.n_features)
        return features

    def update(self, x: np.ndarray, y: float):
        """Online Bayesian update with new observation (x, y)."""
        try:
            phi = self._phi(x).flatten()
            # Posterior precision: S^{-1} = S_0^{-1} + beta * phi phi^T
            # Sherman-Morrison update
            S_inv = np.linalg.pinv(self.S)
            S_inv += self.beta * np.outer(phi, phi)
            self.S = np.linalg.pinv(S_inv)
            self.m = self.S @ (S_inv @ self.m + self.beta * phi * y)
        except np.linalg.LinAlgError:
            # Singular matrix – reset covariance
            self.S = np.eye(self.n_features) * 1.0
            self.m = np.zeros(self.n_features)
        except Exception as e:
            print(f"GP update failed: {e}. Skipping update.")

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Return predictive mean and variance."""
        phi = self._phi(x).flatten()
        mean = phi @ self.m
        var = 1.0 / self.beta + phi @ self.S @ phi.T
        var = max(var, 1e-12)
        return mean, var

    def uncertainty(self, x: np.ndarray) -> float:
        """Return predictive standard deviation (uncertainty)."""
        _, var = self.predict(x)
        return np.sqrt(var)

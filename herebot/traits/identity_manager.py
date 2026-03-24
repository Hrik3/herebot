from importglobal import np,os
from components.directory_handler import STATE_DIR
# ------------------------------------------------------------
# Identity Manager – persistent self-identity representation
# ------------------------------------------------------------
class IdentityManager:
    """
    Maintains a stable self-identity vector that is reinforced
    over successful interactions.
    """

    def __init__(self, emb_dim: int = 64, learning_rate: float = 0.05):
        self.save_dir = STATE_DIR                       # <<< ADD THIS
        self.IDENTITY_FILE = os.path.join(self.save_dir, "identity.npz")  # <<< MOVE HERE

        self.emb_dim = emb_dim
        self.learning_rate = learning_rate

        # Identity embedding – starts as zero vector (neutral)
        self.identity = np.zeros(emb_dim)

        # Confidence in identity (0-1)
        self.confidence = 0.1

        # Number of reinforcement events
        self.reinforcement_count = 0

        self._load()

    def _load(self):
        """Load identity from disk."""
        if os.path.exists(self.IDENTITY_FILE):
            try:
                data = np.load(self.IDENTITY_FILE)
                self.identity = data['identity']
                self.confidence = data['confidence']
                self.reinforcement_count = data['reinforcement_count']
                self.emb_dim = len(self.identity)
            except Exception as e:
                print(f"Could not load identity: {e}")
    
    def save(self):
        """Save identity to disk (atomic)."""
        os.makedirs(self.save_dir, exist_ok=True)

        tmp = self.IDENTITY_FILE + ".tmp.npz"

        with open(tmp, "wb") as f:
            np.savez(
                f,
                identity=self.identity,
                confidence=self.confidence,
                reinforcement_count=self.reinforcement_count
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.IDENTITY_FILE)
    
    def reinforce(self, factor_vector: np.ndarray, reward: float):
        """
        Update identity embedding towards factor_vector when a positive
        outcome occurs. Automatically expands identity if factor_vector is longer.
        """
        # If factor_vector is longer than current identity, expand identity
        # project factor_vector to identity dimension
        if len(factor_vector) != len(self.identity):
            projected = np.zeros_like(self.identity)
            n = min(len(projected), len(factor_vector))
            projected[:n] = factor_vector[:n]
            factor_vector = projected

        
        aligned = np.zeros_like(self.identity)
        n = min(len(aligned), len(factor_vector))
        aligned[:n] = factor_vector[:n]

        norm = np.linalg.norm(aligned)
        if norm > 1e-12:
            direction = aligned / norm
        else:
            direction = np.zeros_like(aligned)
        
        # Update identity (soft update)
        alpha = self.learning_rate * reward * (1 - self.confidence)
        self.identity = (1 - alpha) * self.identity + alpha * direction
        
        # Normalize identity
        norm_id = np.linalg.norm(self.identity)
        if norm_id > 0:
            self.identity /= norm_id
        
        # Increase confidence slowly
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.01 * reward)
        
        self.save()
    
    def detect_drift(self, current_vector: np.ndarray, threshold: float = 0.3) -> float:
        """
        Returns the angular distance between current state vector and stored identity.
        Handles vectors of different lengths via truncation.
        """
        if np.linalg.norm(self.identity) == 0:
            return 0.0
        min_len = min(len(current_vector), len(self.identity))
        vec = current_vector[:min_len]
        ident = self.identity[:min_len]
        sim = np.dot(vec, ident) / (np.linalg.norm(vec) * np.linalg.norm(ident) + 1e-12)
        return 1.0 - sim
    
    def get_stability_bonus(self, current_vector: np.ndarray) -> float:
        """
        Returns a bonus factor (0-1) based on how aligned the current state
        is with the identity.
        """
        drift = self.detect_drift(current_vector)
        bonus = 1.0 - drift
        return max(0, min(1, bonus))

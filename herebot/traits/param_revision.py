from importglobal import np,List,Dict,Optional
from config_manager import ConfigManager
# ------------------------------------------------------------
# Consciousness Model – Global Workspace with Attention
# ------------------------------------------------------------
class ParameterRevision:
    """
    Simulates a global workspace:
    - Parallel processors generate "coalitions" (hypotheses, predictions, questions).
    - A winner-take-all competition selects the most salient coalition.
    - The winner is broadcast globally, influencing all modules.
    Currently just updates Parameters with basic np maths
    config‑aware (placeholder for future parameters).
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.config_manager = config_manager
        self.workspace = []      # current active coalitions
        self.global_broadcast = None  # current conscious content
        self.attention_weights = {}   # module -> weight

    def update_parameters_from_config(self):
        """Reload configuration parameters."""
        if self.config_manager:
            # You could enable/disable features here, but we'll keep it simple for now.
            pass

    def update(self, coalitions: List[Dict], factors: Dict[str, float]):
        """
        coalitions: list of candidate hypotheses / actions / queries
        factors: current factor vector (curiosity, threat, etc.)
        Computes salience and selects winner.
        """
        salience = []
        for c in coalitions:
            # salience = curiosity * (1 - confidence) + threat * negativity + ...
            s = factors.get('curiosity', 0) * c.get('curiosity_relevance', 1)
            s += factors.get('threat', 0) * c.get('threat_relevance', 0)
            s += np.random.rand() * 0.1   # stochasticity
            salience.append(s)
        if salience:
            idx = np.argmax(salience)
            self.global_broadcast = coalitions[idx]
        else:
            self.global_broadcast = None
        return self.global_broadcast

    def get_current_goal(self) -> Optional[str]:
        """Extract a natural language goal from the broadcast."""
        if self.global_broadcast:
            return self.global_broadcast.get('query') or self.global_broadcast.get('hypothesis')
        return None

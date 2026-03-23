from .importglobal import List,Dict,np,dataclass,field
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

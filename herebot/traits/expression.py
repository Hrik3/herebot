class Expression:
    """
    Tree‑structured expression built from symbols and operators.
    """
    def __init__(self, root, left=None, right=None):
        self.root = root          # can be a symbol string or an Operator
        self.left = left          # left child (Expression or None)
        self.right = right        # right child (Expression or None)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __repr__(self):
        if self.is_leaf():
            return str(self.root)
        return f"({self.left} {self.root} {self.right})"
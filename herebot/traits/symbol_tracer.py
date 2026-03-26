from importglobal import Optional, Dict, Any, Set, Callable, List,json,os,Tuple
from traits.expression import Expression
# =============================================================================
# SYMBOL REGISTRY – Formal symbol grounding
# =============================================================================
class Symbol:
    """A formal symbol with type, arity, and semantics."""
    def __init__(self, name: str, sym_type: str, arity: Optional[int] = None, semantics: Optional[Callable] = None):
        self.name = name
        self.sym_type = sym_type  # one of: 'number', 'operator', 'variable', 'predicate', 'relation'
        self.arity = arity
        self.semantics = semantics  # callable for operators, value for numbers, etc.

    def __repr__(self):
        return f"Symbol({self.name}, {self.sym_type})"

class SymbolRegistry:
    """Registry of all formal symbols known to the system."""
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}

    def register(self, symbol: Symbol):
        self.symbols[symbol.name] = symbol

    def get(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def has(self, name: str) -> bool:
        return name in self.symbols

    def get_all_of_type(self, sym_type: str) -> List[Symbol]:
        return [s for s in self.symbols.values() if s.sym_type == sym_type]

    def save(self, path: str):
        """Save symbol metadata (semantics not included)."""
        data = {}
        for name, sym in self.symbols.items():
            data[name] = {
                'name': sym.name,
                'sym_type': sym.sym_type,
                'arity': sym.arity,
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load symbol metadata (semantics must be re‑registered separately)."""
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        for name, d in data.items():
            sym = Symbol(d['name'], d['sym_type'], arity=d['arity'], semantics=None)
            self.symbols[name] = sym
# =============================================================================
# RULE SCHEMA LEARNING – Variable‑based rule generalization
# =============================================================================
class Variable:
    """Represents a variable symbol in a rule schema (e.g., a, b)."""
    def __init__(self, name: str, type_constraint: Optional[str] = None):
        self.name = name
        self.type_constraint = type_constraint  # e.g., 'Number', 'Word', etc.

    def __repr__(self):
        return f"Var({self.name})"

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name and self.type_constraint == other.type_constraint

    def __hash__(self):
        return hash((self.name, self.type_constraint))

class RuleSchema:
    """
    A parameterized rule: For all variables X1..Xn, conclusion holds if premises are true.
    Example: For all a,b: Add(a,b) = Sum(a,b)
    """
    def __init__(self, conclusion: Expression, premises: List[Expression] = None,
                 variables: List[Variable] = None, confidence: float = 0.5,
                 evidence: int = 0):
        self.conclusion = conclusion
        self.premises = premises or []
        self.variables = variables or []
        self.confidence = confidence
        self.evidence = evidence
        self.id = f"schema_{id(self)}"

    def instantiate(self, bindings: Dict[str, Any]) -> Tuple[Expression, List[Expression]]:
        """Replace variables with concrete values in conclusion and premises."""
        def substitute(expr: Expression) -> Expression:
            if expr.is_leaf():
                if isinstance(expr.root, Variable):
                    root = bindings[expr.root.name]
                else:
                    root = expr.root
                return Expression(root)
            else:
                left = substitute(expr.left) if expr.left else None
                right = substitute(expr.right) if expr.right else None
                return Expression(expr.root, left, right)

        new_conc = substitute(self.conclusion)
        new_prems = [substitute(p) for p in self.premises]
        return new_conc, new_prems

    def __repr__(self):
        vars_str = ','.join(str(v) for v in self.variables)
        return f"RuleSchema({vars_str}: {self.conclusion} <- {self.premises})"

class SchemaLearner:
    """
    Induces rule schemas from observed expressions and user feedback.
    Uses anti‑unification to find common patterns with variables.
    """
    def __init__(self, engine: 'ReasoningEngine'):
        self.engine = engine
        self.schema_counter = 0

    def anti_unify(self, expr1: Expression, expr2: Expression) -> Optional[Tuple[Expression, List[Variable]]]:
        """
        Find the most specific generalization of two expressions, introducing variables.
        Returns (generalized_expression, list_of_variables) or None if no common structure.
        """
        # If both are leaves
        if expr1.is_leaf() and expr2.is_leaf():
            if expr1.root == expr2.root:
                # same constant → no variable
                return expr1, []
            else:
                # different constants → introduce a variable
                var_name = f"x{self.engine.variable_counter}"
                self.engine.variable_counter += 1
                var = Variable(var_name)
                return Expression(var), [var]

        # If both are operators with same root and same structure
        if (not expr1.is_leaf() and not expr2.is_leaf() and
                expr1.root == expr2.root and
                ((expr1.left is None) == (expr2.left is None)) and
                ((expr1.right is None) == (expr2.right is None))):

            # Recursively anti‑unify children
            new_left = None
            new_right = None
            all_vars = []

            if expr1.left is not None:
                left_gen, left_vars = self.anti_unify(expr1.left, expr2.left)
                if left_gen is None:
                    return None
                new_left = left_gen
                all_vars.extend(left_vars)

            if expr1.right is not None:
                right_gen, right_vars = self.anti_unify(expr1.right, expr2.right)
                if right_gen is None:
                    return None
                new_right = right_gen
                all_vars.extend(right_vars)

            return Expression(expr1.root, new_left, new_right), all_vars

        return None

    def learn_from_pair(self, expr1: Expression, expr2: Expression, result_expr: Expression = None):
        """
        Given two similar expressions, try to induce a schema.
        If a result_expr is provided (e.g., from an equation), we learn a rule.
        """
        gen, vars_found = self.anti_unify(expr1, expr2)
        if gen is None or not vars_found:
            return None

        # If we have a result expression, anti‑unify that too?
        # For now, we assume expr1 and expr2 are two instances of a pattern.
        # We'll create a schema: For all vars, pattern holds? But we need a conclusion.
        # In the example "1+1=2", we have two expressions: the left side (Add(1,1)) and the right side (2).
        # So we anti‑unify left1 and left2? Actually we have one equation: left = right.
        # To learn Add(a,b) = Sum(a,b), we need multiple examples of the form left = right.
        # We'll collect equations.

        # For now, we just store the generalized pattern as a potential schema.
        schema = RuleSchema(gen, variables=vars_found, confidence=0.3)
        self.engine.schemas.append(schema)
        return schema

    def incorporate_equation(self, left_expr: Expression, right_expr: Expression):
        """
        Called when an equation (e.g., "2 + 3 = 5") is observed.
        Tries to match existing schemas or induce new ones.
        """
        # Check if we already have a schema that can be instantiated to this equation
        for schema in self.engine.schemas:
            # Try to bind variables in conclusion to left and right?
            # For a schema of the form Add(a,b) = Sum(a,b), the conclusion is the whole equation.
            # But our schema stores the conclusion as a single expression (the left side?) – we need to store both sides.
            # Better: store a rule as (antecedent, consequent) or as an equivalence.
            # For simplicity, we'll store the left pattern and right pattern separately.

            # For now, skip; we'll implement later with more structure.
            pass

        # Attempt to anti‑unify left with something? Not yet.

        # For now, we just note the equation.
        pass

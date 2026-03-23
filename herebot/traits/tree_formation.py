from importglobal import Optional, Dict, Any, Set, Callable, List,re
from .expression import Expression
from symbol_tracer import Variable
# =============================================================================
# TYPED SYMBOL & OPERATOR CLASSES
# =============================================================================
class SymbolType:
    """Represents a type in the type hierarchy."""
    def __init__(self, name: str, parent: Optional['SymbolType'] = None, 
                 properties: Dict[str, Any] = None):
        self.name = name
        self.parent = parent          # for single inheritance
        self.properties = properties or {}
        self.instances = set()        # words/phrases that are of this type

    def is_subtype_of(self, other: 'SymbolType') -> bool:
        if self == other:
            return True
        if self.parent is None:
            return False
        return self.parent.is_subtype_of(other)

    def __repr__(self):
        return f"SymbolType({self.name})"

class TypeHierarchy:
    """Manages a tree of types and assigns types to symbols."""
    def __init__(self):
        self.types: Dict[str, SymbolType] = {}
        # Predefine some basic types
        self.add_type("Entity")           # root
        self.add_type("Word", parent="Entity")
        self.add_type("Phrase", parent="Entity")
        self.add_type("Number", parent="Entity")
        self.add_type("Operator", parent="Entity")
        self.add_type("State", parent="Entity")

    def add_type(self, name: str, parent: Optional[str] = None, 
                 properties: Dict[str, Any] = None) -> SymbolType:
        if name in self.types:
            return self.types[name]
        parent_type = self.types.get(parent) if parent else None
        typ = SymbolType(name, parent_type, properties)
        self.types[name] = typ
        return typ

    def get_type(self, name: str) -> Optional[SymbolType]:
        return self.types.get(name)

    def assign_type(self, symbol: str, type_name: str):
        """Mark a symbol (word/phrase) as being of a given type."""
        typ = self.get_type(type_name)
        if typ:
            typ.instances.add(symbol)

    def get_symbols_of_type(self, type_name: str) -> Set[str]:
        typ = self.get_type(type_name)
        if typ is None:
            return set()
        # include subtypes?
        result = set(typ.instances)
        # add instances of all subtypes (optional)
        for t in self.types.values():
            if t.is_subtype_of(typ):
                result.update(t.instances)
        return result

    def infer_type(self, symbol: str) -> Optional[SymbolType]:
        """Find the most specific type that contains this symbol."""
        for typ in self.types.values():
            if symbol in typ.instances:
                return typ
        return None

class Operator:
    """Formal definition of an operator (e.g., +, -, =, >)."""
    def __init__(self, symbol: str, arity: int, precedence: int,
                 evaluator: Optional[Callable] = None):
        self.symbol = symbol
        self.arity = arity          # number of arguments (infix operators have arity 2)
        self.precedence = precedence
        self.evaluator = evaluator  # function that takes arguments and returns a value

    def evaluate(self, *args):
        if self.evaluator is None:
            raise NotImplementedError(f"Operator {self.symbol} has no evaluator")
        return self.evaluator(*args)

    def __repr__(self):
        return f"Operator({self.symbol})"

class SyntaxTree:
    """
    Represents a parse tree for a sentence, with nodes as symbols and operators.
    """
    def __init__(self, root: str, children: List['SyntaxTree'] = None):
        self.root = root          # the symbol at this node
        self.children = children or []

    def to_expression(self, type_hierarchy: TypeHierarchy, 
                      operator_registry: Dict[str, Operator]) -> Optional[Expression]:
        """
        Convert the syntax tree to an expression, using the type system
        and operator definitions.
        """
        # if root is an operator, build an expression with children as sub‑expressions
        if self.root in operator_registry:
            op = operator_registry[self.root]
            if op.arity == 2 and len(self.children) == 2:
                left_expr = self.children[0].to_expression(type_hierarchy, operator_registry)
                right_expr = self.children[1].to_expression(type_hierarchy, operator_registry)
                if left_expr and right_expr:
                    return Expression(op, left_expr, right_expr)
            # unary operators can be added later
        # otherwise it's a leaf symbol
        # Check if it's a variable (single lowercase letter)
        if isinstance(self.root, str):
            # Check if it's a number
            try:
                float(self.root)
                return Expression(self.root)  # treat as constant number
            except ValueError:
                # Not a number
                if re.match(r'^[a-z]$', self.root):
                    # Single lowercase letter -> variable
                    return Expression(Variable(self.root))
        # default: treat as constant symbol
        return Expression(self.root)

import math

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Define function evaluation and derivative mapping
    def eval_func(name, val):
        if name == "square":
            return val ** 2
        elif name == "sin":
            return math.sin(val)
        elif name == "exp":
            return math.exp(val)
        elif name == "log":
            return math.log(val)
        else:
            raise ValueError(f"Unknown function: {name}")
    
    def deriv_func(name, val):
        if name == "square":
            return 2 * val
        elif name == "sin":
            return math.cos(val)
        elif name == "exp":
            return math.exp(val)
        elif name == "log":
            return 1 / val
        else:
            raise ValueError(f"Unknown function: {name}")
    
    # Chain rule: multiply derivatives step by step from inner to outer
    current_val = x
    grad_product = 1.0
    
    # Iterate inner function first (rightmost in list) to outer (leftmost)
    for func in reversed(functions):
        # derivative of outer func evaluated at inner result
        d = deriv_func(func, current_val)
        grad_product *= d
        # update value to pass into outer layer next iteration
        current_val = eval_func(func, current_val)
    
    return grad_product
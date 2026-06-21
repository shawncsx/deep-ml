import torch

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> torch.Tensor:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as torch.Tensor (shape m x n)
        Entry J[i,j] = ∂f_i / ∂x_j
    """
    n = len(x)
    # Evaluate original function to get output dimension m
    f_x = f(x)
    m = len(f_x)
    
    jac = torch.zeros((m, n), dtype=torch.float32)
    
    # Iterate over each input variable x_j
    for j in range(n):
        # Create perturbed point x + h along dimension j
        x_plus = x.copy()
        x_plus[j] += h
        f_plus = f(x_plus)
        
        # Finite difference: df_i / dx_j ≈ (f(x+h e_j) - f(x)) / h
        for i in range(m):
            partial = (f_plus[i] - f_x[i]) / h
            jac[i, j] = partial
    
    return jac
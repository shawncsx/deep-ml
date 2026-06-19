import torch
import math

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> torch.Tensor:
    """
    Compute partial derivatives of multivariable functions using PyTorch autograd.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x^2*y + x*y^2
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x*sin(y)
            'poly3d': f(x,y,z) = x^2*y + y*z^2
            'squared_error': f(x,y) = (x-y)^2
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tensor of partial derivatives [df/dx, df/dy, ...] at point
    """
    # Your code here
    x = point[0]
    y = point[1]

    if func_name == 'poly2d':
        xy = 2*x*y
        xx = x**2
        yy = y**2
        return torch.tensor([xy+yy,xy+xx])
    elif func_name == 'exp_sum':
        xy = math.exp(x+y)
        return torch.tensor([xy,xy])
    elif func_name == 'product_sin':
        return torch.tensor([math.sin(y),x*math.cos(y)])
    elif func_name == 'poly3d':
        z = point[2]
        return torch.tensor([2*x*y,x**2+z**2,2*y*z])
    elif func_name == 'squared_error':
        x_y=2*(x-y)
        return torch.tensor([x_y,-1*x_y])

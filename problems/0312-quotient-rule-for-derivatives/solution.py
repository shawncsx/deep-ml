import torch

def eval_poly(coeffs: list, x: float) -> float:
    """
    计算降幂多项式在x处的值
    coeffs: [a_n,a_{n-1},...,a0] = a_n x^n + ... + a0
    """
    max_power = len(coeffs) - 1
    total = 0.0
    for idx, c in enumerate(coeffs):
        power = max_power - idx
        total += c * (x ** power)
    return total

def diff_poly(coeffs: list) -> list:
    """对降幂系数多项式求导，返回导数的降幂系数"""
    if len(coeffs) == 1:
        # 常数多项式，导数为0（空列表代表零多项式）
        return []
    max_p = len(coeffs) - 1
    deriv_coeffs = []
    for idx, c in enumerate(coeffs):
        power = max_p - idx
        new_c = c * power
        deriv_coeffs.append(new_c)
    # 去掉最后一项（常数项求导结果，不需要）
    return deriv_coeffs[:-1]

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # 1. 求 g', h' 的系数
    g_prime_coeffs = diff_poly(g_coeffs)
    h_prime_coeffs = diff_poly(h_coeffs)
    
    # 2. 计算四个代入值
    g_x = eval_poly(g_coeffs, x)
    h_x = eval_poly(h_coeffs, x)
    g_p_x = eval_poly(g_prime_coeffs, x) if g_prime_coeffs else 0.0
    h_p_x = eval_poly(h_prime_coeffs, x) if h_prime_coeffs else 0.0
    
    # 3. 商求导公式
    numerator = g_p_x * h_x - g_x * h_p_x
    denominator = h_x ** 2
    deriv_val = numerator / denominator
    
    # 返回标量tensor
    return torch.tensor(deriv_val, dtype=torch.float32)
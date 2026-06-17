import torch

def derivative(coeffs: list) -> list:
    """求单个多项式的导数系数列表"""
    # 常数多项式，导数为0
    if len(coeffs) == 1:
        return []
    deriv_coeffs = []
    for power, c in enumerate(coeffs):
        # x^power 求导后变为 power*c * x^(power-1)
        deriv_val = power * c
        deriv_coeffs.append(deriv_val)
    # 去掉常数项（x^0），剩下就是导数多项式
    return deriv_coeffs[1:]

def poly_mult(p1: list, p2: list) -> list:
    """多项式乘法，返回乘积系数"""
    max_deg = len(p1) + len(p2) - 2
    res = [0.0] * (max_deg + 1)
    for i, c1 in enumerate(p1):
        for j, c2 in enumerate(p2):
            res[i + j] += c1 * c2
    return res

def poly_add(p1: list, p2: list) -> list:
    """多项式加法，对齐次数相加"""
    max_len = max(len(p1), len(p2))
    res = []
    for i in range(max_len):
        val1 = p1[i] if i < len(p1) else 0.0
        val2 = p2[i] if i < len(p2) else 0.0
        res.append(val1 + val2)
    return res

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> torch.Tensor:
    """
    用乘积法则计算 (f*g)' 的多项式系数
    
    Args:
        f_coeffs: f(x) 系数列表，下标i是x^i系数
        g_coeffs: g(x) 系数列表，下标i是x^i系数
    
    Returns:
        torch.Tensor，系数保留4位小数，去除末尾无效0；零多项式返回[0.0]
    """
    # 1. 分别求 f' 和 g'
    f_prime = derivative(f_coeffs)
    g_prime = derivative(g_coeffs)
    
    # 2. 计算 f'*g 和 f*g'
    term1 = poly_mult(f_prime, g_coeffs)
    term2 = poly_mult(f_coeffs, g_prime)
    
    # 3. 两项相加得到 (fg)'
    total = poly_add(term1, term2)
    
    # 4. 四舍五入保留4位小数
    rounded = [round(x, 4) for x in total]
    
    # 5. 去除末尾连续的0
    while len(rounded) > 1 and rounded[-1] == 0.0:
        rounded.pop()
    
    # 全零多项式统一返回 [0.0]
    if all(v == 0.0 for v in rounded):
        rounded = [0.0]
    
    # 转为torch张量返回
    return torch.tensor(rounded, dtype=torch.float32)
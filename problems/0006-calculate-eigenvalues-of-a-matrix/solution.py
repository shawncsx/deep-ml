import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # 求解特征值、特征向量
    eig_vals, _ = torch.linalg.eig(matrix)
    # 提取实数部分（实对称矩阵虚部为0）
    real_eig = eig_vals.real
    # 降序排序，丢弃排序下标
    sorted_vals, _ = torch.sort(real_eig, descending=True)
    return sorted_vals
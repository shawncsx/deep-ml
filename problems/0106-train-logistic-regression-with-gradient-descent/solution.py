import torch

def train_logreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Train logistic regression using gradient descent with BCE loss.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features) as Tensor
        y: Binary labels of shape (n_samples,) as Tensor
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
    
    Returns:
        Tuple of (coefficients, losses) where:
        - coefficients: List of learned weights (bias first, then feature weights), rounded to 4 decimals
        - losses: List of BCE loss values at each iteration, rounded to 4 decimals
    
    Notes:
        - Initialize all coefficients to zero
        - Add bias column as FIRST column of X
        - Use sum-based BCE loss: -sum(y*log(p) + (1-y)*log(1-p))
    """
    losses = []
    n_samples = X.shape[0]
    ones = torch.ones((n_samples, 1), device=X.device, dtype=X.dtype)
    X_aug = torch.cat([ones, X], dim=1)
    
    d_plus1 = X_aug.shape[1]
    theta = torch.zeros((d_plus1, 1), dtype=X.dtype, device=X.device)
    y_col = y.reshape(-1, 1)

    for _ in range(iterations):
        y_pred = torch.sigmoid(X_aug @ theta)
        loss = -torch.sum(y_col * torch.log(y_pred) + (1 - y_col) * torch.log(1 - y_pred))
        losses.append(round(loss.item(), 4))

        grad = X_aug.T @ (y_pred - y_col)
        with torch.no_grad():
            theta -= learning_rate * grad


    coefficients = [round(val.item(), 4) for val in theta.flatten()]
    return (coefficients, losses)
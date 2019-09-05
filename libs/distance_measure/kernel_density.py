import torch
def kernel_density_loss(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)
    n = X.size(1)

    expand_X = X.unsqueeze(2).expand(-1, -1, m).permute(0, 2, 1)
    expand_Y = Y.unsqueeze(2).expand(-1, -1, m).permute(2, 0, 1)
    X_Y = expand_X - expand_Y
    X_Y_norm = torch.zeros((m, m))
    for i in range(n):
        X_Y_norm = X_Y_norm + X_Y[:, :, i]**2
    X_Y_norm_rbf = torch.zeros((m, m))
    for sigma in sigma_list:
        X_Y_norm_rbf = X_Y_norm_rbf + torch.exp(-1 * X_Y_norm / (2 * (sigma ** 2))) / np.sqrt(2*np.pi*(sigma**2))
    print("X_Y", X_Y)
    print("X_Y_norm_rbf", X_Y_norm_rbf)
    loss = -1 * torch.sum(torch.log(torch.sum(X_Y_norm_rbf, dim=0) / m)) / m
    return loss
import torch
import numpy as np


def gumbel_softmax(p, device, sample_num=1, tau=0.2):
    g = np.random.uniform(0.0001, 0.9999, [sample_num, p.shape[0]])
    g = -torch.log(-torch.log(torch.Tensor(g)))
    numerator = torch.exp((torch.log(p).repeat(sample_num, 1) + g.to(device)) / tau)
    denominator = torch.sum(numerator, dim=1)
    denominator_repeat = denominator.repeat(p.shape[0], 1).t()
    return numerator / denominator_repeat


class cos_act(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        D = x.shape[1]
        x = (2.0 / D) ** 0.5 * torch.cos(x)
        return x


def normalize_components(components, fake_components):
    num_components = components.shape[1]
    normalize_factor = torch.zeros([num_components])
    components = torch.Tensor(components.float())
    for i in range(num_components):
        normalize_factor[i] = torch.sqrt(torch.var(fake_components[i, :]) / torch.var(components[i, :]))
    return normalize_factor


def normalize_mixing_matrix(real_A, estimated_A):
    real_A = torch.abs(real_A)
    estimated_A = torch.abs(estimated_A)
    factor_real_A = 1/torch.norm(real_A[:, 0])
    factor_estimated_A = 1/torch.norm(estimated_A[:, 0])
    real_A_normalized = real_A * factor_real_A
    estimated_A_normalized = estimated_A * factor_estimated_A
    return real_A_normalized, estimated_A_normalized

if __name__ == '__main__':
    print(gumbel_softmax(torch.Tensor(np.array([0.1, 0.9])), 100, 0.05))

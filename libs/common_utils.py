import torch
import numpy as np
def gumbel_softmax(p, device, sample_num=1, tau=0.2):
    g = np.random.uniform(0.0001, 0.9999, [sample_num, p.shape[0]])
    g = -torch.log(-torch.log(torch.Tensor(g)))
    numerator = torch.exp((torch.log(p).repeat(sample_num, 1)+g.to(device))/tau)
    denominator = torch.sum(numerator, dim=1)
    denominator_repeat = denominator.repeat(p.shape[0], 1).t()
    return numerator/denominator_repeat


def get_real_data():
    data_folder = './data/pairs/'
    data_file_list = ['pair0049.txt', 'pair0050.txt', 'pair0051.txt']
    # data_file_list = ['pair0050.txt']
    k = 2
    n = 6
    T = 365
    batch_size = T-2
    observed_data = np.zeros([T, n])
    i = 0
    for data_file in data_file_list:
        f = np.loadtxt(data_folder+data_file)
        observed_data[:, 2*i:2*(i+1)] = f
        i += 1

    return [observed_data.T, k, n, T, batch_size]

class cos_act(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        D = x.shape[1]
        x = (2.0/D)**0.5*torch.cos(x)
        return x

if __name__ == '__main__':
    print(gumbel_softmax(torch.Tensor(np.array([0.1, 0.9])), 100, 0.05))

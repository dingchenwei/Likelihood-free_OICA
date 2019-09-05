from torch.optim.sgd import SGD
import torch
from torch.optim.optimizer import required
class PGM(SGD):
    def __init__(self, params, proxs, lr=required, reg_lambda=0, momentum=0, dampening=0,
                 nesterov=False):
        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0, nesterov=nesterov)
        super().__init__(params, **kwargs)
        self.lr = lr
        self.reg_lambda = reg_lambda
        if len(proxs) != len(self.param_groups):
            raise ValueError("Invalid length of argument proxs: {} instead of {}".format(len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, closure=None):
        # this performs a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)

        for group in self.param_groups:
            prox = group['prox']

            # here we apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(p.data, self.lr, self.reg_lambda)

def prox_soft(X, step, thresh=0):
    """Soft thresholding proximal operator
    """
    thresh_ = step_gamma(step, thresh)
    return torch.sign(X)*prox_plus(torch.abs(X) - thresh_)

def prox_plus(X):
    """Projection onto non-negative numbers
    """
    below = X < 0
    X[below] = 0
    return X

def step_gamma(step, gamma):
    """Update gamma parameter for use inside of continuous proximal operator.
    Every proximal operator for a function with a continuous parameter,
    e.g. gamma ||x||_1, needs to update that parameter to account for the
    stepsize of the algorithm.
    Returns:
        gamma * step
    """
    return gamma * step
from torch.optim.optimizer import Optimizer

from .registry import OPTIMIZERS


@OPTIMIZERS.register_module
class NoneOptimizer(Optimizer):
    """A clone of torch.optim.SGD.

    A customized optimizer could be defined like CopyOfSGD.
    You may derive from built-in optimizers in torch.optim,
    or directly implement a new optimizer.
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NoneOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        return

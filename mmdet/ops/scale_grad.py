from torch.autograd import Function


class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


def scale_tensor_gard(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)


def scale_tensor_gard1(t, grad_scale):
    return (1-grad_scale) * t.detach() + grad_scale * t

import torch
from torch.autograd import Function


class Sign(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(Sign, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # TODO (cywu): consider passing 0 for tanh(x) > 1 or tanh(x) < -1? 
        # See https://arxiv.org/pdf/1712.05087.pdf. 
        return grad_output, None
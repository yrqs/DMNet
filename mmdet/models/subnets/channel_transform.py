import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, freeze=False):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Parameter(torch.rand(channels), requires_grad=not freeze)

    def forward(self, x, dim=1):
        assert x.size(dim) == self.channel_attention.size(0)
        unsqueeze_size = [1 for i in range(len(x.size()))]
        unsqueeze_size[dim] = x.size(dim)
        return x * self.channel_attention.reshape(unsqueeze_size).expand_as(x)

class MutexChannelAttention(nn.Module):
    def __init__(self, channels, init_value=1., freeze=False):
        super(MutexChannelAttention, self).__init__()
        self.channel_attention = nn.Parameter(torch.ones(channels) * init_value, requires_grad=not freeze)

    def forward(self, x, dim=1):
        assert x.size(dim) == self.channel_attention.size(0)
        unsqueeze_size = [1 for i in range(len(x.size()))]
        unsqueeze_size[dim] = x.size(dim)
        att = self.channel_attention.reshape(unsqueeze_size).expand_as(x)
        att_mutex = F.relu(torch.ones_like(att, device=att.device) - att)
        return x * att, x * att_mutex

class AffineLayer(nn.Module):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, x):
        out = x * self.weight.expand_as(x)
        if self.bias is not None:
            out = out + self.bias.expand_as(x)
        return out
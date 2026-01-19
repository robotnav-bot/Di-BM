import torch
import torch.nn as nn
import torch.nn.functional as F

# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, num_experts=None,):
        super().__init__()

        if num_experts is not None:
            self.block = nn.ModuleList([nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            ) for i in range(num_experts)])

            self.num_experts=num_experts           

        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )
            self.num_experts=None


    def forward(self, x, use_expert_i=None):

        if self.num_experts is not None:
            results = torch.zeros_like(x)
            assert use_expert_i is not None
            weights, selected_experts = torch.topk(F.one_hot(use_expert_i,num_classes=self.num_experts), 1, dim=-1) #(B, 1)


            for i, expert in enumerate(self.block):
                batch_idx, nth_expert = torch.where(selected_experts == i)

                if batch_idx.shape[0]==0:
                    continue
                
                w=weights[batch_idx, nth_expert, None, None]
                e=expert(x[batch_idx])
                results[batch_idx] += w * e

            return results

        else:
            return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)

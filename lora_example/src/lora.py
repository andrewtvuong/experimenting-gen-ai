import torch
import torch.nn as nn
import torch.nn.functional as f

class LoRA(nn.Module):
    def __init__(self, adapted_module, rank, input_size, output_size):
        super(LoRA, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(input_size, rank))
        self.B = nn.Parameter(torch.randn(rank, output_size))
        self.adapted_module = adapted_module
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        delta_weight = self.A @ self.B
        original_output = self.adapted_module(x)
        adapted_output = f.linear(x, delta_weight)
        return original_output + adapted_output

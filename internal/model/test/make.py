import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Parameter(
            torch.rand(2, 3, dtype=torch.bfloat16),
            # torch.arange(6, dtype=torch.long).reshape(2, 3),
            requires_grad=False)
        self.linear[0, 0] = torch.inf
        print(self.linear)

    def forward(self, x):
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear()

    def forward(self, x):
        return self.linear(x)

model = Model()

torch.save(model.state_dict(), 'linear.pt')
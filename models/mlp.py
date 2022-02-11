from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, blocks=2):
        super(MLP, self).__init__()
        self.blocks = blocks
        self.mlp = nn.Sequential(nn.LayerNorm(input_dims),
                                 nn.Linear(input_dims, 4096),
                                 nn.BatchNorm1d(4096),
                                 nn.GELU(),
                                 nn.Linear(4096, 2048),
                                 nn.BatchNorm1d(2048),
                                 nn.GELU(),
                                 nn.Linear(2048, 2048),
                                 nn.BatchNorm1d(2048),
                                 nn.GELU(),
                                 nn.Linear(2048, 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.GELU(),
                                 nn.Linear(1024, input_dims),
                                 nn.BatchNorm1d(input_dims),
                                 nn.GELU())
        self.fc = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        for _ in range(self.blocks):
            x = self.mlp(x) + x
        x = self.fc(x)
        return x

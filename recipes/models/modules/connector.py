from torch import nn

def get_connector(name, input_dim, output_dim, k):
    if name == 'cnn':
        return CNNConnector(input_dim, output_dim, k)
    elif name == 'linear':
        return LinearConnector(input_dim, output_dim, k, pool_layer=False)
    elif name == 'linear-pool':
        return LinearConnector(input_dim, output_dim, k, pool_layer=True)
    elif name == 'linear-pool-mlp':
        return LinearMLPConnector(input_dim, output_dim, k)
    else:
        raise NotImplementedError

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5, stride=k, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k, pool_layer=False):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        if pool_layer:
            self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.pool_layer = pool_layer

    def forward(self, x):
        x = self.layer(x)
        if self.pool_layer:
            x = x.transpose(1, 2)

            x = self.pool(x)
            x = x.transpose(1, 2)

        return x

class LinearMLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super(LinearMLPConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.linear2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)

        x = self.pool(x)
        x = x.transpose(1, 2)

        x = self.linear2(x)

        return x

from torch import nn
from torch.nn.init import kaiming_normal_


class Linear(nn.Linear):
    "Linear Layer with kaiming normal normalization"
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 norm_gain: float = 0.,
                 mode: str = 'fan_in'):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias)
        kaiming_normal_(self.weight.data, mode=mode, a=norm_gain)
        if bias:
            self.bias.data.zero_()


class SimpleDeepNet(nn.Module):
    def __init__(self, in_features: int = 310,
                 out_features: int = 1,
                 neurons: int = 25,
                 n_layers: int = 1,
                 use_batch_norm: bool = False,
                 dropout: float = 0.):
        super().__init__()
        self.loss_func = nn.MSELoss()
        layers = [Linear(in_features, neurons)]
        for n in range(n_layers):
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons))
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))
            if n < n_layers - 1:
                layers.append(Linear(neurons, neurons))
        layers.append(Linear(neurons, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, features, target=None):
        prediction = self.model(features)
        output = {'prediction': prediction}
        if target is not None:
            loss = self.loss_func(prediction, target)
            output['loss'] = loss
        return output

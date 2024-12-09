import torch.nn as nn

class ClassifierNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, dropout_p):
        super(ClassifierNet, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activations[activation])
            layers.append(nn.Dropout(p=dropout_p))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class CliqueDataset(torch.utils.data.Dataset):
    def __init__(self, cliques, labels):
        self.cliques = cliques
        self.labels = labels

    def __len__(self):
        return len(self.cliques)

    def __getitem__(self, idx):
        return self.cliques[idx], self.labels[idx]


def train_classifier_net(train_dataset, device, batch_size, num_epochs, learning_rate, hidden_dims, activation, dropout_p, criterion):
    from utils.model import ClassifierNet
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    classifier_net = ClassifierNet(
        input_dim=train_dataset[0][0].shape[0],
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_p=dropout_p
    ).to(device)
    optimizer = optim.Adam(classifier_net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        classifier_net.train()
        for cliques, labels in train_loader:
            cliques, labels = cliques.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier_net(cliques)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return classifier_net

import torch.nn as nn


class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        assert hidden_dim > 256, "hiddem dim not valid"
        super(MLPRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)

        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.tanh(x)
        return x

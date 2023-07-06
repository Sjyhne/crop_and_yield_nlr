import torch
from torch import nn
from torch.nn import functional as F

class SingleImageCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(SingleImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 12 * 12, output_dim)  # Adjust the input dimension here based on your image size and CNN architecture

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1, 3)
        x = F.relu(self.fc(x))
        return x


class YieldModel(nn.Module):
    def __init__(self, timesteps, cnn_input_dim, feature_dim):
        super(YieldModel, self).__init__()
        self.timesteps = timesteps
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=7, stride=7)
        self.scnn = SingleImageCNN(cnn_input_dim, 64)
        self.td_scnn = nn.ModuleList([self.scnn for _ in range(timesteps)])
        self.gru = nn.GRU(input_size=133, hidden_size=128, batch_first=True)  # Adjust the input size here
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, cnn_input, weather_input, feature_input):
        weather_input = F.relu(self.conv1d(weather_input)).permute(0, 2, 1)

        feature_input = feature_input.unsqueeze(1).repeat(1, self.timesteps, 1)

        cnn_output = []
        for i in range(self.timesteps):
            cnn_output.append(self.td_scnn[i](cnn_input[:, i]))

        cnn_output = torch.stack(cnn_output, dim=1)

        x = torch.cat([cnn_output, feature_input, weather_input], dim=2)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Define a simple CNN for MNIST
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # output shape: (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # output shape: (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)            # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Quick test for the network
if __name__ == '__main__':
    model = MNISTNet()
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    print("Output shape:", output.shape)

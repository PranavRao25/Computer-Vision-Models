import torch.nn as nn
import torch.nn.functional as F


class UnnormalisedLeNet(nn.Module):
    """
      LeNet Architecture
    """

    # Input: 3 * 32 * 32
    # Conv1: Cout=6 Hout=32 Wout=32 filter: K=5 Padding = 2 Stride = 1 : 6 * 32 * 32
    # ReLU: 6 * 32 * 32
    # MaxPool: K=2 stride=2 : 6 * 16 * 16
    # Conv2: Cout=16 K=5 padding=2 stride=1 : 16 * 12 * 12
    # ReLU: 50 * 14 * 14 : 16 * 12 * 12
    # MaxPool: K=2 stride=2 : 16 * 6 * 6
    # Flatten
    # FC: 120 -> 84
    # ReLU: 84
    # FC: 84 -> 10

    def __init__(self):
        super(UnnormalisedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(6 * 6 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.final = nn.Softmax()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.final(x)
        return x


class NormalisedLeNet(nn.Module):
  """
    Normalised LeNet Architecture
  """

  # Input: 3 * 32 * 32
  # Conv1: N=batch_size Cin=1 H=28 W=28 Cout=20 Hout=28 Wout=28 filter: Cout * Cin * K * K K=5 Padding = 2 Stride = 1
  # Batch normalise
  # ReLU: 6 * 32 * 32
  # MaxPool: K=2 stride=2
  # Conv2: Cout=50 K=5 padding=2 stride=1
  # Batch normalise
  # ReLU: 16 * 12 * 12
  # MaxPool: K=2 stride=2
  # Flatten
  # FC: 120 -> 84
  # ReLU: 84
  # FC: 84 -> 10

  def __init__(self):
    super(NormalisedLeNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
    self.norm1 = nn.BatchNorm2d(6)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.norm2 = nn.BatchNorm2d(16)
    self.fc1 = nn.Linear(6 * 6 * 16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.final = nn.Softmax()

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), (2, 2), stride=2)
    x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), (2, 2), stride=2)
    x = x.view(-1, 6 * 6 * 16)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = self.final(x)
    return x



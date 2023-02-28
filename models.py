import pdb
import torch as tc
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.name = "CNN"

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.full1 = nn.Linear(in_features=64 * 16 * 16, out_features=128)
        self.full2 = nn.Linear(in_features=128, out_features=64)
        self.full3 = nn.Linear(in_features=64, out_features=4)

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def get_num_params(self):
        num = sum([p.numel() for p in self.parameters()])
        return num

    def forward(self, inp):

        assert len(inp.shape) == 4 and inp.shape[1] == 1, "Wrong input dimensions!"

        # apply convolutions and pooling
        x = self.pool(self.relu(self.conv1(inp)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout1(x)

        # apply fully connected layers
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.full1(x))
        x = self.dropout2(x)
        x = self.relu(self.full2(x))
        x = self.full3(x)

        return x

import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

train_dataset = MNIST(root="data",
                      train=True,
                      download=True,
                      transform=transforms.ToTensor())

test_dataset = MNIST(root="data",
                     train=False,
                     download=True,
                     transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             shuffle=False)


class StackedCNN(nn.Module):

    def __init__(
            self,
            input_shape: int,
            output_shape: int,
            hidden_units: int
    ):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),

            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.linear_stack(x)
        return x

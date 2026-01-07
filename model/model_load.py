import torch
import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, input):
    super().__init__()

    self.feature = nn.Sequential(
        nn.Conv2d(input,32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32,64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )
    self.fatten = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64*5*5, 128),
        nn.ReLU(),
        nn.Linear(128, 10),

    )

  def forward(self, x):
      x = self.feature(x)
      x = self.fatten(x)
      return x
  
  

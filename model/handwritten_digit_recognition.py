

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)



train_loader = DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, pin_memory=True)

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

learing_rate = 0.01
epochs = 50

model = MyModel(1)
model.to(dev)


loss_fun = nn.CrossEntropyLoss()


optimzer = torch.optim.Adam(model.parameters(), lr = learing_rate)

for epoch in range(epochs):

  total_epoch_loss = 0

  for batch_features , batch_labels in train_loader:


    batch_features, batch_labels = batch_features.to(dev), batch_labels.to(dev)

    y_pred = model(batch_features)

    loss = loss_fun(y_pred, batch_labels)

    optimzer.zero_grad()
    loss.backward()

    optimzer.step()

    total_epoch_loss = total_epoch_loss + loss.item()


  avg_loss = total_epoch_loss/len(train_loader)
  print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')

model.eval()



# evaluation on test data
total = 0
correct = 0

with torch.no_grad():

  for batch_features, batch_labels in test_loader:

    # move data to gpu
    batch_features, batch_labels = batch_features.to(dev), batch_labels.to(dev)

    outputs = model(batch_features)

    _, predicted = torch.max(outputs, 1)

    total = total + batch_labels.shape[0]

    correct = correct + (predicted == batch_labels).sum().item()

print(correct/total)

# evaluation on training data
total = 0
correct = 0

with torch.no_grad():

  for batch_features, batch_labels in train_loader:

    # move data to gpu
    batch_features, batch_labels = batch_features.to(dev), batch_labels.to(dev)

    outputs = model(batch_features)

    _, predicted = torch.max(outputs, 1)

    total = total + batch_labels.shape[0]

    correct = correct + (predicted == batch_labels).sum().item()

print(correct/total)

Model_path = 'model.pth'
torch.save(model.state_dict(), Model_path)



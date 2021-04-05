# -*- coding: UTF-8 -*-

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import random
from collections import namedtuple

import tednet as tdt
import tednet.tnn.tensor_ring as tr

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
seed = 233
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
Input_Size = np.prod([28, 28])
Hidden_Size = 256

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/hdd/panyu/project_jupyter/tednet/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/hdd/panyu/project_jupyter/tednet/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=256, shuffle=True, **kwargs)


class ClassifierTR(nn.Module):
    def __init__(self, num_class=10):
        super(ClassifierTR, self).__init__()
        in_shape = [28, 28]
        hidden_shape = [16, 16]

        self.hidden_size = Hidden_Size

        self.lstm = tr.TRLSTM(in_shape, hidden_shape, [5, 5, 5, 5])
        self.fc = nn.Linear(self.hidden_size, num_class)

    def forward(self, x, state):
        input_shape = x.shape
        batch_size = input_shape[0]
        seq_size = input_shape[1]
        x = x.view(batch_size, seq_size, -1)
        x = x.permute(1, 0, 2)
        _, x = self.lstm(x, state)
        x = self.fc(x[0])
        return x


def train(model, device, train_loader, optimizer, epoch, log_interval=200):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        batch_size = data.shape[0]
        state = LSTMState(torch.zeros(batch_size, Hidden_Size, device=device),
                          torch.zeros(batch_size, Hidden_Size, device=device))
        output = model(data, state)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            batch_size = data.shape[0]
            state = LSTMState(torch.zeros(batch_size, Hidden_Size, device=device),
                              torch.zeros(batch_size, Hidden_Size, device=device))
            output = model(data, state)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Define a TR-LSTM
model = ClassifierTR()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.00016667)

for epoch in range(20):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

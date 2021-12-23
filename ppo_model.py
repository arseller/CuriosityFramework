import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir='./models', game=None, device=torch.device('cpu')):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_{}.pt'.format(game))

        self.conv1 = nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        conv_shape = self.calc_conv_shape(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.pi = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state, h_a):

        x1 = F.relu(self.conv1(state))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        x_flat = x4.view((x4.size()[0], -1))
        rec = self.gru(x_flat, h_a.view((-1, 256)))

        pi = self.pi(rec)

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)

        return dist, h_a

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def calc_conv_shape(self, input_dim):
        tmp = torch.zeros(1, *input_dim)
        dim = self.conv1(tmp)
        dim = self.conv2(dim)
        dim = self.conv3(dim)
        dim = self.conv4(dim)
        return int(np.prod(dim.size()))

    def init_hidden(self, h_dim=256):
        h = torch.zeros((1, 1, h_dim), dtype=torch.float32).to(self.device)
        return h


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir='./models', game=None, device=torch.device('cpu')):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_{}.pt'.format(game))

        self.conv1 = nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        conv_shape = self.calc_conv_shape(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state, h_c):
        x1 = F.relu(self.conv1(state))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        x_flat = x4.view((x4.size()[0], -1))
        rec = self.gru(x_flat, h_c.view((-1, 256)))

        value = self.v(rec)

        return value, h_c

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def calc_conv_shape(self, input_dim):
        tmp = torch.zeros(1, *input_dim)
        dim = self.conv1(tmp)
        dim = self.conv2(dim)
        dim = self.conv3(dim)
        dim = self.conv4(dim)
        return int(np.prod(dim.size()))

    def init_hidden(self, h_dim=256):
        h = torch.zeros((1, 1, h_dim), dtype=torch.float32).to(self.device)
        return h

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


from FIO import Fourier_Int_Op

class FNO(nn.Module):
  def __init__(self, n_modes, channels, activation = F.relu):
    super().__init__()
    self.activation = activation

    # Projects up to channel space
    self.proj_up = nn.Linear(2, channels)

    # Projects down to target space
    self.proj_down = nn.Linear(channels, 1)

    # Four Fourier Layers, as described by the paper in Figure 2
    self.fourier_conv1 = Fourier_Int_Op(channels, channels, n_modes)
    self.fourier_conv2 = Fourier_Int_Op(channels, channels, n_modes)
    self.fourier_conv3 = Fourier_Int_Op(channels, channels, n_modes)
    self.fourier_conv4 = Fourier_Int_Op(channels, channels, n_modes)

    # Note that Conv1d with kernel size 1 is the same as a linear layer (the W 
    # linear transform in Figure 2 in the paper), we just need to permute last 
    # two dimensions. This can yield better results in practice than a dense
    # nn.Linear layer 
    self.conv1 = nn.Conv1d(channels, channels, 1)
    self.conv2 = nn.Conv1d(channels, channels, 1)
    self.conv3 = nn.Conv1d(channels, channels, 1)
    self.conv4 = nn.Conv1d(channels, channels, 1)
        
  def forward(self, x):
    batch_size, size_x = x.shape[0:2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1)
    gridx = gridx.repeat([batch_size, 1, 1])

    x = torch.cat((x, gridx), dim=-1)

    # Project up to channel space
    x = self.proj_up(x)
    x = x.permute(0, 2, 1)

    # FNO Layer 1
    x_down = self.conv1(x)
    x_fourier = self.fourier_conv1(x)
    x = x_down + x_fourier
    x = self.activation(x)

    # FNO Layer 2
    x_down = self.conv2(x)
    x_fourier = self.fourier_conv2(x)
    x = x_down + x_fourier
    x = self.activation(x)

    # FNO Layer 3
    x_down = self.conv3(x)
    x_fourier = self.fourier_conv3(x)
    x = x_down + x_fourier
    x = self.activation(x)

    # FNO Layer 4
    x_down = self.conv4(x)
    x_fourier = self.fourier_conv4(x)
    x = x_down + x_fourier
    x = self.activation(x)

    # Project down to real space
    x = x.permute(0, 2, 1)
    x = self.proj_down(x)

    return x    


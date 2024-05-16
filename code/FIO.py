import torch
from torch import nn
    
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

class Fourier_Int_Op(nn.Module):
  def __init__(self, in_channels, out_channels, n_modes):
    """
    Inputs:
    in_channels = number of channels in the input to the convolution
    out_channels = number of channels in the output of the convolution
    n_modes = number of Fourier modes used
    """
    super().__init__()
    print("initializing Fourier Integral Operator")
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_modes = n_modes
    self.weights = nn.Parameter(torch.rand(in_channels, out_channels, self.n_modes, dtype=torch.cfloat)/(in_channels*out_channels))
  
  def forward(self, x):    
    batch_size, _, length = x.shape

    # Send up to Fourier Space
    trans_x = torch.fft.rfft(x)

    # Linearly transform the lowest n_modes modes
    out = torch.zeros([batch_size, self.out_channels, x.size(-1)//2 + 1],dtype=torch.cfloat)     
    out[:,:,:(self.n_modes)] = torch.einsum("xij,ykj->xkj", trans_x[:,:,:(self.n_modes)],  self.weights)

    # Bring back to normal space
    inv_trans_x = torch.fft.irfft(out, n=length)
    return inv_trans_x
  

     

  
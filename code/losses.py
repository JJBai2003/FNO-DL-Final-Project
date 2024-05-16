import torch
import torch.nn as nn
import torch.nn.functional as F

# implementing relative L^2 loss, which is ||u_pred - truth||_{L^2} / ||truth||_{L^2}
def relative_l2_loss(x, y):
    err = torch.norm(torch.flatten(x) - torch.flatten(y), dim=-1)
    norm = torch.norm(torch.flatten(y), dim=-1)
    rel_err = err/norm
    return rel_err

# implementing relative H^1 loss, which is ||u_pred - truth||_{H^1} / ||truth||_{H^1}
# H^1 is the Sobolev Space W^{1,2}, the space of functions in L^2 with weak derivative
# also in L^2. The H^1 norm is ||f||_{H^1} = (||f||_{L^2} + ||dfdx||_{L^2})^{1/2}
def relative_h1_loss(x,y):
    dx = 1/x.size(-1) 

    out_x = {}
    out_y = {}
    out_x[0] = x
    out_y[0] = y
    out_x[1] = first_deriv(x, dx)
    out_y[1] = first_deriv(y, dx)

    sq_err = torch.norm(out_x[0] - out_y[0], dim=-1)**2
    sq_err += torch.norm(out_x[1] - out_y[1], dim=-1)**2

    sq_norm = torch.norm(out_y[0], dim = -1)**2 
    sq_norm += torch.norm(out_y[1], dim=-1)**2

    rel_err = torch.sum((sq_err/sq_norm)**0.5)

    return rel_err

# Note: The method below approximates a derivative using the finite element method
# First derivatives are calculated by f(x+h)-f(x-h)/2h for finite, small h

def first_deriv(x, dx):

    # Calculating spatial derivatives, note that boundaries have to be
    # handled seperately
    dudx = torch.zeros(x.shape)
    dudx[:,0] = (x[:,1] - x[:,0])/dx
    dudx[:,-1] = (x[:,-1] - x[:,-2])/dx

    for i in range(x.shape[1]-2):
        dudx[:,(i+1)] = (x[:,(i+2)] - x[:,i])/(2*dx)

    return dudx

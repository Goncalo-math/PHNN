import numpy as np
import torch
import math
import scipy, scipy.misc, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from scipy.optimize import fsolve
from torch.autograd import grad


def integrate_model(model, t_span, y0, fun=None, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  fun = default_fun if fun is None else fun
  return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def rk4(fun, y0, t, dt, *args, **kwargs):
  dt2 = dt / 2.0
  k1 = fun(y0, t, *args, **kwargs)
  k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
  k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
  k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy


def L2_loss(u, v):
  return (u-v).pow(2).mean()


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl

def alpha(x,y):
    return [x[0]*math.exp(0.5*y[1]*x[1]), x[1]*math.exp(-0.5*y[0]*x[0])]

def beta(x,y):
    return x[0]*torch.exp(-0.5*y[1]*x[1]), x[1]*torch.exp(0.5*y[0]*x[0])

def get_model(integrator):
  if(integrator == 'Euler'):
    model = torch.load("model_Euler_noise.pt")
  if(integrator == 'PHI'):
    model = torch.load("model_PHI_noise.pt")
  return model


def funct(y,x,h,model):
  torch_y = torch.tensor(y, requires_grad=True, dtype=torch.float32)
  hamilt = model.forward_fn(torch_y)
  grad_Hy = grad(hamilt.sum(), torch_y, create_graph=True)[0].detach().numpy()
  return alpha(y,h*grad_Hy)-x

def integrate(model, y0, num_steps, h, integrator):
  result = torch.empty((num_steps, 2), requires_grad=False)
  result[0,:] = torch.tensor(y0)

  if(integrator == 'Euler'):
    for i in range(1,num_steps):

      x_hat = result[i-1,:]
      x_hat.requires_grad_(True)
      # q,p = torch.tensor(result0[i-1,0], requires_grad=True, dtype=torch.float32), torch.tensor(result0[i-1,1], requires_grad=True, dtype=torch.float32)

      hamilt = model.forward_fn(x_hat)
      dHdx = grad(hamilt.sum(), x_hat, create_graph=True)[0]

      result[i, 1] = result[i-1,1] - x_hat[0]*x_hat[1]*dHdx[0]*h
      result[i, 0] = result[i-1,0] + x_hat[0]*x_hat[1]*dHdx[1]*h

  if(integrator == 'PHI'):
    for i in range(1,num_steps):
      
      x_hat = result[i-1,:]

      root = fsolve(funct, x_hat.detach().numpy(), args=(x_hat.detach().numpy(), h, model))

      y = torch.tensor(root, requires_grad=True, dtype=torch.float32)

      hamilt = model.forward_fn(y) 
      dhdy = grad(hamilt.sum(), y, create_graph=True)[0]

      q_next, p_next = beta(y,h*dhdy)

      result[i, 1:] = p_next
      result[i, :1] = q_next

  return result





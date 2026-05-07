import numpy as np
import torch
import autograd
import autograd.numpy as anp
import scipy, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from scipy.optimize import fsolve, least_squares
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


def get_model(integrator):
  if(integrator == 'Euler'):
    model = torch.load("model_Euler_teste.pt")
  if(integrator == 'PHI'):
    model = torch.load("model_PHI.pt")
  return model


def jIso(x):
  M = np.matrix([[0, -x[2], x[1]], [x[2], 0, x[0]], [-x[1], -x[0], 0]])
  return M

def jInv(M):
  return [M[1,2],M[0,2],M[1,0]]


def alpha(x,y):
    I = np.identity(3)
    A = jIso(x)
    X = jIso(y)
    return (I+A/4)@X@(I-A/4)


def beta(x,y):
    I = np.eye(3)
    A = jIso(x)
    X = jIso(y)
    return (I-A/4)@X@(I+A/4)


def funct(y,x,h,model):
  torch_y = torch.tensor(y, requires_grad=True, dtype=torch.float32)
  hamilt = model.forward_fn(torch_y)
  grad_Hy = grad(hamilt.sum(), torch_y, create_graph=True)[0].detach().numpy()
  final = x - jInv(alpha(y,h*grad_Hy))
  return final

def integrate(model, y0, num_steps, h, integrator):
  result = torch.zeros((num_steps, 3), requires_grad=False)
  result[0,:] = torch.tensor(y0)

  if(integrator == 'Euler'):
    for i in range(1,num_steps):

      x_hat = result[i-1,:]
      x_hat.requires_grad_(True)


      hamilt = model.forward_fn(x_hat)

      dHdx = grad(hamilt.sum(), x_hat, create_graph=True)[0]

      result[i, 0] = result[i-1,0] + h*(x_hat[1]*dHdx[2] - x_hat[2]*dHdx[1])
      result[i, 1] = result[i-1,1] + h*(x_hat[2]*dHdx[0] - x_hat[0]*dHdx[2])
      result[i, 2] = result[i-1,2] + h*(x_hat[0]*dHdx[1] - x_hat[1]*dHdx[0])

  if(integrator == 'PHI'):
    for i in range(1,num_steps):
      
      x_hat = result[i-1,:].detach().numpy()

      root2 = least_squares(funct, x_hat, args=(x_hat,h,model), ftol=2.23e-16)
      root = root2.x

      y = torch.tensor(root, requires_grad=True, dtype=torch.float32)

      hamilt = model.forward_fn(y) 
      dhdy = grad(hamilt.sum(), y, create_graph=True)[0].detach().numpy()

      r_next = torch.tensor(jInv(beta(root,h*dhdy)), dtype=torch.float32)
      result[i, :] = r_next


  return result





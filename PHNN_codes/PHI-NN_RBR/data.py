import autograd
import autograd.numpy as anp
import numpy as np
import scipy.integrate, math
from scipy.integrate import odeint
solve_ivp = scipy.integrate.solve_ivp
from scipy.optimize import fsolve
from utilis import jInv, beta, alpha, jIso


def hamiltonian_fn(coords):
    q, p, z = anp.split(coords,3)
    I1, I2, I3 = 1,math.pi,100
    H = 0.5*((I2+I3)*q**2 + (I1+I3)*p**2 + (I1+I2)*z**2)
    return H

def poisson_fn(coords):
    q, p, z = anp.split(coords,3)
    J = [[0, -z[0], p[0]],[z[0], 0, -q[0]],[-p[0], q[0], 0]]
    return J

def dynamics_fn(coords,t):
    J = poisson_fn(coords)
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    S = J@dcoords
    return S

def funct_np(y,x,h):
  gradH = autograd.grad(hamiltonian_fn)(y)
  return jInv(alpha(jIso(y),jIso(h*gradH))-jIso(x))

def get_trajectory(t_span=[0,1/14], timescale=100, radius=None, y0=None, integrator = 'Euler', noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(3)*2.-1

    if integrator == 'Euler':
        spring_ivp = odeint(dynamics_fn, y0, t_eval)
        
    if integrator == 'PHI':
        spring_ivp = np.empty((timescale,3))
        spring_ivp[0,:] = y0
        h = 1/(14*99)
        for i in range(1,timescale):
            x_hat = spring_ivp[i-1,:]

            root = fsolve(funct_np, x_hat, args=(x_hat, h))

            gradH = autograd.grad(hamiltonian_fn)(root)

            spring_ivp[i, :] = jInv(beta(root,h*gradH))


    q, p, z = spring_ivp[:,0], spring_ivp[:,1], spring_ivp[:,2]
    dydt = np.zeros(spring_ivp.shape)
    for i in range(spring_ivp.shape[0]):
        dydt[i,:] = dynamics_fn(spring_ivp[i][:],None)
    dydt = np.stack(dydt).T
    dqdt, dpdt, dzdt = np.split(dydt,3)

    # add noise
    # q += np.random.randn(*q.shape)*noise_std
    # p += np.random.randn(*p.shape)*noise_std
    # z += np.random.randn(*z.shape)*noise_std
    return q, p,z, dqdt, dpdt, dzdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, integrator = 'Euler',**kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, z, dx, dy, dz, t = get_trajectory(integrator = integrator,**kwargs)
        xs.append( np.stack( [x, y, z]).T )
        dxs.append( np.stack( [dx, dy, dz]).T )
    
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
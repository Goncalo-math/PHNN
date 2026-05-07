import autograd
import autograd.numpy as anp
import numpy as np
import scipy.integrate
from scipy.integrate import odeint
solve_ivp = scipy.integrate.solve_ivp


def hamiltonian_fn(coords):
    q, p = anp.split(coords,2)
    H = q - anp.log(q) + p - 2*anp.log(p)
    return H

def poisson_fn(coords):
    q, p = anp.split(coords,2)
    J = np.kron([[0,1],[-1,0]], [q*p])
    return J

def dynamics_fn(coords,t):
    J = poisson_fn(coords)
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    S = J@dcoords
    return S

def get_trajectory(t_span=[0,6], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1 + [1,2]

    spring_ivp = odeint(dynamics_fn, y0, t_eval)
    q, p = spring_ivp[:,0], spring_ivp[:,1]
    dydt = np.zeros(spring_ivp.shape)
    for i in range(spring_ivp.shape[0]):
        dydt[i,:] = dynamics_fn(spring_ivp[i][:],None)
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
import autograd
import autograd.numpy as anp
import numpy as np
import scipy.integrate
from scipy.integrate import odeint
solve_ivp = scipy.integrate.solve_ivp



def hamiltonian_fn(coords):
    q, p = anp.split(coords,2)
    H = (1-anp.cos(q)) + p**2 # pendulum hamiltonian
    return H

def dynamics_fn(coords,t):
    n = 1
    J = np.kron([[0,1],[-1,0]], np.identity(n))
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    S = J@dcoords
    return S

def get_trajectory(t_span=[0,6], h=1/15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/h))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*4.-2
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    # spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
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

def get_dataset(seed=0, samples=50, test_split=0.5, h = 1/15, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(h = h, **kwargs)
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
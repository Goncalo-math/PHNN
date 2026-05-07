import numpy as np
import torch, argparse, math
import matplotlib.pyplot as plt
from HNN import MLP, HNN
from scipy.optimize import fsolve
from data import get_dataset, dynamics_fn
from utilis import L2_loss, get_model, alpha, beta, funct, integrate
from scipy.integrate import odeint
from torch.autograd import grad


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='circ', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--delta_t', default=1/15, type=float, help='time step')
    parser.add_argument('--integrator', default='PHI', type=str, help='integrator used: Euler or PHI')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data

  data = get_dataset(seed=args.seed, h = args.delta_t)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])



  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    
    # train step

    trajectories = torch.empty((x.shape[0], 2), requires_grad=False)

    if(args.integrator == 'Euler'):
      for i in range(x.shape[0]):

        if(i%90) == 0:
          q = torch.tensor([x.data[i,0]], requires_grad=True, dtype=torch.float32)
          p = torch.tensor([x.data[i,1]], requires_grad=True, dtype=torch.float32)

        else:
          q = torch.tensor([x.data[i,0]], requires_grad=True, dtype=torch.float32)
          p = torch.tensor([x.data[i,1]], requires_grad=True, dtype=torch.float32)
          x_hat = torch.cat((q,p))

          hamilt = model.forward_fn(x_hat)
          dpdt = -q*p*grad(hamilt.sum(), q, create_graph=True)[0]
          dqdt = q*p*grad(hamilt.sum(), p, create_graph=True)[0]

          p_next = x[i-1,1] + dpdt*1/15
          q_next = x[i-1,0] + dqdt*1/15

          p = p_next
          q = q_next

        trajectories[i, 1:] = p
        trajectories[i, :1] = q

    if(args.integrator == 'PHI'):
      for i in range(x.shape[0]):

        if(i%90) == 0:
          q = torch.tensor([x.data[i,0]], requires_grad=True, dtype=torch.float32)
          p = torch.tensor([x.data[i,1]], requires_grad=True, dtype=torch.float32)

        else:

          x_hat = x[i-1,:]

          root = fsolve(funct, x_hat.detach().numpy(), args=(x_hat.detach().numpy(), 1/15, model))

          y = torch.tensor(root, requires_grad=True, dtype=torch.float32)

          hamilt = model.forward_fn(y) 
          dhdy = grad(hamilt.sum(), y, create_graph=True)[0]

          q_next, p_next = beta(y,1/15*dhdy)

          p = p_next
          q = q_next


        trajectories[i, 1:] = p
        trajectories[i, :1] = q


    # mse = torch.nn.MSELoss()
    loss = L2_loss(x, trajectories)

    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # run test data
    test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)

    test_loss = L2_loss(test_x, test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats


if __name__ == "__main__":
    # Initial data
  num_steps = 1000

  t_span = [0,6*math.pi]
  y0 = np.asarray([1, 1])
  y1 = np.asarray([0.8, 0.8])

  h = t_span[1]/num_steps

  #Integrate problem
  real0 = odeint(dynamics_fn, y0, np.linspace(t_span[0], t_span[1], num_steps))
  real1 = odeint(dynamics_fn, y1, np.linspace(t_span[0], t_span[1], num_steps))

  fig1, ax = plt.subplots()
  # ax.set_title('Ideal pendulum')
  ax.plot(real0[:,0], real0[:,1],'-k')
  ax.plot(real1[:,0], real1[:,1],'-k')

  ## Poisson Neural Network
  args = get_args()
  model, stats = train(args)

## Integrate the model
 
  result0 = integrate(model, y0,num_steps, h, args.integrator)

  ax.plot(result0[:,0].detach().numpy(), result0[:,1].detach().numpy(),'--g')

  result1 = integrate(model, y1,num_steps, h, args.integrator)

  ax.plot(result1[:,0].detach().numpy(), result1[:,1].detach().numpy(),'--g')

  ax.set_xlabel('$x^1$')
  ax.set_ylabel('$x^2$')
  ax.legend(['Real','_Hiden', 'PHI-PHI'], loc = 'upper right', fontsize = 15)
  plt.show()




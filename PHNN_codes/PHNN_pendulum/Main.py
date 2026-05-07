import numpy as np
import torch, argparse, math
import matplotlib
import matplotlib.pyplot as plt
markers_array = matplotlib.markers.MarkerStyle.markers.keys()
from HNN import MLP, HNN
from data import get_dataset, dynamics_fn
from utilis import L2_loss, integrate_model
from scipy.integrate import odeint


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
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

  data = get_dataset(seed=args.seed)
  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])
  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
    # train step
    dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
    # print(dxdt.shape)
    # print(dxdt_hat.shape)
    loss = L2_loss(dxdt, dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # run test data
    test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)

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
  h = 2*math.pi/num_steps
  t_span = [0,20]
  y0 = np.asarray([1, 1])
  y1 = np.asarray([0.5, 0.5])
  #Integrate problem
  real0 = odeint(dynamics_fn, y0, np.linspace(t_span[0], t_span[1], num_steps))
  real1 = odeint(dynamics_fn, y1, np.linspace(t_span[0], t_span[1], num_steps))

  fig1, ax = plt.subplots()
  #ax.set_title('Ideal pendulum with HNN and real solution')
  ax.plot(real0[:,0], real0[:,1],'-k')
  ax.plot(real1[:,0], real1[:,1],'-k')

  ## Poisson Neural Network
  args = get_args()
  model, stats = train(args)

  # Integrate the model
  kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], num_steps), 'rtol': 1e-12}
  pnn_ivp = integrate_model(model, t_span, y0, **kwargs)

  result0 = pnn_ivp['y'].T

  ax.plot(result0[:,0], result0[:,1],'--r',)

  pnn_ivp = integrate_model(model, t_span, y1, **kwargs)

  result1 = pnn_ivp['y'].T

  ax.plot(result1[:,0], result1[:,1],'--r')

  ax.set_xlabel('q')
  ax.set_ylabel('p')
  ax.legend(['Real','_Hide' ,'HNN'],fontsize=15, loc = 'upper right')
  plt.show()




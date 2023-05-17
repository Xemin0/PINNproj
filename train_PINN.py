# Essential Modules
import jax
import jax.numpy as jnp

###### List Current Device
print('-'*20)
print('Jax Default Backend:', jax.default_backend())
print('-'*20)

import numpy as np

# Cross Ref
from model.MLP import init_network_params, normalized_predict
from utils.save_load import save_params, load_params, validate_params
from data.load_data import load_data
from model.PINN import plot_trajectory_PINN, plot_losses, trainPINN, total_PINN_loss

# Miscellaneous
import argparse
import re


# Argument Parsing
parser = argparse.ArgumentParser(description = 'PINN for 2D Lorentz Force in JAX')
parser.add_argument('--datapath', type = str, default = './Datasets', help = 'Base path where to load the train and test data')
parser.add_argument('--lr', type = float, default = 0.002, help = 'Learning Rate for Adam')
parser.add_argument('--adam', type = int, default = 10, help = 'Number of Adam optimization steps')
parser.add_argument('--lbfgs', type = int, default = 0, help = 'Number of L-BFGS optimization steps')
parser.add_argument('--savemodel', type = bool, default = False, help = 'Whether to save the trained model in JAX, default path = ./PINN_trained')
parser.add_argument('--savefig', type = bool, default = False, help = 'Whether to save the plotted figures, default path = ./PredictionsLosses')
parser.add_argument('--inverseprob', type = bool, default = True, help = 'Whether to solve the inverse problem inferring m and q')
parser.add_argument('--lamda', type = str, default = '5.0,5.0,1.0', help = 'Weight Coefficients for PINN loss, x_loss, v_loss and f_loss')

args = parser.parse_args()

# Processing Lambda for Weighted Loss
def parse_lambda(lambdas):
    r'''Parse Given Weights List for Loss (string -> float)'''
    return list(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', lambdas)))

lamda = parse_lambda(args.lamda)

print(f'{lamda=}')

'''
Data Preparation

Using PINN N(t) to approximate 
future spatial states/phase flow of charged particle in an ElectroMagnetic Potential

The output in this specific problem is expected to be of shape (None, 2)
'''
# Of shape (num_pts, dims) = (1500, 4)
# where the first two columns are Velocity v
# the second two columns are Spatial Location x
data_train = load_data('train', base_path = args.datapath)
data_test = load_data('test', base_path = args.datapath)

v_train, x_train = data_train[:, :2], data_train[:, 2:4]
v_test, x_test = data_test[:, :2], data_test[:, 2:4]

# 1500 pts with grid size 0.1
lb, ub = (0, 150)
stepsz = 0.1
t = jnp.arange(lb, ub, stepsz).reshape((-1, 1))


'''
Initialize the Network

input shape : (None, 1) Time t
output shape: (None, 2) Spatial Locations X; (None, 4) Phase Flow Z

8 hidden layers with 50 neurons each

1. To make sure mass m is positive we predict logm instead of m
2. For Inverse Problem: make logm and q trainable variables (jax.Array);
   Otherwise keep them constants (untraceable) numpy.ndarray 
** Expected m = 1.0, q = 1.0 **
'''
layers = [1] + [50]*8 + [2]
pinn_initial_params = init_network_params(layers, initializer = 'xavier_normal')

if args.inverseprob:
    mq = jax.random.normal(jax.random.PRNGKey(0), [1])
else:
    mq = np.array([1.0])


# For the inverse problem include Mass-to-Charge Ratio in trainable variables
all_initial_params = [*pinn_initial_params, *[mq]]

# Check initialzied m and q
print(f'\n{mq=}')


'''
Train the PINN 
'''
updated_params, losses, opt_state = trainPINN(all_initial_params, t, lb, ub, [v_train, x_train], lamda = lamda,\
                                     learning_rate = args.lr, opt_state = None,\
                                     iter_adam = args.adam, iter_lbfgs = args.lbfgs,\
                                     loss_fn = total_PINN_loss)

if args.savemodel:
    filename = f'./PINN_trained/{args.adam}Adam_{args.lbfgs}LBFGS'
    save_params(filename, filename, updated_params)

## Sample code to load and validate the model back
#loaded = load_params(path_p = filename, path_t = filename)
#
#print('Check if the loaded params are the same as the original:',validate_params(updated_params, loaded))


'''
Plotting Trajectory and Losses
'''
plot_trajectory_PINN(normalized_predict, updated_params[:-1], t, [x_train, x_train],\
                    lb, ub, text = f'{args.adam}Adams_{args.lbfgs}LBFGS', savefig = args.savefig)

plot_losses(losses, semilogy = True, savefig = args.savefig)

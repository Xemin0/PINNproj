# Essential Modules                                                                                            
import jax.numpy as jnp 
from jax import value_and_grad, jit, grad

import numpy as np

# Optimizers
from jax.example_libraries.optimizers import adam
# L-BFGS from JAXopt
# Reference: https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGS.html
from jaxopt import LBFGS


# Cross Ref
from utils.electro_mag import f_lorentz

# Miscellaneous
import matplotlib.pyplot as plt 
from matplotlib import colormaps
from os.path import join

'''
PINN Total Loss
(DataLoss(Initial, Boundary, Residual) + EqnLoss)

'''
@jit
def total_PINN_loss(params, mq, t, v_true, x_true,\
                    lb, ub, lamda = (1.0, 1.0, 1.0)):
    '''
    Total Loss of PINN:
        - MSE(x_pred, x)
        - Residual of Eqn
        - MSE(v_pred, v)
    '''
    # Forward pass of the Network
    residual, x_pred, v_pred, _ = f_lorentz(params, mq, t, lb, ub)
    
    x_loss = jnp.mean(jnp.sum((x_pred - x_true) ** 2, axis = 1))
    
    v_loss = jnp.mean(jnp.sum((x_pred - x_true) ** 2, axis = 1))
    
    f_loss = jnp.mean(jnp.sum(residual ** 2, axis = 1) )
    
    # Unpack the weights for the losses
    lamda_x, lamda_v, lamda_f = lamda
    return lamda_x * x_loss + lamda_v * v_loss + lamda_f * f_loss,\
            (x_loss, v_loss, f_loss)



'''
Overwrite the Train Steps for PINN

Inputs:
    *** Data ***
    - all_params         : [params, mq] 
                           ### Use [*a, *b] to combine elements from list a and b ###
    - t                  : Inputs of shape (num_pts, 1)
    - lb                 : Lower bound of the input data (time t)
    - ub                 : Upper bound of the input data (time t)
    - data               : [v_true, x_true]
    *** Optimizers ***
    - learning_rate      :
    - opt_state          : 
    - iter_adam          : Number of iterations for Adam steps
    - iter_lbfgs         : Number of iterations for L-BFGS steps ## will only be used for minimization
    - loss_fn            : Customized LossFunction for PINN
    
Outputs:
    - updated_params     : [params] 
    - mq                 : Mass-to-Charge Ratio
    - loss_list          : A dictionary of all Train Losses
    - opt_states         : [opt_state1, opt_state2] for params and mq
    
    ### Adversarial Training for the Inverse Problem ###
        - Minimize the Total Loss w.r.t. Model's parameters
        - Maximize the Total Loss (Residuals) w.r.t. mq the Mass-to-Charge Ratio
'''
MAXITER = 2000
N_GRAD = 8

def trainAdPINN(all_params, t, lb, ub, data, lamda = (1.0, 1.0, 1.0),\
             learning_rate = 2e-3, opt_state = None,\
             iter_adam = 10, iter_lbfgs = 0,\
             loss_fn = total_PINN_loss):
    # Unpack the parameters
    params = all_params[:-1]
    mq = all_params[-1]
    
    # Initialize the Optimizers
    # For model's params
    adam_init, adam_update, adam_get_params = adam(learning_rate)
    # For mq
    adam_initmq, adam_updatemq, adam_get_paramsmq = adam(learning_rate)
    if None == opt_state:
        opt_state = adam_init(params)
        opt_statemq = adam_initmq(mq)
    
    # For model's params only
    lbfgs_solver = LBFGS(fun = loss_fn, has_aux = True, maxiter = MAXITER, history_size = N_GRAD)
                        # linesearch = 'hager-zhang', stepsize = 4e-1 #
        
    updated_params = params
    loss_list = {
        'total_losses' : [],
        'x_losses' : [],
        'v_losses' : [],
        'f_losses' : [],
    }
    
    # Unpack the input data
    v_true, x_true = data
    
    ''' Adam '''
    print(f'\nBegin {iter_adam} Adam Steps....')
    for i in range(iter_adam):
        # Optimizer Step - Weights and States Update
        # For model's params
        losses, g = value_and_grad(loss_fn, argnums = 0, has_aux = True)(\
                                        updated_params, mq, t, v_true, x_true,\
                                        lb, ub, lamda)
        # For mq # Discard the Auxilary data returned
        g_mq, _ = grad(loss_fn, argnums = 1, has_aux = True)(\
                                        updated_params, mq, t, v_true, x_true,\
                                        lb, ub, lamda)
        # For model's params
        opt_state = adam_update(i, g, opt_state)
        updated_params = adam_get_params(opt_state)
        
        # For mq
        #print('grad for mq:', g_mq, end = '')
        opt_statemq = adam_updatemq(i, -g_mq, opt_statemq)
        mq = adam_get_paramsmq(opt_statemq)
        
        # Upack the losses = (value, auxiliary_data)
        total_loss, (x_loss, v_loss, f_loss) = losses
        
        loss_list['total_losses'].append(total_loss)
        loss_list['x_losses'].append(x_loss)
        loss_list['v_losses'].append(v_loss)
        loss_list['f_losses'].append(f_loss)
        
        print(f'\r[Train Adam Step: {i+1}/{iter_adam}]\tTotalLoss:{total_loss:.5f}\tXLoss:{x_loss:.5f}\tVLoss:{v_loss:.5f}\tFLoss:{f_loss:.5f}\t mq = {mq[0]:.5f}\t mq_grad = {g_mq[0]:.5f}', end = '')
    print('\n' + '-'*20)
    
    ''' L-BFGS '''
    print(f'\nBegin {iter_lbfgs} L-BFGS Steps....')
    for i in range(iter_lbfgs):
        # Optimizer Step - Weights and States Update
        updated_params, opt_state = lbfgs_solver.run(updated_params, mq, t, v_true, x_true,\
                                                    lb, ub, lamda)
        total_loss = opt_state.value
        x_loss, v_loss, f_loss = opt_state.aux
        
        loss_list['total_losses'].append(total_loss)
        loss_list['x_losses'].append(x_loss)
        loss_list['v_losses'].append(v_loss)
        loss_list['f_losses'].append(f_loss)
        
        print(f'\r[Train LBFGS Step: {i+1}/{iter_lbfgs}]\tTotalLoss:{total_loss:.5f}\tXLoss:{x_loss:.5f}\tVLoss:{v_loss:.5f}\tFLoss:{f_loss:.5f}\t mq = {mq[0]:.5f}', end = '')
    print('\n' + '-'*20)
    return updated_params, mq, loss_list, [opt_state, opt_statemq]

'''
Plotting Subroutines

** Need to be adjusted ?? **
'''
def plot_trajectory_PINN(x_net, params, t, x_data,\
                         lb, ub, text = 'Positional Predictions', savefig = False):
    '''
    Plot the Predicted (Spatial) Trajectory (by PINN) against the Ground Truth one
    
    Inputs:
        - x_net : The forward pass of the approximation network
        - params:
        - t     : Time, of shape (num_pts, 1)
        - x_true: Ground truth spatial locations, of shape (num_pts, dims = 2)
        - lb    : Lower bound of x_net inputs (time t)
        - ub    : Upper bound of x_net inputs (time t)
        
    Outputs:
        - x_pred: Predicted spatial locations
    '''
    x_train, x_test = x_data
    x_pred = x_net(params, t, lb, ub)
    steps = len(x_test)
    print('x_pred shape', x_pred.shape)
    
    plt.figure(figsize=[6.4 * 1, 5.5 * 4])
    # Plot spatial locations
    plt.subplot(411)
    # Test data
    plt.plot(x_test[:, 0], x_test[:, 1], 'b-', label = 'Ground Truth', zorder = 0, linewidth = 3)
    # Predicted data
    plt.plot(x_pred[:, 0], x_pred[:, 1], 'r-', label = 'Predicted Flow', zorder = 1, linewidth = 2)
    # Train data
    plt.plot(x_train[:, 0], x_train[:, 1], 'k--', label = 'Learnt Data', zorder = 2, linewidth = 2)
    plt.xlabel(f'$x_1$')
    plt.xlabel(f'$x_2$')
    plt.title(text)
    #plt.axis('square')
    plt.legend()
    plt.axis('equal')
    plt.subplots_adjust(right=1.2)
    
    plt.subplot(412)
    plt.plot(np.arange(steps), x_test[:, 0], color='b', label='Ground Truth')
    plt.plot(np.arange(steps), x_pred[:, 0], color='r', label='PNN Predicted')
    plt.xlabel(r'Step', fontsize=13)
    plt.ylabel(r'$x_1$', fontsize=13)
    plt.title(f'$x_1$ v.s. Time Steps')
    plt.legend(fontsize=13)
    
    plt.subplot(413)
    plt.plot(np.arange(steps), x_test[:, 1], color='b', label='Ground Truth')
    plt.plot(np.arange(steps), x_pred[:, 1], color='r', label='PNN Predicted')
    plt.xlabel(r'Step', fontsize=13)
    plt.ylabel(r'$x_2$', fontsize=13)
    plt.title(f'$x_2$ v.s. Time Steps')
    plt.legend(fontsize=13)
    
    plt.subplot(414)
    # Calculate the MSE w.r.t. time t for the predicted results (for the test predictions)
    def MSE(y_pred, y_true):
        diff = y_pred - y_true
        if 1 == y_pred.ndim:
            return jnp.mean(diff ** 2)
        elif 2 == y_pred.ndim:
            return jnp.mean(jnp.sum(diff ** 2, axis = -1))
    
    num_test = len(x_test)
    print('Number of Test Prediciton Points:', num_test)
    ### Taking Extra Long Time ###
    #MSE_losses = [MSE(x_test[:i+1], x_pred[:i+1]) for i in range(num_test)]
    
    diff = x_pred - x_test
    squared_sum = jnp.sum(diff ** 2, axis = -1)
    MSE_losses = [jnp.mean(squared_sum[:i+1]) for i in range(num_test)]  
    
    plt.plot(np.arange(num_test), MSE_losses)
    plt.title('Predicted MSE Loss v.s. Time')
    
        
    np.save('./PredictedLosses/PINNloss', np.array(MSE_losses))
    
    if savefig:
        plt.savefig(text + '.pdf')
    #plt.show()
    plt.close()
    return x_pred

from matplotlib import colormaps

def plot_losses(losses, semilogy = False, text = 'All Losses of PINN',\
                    colormap = 'tab10', savefig = False):
    '''** Expect the input to be a dictionary **'''
    # Create a Figure
    fig, ax = plt.subplots()
    # Setup Cycling Colors
    #cmap = get_cmap(colormap)
    cmap = colormaps[colormap]
    colors = cmap.colors
    ax.set_prop_cycle(color = colors)
    
    if semilogy:
        for key, loss_history in losses.items():
            ax.semilogy(loss_history, label = key)
    else:
        steps = len(list(losses.values())[0])
        steps = np.arange(steps)
        for key, loss_history in losses.items():
            ax.plot(steps, loss_history, '-', label = key)
    
    plt.legend()
    
    if savefig:
        plt.savefig(text + '.pdf')
        
    plt.close()

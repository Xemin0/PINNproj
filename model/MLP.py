# Essential Modules
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jax.nn import tanh

# Optimizers
from jax.example_libraries.optimizers import adam
# L-BFGS from JAXopt
# Reference: https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGS.html
from jaxopt import LBFGS

'''
Generic MLP Neural Network in JAX for Function Approximations
    
    - Variable Number of Layers upon Initialization with Tanh Activation (No activation at the output)
    - Optimizers: Adam ; Adam + L-BFGS (L-BFGS)
    - Metrics   : MSE; Customized
    - 

    *Reference*: 
    - MY HW_L6
'''
# Initialize a single linear layer's parameters
def init_layer_params(Fan_in, Fan_out, key, scale, method = 'normal'):
    w_key, b_key = jax.random.split(key)
    if 'uniform' == method:
        scale *= 3
        return jax.random.uniform(w_key, shape = (Fan_in, Fan_out), minval = -scale, maxval = scale), jax.random.uniform(b_key, shape = (Fan_out,))
                                                                                            #jnp.zeros(Fan_out, dtype = jnp.float32)
    elif 'normal' == method:
        return scale * jax.random.normal(w_key, (Fan_in, Fan_out)), jax.random.normal(b_key, (Fan_out,))
                                                            #jnp.zeros(Fan_out, dtype = jnp.float32)

# Initialize all layers' parameters in the network
# use Xavier Initialization by default for tanh
def init_network_params(sizes, key = jax.random.PRNGKey(0), initializer = 'xavier_normal'):
    assert initializer.lower() in (
        'xavier_normal',
        'he_normal',
        'xavier_uniform',
        'he_uniform'
    ), ('\nmust be one of {xavier_normal, he_normal, xavier_uniform, he_uniform}\n-----\n' + f"Unknown Initialization Strategy, '{initializer}' requested")
    
    name, method = initializer.lower().split('_')
    if 'he' == name:
        scale = lambda a,b: jnp.sqrt(2.0/(a))
    elif 'xavier' == name:
        scale = lambda a,b: jnp.sqrt(2.0/(a+b))
    keys = jax.random.split(key, len(sizes))
    return [init_layer_params(m, n , k, scale(m,n), method) \
           for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

'''
# Forward Pass
'''
# *** the final output layer has no activation
@jit
def predict(params, X):
    for w, b in params:
        Y = jnp.dot(X, w) + b
        X = tanh(Y)
    return Y

@jit
def normalized_predict(params, X, lb, ub):
    # re-range the input from (lb, ub) to 
    # (-1, 1)
    X = 2*(X - lb)/(ub - lb) - 1
    for w, b in params:
        Y = jnp.dot(X, w) + b
        X = tanh(Y)
    return Y


'''
# The MSE loss
'''
@jit
def mse_loss(params, x, y_true):
    y_pred = vmap(predict, in_axes = (None, 0))(params, x) # [Batch_size, FeatureSize]
    diff = y_pred - y_true
    return jnp.mean(jnp.sum(diff ** 2, axis = 1))

@jit
def normalized_mse_loss(params, x, y_true, lb, ub):
    y_pred = vmap(normalized_predict, in_axes = (None, 0, None, None))(params, x, lb, ub)
    diff = y_pred - y_true
    return jnp.mean(jnp.sum(diff ** 2, axis = 1))
    
    
'''
Train Step
'''
MAXITER = 1000
N_GRAD = 8
def trainApprox(params, x, y, LEARNING_RATE = 2e-3, opt_state = None,\
                iter_adam = 20, iter_lbfgs = 0,\
                loss_fn = mse_loss):
    # Initialize the Optimizers
    adam_init, adam_update, adam_get_params = adam(LEARNING_RATE)
    if opt_state == None:
        opt_state = adam_init(params)
    
    lbfgs_solver = LBFGS(fun = loss_fn, maxiter = MAXITER, history_size = N_GRAD) #linesearch = 'hager-zhang', stepsize = 4e-1)
    
    updated_params = params
    loss_list = []
    ''' Adam'''
    for i in range(iter_adam):
        loss, g = value_and_grad(loss_fn, argnums = 0)(updated_params, x, y)
        opt_state = adam_update(i, g, opt_state)
        updated_params = adam_get_params(opt_state)
        loss_list.append(loss)
        print(f'\r[Train Adam Step:{i+1}/{iter_adam}]\tLoss:{loss:.4f}', end = '')
    print('\n' + '-'*30)
    
    '''LBFGS'''
    for i in range(iter_lbfgs):
        updated_params, opt_state = lbfgs_solver.run(updated_params, x = x, y_true = y)
        loss = opt_state.value
        loss_list.append(loss)
        print(f'\r[Train LBFGS Step:{i+1}/{iter_lbfgs}]\tLoss:{loss:.4f}', end = '')
    print('\n' + '-'*30)
    return updated_params, loss_list, opt_state

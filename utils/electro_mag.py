# Essential Modules
import jax.numpy as jnp
from jax import jit

# Cross Reference
from .auto_diff import value_and_egrad, value_and_vec_ehessian
from model.MLP import normalized_predict

'''
Problem Specific Utilities
'''
@jit
def electric_field(x):
    '''*** BUT THERE IS A SINGULARITY AT THE ORIGIN ***'''
    '''
    ** Solution 1: Initialize network's biases as non-zeros
    ** Solution 2: Replace zero rows in the input with eps = 1e-7
                   *** Must be Differentiable ***
                   *** Better be Jittable     ***
    '''
    # Replace all Rows of Zeros in the input with eps = 1e - 7
    # ** CHANGE VARIABLE NAME **
    # new_x will be used in the actual function evaluation
    # while x the input will still be used to create mask for jnp.where
    # ** Need to use TWO jnp.where to guarantee correct values
    # ** for both the function evaluation and the gradient
    eps = 1e-7
    new_x = jnp.where(
            jnp.all(x == 0, axis = 1, keepdims = True),  # Mask : Locations (where x == 0) of entries to be replaced
            jnp.ones_like(x) * eps ,                     #        The new values (inside the mask)
            x)                                           #        Original       (outside the mask)
    return jnp.where(
                jnp.all(x == 0, axis = 1, keepdims = True),
                jnp.ones_like(x) / eps,
                new_x / (100.0 * (jnp.sum(new_x ** 2, axis = 1, keepdims = True) ** 1.5))
            )
    #return x / (100.0 * (jnp.sum(x ** 2, axis = 1, keepsdims = True) ** 1.5))

@jit
def magnetic_force(v):
    '''v × B = (v2, -v1) * B3'''
    return v[:, [1, 0]] * jnp.array([1, -1])\
            * jnp.sum(v ** 2, axis = 1, keepdims = True) ** 0.5

'''
### Residual Computation (Defined by the Differential Eqn in Physics) 
# Requires 
#      - forward pass of x_net
#      - Element-wise Gradient(First Order Derivative)
#      - Element-wise Hessian (Second Order Derivative) 
#
# input:
#      - t of shape (batch_size, 1)
# output:
#      - x of shape (batch_size, dims = 2)
#
# the Physics Eqn (ODE - Lorentz Force)
# mx'' = qE + x' × B
#
# Thus f_lorentz = mx'' - (qE + x' × B)
#
#  ** 1. To make sure mass m is positive we predict logm instead of m **
#        While adding a Regularization term in the loss function to drive the value of logm down
### **** Need Element-Wise Derivatives ****
'''
@jit
def f_lorentz(params, logm, q, t, lb, ub):
    # Forward Wrapper of the Net to exclude unnecessary arguments
    # which will normalize the input
    def x_net(t):
        return normalized_predict(params, t, lb, ub)
    ############# JVP + HVP
    # x = x_net(t) 
    #x_pred = x_net(t)
    #print('finished evaluation')
    # v = x'
    # a = x'' 
    #v_pred, a_pred = egrad_and_ehessian(x_net, t)
    #print('finished egrad, ehessian')
    ############## VMAP + reshape
    # x = x_net(t)
    # v = x'
    x_pred, v_pred = value_and_egrad(x_net, t)
    # a = x''
    _, a_pred = value_and_vec_ehessian(x_net, t)
    ##############
    
    # residual = LHS - RHS
    '''
    For the inverse problem, since we would require m to be always positive
    thus, instead of directly predicting the m, we will predict logm
    and to avoid large m, q values we will add a penalty term to drive their values down
    '''
    residual = jnp.exp(logm) * a_pred - q * (electric_field(x_pred) + magnetic_force(v_pred))
    #print('finished residual')
    return residual, x_pred, v_pred, a_pred

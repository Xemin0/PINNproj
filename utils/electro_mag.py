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
# mx''/q = E + x' × B
#
# Thus f_lorentz = mx''/q - (E + x' × B)
#
#  ** 1. To make sure mass m is positive we predict logm instead of m **
#        While adding a Regularization term in the loss function to drive the value of logm down
# ------------ UPDATED --------------
# Instead of predicting m and q at the same time we will just predict the mass-to-charge ratio
### **** Need Element-Wise Derivatives ****
'''
@jit
def f_lorentz(params, mq, t, lb, ub):
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
    residual = mq[0] * a_pred - (electric_field(x_pred) + magnetic_force(v_pred))
    #print('finished residual')
    return residual, x_pred, v_pred, a_pred


@jit
def f_lorentzPhaseFlow(params, mq, t, lb, ub):
    '''
    # Forward Wrapper of the Net to exclude unncessary arguments
    # which will normalize the input
    # 
    # flow_net(t) = (V_pred, X_pred)
    #
    ### For the inverse problem we willl predict the mass-to-charge ratio mq ###
    '''
    def flow_net(t):
        return normalized_predict(params, t, lb, ub)
    
    # (V,X)_pred    = flow_net(t)
    # (V', X')_pred = (V, X)_pred'
    flow_pred, flow_pred_d1 = value_and_egrad(flow_net, t)
    # (V'', X'')_pred = (V, X)_pred''
    _, flow_pred_d2 = value_and_vec_ehessian(flow_net, t)
    
    ######### _t for Ground Truth, _p for Predictions
    #               V_t, X_t = flow_true
    #               V_p, X_p = flow_pred
    #        V_p' , X_p'
    # V_p'', X_p''     
    #
    ######### 
    # So here we will return two residual f_loss1(X_p), f_loss2(V_p)
    # V_p = flow_pred[ : dim ] ; X_p = flow_pred[-dim : ]
    #
    # residual = LHS - RHS
    #          = mq * a - (E + v × B)
    # Dimension of the Phase Flow, in this case 4, then V and X both have a dimension of 2
    flow_dim = flow_pred.shape[-1]
    dim = flow_dim // 2
    
    # f_loss1(X_p)
    residual1 = mq[0] * flow_pred_d2[:,-dim:] - (electric_field(flow_pred[:, -dim:]) + magnetic_force(flow_pred_d1[:, -dim:]))
    # f_loss2(V_p)
    residual2 = mq[0] * flow_pred_d1[:,:dim] - (electric_field(flow_pred[:, -dim:]) + magnetic_force(flow_pred[:, :dim]))
    
    return residual1, residual2, flow_pred, flow_pred_d1, flow_pred_d2


'''
PINN Total Loss for Phase Flows
'''
def total_PINN_pfLoss(params, mq, t, flow_true,\
                    lb, ub, lamda = (1.0, 1.0, 1.0)):
    '''
    Total Loss for PINN that Predicts Phase Flow (**Difficult to Design**)
    *** Check f_lorentzPhaseFlow for details ***
    For (V_t, X_t) = flow_true ; (V_p, X_p) = flow_pred
    
        - pf_loss                        : MSE(flow_pred, flow_true) * lamda0
        - f_loss1 * lamda1               : based on X_p
        - f_loss2 * lamda2               : based on V_p
        - approx_loss                    : MSE(V_p, X_p') * lamda0
    '''
    # Dimension of the Phase Flow, in this case 4, then V and X both have a dimension of 2
    flow_dim = flow_true.shape[-1]
    dim = flow_dim // 2
    
    residual1, residual2, flow_pred, flow_pred_d1, flow_pred_d2 = f_lorentzPhaseFlow(params, mq, t, lb, ub)
    # Unpack the weights for the losses
    lamda_pf, lamda_f1, lamda_f2 = lamda
    
    pf_loss = jnp.mean(jnp.sum((flow_pred - flow_true) ** 2, axis = 1))
    
    f_loss1 = jnp.mean(jnp.sum(residual1 ** 2, axis = 1))
    
    f_loss2 = jnp.mean(jnp.sum(residual2 ** 2, axis = 1))
    
    approx_loss = jnp.mean(
                    jnp.sum((flow_pred[:, :dim] - flow_pred_d1[:, -dim:]) ** 2, axis = 1)
                    )
    
    return lamda_pf * (pf_loss + approx_loss) + lamda_f1 * f_loss1 + lamda_f2 * f_loss2,\
            (pf_loss, f_loss1, f_loss2, approx_loss)

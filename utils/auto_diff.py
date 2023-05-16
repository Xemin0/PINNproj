import jax.numpy as jnp
from jax import jvp, vmap, hessian

'''
Customized Auto-Differentiations:
(MAY NOT BE OPTIMIZED FOR MEMORIES;
Consider using both Foward- and Reverse- Modes;
Forward Mode has less overhead in this case 
      scalar input -> vector output,
thus keeping `jacfwd` on the outside????)

**Equivalences**

  - Element-wise Gradient 
    (First Order Derivative that Retains the Output Size):
          - y, dydx = jvp(f, (x,), (jnp.ones(x.shape),))
          - dydx = jnp.tensordot(jacfwd(f)(x), jnp.ones(x.shape))
          - dydx = jnp.einsum('ijmn,mn->ij',jacfwd(f)(x), jnp.ones(x.shape))
          -
          ## v.s. JVP ??
          - dydx = vmap(jacfwd(f))(x).reshape(y.shape)  
          
  - Element-Wise Hessian (hvp recommended)
    (Second Order Derivative that Retains the Output Size):
            # Forward-over-Reverse Mode
          - (df = lambda a: jnp.tensordot(jacrev(f)(a), jnp.ones(a.shape)))
            dydx, ddydx2 = jvp(df, (x,), (jnp.ones(x.shape), ))
          - ddydx2 = jnp.tensordot(
                             jnp.tensordot(hessian(f)(x), jnp.ones(x.shape)),
                             jnp.ones(x.shape)
                         )   ### Hessian @ V @ V  ***** !! OOM WARNING !! *****
          -
          ## vmap + reshape much faster than hvp
          - ddydx2 = vmap(hessian(f), in_axes = 0)(x).reshape(y.shape)
'''

#@jit
def value_and_egrad(f, x):
    '''Returns both the Evaluation and the Element-wise Gradients'''
    return jvp(f, (x,), (jnp.ones(x.shape),))



#@jit
def value_and_vec_ehessian(f, x):  ### vmap + reshape
    '''Element-wise Hessians'''
    #return jnp.tensordot(
    #                jnp.tensordot(hessian(f)(x), jnp.ones(x.shape)),
    #                jnp.ones(x.shape)
    #            )
    y = f(x)
    return y, vmap(hessian(f))(x).reshape(y.shape)

#@jit
def egrad_and_ehessian(f, x): ### hvp function???
    '''Returns both the Element-wise Gradients and the Element-wise Hessian'''
    df = lambda a: jnp.tensordot(jacrev(f)(a), jnp.ones(a.shape))
    return jvp(df, (x,), (jnp.ones(x.shape), ) )

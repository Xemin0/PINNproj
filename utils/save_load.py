import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_leaves, tree_map, tree_flatten, tree_unflatten, tree_all

import pickle

#from os.path import join
#import glob

'''
- Save and Load the Model in JAX
- Validate if 2 sets of params (the original and the loaded) are the same

Reference: MY HW_L6
'''

def save_params(path_p:str, path_t:str, params)->None:
    '''
    Save the model in JAX
    
    Args:
        path_p (str)      : (.npy) Path to save the Parameters as tree_leaves
        path_t (str)      : (.pkl) Path to save the Parameters' PyTree structure
        params (jax.Array): Parameters to be saved
    '''
    if 'npy' != path_p.split('.')[-1]:
        path_p = path_p + '.npy'
    if 'pkl' != path_t.split('.')[-1]:
        path_t = path_t + '.pkl'
    
    with open(path_p, 'wb') as f:
        for p in tree_leaves(params):
            jnp.save(f, p, allow_pickle=False)
    print(f'Saved the Parameters Leaves at: {path_p}')
    
    p_tree_struct = tree_map(lambda t: 0, params)
    with open(path_t, 'wb') as f:
        pickle.dump(p_tree_struct, f)
    print(f'Save the Parameters PyTree Structure at: {path_t}')
    
        
        
def load_params(path_p:str, path_t:str):
    '''
    Load the model in JAX
    
    Args:
        path_p (str)      : (.npy) Path to load the Parameters as tree_leaves
        path_t (str)      : (.pkl) Path to load the Parameters' PyTree structure
    '''
    if 'npy' != path_p.split('.')[-1]:
        path_p = path_p + '.npy'
    if 'pkl' != path_t.split('.')[-1]:
        path_t = path_t + '.pkl'
        
    with open(path_t, 'rb') as f:
        tree_struct = pickle.load(f)
    print(f'Loaded the Parameters Leaves at: {path_p}')
    
    leaves, treedef = tree_flatten(tree_struct)
    with open(path_p, 'rb') as f:
        flat_params = [jnp.load(f) for _ in leaves]
    print(f'Loaded the Parameters PyTree Structure at: {path_t}')
    return tree_unflatten(treedef, flat_params)



def validate_params(original, loaded) -> bool:
    ''' Check if the two sets of params are the same'''
    return tree_all(
                tree_map(jnp.array_equal, original, loaded)
            )

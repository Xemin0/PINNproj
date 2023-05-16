import numpy as np
from os.path import join
'''
Loading Data

Default 'train.txt' and 'test.txt' from '../Datasets'
'''
def load_data(dataset:str, base_path = '../Datasets/'):
    if dataset in ('train', 'test'):
        data = np.loadtxt(join(base_path, dataset + '.txt'), delimiter = ' ')
        print(f'Successfully Loaded {dataset} data...')
    else:
        raise ValueError("Input str must be either 'train' or 'test'")
        
    return data

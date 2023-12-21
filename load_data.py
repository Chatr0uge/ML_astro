import numpy as np
import os
from tqdm import tqdm
from typing import Iterable


def slice_image(data : np.ndarray, size : int) -> Iterable:
    """
    slice_image Slice an image into smaller images of size size x size

    Parameters
    ----------
    data : np.ndarray
        image to be sliced
    size : int
        size of the sliced images

    Returns
    -------
    Iterable
        array of sliced images
    """    
    """"""
    # memory allocation : 
    sliced_data = np.zeros(shape = (data.shape[0] * data.shape[1] // size**2, size, size), dtype = np.float32)
    
    # slicing : 
    for i in range(data.shape[0] // size): 
        for j in range(data.shape[1] // size) : 
            sliced_data[i * data.shape[1] // size + j] = data[i * size : i * size + size, j * size : j * size + size]
            
    return sliced_data


def process_data(dir_path : str) -> Iterable: 
    """
    process_data Extract data from a given set of Quijote Simulation

    Parameters
    ----------
    dir_path : str
        
        path of the quijote simulations

    Returns
    -------
    Iterable
    
        array of quijote Simulations 
    """    
    """"""
    
    # memory allocation :     
    print('Processing Data in ' + dir_path)
    j = 0 
    total_data = []
    
    for j, file in enumerate(tqdm(os.listdir(dir_path)[2:], colour = 'cyan', desc= 'Extracting and slicing data')) : 
        for i, cubes in enumerate(os.listdir(dir_path + '/' +  file + '/')) : 
            
            cube_dir = dir_path + '/' +  file + '/' + cubes 
                            
            data = np.load(cube_dir)
            
            for k in range(0,256,2) :
                           
                for image in slice_image(np.mean(data[:,:,k:k + 2], axis=2), 64) : 
                    shape
                    total_data.append(image)
        
        
    return total_data
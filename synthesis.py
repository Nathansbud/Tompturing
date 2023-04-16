from skimage.io import imread
import numpy as np

def quilt(texture_path: str):
    texture = imread(texture_path)
    print(texture.shape)
    
    #TODO: The paper :)

    return texture

def quilt_and_transfer(texture_path: str, transfer_path: str):
    quilted = quilt(texture_path)
    transfer = imread(transfer_path)
    print(transfer.shape)

    #TODO: The paper :)

    return transfer

    

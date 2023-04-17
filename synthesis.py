from skimage.io import imread
import numpy as np

def quilt(block_size: int, texture_path: str, scale: int):
    texture = imread(texture_path)
    th, tw, tc = texture.shape

    # tile must be within the bounds of the texture
    if not (0 < block_size < th and 0 < block_size < tw):
        print(f"Block size must be between 0 and texture min dimension; {block_size} not within [0, {min(th, tw)}])")
        exit(1)
    
    print(texture.shape)
    
    #TODO: The paper :)

    return texture

def quilt_and_transfer(block_size: int, texture_path: str, transfer_path: str, scale: int):
    quilted = quilt(block_size, texture_path)
    transfer = imread(transfer_path)
    print(transfer.shape)

    #TODO: The paper :)

    return transfer

    

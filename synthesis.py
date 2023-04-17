from skimage.io import imread
import numpy as np
from typing import List
import math

def get_texture_blocks(texture: np.ndarray, block_size: int) -> List[np.ndarray]:
    """Returns a list of blocks (of size block_size) from the input texture"""
    th, tw, tc = texture.shape
    texture_blocks = [texture[i:i+block_size, j:j+block_size] for j in range(tw - block_size) for i in range(th - block_size)]
    return texture_blocks

def get_random_block(blocks: List[np.ndarray]) -> np.ndarray:
    """Returns a random block out of the provided list of blocks"""
    index = np.random.randint(len(blocks))
    return blocks[index]

def find_good_block(img_segment: np.ndarray, blocks: List[np.ndarray]) -> np.ndarray:
    """Returns a random block from the input texture which fits the image segment and satisfies the overlap constraints"""
    h, w, c = img_segment.shape
    downscaled_blocks = [np.copy(block)[:h, :w, :c] for block in blocks] # Scale down blocks to img_segment size
    l2_norms = []
    for block in downscaled_blocks:
        # multiply with mask to only calculate error for overlapping part:
        l2_norm = np.sum(np.square(block - img_segment) * (img_segment >= 0))
        l2_norms.append(l2_norm)
    
    best_norm = min(l2_norms)
    tolerance = 0.1 * best_norm # tolerance: must be within 0.1 times of the best overlapping block's error
    good_blocks = []
    for i, block in enumerate(downscaled_blocks):
        if l2_norms[i] <= best_norm + tolerance:
            good_blocks.append(block)
    return get_random_block(good_blocks)



def quilt(block_size: int, texture_path: str, scale: float):
    texture = imread(texture_path)
    th, tw, tc = texture.shape

    # tile must be within the bounds of the texture
    if not (0 < block_size < th and 0 < block_size < tw):
        print(f"Block size must be between 0 and texture min dimension; {block_size} not within [0, {min(th, tw)}])")
        exit(1)
    
    print(texture.shape)
    
    texture = np.array(texture)
    texture_blocks = get_texture_blocks(texture, block_size)
    
    outh, outw, outc = int(th * scale), int(tw * scale), tc # Dimensions of output texture
    quilted_img = -100 * np.ones((outh, outw, outc)) # Placeholder array for quilted texture
    ## NOTE: I filled it with negative values instead of zeros to help in creating masks for the L2 norm calculations in find_good_block

    quilted_img[:block_size, :block_size, :] = get_random_block(texture_blocks) # Place a random block on the top left

    overlap = math.ceil(block_size / 6) # The paper said the overlap was 1/6th of the block size
    # Going through the image to be synthesized in raster scan order
    for row in range(block_size - overlap, outh, block_size - overlap):
        for col in range(block_size - overlap, outw, block_size - overlap):
            remainingY = outh - row
            remainingX = outw - col
            img_segment = quilted_img[row: row + max(block_size, remainingY), col: col + max(block_size, remainingX)]
            selected_block = find_good_block(img_segment, texture_blocks)

            # TODO: min_err_boundary_cut (seamcarve) and then mould selected_block based on that seam and place into quilted_img

    return texture

def quilt_and_transfer(block_size: int, texture_path: str, transfer_path: str, scale: float):
    quilted = quilt(block_size, texture_path)
    transfer = imread(transfer_path)
    print(transfer.shape)

    #TODO: The paper :)

    return transfer

    

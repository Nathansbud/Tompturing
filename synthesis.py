from skimage.io import imread, imshow
import numpy as np
from typing import List, Tuple
import math
import matplotlib.pyplot as plt

def get_texture_blocks(texture: np.ndarray, block_size: int) -> List[np.ndarray]:
    """Returns a list of blocks (of size block_size) from the input texture"""
    print("get texture blocks")
    th, tw, tc = texture.shape
    texture_blocks = [texture[i:i+block_size, j:j+block_size] for j in range(0, tw - block_size, 3) for i in range(0, th - block_size, 3)]
    return texture_blocks

def get_random_block(blocks: List[np.ndarray]) -> np.ndarray:
    """Returns a random block out of the provided list of blocks"""
    print("get random block")
    index = np.random.randint(len(blocks))
    return blocks[index]

def find_good_block(img_segment: np.ndarray, blocks: List[np.ndarray], overlap: int, row: int, col: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a random block from the input texture which fits the image segment and satisfies the overlap constraints"""
    print("find good block")
    h, w, c = img_segment.shape
    downscaled_blocks = [np.copy(block)[:h, :w, :c] for block in blocks] # Scale down blocks to img_segment size
    l2_norms = []
    print("downscaled blocks len:", len(downscaled_blocks))
    for block in downscaled_blocks:
        # l2_norm = 0
        l2_norm = np.sum(np.square(block - img_segment) * (img_segment >= 0)) 
        # if row > 0: # there is overlap on top
        #     l2_norm += np.sum(np.square(block[:overlap,:,:] - img_segment[:overlap,:,:]))
        # if col > 0: # there is overlap on the left
        #     l2_norm += np.sum(np.square(block[:,:overlap,:] - img_segment[:,:overlap,:]))
        # if row > 0 and col > 0:
        #     l2_norm -= np.sum(np.square(block[:overlap,:overlap,:] - img_segment[:overlap,:overlap,:]))
        l2_norms.append(l2_norm)
    
    best_norm = min(l2_norms)
    tolerance = 0.1 * best_norm # tolerance: must be within 0.1 times of the best overlapping block's error
    good_blocks = []
    for i, block in enumerate(downscaled_blocks):
        if l2_norms[i] <= best_norm + tolerance:
            good_blocks.append(block)
    selected_block =  get_random_block(good_blocks)
    return selected_block, np.sum(np.square(selected_block - img_segment) * (img_segment >= 0), axis=-1)

def min_err_boundary_cut(overlap_img: np.ndarray) -> np.ndarray:
    print("min err boundary cut")
    # Perform seamcarve
    height, width = overlap_img.shape
    for row in range(1, height):
        for col in range(width):
            if col == 0:
                overlap_img[row,col] += min(overlap_img[row-1, col], overlap_img[row-1,col+1])
            elif col == width - 1:
                overlap_img[row,col] += min(overlap_img[row-1, col], overlap_img[row-1,col-1])
            else:
                overlap_img[row,col] += min(overlap_img[row-1,col-1], overlap_img[row-1, col], overlap_img[row-1,col+1])
    
    # Get the path
    mask = np.zeros((height, width))
    min_idx = 0
    for i in range(height-1, -1, -1):
        if i == height - 1:
            min_idx = np.argmin(overlap_img[i,:])
            row_mask = [1 for _ in range(min_idx)] + [0 for _ in range(width - min_idx)]
            mask[i,:] = np.array(row_mask)
        else:
            if min_idx == 0:
                check_vals = overlap_img[i,min_idx:min_idx+2]
            elif min_idx == width - 1:
                check_vals = overlap_img[i, min_idx-1:]
            else:
                check_vals = overlap_img[i, min_idx-1:min_idx+2]
            if min_idx != 0:
                min_idx -= 1
            min_idx = min_idx + np.argmin(check_vals)
            row_mask = [1 for _ in range(min_idx)] + [0 for _ in range(width - min_idx)]
            mask[i,:] = np.array(row_mask)
    return mask.astype(np.uint8)


def quilt(block_size: int, texture_path: str, scale: float):
    texture = imread(texture_path)[:,:,:3] # NOTE: removing alpha channel if it exists
    
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
    for row in range(0, outh, block_size - overlap):
        remainingY = outh - row
        print(row)
        for col in range(0, outw, block_size - overlap):
            if row == 0 and col == 0:
                continue
            remainingX = outw - col
            img_segment = quilted_img[row: row + min(block_size, remainingY), col: col + min(block_size, remainingX)]
            selected_block, overlap_error = find_good_block(img_segment, texture_blocks, overlap, row, col)

            # min_err_boundary_cut (seamcarve) and then mould selected_block based on that seam and place into quilted_img
            if row == 0: # overlap only on the left
                overlap_error = overlap_error[:, :overlap]
                mask = min_err_boundary_cut(overlap_error)
                mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
                # previously empty (-100) image segment becomes 0:
                img_segment[:,overlap:] = 0
                # between path and overlap vertical line becomes 0 in image segment:
                img_segment[:,:overlap] *= mask
                # block to the left of the path becomes 0:
                flipped_mask = mask + (mask != 1) - (mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block[:,:overlap] *= flipped_mask
                img_segment += ragged_block
                # pass copy into min_err_boundary_cut
            elif col == 0: # overlap only on top
                overlap_error = overlap_error[:overlap, :]
                mask = min_err_boundary_cut(overlap_error.T).T
                mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
                # previously empty (-100) image segment becomes 0
                img_segment[overlap:,:] = 0
                # between path and overlap horizontal line becomes 0 in image segment:
                img_segment[:overlap, :] *= mask
                # block above path becomes 0:
                flipped_mask = mask + (mask != 1) - (mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block[:overlap,:] *= flipped_mask
                img_segment += ragged_block
            else: # overlap on top and left
                overlap_error_left = overlap_error[:, :overlap]
                mask_left = min_err_boundary_cut(overlap_error_left)
                mask_left = np.repeat(mask_left[:,:,np.newaxis], 3, axis=2)
                overlap_error_top = overlap_error[:overlap, :]
                mask_top = min_err_boundary_cut(overlap_error_top.T).T
                mask_top = np.repeat(mask_top[:,:,np.newaxis], 3, axis=2)
                # previously empty (-100) image segment becomes 0
                img_segment[overlap:, overlap:] = 0
                # apply masks to img_segment
                total_mask = np.zeros_like(img_segment)
                total_mask[:, :overlap] += mask_left
                total_mask[:overlap, :] += mask_top
                total_mask[total_mask > 1] = 1
                img_segment *= total_mask
                total_mask_flipped = total_mask + (total_mask != 1) - (total_mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block *= total_mask_flipped.astype(np.uint8)
                img_segment += ragged_block

            if remainingX <= block_size:
                break # make sure to break if we need to fill a smaller-than-block sized portion at the end
        if remainingY <= block_size:
            break # make sure to break if we need to fill a smaller-than-block sized portion at the end
    
    imshow(np.uint8(quilted_img))
    plt.savefig('result.png')
    plt.show()
    return quilted_img
    
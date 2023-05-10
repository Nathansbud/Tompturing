from skimage.io import imread, imshow
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List

# Declaring as global variable so we don't have to pass this as a function argument
# (large size so saves space on stack --> enhances speed)
texture_blocks = None

def get_intensity_map(img: np.ndarray):
    """Returns an intensity map for the provided image"""
    return rgb2gray(img)

def get_orientation_angle_map(img: np.ndarray):
    """Returns a map of orientation angles (gradients) for the provided image"""
    return np.gradient(img)

def get_correspondence_function(key: str):
    """Returns the function which can create a correspondence map based on the desired type of
    correspondence (specified in the key)"""
    if key == "intensity": return get_intensity_map
    elif key == "orientation_angles": return get_orientation_angle_map
    elif not key:
        raise Exception(f"{key} is not a valid correspondence function!")

def get_texture_blocks(texture: np.ndarray, block_size: int):
    """Generates a list of blocks (of size block_size) from the input texture and stores them in texture_blocks"""
    print("get texture blocks")
    th, tw, tc = texture.shape
    global texture_blocks
    texture_blocks = [texture[i:i+block_size, j:j+block_size,:] for j in range(tw - block_size) for i in range(th - block_size)]
    # print(len(texture_blocks))

def get_random_block(blocks: List[np.ndarray]):
    """Returns a random block out of the provided list of blocks"""
    index = np.random.randint(len(blocks))
    return blocks[index]

def get_top_left_block(block_size: int, transfer_img_section: np.ndarray, correspondence: str):
    """Identifies the texture block to be used at the top-left location of the constructed result"""
    print("get top left block")
    correspondence_function = get_correspondence_function(correspondence)

    transfer_correspondence_map = correspondence_function(transfer_img_section)
    l2_norms = []
    for block in texture_blocks:
        block_correspondence_map = correspondence_function(block)
        l2_norm = np.sum(np.square(block_correspondence_map - transfer_correspondence_map))
        l2_norms.append(l2_norm)
    best_norm_index = np.argmin(l2_norms)
    return texture_blocks[best_norm_index]

def find_good_block(transfer_result_segment: np.ndarray, transfer_img_segment: np.ndarray, correspondence: str, alpha: float, iter_num: int, overlap: int, row: int, col: int):
    """Finds the best block to place at the specified location of the constructed result based on
    the constraints defined with the overlap error, correspondence error, and overlay error.
    Returns this block along with an error map for the overlapping section."""
    print("find good block")
    # NOTE: transfer_result_segment represents the portion of the transfer result that is currently
    # being constructed, for which we are trying to find a good block. On the other hand,
    # transfer_img_segment represents the corresponding portion of the transfer (target) image
    # provided by the user.
    h, w, c = transfer_result_segment.shape
    downscaled_blocks = [np.copy(block)[:h, :w, :c] for block in texture_blocks]
    correspondence_function = get_correspondence_function(correspondence)
    transfer_img_segment_correspondence_map = correspondence_function(transfer_img_segment)
    errors = []
    for block in downscaled_blocks:
        overlap_error = 0
        if row > 0: # there is overlap on top
            overlap_error += np.sum(np.square(block[:overlap,:,:] - transfer_result_segment[:overlap,:,:]))
        if col > 0: # there is overlap on the left
            overlap_error += np.sum(np.square(block[:,:overlap,:] - transfer_result_segment[:,:overlap,:]))
        if row > 0 and col > 0: # make sure not to double count the top-left corner's overlap
            overlap_error -= np.sum(np.square(block[:overlap,:overlap,:] - transfer_result_segment[:overlap,:overlap,:]))
        block_correspondence_map = correspondence_function(block)
        correspondence_error = np.sum(np.square(block_correspondence_map - transfer_img_segment_correspondence_map))
        if iter_num > 0:
            previous_overlay_error = np.sum(np.square(block[overlap:, overlap:, :] - transfer_result_segment[overlap:, overlap:, :]))
            error = (alpha * (overlap_error + previous_overlay_error)) + ((1 - alpha) * correspondence_error)
        else:
            error = (alpha * overlap_error) + ((1 - alpha) * correspondence_error)
        errors.append(error)

    best_error_index = np.argmin(errors)
    selected_block = downscaled_blocks[best_error_index]

    error_map = np.sum(np.square(selected_block - transfer_result_segment) * (transfer_result_segment > 0), axis=-1)
    return selected_block, error_map


def min_err_boundary_cut(overlap_img: np.ndarray) -> np.ndarray:
    """Identifies the minimum error cut across the provided overlap segment. Then constructs a
    path for that cut and generates (and returns) a boolean mask which can be used to """
    print("min err boundary cut")
    # Find minimum error boundary cut
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


def transfer(block_size: int, texture_img: np.ndarray, transfer_img: np.ndarray, alpha: float, iter_num: int, correspondence: str, prev_result: np.ndarray):
    """Performs one iteration of the transfer process"""
    th, tw, tc = texture_img.shape
    outh, outw, outc = transfer_img.shape

    if not (0 < block_size < th and 0 < block_size < tw):
        print(f"Block size must be between 0 and texture min dimension; {block_size} not within [0, {min(th, tw)}]")
        exit(1)
    
    transfer_result = prev_result
    get_texture_blocks(texture_img, block_size)

    transfer_result[:block_size, :block_size, :] = get_top_left_block(block_size, transfer_img[:block_size, :block_size, :], correspondence)
    
    overlap = math.ceil(block_size / 6)
    for row in range(0, outh, block_size - overlap):
        remainingY = outh - row
        print(row)
        for col in range(0, outw, block_size - overlap):
            if row == 0 and col == 0:
                continue
            remainingX = outw - col
            
            transfer_result_segment = transfer_result[row: row + min(block_size, remainingY), col: col + min(block_size, remainingX)]
            transfer_img_segment = transfer_img[row: row + min(block_size, remainingY), col: col + min(block_size, remainingX)]
            selected_block, overlap_error = find_good_block(transfer_result_segment, transfer_img_segment, correspondence, alpha, iter_num, overlap, row, col)

            if row == 0: # overlap only on the left
                overlap_error = overlap_error[:, :overlap]
                mask = min_err_boundary_cut(overlap_error)
                mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
                transfer_result_segment[:,overlap:] = 0
                transfer_result_segment[:,:overlap] *= mask
                flipped_mask = mask + (mask != 1) - (mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block[:,:overlap] *= flipped_mask
                transfer_result_segment += ragged_block
            elif col == 0: # overlap only on top
                overlap_error = overlap_error[:overlap,:]
                mask = min_err_boundary_cut(overlap_error.T).T
                mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
                transfer_result_segment[overlap:,:] = 0
                transfer_result_segment[:overlap, :] *= mask
                flipped_mask = mask + (mask != 1) - (mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block[:overlap, :] *= flipped_mask
                transfer_result_segment += ragged_block
            else: # overlap on top and left
                overlap_error_left = overlap_error[:, :overlap]
                mask_left = min_err_boundary_cut(overlap_error_left)
                mask_left = np.repeat(mask_left[:,:,np.newaxis], 3, axis=2)
                overlap_error_top = overlap_error[:overlap, :]
                mask_top = min_err_boundary_cut(overlap_error_top.T).T
                mask_top = np.repeat(mask_top[:,:,np.newaxis], 3, axis=2)
                # previously empty (-100) image segment becomes 0
                transfer_result_segment[overlap:, overlap:] = 0
                # apply masks to img_segment
                total_mask = np.zeros_like(transfer_result_segment)
                total_mask[:, :overlap] += mask_left
                total_mask[:overlap, :] += mask_top
                total_mask[total_mask > 1] = 1
                transfer_result_segment *= total_mask
                total_mask_flipped = total_mask + (total_mask != 1) - (total_mask == 1)
                ragged_block = np.copy(selected_block)
                ragged_block *= total_mask_flipped.astype(np.uint8)
                transfer_result_segment += ragged_block

            if remainingX <= block_size:
                break
        if remainingY <= block_size:
            break
    imshow(transfer_result)
    plt.show()
    return transfer_result


def iterative_transfer(block_size: int, texture_path: str, transfer_path: str, correspondence: str, num_iters: int=2):
    """Iteratively performs the texture transfer process"""
    # NOTE: removing alpha channel (if present)
    texture_img = imread(texture_path)[:,:,:3] / 255.0
    transfer_img = imread(transfer_path)[:,:,:3] / 255.0 if transfer_path else None

    print(f"texture shape: {texture_img.shape}")
    print(f"transfer shape: {transfer_img.shape}")

    ## Iteratively running the transfer function:
    transfer_result = np.zeros_like(transfer_img)
    for iter_num in range(num_iters):
        print(f"Iteration: {iter_num} of {num_iters-1}")
        ALPHA = (0.8 * iter_num / (num_iters-1)) + 0.1
        if iter_num > 0:
            block_size = int(block_size * 0.67)
        print(f"block_size: {block_size}")
        transfer_result = transfer(block_size, texture_img, transfer_img, ALPHA, iter_num, correspondence, transfer_result)

    imshow(transfer_result)
    plt.savefig('transfer_result.png')
    plt.show()
    return transfer_result

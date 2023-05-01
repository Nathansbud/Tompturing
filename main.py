import argparse
import os
import sys

from synthesis import quilt
from transfer import iterative_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Tompturing: Quilting for Texture Synthesis & Transfer")
    parser.add_argument("-t", "--texture", default=None, help="Texture to quilt")
    parser.add_argument("-d", "--transfer", default=None, help="Image to transfer texture to")

    parser.add_argument("-b", "--block_size", metavar="n", nargs=1, default=16, type=int, help="Block size (n x n)")
    parser.add_argument("-s", "--scale", default=2.0, nargs=1, type=float, help="Scaling factor to apply while quilting")
    parser.add_argument("-c", "--correspondence", choices=["luminance", "intensity"], help="Correspondence function to use")
    parser.add_argument("-a", "--alpha", default=1.0, help="Alpha to use for correspondence blending")
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    texture, transfer = args.texture, args.transfer
    
    scale = args.scale
    if scale:
        try:
            scale = float(scale)
        except ValueError as e:
            print("Provided scale must be a floating point value")
            exit(1)
    
    if texture and transfer:
        if os.path.isfile(texture) and os.path.isfile(transfer):
            iterative_transfer(args.block_size[0], texture, transfer, args.correspondence)
        else:
            print("Provided texture and transfer images must be valid filepaths!")
    elif texture:
        if os.path.isfile(texture): 
            quilt(args.block_size[0], texture, args.scale)
        else: 
            print("Provided texture must be a valid filepath!")
    elif transfer:
        print("A texture must be provided to perform style transfer!")

import argparse
import os
import sys

from synthesis import quilt, quilt_and_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Tompturing: Quilting for Texture Synthesis & Transfer")
    parser.add_argument("-t", "--texture", default=None, help="Texture to quilt")
    parser.add_argument("-d", "--transfer", default=None, help="Image to transfer texture to")

    parser.add_argument("-b", "--block_size", metavar="n", nargs=1, default=16, type=int, help="Block size (n x n)")

    # we might want user parameters of error and error tolerance
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    texture, transfer = args.texture, args.transfer
    
    if texture and transfer:
        if os.path.isfile(texture) and os.path.isfile(transfer):
            quilt_and_transfer(args.block_size[0], texture, transfer)
        else:
            print("Provided texture and transfer images must be valid filepaths!")
    elif texture:
        if os.path.isfile(texture): 
            quilt(args.block_size[0], texture)
        else: 
            print("Provided texture must be a valid filepath!")
    elif transfer:
        print("A texture must be provided to perform style transfer!")

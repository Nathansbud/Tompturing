# Tompturing

Implementation of [Image Quilting for Texture Synthesis and Transfer](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) (Efros & Freeman, 2001) for CSCI 1430

## Running the code

The code for this project can be run by running `main.py` from the command line. The command line interface is defined as:

```
usage: main.py [-h] [-t TEXTURE] [-d TRANSFER] [-b n] [-s SCALE]
               [-c {intensity,orientation_angles}] [-a ALPHA]

Tompturing: Quilting for Texture Synthesis & Transfer

optional arguments:
  -h, --help            show this help message and exit
  -t TEXTURE, --texture TEXTURE
                        Texture to quilt
  -d TRANSFER, --transfer TRANSFER
                        Image to transfer texture to
  -b n, --block_size n  Block size (n x n)
  -s SCALE, --scale SCALE
                        Scaling factor to apply while quilting
  -c {intensity,orientation_angles}, --correspondence {intensity,orientation_angles}
                        Correspondence function to use
```

Therefore, an example of how one might run the code for texture transfer with a block size of 20:

```
python main.py -t <path_to_input_texture> -d <path_to_target_image> -b 20
```

## Report

Check out our [final report](https://drive.google.com/file/d/1wgyOiR6VMBXk40bdzidyG8-Dhfxml5zR/view?usp=sharing) (which also contains many sample results)!

## Credits

This project is aimed effectively blending tilings of arbitrary textures, hence we use images from a variety of sources. Credit for these textures (housed under `textures/`) is attributed to free and non-commercial use repositories, including: [texturelib](http://texturelib.com/) (TL), [Paul Bourke](http://paulbourke.net/) (PB), and several others.

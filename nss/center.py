#!/usr/bin/env python3
'''
Nick's star stacker (nss) originally written in python on 2017-09-02

A brute force image processing method to take star fields shot on a tripod and
align the star fields to the reference starfield.  This imitates what Nick was
doing by hand in Photoshop.

-------------------------------------------------------------------------------
Key terms:


target    - the target image, the other images are manipulated to align with
            this image

src_image - a source image, it is broken up into smaller sub-images called
            patches and are translated and rotate to find the best fit against
            the target

patch     - a sub-image from the target image, it is translated and rotated
            until the best fit is found against the reference image

radius    - the radius of a patch in pixels, a patch is always 2 * r + 1 pixels
            in width and height

aligned   - the final aligned output image made out of the patches

parameter
cache     - a cache for patch parameters (src_pos, rotation, translation), these
            are used as starting parameters for the gradient ascent search for
            best fit.  By starting nearby parameters, the gradient ascent will
            find best fit in fewer iterations.

m,n       - matrix notation, m = row index, n = column index, 0,0 is the upper
            left element in the matrix, 5,5 is lower right

-------------------------------------------------------------------------------
The algorithm

step 1 - coarse parameter search

The first step is the do exhaustive alignments with patches on a very coarse
grid between the source image and the target.  These parameters are stored and
used for the next step as starting parameters

set 2 - fine patch alignment

Given the coarse grid parameters, break the src image into small patches and
perform a gradient ascent search for best fit, start the search with the
coarse parameters at the starting location
'''

import argparse
import copy
import os
import random
import sys

import matplotlib.pyplot as plt

from nss import utils

def make_centered_filename(filename):
    """
    In:  C:\some\path\to\image.tif
    OUT: C:\some\path\to\centered\image.tif
    """
    basename = os.path.basename(filename)
    dir_prefix = os.path.dirname(filename)
    return os.path.join(dir_prefix, "centered", basename)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--plot",
        action = "store_true",
        help = "plot a summary at the end"
    )

    parser.add_argument(
        "-s",
        "--size",
        default = 121,
        type = int,
        help = "The grid size used to detect the moon."
    )

    parser.add_argument(
        "-o",
        "--output",
        default = None,
        type = str,
        help = "The output image to write."
    )

    parser.add_argument(
        "source",
        default = None,
        type = str,
        help = "The source image to be aligned with the target"
    )

    args = parser.parse_args()

    source_fn = args.source

    with utils.timeit("Reading images ...\n"):
        source = utils.TiffFile().read(source_fn)
        print(f"    Source: {source_fn}")

    with utils.timeit("Detecting moon in source: "):
        source_circle = utils.detect_moon(source.array, args.size, plot=args.plot)
        print(f"Source: {source_circle}")
        if args.plot:
            plt.title("Source Image")
            plt.show()
        
    M, N = source.array.shape[0:2]
    delta_m = M / 2.0 - source_circle.center[0]
    delta_n = N / 2.0 - source_circle.center[1]
    source.array = utils.apply_alignment(source.array, delta_m, delta_n, 0.0, source_circle.center)
    source_circle = utils.Circle(center=(M / 2.0, N / 2.0), radius=source_circle.radius)
    centered_fn = args.output
    if centered_fn is None:
        centered_fn = make_centered_filename(source_fn)
    source.saveas(centered_fn)
    print(f"Wrote centered image: {centered_fn}")

if __name__ == "__main__":
    main()

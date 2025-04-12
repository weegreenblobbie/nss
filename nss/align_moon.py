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
import datetime
import os
import pickle
import random
import sys
import warnings

import PIL.Image
import scipy
import scipy.signal
import skimage

from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.signal import correlate2d

import scipy.ndimage
from   scipy.ndimage import median_filter as median

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tifffile
import tifffile
from tifffile import imwrite
from tifffile import TiffFile

from nss import utils

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--cache",
        default = None,
        type = str,
        help = "Reads a previously computed coarse alignment cache file."
    )

    parser.add_argument(
        "-t",
        "--target",
        required = True,
        type = str,
        help = "Specifies the target image that all other images are aligned to"
    )

    parser.add_argument(
        "-m",
        "--mask",
        type = str,
        help = "Specifies the mask image, pixels that are black are ignored during processing"
    )

    parser.add_argument(
        "-o",
        "--output",
        type = str,
        default = None,
        help = "Specifies the aligned output image name, default is {prefix}-aligned.tiff"
    )

    parser.add_argument(
        "-p",
        "--plot",
        action = "store_true",
        default = False,
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
        "source",
        type = str,
        help = "the source image to be aligned with the target"
    )

    args = parser.parse_args()

    target_fn = args.target
    source_fn = args.source
    mask_fn   = args.mask
    coarse_cache_fn = args.cache
    output_fn = args.output

    assert os.path.isfile(target_fn), "Could not find file " + target_fn
    assert os.path.isfile(source_fn), "Could not find file " + source_fn

    assert target_fn != source_fn, "Target and Source are the same file!"

    if mask_fn:
        assert os.path.isfile(mask_fn), "Could not find file " + mask_fn

    if output_fn is None:
        path = os.path.dirname(source_fn)
        prefix = os.path.splitext(os.path.basename(source_fn))[0]
        output_fn = os.path.join(path, prefix + "-aligned.tiff")

    mask = None

    with utils.timeit("Reading images ...\n"):
        utils.log(f"    Target: {target_fn}\n")
        with TiffFile(target_fn) as tif:
            target = tif.asarray().astype(np.float32)
            target_tags = tif.pages[0].tags
        utils.log(f"    Source: {source_fn}\n")
        with TiffFile(source_fn) as tif:
            source = tif.asarray().astype(np.float32)
            colormap = tif.pages.first.colormap
            photometric = tif.pages.first.photometric
            iccprofile = tif.pages.first.iccprofile

    assert target.shape == source.shape, f"Image shapes don't match, can not align: {target.shape} != {source.shape}"

    with utils.timeit("Detecting moon in target ... "):
        # TODO: read target location from cache.
        target_circle = utils.detect_moon(target, args.size, plot=args.plot)
        utils.log(f"{target_circle} ")
        if args.plot:
            plt.title("Target Image")

    with utils.timeit("Detecting moon in source ... "):
        # TODO: read target location from cache.
        source_circle = utils.detect_moon(source, args.size, plot=args.plot)
        utils.log(f"{source_circle} ")
        if args.plot:
            plt.title("Source Image")

    if args.plot:
        plt.show()
        utils.log("Quitting after showing plots.")
        return

    with utils.timeit("Aligning images\n"):
        alignment = utils.align_moon_images(
            target,
            source,
            target_circle,
            source_circle
        )
        print(f"final score: {alignment.score:.5g}")

    # Apply alignment.
    delta_m, delta_n = alignment.heading
    angle = round(alignment.angle, 4)
    print(f"Applying:\n    translate: {delta_m},{delta_n}, rotate: {angle}")
    aligned = utils.apply_alignment(source, delta_m, delta_n, angle, target_circle.center)

    with utils.timeit(f"Saving {output_fn} ... "):
        aligned -= np.nanmin(aligned)
        aligned /= np.nanmax(aligned)
        aligned *= (1 << 16) -1
        aligned = aligned.astype(np.uint16)

        imwrite(output_fn, aligned,
            photometric=photometric,
            colormap=colormap,
            iccprofile=iccprofile,
            software="https://github.com/weegreenblobbie/nss",
        )


if __name__ == "__main__":
    main()

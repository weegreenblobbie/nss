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

from nss import utils

def make_aligned_filename(filename):
    prefix, _ = os.path.splitext(filename)
    return prefix + "-aligned.tiff"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--center",
        action="store_true",
        help = "Centers the moon in the target image before aligning."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-t",
        "--target",
        default = None,
        type = str,
        help = "Specifies the target image that all other images are aligned to"
    )

    group.add_argument(
        "-u",
        "--use-target-cache",
        action="store_true",
        help = "Uses the previously saved target cache."
    )

    parser.add_argument(
        "--save-target-cache",
        action="store_true",
        help = "Preprocesses the target image and saves the results locally."
    )

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
        "source",
        default = None,
        nargs = "?",
        type = str,
        help = "the source image to be aligned with the target"
    )

    args = parser.parse_args()

    target_fn = args.target
    source_fn = args.source

    if args.target and args.source:
        assert target_fn != source_fn, "Target and Source are the same file!"

    if args.use_target_cache:
        assert args.save_target_cache is False

    if args.save_target_cache:
        assert args.use_target_cache is False

    with utils.timeit("Reading images ...\n"):

        if args.use_target_cache:
            target = utils.TiffFile()
            data = utils.read_dict("target.pkl")
            target.array = data["array"]
            target_circle = data["circle"]
            target_fn = data["target_fn"]
        else:
            target = utils.TiffFile().read(target_fn)

        print(f"    Target: {target_fn}")
        if not args.save_target_cache:
            print(f"    Source: {source_fn}")
            source = utils.TiffFile().read(source_fn)

    if not args.use_target_cache:
        with utils.timeit("Detecting moon in target: "):
            target_circle = utils.detect_moon(target.array, args.size, plot=args.plot)
            print(f"Target: {target_circle}")
            if args.plot:
                plt.title("Target Image")

        if args.center:
            M, N = target.array.shape[0:2]
            delta_m = M / 2.0 - target_circle.center[0]
            delta_n = N / 2.0 - target_circle.center[1]
            target.array = utils.apply_alignment(target.array, delta_m, delta_n, 0.0, target_circle.center)
            target_circle = utils.Circle(center=(M / 2.0, N / 2.0), radius=target_circle.radius)
            centered_fn = make_aligned_filename(target_fn)
            target.saveas(centered_fn)
            print(f"Wrote centered target: {centered_fn}")

    else:
        print("Detected moon in target:")
        print(f"Target: {target_circle}")

    # TODO: Derive this from the detected moon radius.
    window_size = 101

    if args.save_target_cache:
        utils.log("Preprocessing target image: ")
        target.array = utils.align_moon_image_preprocess(target.array, target_circle, window_size, "target")
        utils.save_dict("target.pkl", dict(array=target.array, circle=target_circle, target_fn=target_fn))
        print("Wrote target.pkl")
        utils.log("Quitting caching target data.")
        return

    with utils.timeit("Detecting moon in source: "):
        source_circle = utils.detect_moon(source.array, args.size, plot=args.plot)
        print(f"Source: {source_circle}")
        if args.plot:
            plt.title("Source Image")

    if args.plot:
        plt.show()
        utils.log("Quitting after showing plots.")
        return

    with utils.timeit("Aligning images\n"):
        alignment = utils.align_moon_images(
            target.array,
            source.array,
            target_circle,
            source_circle,
            window_size,
            target_preprocessed=args.use_target_cache,
        )
        print(f"final score: {alignment.score:.5g}")

    # Apply alignment.
    delta_m, delta_n = alignment.heading
    angle = round(alignment.angle, 4)
    print(f"Applying:\n    translate: {delta_m},{delta_n}, rotate: {angle}")
    source.array = utils.apply_alignment(source.array, delta_m, delta_n, angle, target_circle.center)

    output_fn = make_aligned_filename(source_fn)
    with utils.timeit(f"Saving {output_fn} ... "):
        source.saveas(output_fn)

if __name__ == "__main__":
    main()

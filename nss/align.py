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
import tifffile.tifffile
from tifffile.tifffile import imread, imwrite

SCALE = 1.0

RADIUS = int(SCALE * 200)

COARSE_CACHE_PKL = 'coarse_param_cache.pkl'


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
        "--moon",
        action = "store_true",
        help = "Optimize processing for moon images"
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
        "-r",
        "--radius",
        default = RADIUS,
        type = int,
        help = "The patch radius to use for alignment"
    )

    parser.add_argument(
        "-s",
        "--save-cache",
        action = "store_true",
        default = False,
        help = "saves the coarse param search object"
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

    if coarse_cache_fn:
        assert os.path.isfile(coarse_cache_fn), "Could not find file " + coarse_cache_fn

    if output_fn is None:
        path = os.path.dirname(source_fn)
        prefix = os.path.splitext(os.path.basename(source_fn))[0]
        output_fn = os.path.join(path, prefix + "-aligned.tiff")

    mask = None

    with timeit("Reading images ... "):
        print(f"target = {target_fn}")
        target = imread(target_fn).astype(np.float32)
        source = imread(source_fn).astype(np.float32)

        assert target.shape == source.shape

        if mask_fn:
            mask = imread(mask_fn).astype(np.float32)
            mask = mask[:,:, 0]

#~    with timeit():
#~        r = coarse_align_radius_search(target)

    with timeit("Coarse alignment parameter search ... "):

        if coarse_cache_fn and os.path.isfile(coarse_cache_fn):
            log("\n    reading coarse align cache %s ..." % coarse_cache_fn)

            with open(coarse_cache_fn, 'rb') as fd:
                param_cache = pickle.load(fd)
                log("\n")

        else:

            param_cache = coarse_align_param_search(
                target,
                source,
                mask,
                args.radius,
                moon=args.moon,
            )

            if args.save_cache:
                log("\n    writing %s ..." % COARSE_CACHE_PKL)
                with open(COARSE_CACHE_PKL, 'wb') as fd:
                    pickle.dump(param_cache, fd)
                log("\n")

        print_param_cache(param_cache)

    radius = args.radius

    with timeit("Aligning images\n"):
        target, source, aligned = align_images(
            target,
            source,
            mask,
            radius,
            param_cache
        )

    with timeit(f"Saving {output_fn} ... "):
        aligned.savetiff(output_fn)

    #-------------------------------------------------------------------------
    # Compute final fit

    tgt = unpad_array(target, radius)
    s = unpad_array(to_gray(aligned.get_rgb()), radius)
    fit = xcorr_score(tgt, s)
    print("Final fit = %.3f" % fit)

    if args.plot:


        plt.figure()
        imshow(tgt)
        plt.title('target')

#~        src = unpad_array(source, radius)
#~        plt.figure()
#~        imshow(src)
#~        plt.title('src')

        plt.figure()
        imshow(s)

        plt.title('aligned.rgb %s, fit = %.3f' % (repr(s.shape), fit))

        c = unpad_array(aligned._count, radius)

        plt.figure()
        imshow(c)
        plt.title('aligned.counts')

        valid = (~np.isnan(tgt)) * (~np.isnan(s))

        tgt = zscore(tgt, valid)
        s   = zscore(s, valid)

        diff = abs(tgt - s)
        diff[~valid] = 0.0

        plt.figure()
        imshow(diff)
        plt.title('fit error: min=%.4f, max=%.4f' % (np.nanmin(diff), np.nanmax(diff)))

        plt.show()


def imresize(array, scale):
    """
    resizes an image
    """

    in_dtype = array.dtype

    new_shape = list(array.shape)

    for i in [0, 1]:
        new_shape[i] = int(new_shape[i] * scale)

    anti_aliasing = scale < 1.0

    out = skimage.transform.resize(array, new_shape, anti_aliasing = anti_aliasing)

    assert out.dtype in [np.float32, np.float64], \
        f"output array dtype changed by imresize! ({in_dtype} != {out.dtype})"

    return out.astype(in_dtype)


def align_images(target, source, mask, radius, param_cache):

    if SCALE > 1.0:
        with timeit(f"    scaling by {SCALE} ..."):

            param_cache = param_cache_resize(param_cache, SCALE)

            target = imresize(target, SCALE).astype(np.float32)
            source = imresize(source, SCALE).astype(np.float32)

            if mask is not None:

                mask = imresize(mask, SCALE).astype(np.float32)

                #----------------------------------------------------------------------
                # convert the mask into a boolean mask

                peak = np.max(mask)

                invalid = mask < peak

                # write nans into the black (masked out) region of the target image

                target[invalid] = np.nan

    #--------------------------------------------------------------------------
    # convert and pad images

    target = pad_array(to_gray(target), radius)
    source = pad_array(source, radius)

    # working aligned buffer

    aligned = Aligned(source.shape)

    M, N = target.shape[0:2]

    width = 2 * radius + 1

    m_remainder = M - width
    n_remainder = N - width

    num_m = (m_remainder + width -1) // width + 1
    num_n = (n_remainder + width -1) // width + 1

    m_axis = np.linspace(radius, M - radius - width, num_m).astype(np.uint32)
    n_axis = np.linspace(radius, N - radius - width, num_n).astype(np.uint32)

    # DEBUG show how start windows overlap

    if False:

        tmp = np.zeros(target.shape, np.float32)

        valid = ~np.isnan(target)

        tmp[valid] = 1.0

        for m in m_axis:
            for n in n_axis:

                m0 = m
                m1 = m0 + width

                n0 = n
                n1 = n0 + width

                tmp[m0 : m1, n0 : n1] = 2.0

        plt.figure()
        imshow(tmp)
        plt.title("align_images patches")

        plt.show()

        xxxxxxxxxx

    patch_count = len(m_axis) * len(n_axis)

    size = 2 * radius + 1

    jobs = []

    for m in m_axis:
        for n in n_axis:
            jobs.append( (m,n) )

    np.random.shuffle(jobs)

    count = 0

    while len(jobs) > 0:

        count += 1

        m, n = jobs.pop()

        log("    patch %3d / %3d " % (count, patch_count))

        m0 = m
        m1 = m + size
        n0 = n
        n1 = n + size

        src = source[m0 : m1, n0 : n1, :]

        src_pos = (m,n)
        src_off = 0, 0
        angle = 0.0

        c = cache_lookup(param_cache, (m-radius,n-radius))

        if c:
            src_off = c['heading']
            angle = c['angle']

        patch, best = gradient_ascent(
            src,
            target,
            src_pos,
            src_off,
            angle
        )

        if best.score > 0.500:
            param_cache[(m-radius,n-radius)] = dict(
                angle = best.angle,
                heading = best.heading,
                score = best.score,
            )

        aligned.add_patch(patch)

    with open('param_cache.pkl', 'wb') as fd:
        pickle.dump(param_cache, fd)
    log("\n")

    return target, source, aligned


def gradient_ascent(
        src,
        tgt,
        src_pos,
        src_off,
        angle,
        min_iterations=36,
        max_iterations=72,
        min_score=1.0,
        show_pdone=False,
        count=None,
        max_count=None,
    ):

    d0 = datetime.datetime.now()

    patch = Patch(src, src_pos)

    da = 0.050  # delta angle
    da_pool = np.linspace(-3*da, 3*da, 7)

    m_off, n_off = src_off

    best = None

    it = 0 # iterations

    score = 0.0

    # Perform at least the minimum number of iterations and no more than the
    # maximum.  If the score goes above the minimum alignment score bail out
    # early.
    while it < min_iterations or (score < min_score and it < max_iterations):

        it += 1

        m, n = src_pos

        m -= m_off
        n -= n_off

        if m < 1:
            m = 1
            m_off +=1

        if n < 1:
            n = 1
            n_off += 1

        if best is None:
            best = patch.score(tgt, (m    , n), angle)

        scores = [best]

        # Translate in the 8 directions.

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                scores.append(patch.score(tgt, (m + i, n + j), angle))

        # Rotate the patch about it's center

        scores.append(patch.score(tgt, (m, n), angle - da))
        scores.append(patch.score(tgt, (m, n), angle + da))

        # Shift and rotate the patch randomly.

        rm = np.random.randint(-3,3)
        rn = np.random.randint(-3,3)
        ra = np.random.choice(da_pool)

        m0 = m + rm
        n0 = n + rn

        if m0 < 0:
            m0 = 0
        if n0 < 0:
            n0 = 0

        scores.append(patch.score(tgt, (m0, n0), angle + ra))

        # Find best score and parameters

        best = max(scores)

        angle = best.angle
        m_off, n_off = best.heading #(-best.heading[0], -best.heading[1])

        score = best.score

        if show_pdone:
            count += 1
            pdone = 100.0 * count / max_count
            log('\b\b\b\b\b\b%5.1f%%' % pdone)
#~            for s in scores:
#~                log("%s\n" % s)

#~        print(
#~            "Step %2d: score = %.3f, da = %.3f, angle = %6.3f, heading = %3d,%3d" % (
#~                k+1,
#~                peak.score,
#~                da,
#~                peak.angle,
#~                peak.heading[0],
#~                peak.heading[1],
#~            )
#~        )

#~    print(
#~        "Final  : score = %.3f, da = %.3f, angle = %6.3f, heading = %3d,%3d" % (
#~            peak.score,
#~            da,
#~            peak.angle,
#~            peak.heading[0],
#~            peak.heading[1],
#~        )
#~    )

    if not show_pdone:
        d1 = datetime.datetime.now()
        log(
            "score = %6.3f, heading (%3d,%3d), angle %7.4f, %2d iterations, took %5.2f seconds\n" % (
                best.score,
                best.heading[0],
                best.heading[1],
                best.angle,
                it,
                (d1 - d0).total_seconds(),
            )
        )

    patch.set_final_params(best.angle, best.tgt_pos)

    return patch, best


def plot_patches(data, windows, show=True, title="windows"):

    data = np.array(data)
    data[np.isnan(data)] = 0.0
    data **= 0.5
    black = "black"
    white = "white"
    red = "red"

    plt.figure(facecolor=black, figsize=(16,8))
    plt.title(title, color=white)
    ax = plt.gca()
    imshow(data)

    for win in windows:
        m0, n0 = win["pos"]
        size = win["size"]
        rect = plt.Rectangle((m0, n0), size, size, ls='-', color=white, fill=False)
        ax.add_patch(rect)
        ax.text(m0, n0 + size/2.0, f"%.3f" % win["score"], color=red)

    if show:
        plt.show()


def coarse_align_param_search(target, source, mask, patch_radius, **kwargs):

    moon = kwargs.get("moon", False)

    param_cache = dict()

    with timeit("\n    converting gray scale ... "):
        target = to_gray(target)
        source = np.array(source)

        if mask is not None:

            #----------------------------------------------------------------------
            # convert the mask into a boolean mask

            peak = np.max(mask)

            invalid = mask < peak

            # write nans into the black (masked out) region of the target image

            target[invalid] = np.nan

    with timeit("    padding ... "):

        pad_size = patch_radius

        target = pad_array(target, pad_size)
        source = pad_array(source, pad_size)
        source_gray = to_gray(source)

    M, N = target.shape[0:2]

    patch_width = 2 * patch_radius + 1

    #-------------------------------------------------------------------------
    # Perform searches in windows with maximum deltas.

    src = np.sqrt(np.array(source_gray))
    src -= np.nanmean(src)
    
    candidates = []

    for m in range(0, M - patch_radius, patch_radius):
        for n in range(0, N - patch_radius, patch_radius):

            m0 = m
            m1 = m0 + patch_width

            n0 = n
            n1 = n0 + patch_width

            p = np.array(src[m0 : m1, n0 : n1])

            pixel_range = np.nanmax(p) - np.nanmin(p)
            if moon:
                # Boost the score of patches based on the sum of the
                # pixels in the patch, this should favor patches including
                # parts the moon rather than patches with only stars.
                score = pixel_range * np.nansum(p.flatten())
            else:
                score = pixel_range

            candidates.append(dict(score=score, pos=(m,n), size=patch_width))

    candidates = sorted(candidates, key = lambda x: x["score"], reverse = True)

    # Normalize the scores.
    max_score = max(candidates[0]["score"], 1.0)
    print(f"Max score before normalizing: {max_score}")

    for can in candidates:
        can["score"] /= max_score

    #-------------------------------------------------------------------------
    # Downsample the candidate so they they are at least 2 radii away.

    sparse_candidates = [ candidates[0] ]

    for can in candidates[1:]:
        score = can["score"]
        if moon:
            if score < 0.10:
                continue
        elif score < 0.50:
            continue

        can_pos = can["pos"]

        distances = []

        for sparse in sparse_candidates:
            spar_pos = sparse['pos']
            distances.append(
                np.hypot(
                    can_pos[0] - spar_pos[0],
                    can_pos[1] - spar_pos[1]
                )
            )

        if np.min(distances) > 3 * patch_radius:
            sparse_candidates.append(can)

    # plot_patches(source_gray, sparse_candidates)

    #-------------------------------------------------------------------------
    # Now take the top N windows.

    # course align every 1000 pixels or at least 4 windows

    size = 1000
    num_m = 1
    num_n = 1

    while num_m * num_n < 4:
        num_m = (M + size - 1) // size
        num_n = (N + size - 1) // size
        size -= 1

    num_windows = num_m * num_n

    random.shuffle(sparse_candidates)

    candidates = sparse_candidates[0:num_windows]

    num_windows = len(candidates)

    #-------------------------------------------------------------------------
    # Plot target patches that will be aligned with src patches.

    #plot_patches(source, candidates, title="coarse param search windows")
    #xxxxxxx

    #--------------------------------------------------------------------------
    # slide patches around the corse window locations

    search_window_size = 2 * patch_width

    window_iterations = 10
    random_walk_iterations = 1000
    gradient_assent_iterations = 100

    count = 0
    max_count = num_windows * window_iterations * (random_walk_iterations + gradient_assent_iterations)

    log(f"    patchpatch_width_size = {patch_width}\n")
    log(f"    num_windows = {num_windows}\n")
    log(f"    num_iterations = {max_count}\n")
    log( "    coarse param search: %5.1f%%" % 0.0)

    best_scores = []

    for can in candidates:
        for i in range(window_iterations):
            src_pos = can['pos']

            m0, n0 = src_pos
            m1 = m0 + patch_width
            n1 = n0 + patch_width

            src = source[m0 : m1, n0 : n1, :]

            src_patch = Patch(src, src_pos)

            m, n = src_pos
            angle = 0.0
            walk_scores = []

            # Randomly walk.
            for j in range(random_walk_iterations):

                rm = np.random.choice([-1, 0, 1])
                rn = np.random.choice([-1, 0, 1])
                ra = np.random.choice([-0.05, 0.0, 0.05])

                m += rm
                n += rn
                angle += ra

                if m < 0:
                    m = 0
                if n < 0:
                    n = 0

                walk_scores.append(src_patch.score(target, (m, n), angle))

                count += 1
                pdone = 100.0 * count / max_count
                log('\b\b\b\b\b\b%5.1f%%' % pdone)

            # Perform gradient_ascent at the best location so far.

            best = max(walk_scores)

            _, best = gradient_ascent(
                src,
                target,
                src_pos,
                best.heading,
                best.angle,
                max_iterations = gradient_assent_iterations,
                min_score = 1.0,
                show_pdone = True,
                count = count,
                max_count = max_count,
            )

            count += gradient_assent_iterations

            best_scores.append(best)

        #---------------------------------------------------------------------
        # Pull out maximum score.
        best = max(best_scores)
        score = best.score

        # add result to cache

        if not np.isnan(score) and score > 0.5:

            param_cache[best.src_pos] = dict(
                heading = best.heading,
                angle = best.angle,
                score = best.score,
            )

        if score > 1.0001 or score < -1.0001:
            # Debug if we get scores outside what isn't possible.

            log("Score {} is outside what I thought was possible: +/- 1.0".format(s))
            log("Throwing up some debug plots")

            plt.figure()
            imshow(src_patch._gray)
            plt.title('src_patch')
            plt.colorbar()

            plt.figure()
            imshow(scores)
            plt.title('translation scores')
            plt.colorbar()

            plt.figure()
            plt.plot(angle_axis, angle_scores)
            plt.grid(True)
            plt.title("angle scores")

            plt.title(
                'coarse patch scores: peak = %.3f, heading = %s, angle = %.3f' % (
                    s,
                    repr(heading),
                    angle
                )
            )

            plt.show()

            xxxxxxx

    log('\b\b\b\b\b\b%5.1f%%\n' % 100.0)

    assert param_cache, "Course alignment failed, need to adjust parameters in the code!"

    return param_cache




if __name__ == "__main__":
    with timeit("Total processing: "):
        main()

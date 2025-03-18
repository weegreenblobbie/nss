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

The frist step is the do exhustive alignments with patches on a very coarse
grid between the source image and the target.  These parameters are stored and
used for the next step as starting paramters

set 2 - fine patch alignment

Given the coarse grid parameters, break the src image into small patches and
peform a gradient asscent search for best fit, start the search with the
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
import scipy.misc
import scipy.signal
import skimage

from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.signal import correlate2d

import scipy.ndimage
import scipy.ndimage.filters
from   scipy.ndimage.filters import median_filter as median

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
        help = "Reads a previously computed coarse aligment cache file."
    )

    parser.add_argument(
        "-t",
        "--target",
        required = True,
        type = str,
        help = "specifes the target image that all other images are alinged to"
    )

    parser.add_argument(
        "-m",
        "--mask",
        type = str,
        help = "specifes the mask image, pixels that are black are ignored during processing"
    )

    parser.add_argument(
        "-o",
        "--output",
        type = str,
        default = None,
        help = "specifes the aligned output image name, default is {prefix}-aligned.tiff"
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

    if mask_fn:
        assert os.path.isfile(mask_fn), "Could not find file " + mask_fn

    if coarse_cache_fn:
        assert os.path.isfile(coarse_cache_fn), "Could not find file " + coarse_cache_fn

    if output_fn is None:
        path = os.path.dirname(source_fn)
        prefix = os.path.splitext(os.path.basename(source_fn))[0]
        output_fn = os.path.join(path, prefix + "-aligned.tiff")

    if os.path.isfile(output_fn):
        log(f"quitting ealry since output already exists: {output_fn}")
        return 0

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
                60,
            )

            if args.save_cache:
                log("\n    writing %s ..." % COARSE_CACHE_PKL)
                with open(COARSE_CACHE_PKL, 'wb') as fd:
                    pickle.dump(param_cache, fd)
                log("\n")

        print_param_cache(param_cache)

    radius = RADIUS

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


def imshow(array):

    peak = np.nanmax(array)

    if peak < 1.0:
        peak = 1.0

    # gray scale
    if array.ndim == 2:
        plt.imshow(array / peak, cmap = 'gray', interpolation='nearest')

    # color
    elif array.ndim == 3:
        plt.imshow(array / peak, interpolation='nearest')

    else:
        raise RuntimeError("don't know how to handle array with shape %s" % repr(array.shape))


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
        min_iterations=7,
        max_iterations=36,
        min_score=0.6,
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

        # Shift and rotate the patch randomally

        rm = np.random.randint(-3,3)
        rn = np.random.randint(-3,3)
        ra = np.random.choice(da_pool)

        scores.append(patch.score(tgt, (m + rm, n + rn), angle + ra))

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


def coarse_align_param_search(target, source, mask, patch_radius):

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

    candidates = []

    for m in range(0, M - patch_radius, patch_radius):
        for n in range(0, N - patch_radius, patch_radius):

            m0 = m
            m1 = m0 + patch_width

            n0 = n
            n1 = n0 + patch_width

            p = source_gray[m0 : m1, n0 : n1]

            delta = np.nanmax(p) - np.nanmin(p)

            candidates.append(dict(delta=delta, pos=(m,n)))

    candidates = sorted(candidates, key = lambda x: x['delta'], reverse = True)

    #-------------------------------------------------------------------------
    # Downsample the candidate so they they are at least 2 radii away.

    sparse_candidates = [ candidates[0] ]

    for can in candidates[1:]:

        can_pos = can['pos']

        p = can['delta'] / sparse_candidates[0]['delta']

        if p < 0.50: continue

        distances = []

        for sparse in sparse_candidates:
            spar_pos = sparse['pos']
            distances.append(
                np.hypot(
                    can_pos[0] - spar_pos[0],
                    can_pos[1] - spar_pos[1]
                )
            )

        if np.min(distances) > 4 * patch_radius:
            sparse_candidates.append(can)

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

    #-------------------------------------------------------------------------
    # Plot target patches that will be aligned with src patches.
    if False:

        array = np.ones(target.shape)
        array += source_gray

        invalid = np.isnan(target)

        array[invalid] = 0.0

        for can in candidates:
            m0, n0 = can['pos']
            m1 = m0 + patch_width
            n1 = n0 + patch_width
            array[m0 : m1, n0 : n1] += 100.0

        plt.figure()
        imshow(array)
        plt.title("coarse param search windows")

        plt.show()
        xxxxxxx

    #--------------------------------------------------------------------------
    # slide patches around the corse window locations

    search_window_size = 2 * patch_width

    window_iterations = 10
    random_walk_iterations = 1000
    gradient_assent_iterations = 100

    count = 0
    max_count = num_windows * window_iterations * (random_walk_iterations + gradient_assent_iterations)

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

            # Randomally walk.
            for j in range(random_walk_iterations):

                rm = np.random.choice([-1, 0, 1])
                rn = np.random.choice([-1, 0, 1])
                ra = np.random.choice([-0.05, 0.0, 0.05])

                m += rm
                n += rn
                angle += ra

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
        # Pull out maxium score from transations.
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
            # Debug if we get scores ourside what isn't possible.

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


class RegScore(object):
    '''
    A class to hold the registration score along with registration parameters
    so they can be tracked together.
    '''

    def __init__(self, score, src_pos, tgt_pos, angle):
        self.score = score
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.heading = (src_pos[0] - tgt_pos[0], src_pos[1] - tgt_pos[1])
        self.angle = angle


    def __lt__(self, rhs):
        return self.score < rhs.score


    def __gt__(self, rhs):
        return self.score > rhs.score


    def __ge__(self, rhs):
        return self.score >= rhs.score

    def __str__(self):
        return "score=%.3f, src_pos=%s, tgt_pos=%s, heading=%s, angle=%s" % (
            self.score,
            self.src_pos,
            self.tgt_pos,
            self.heading,
            self.angle,
        )



class Patch(object):
    '''
    A patch is a square sub image taking out of the source image.  It is scored
    using a cross correlation of the overlapping pixels.

    This class copies the source image in RGB, after the final translation
    and rotation is found, the RGB pixles are given to the Stacker class to be
    written to the final aligned image.
    '''


    def __init__(self, rgb, src_pos):

        assert \
            rgb.ndim == 3, \
            "Error, expecting patch to be a color image, got shape = %s" % \
                repr(rgb.shape)

        self._rgb = np.array(rgb)
        self._gray = to_gray(rgb)
        self._gray[np.isnan(self._gray)] = 0.0
        self._cache = dict()
        self._src_pos = src_pos
        self._tgt_pos = None
        self._angle = None
        self._heading = None

        self._rotate_kwargs = dict(
            reshape = False,
            mode = 'constant',
            cval = np.nan
        )

        self.shape = self._gray.shape


    def color_array(self):
        return self._rgb


    def tgt_pos(self):
        '''
        Returns the final target position
        '''

        assert self._tgt_pos is not None, "oops, patch._tgt_pos is None"

        return self._tgt_pos


    def set_final_params(self, angle, tgt_pos):

        assert self._tgt_pos is None, "oops, patch._tgt_pos is already set"

        self._tgt_pos = tgt_pos
        self._angle = angle

        # rotate each color channel

        invalid = np.isnan(self._rgb[:,:,0])

        for ax in range(self._rgb.shape[2]):

            array = self._rgb[:,:, ax]

            # rotation is implemented in terms a an image interpolation, so
            # any nans will propagte, so must turn them into zeros

            array[invalid] = 0.0

            array = ndimage.rotate(array, angle, **self._rotate_kwargs)

            self._rgb[:,:, ax] = array


    def rotate(self, angle):

        key = int(angle * 1000)

        if key in self._cache:
            return self._cache[key]

        # cache miss

        if key == 0:
            return self._gray

        p = ndimage.rotate(self._gray, angle, **self._rotate_kwargs)

        self._cache[key] = p

        return p


    def score(self, tgt, tgt_pos, angle = 0.0):

        assert \
            tgt.ndim == 2, \
            "Error, expecting target array to be gray scale, (shape = %s)" % \
                repr(tgt.shape)

        H, W = tgt.shape  # height and width of target image aligning to

        size = self._gray.shape[0]  # size of the patch

        radius = size // 2

        m, n = tgt_pos # upper left corner on target image

        # compute indices into target image and limit at image boundary

        m0 = m
        m1 = m0 + size

        if m1 > H:
            m1 = H
            m0 = m1 - size

        n0 = n
        n1 = n0 + size

        if n1 > W:
            n1 = W
            n0 = n1 - size

        # extract patch from target

        t = tgt[m0 : m1, n0 : n1]

        # rotate the patch to angle

        p = self.rotate(angle)

        score = xcorr_score(p, t)

        if False:
            plt.figure()
            imshow(p)
            plt.title('patch')
            plt.figure()
            imshow(t)
            plt.title('target (score=%f)' % score)
            plt.show()

        return RegScore(score, self._src_pos, tgt_pos, angle)

    def plot(self, title):
        plt.figure()
        imshow(self._gray)
        plt.title(title)
        plt.show()



class Aligned(object):
    '''
    This class stores an empty RGB image on construction.  The algorithms take
    pactches and align them to the reference images, then the patches are given
    to this class and the rgb values are copied out of the patch and stored.

    This class can be pickled to disk and the resulting tiff can also be saved.
    '''


    def __init__(self, shape):
        assert len(shape) >= 2
        self._rgb = np.nan * np.ones(shape, np.float32)
        self._count = np.ones(shape[0:2], np.float32)


    def add_patch(self, patch):

        array = patch.color_array()

        M, N = array.shape[0:2]

        m, n = patch.tgt_pos() # upper left corner of patch

        target = self._rgb[m : m + M, n : n + N, :]

        # Only write out pixels that are not NaN, rotating patches introduce
        # NaNs.

        valid = ~np.isnan(array)

        target[valid] = array[valid]

        target = self._count[m : m + M, n : n + N]

        target[valid[:,:,0]] += 1.0


    def get_rgb(self):

        rgb = np.array(self._rgb)

        return rgb


    def pickle(self, filename):

        d = dict(
            rgb = self._rgb,
            count = self._count,
        )

        with open(filename, 'wb') as fd:
            pickle.dump(d, fd)

        print("wrote %s" % filename)


    def savetiff(self, filename, resize = 1.0):

        rgb = self.get_rgb()

        rgb -= np.nanmin(rgb)
        rgb /= np.nanmax(rgb)

#~        invalid = np.isnan(rgb)

#~        rgb[invalid] = 0.0

#~        rgb = imresize(rgb, resize)

        # Convert float32 (0.0 to 1.0) to uint16.

        rgb *= (1 << 16) - 1

        rgb = rgb.astype(np.uint16)

        rgb = unpad_array(rgb, RADIUS)

        imwrite(filename, rgb, photometric = 'rgb')


    @classmethod
    def from_pickle(cls, filename):

        with open(filename, 'rb') as fd:
            d = pickle.load(fd)

        obj = cls(d['rgb'].shape)

        obj._rgb = d['rgb']
        obj._count = d['count']

        print("read %s" % filename)

        return obj


class timeit(object):

    def __init__(self, msg=None):
        if msg:
            log(msg)

    def __enter__(self):
        self.d0 = datetime.datetime.now()

    def __exit__(self, *args):
        delta = datetime.datetime.now() - self.d0
        total_seconds = delta.total_seconds()
        if total_seconds > 60.0:
            delta = str(delta)
        else:
            delta = "%.2f seconds" % total_seconds
        log(f"took {delta}\n")


def to_gray(rgb):

    assert rgb.ndim == 3, "oops, expecting color image, got shape %s" % repr(rgb.shape)

    #return np.sqrt(rgb[...,0] ** 2 + rgb[...,1] ** 2 + rgb[...,2] ** 2)

    # Luminance in the Adobe Color colorspace
    #     https://community.adobe.com/t5/photoshop-ecosystem-discussions/how-to-measure-true-luminosity-of-a-color-sample-by-script/td-p/12094302
    return 0.29999 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.1131 * rgb[..., 2]


def log(msg):

    sys.stdout.write(msg)
    sys.stdout.flush()


def zscore(array, valid):

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')

        array = np.array(array)

        array[~valid] = np.nan

        mu = np.nanmean(array)
        sig = np.nanstd(array)

        if np.isnan(mu) or np.isnan(sig):
            return np.zeros_like(array)

        out = (array - mu) / sig

        return out


def pad_array(array, radius):

    if array.ndim == 2:
        return _pad_array_2d(array, radius)

    else:
        return _pad_array_3d(array, radius)


def _pad_array_2d(array, radius):

    M, N = array.shape

    out = np.nan * np.ones((M + 2 * radius, N + 2 * radius), array.dtype)

    out[radius : radius + M, radius : radius + N] = array

    return out


def _pad_array_3d(array, radius):

    M, N, O = array.shape

    out = np.nan * np.ones((M + 2 * radius, N + 2 * radius, O), array.dtype)

    out[radius : radius + M, radius : radius + N, :] = array

    return out


def unpad_array(array, radius):

    if array.ndim == 2:
        return _unpad_array_2d(array, radius)

    else:
        return _unpad_array_3d(array, radius)


def _unpad_array_2d(array, radius):

    M, N = array.shape

    return array[
        radius : radius + M - 2 * radius,
        radius : radius + N - 2 * radius
    ]


def _unpad_array_3d(array, radius):

    M, N, _ = array.shape

    return array[
        radius : radius + M - 2 * radius,
        radius : radius + N - 2 * radius,
        :
    ]


def param_cache_resize(param_cache, s):

    out = dict()

    for k,v in param_cache.items():

        new_k = (int(k[0] * s), int(k[1] * s))
        new_v = dict(
            heading = (int(v['heading'][0] * s), int(v['heading'][1] * s)),
            angle = v['angle'],
            score = v['score'],
        )

        out[new_k] = new_v

    return out


def cache_lookup(param_cache, pos):

    keys = list(param_cache.keys())

    if not keys:
        return dict(heading=[0,0], angle=0.0)

    dist = np.zeros((len(keys)),np.float32)

    for i in range(len(keys)):

        loc = keys[i]

        dist[i] = np.sqrt((loc[0] - pos[0])**2 + (loc[1] - pos[1])**2)

    k = keys[dist.argmin()]

    return param_cache[k]


def print_param_cache(param_cache):

    print("param_cache:")

    keys = sorted(list(param_cache.keys()))

    for k in keys:

        v = param_cache[k]

        print(
            "    (%5d,%4d): score %.3f heading (%3d,%3d) angle %7.3f" % (
                k[0],
                k[1],
                v['score'],
                v['heading'][0],
                v['heading'][1],
                v['angle']
            )
        )


def xcorr_score(p, t):
    '''
    xcorr that deals with nans.

    The patch and target arrays are z-scored which scales each image in units of
    standard deviation.  This way, we are comparing the shapes in the images and
    relative scaling doesn't matter.

    score = average(patch * target)

    When shapes match well, the score appraoches 1.0, when they mismatch they
    approach 0, and if they match exactly oppisote, approach -1.0.
    '''

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in less')

        valid = ~np.isnan(p) * ~np.isnan(t)

        p = zscore(p, valid)
        t = zscore(t, valid)

        s = np.nanmean(p * t)

        if np.isnan(s) or np.isinf(s):
            return 0.0

        # deweight scores as the number of valid pixels drops off

        valid_count = np.sum(valid.astype(np.int32))

        if valid_count == 0:
            return 0.0

        weight = float(valid_count) / p.size

        if weight < 0.50:
            weight *= 2.0

        s *= weight

        if s > 1.0001 or s < -1.0001:

            print("Score {} out of bounds".format(s))
            print("valid_count = %s" % (repr(valid_count)))
            print("weight = %s" % (repr(weight)))
            print("s = %s" % (repr(s)))

            plt.figure()
            imshow(p)
            plt.title('p (%s, %s)' % (np.nanmin(p), np.nanmax(p)))

            plt.figure()
            imshow(t)
            plt.title('t (%s, %s)' % (np.nanmin(t), np.nanmax(t)))

            tmp = p * t

            plt.figure()
            imshow(tmp)
            plt.title('p*t (%s, %s)' % (np.nanmin(tmp), np.nanmax(tmp)))

            plt.show()

            raise RuntimeError(
                "score %s is outside expected range (-1 : 1)" % s
            )

        s = np.sign(s) * (s**2)

    return s



if __name__ == "__main__":
    main()

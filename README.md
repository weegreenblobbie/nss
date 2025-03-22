# Nick's Star Stacker

# Example Usage

Make a python virtual env and install the requirements.

These examples are in **windows powershell** format:

Align one TIFF against the target:

```
python align.py -t "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\20250313_232842.99.tif" `
                   "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\20250313_232746.99.tif"
```

Align multiple TIFFs against the target:
```
python align.py -t TARGET_TIF TIFF+
```

Align a bunch of files using python multiprocessing:
```
python nss.py -t "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\20250313_232842.99.tif" `
                 "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\*.tif"
```

# Roadmap

See [ROADMAP.md](ROADMAP.md).

# Origin Story

I fell in love with photographing the night sky many years ago, going out to the
dark mountains and setting up my tripod and camera with a wide angle lens. 
Using an intervalometer I would trigger my camera and record hundreds of images
at high iso to capture the Milky Way.

How does one process those images and get good results?  At the time,
[Sequator](https://sites.google.com/view/sequator/) was the best choice that
made the process easy.  The only problem was that it didn't handle complicated
silhouette in the foreground like pine tress.

So, naively I ventured into writing some python code to align TIFF images using
numpy.

# How stacking stars works

Given a sequence of exposures that are:
- captured while camera on a tripod
- camera operating in manual, using the same exposure settings
- camera triggered using an intervalometer
- the images are all the same dimensions

## Step 1, picking a target image, all other images are aligned to it

## Step 2. Invoke the program specifying the target and list of images to align

## Step 3. Perform an exhaustive parameter search

It's currently called [`coarse_align_param_search()`](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L589).

First we create a bunch of "patches", each patch is a square that is
`2 * patch_radius + 1`, where the radius is currently hard coded to `60` pixels.
So as currently written, the patches are squares that are 121 pixels wide.

For each patch, we create candidates that are sorted based on the patch's pixel
max - mix min, so that we try to align patches with the largest dynamic range
first.  The hope is that by doing it this way, we process a patch that actually
contains stars.

Next, these patches are down sampled further into a smaller set so we don't
waste a ton of processing time.  I 
[skip patches](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L653-L655)
who's deltas (i.e. pixel range) are less than 50% of the maximum patch delta (stored in candidates[0]).

This is surely **currently a bug** as I don't guarantee the result of this division
is in the 0.0 to 1.0 bounds, so it likely never continues and we end up with
more patches than intended.

I also make sure the [remaining patches are 4 radii](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L668-L669)
away from each other, so about 240 pixels away (4 * 60).

Next, I take the first N windows, where N is a path about every 1000 pixels of
the input image.  So for example, if the input image were 1000x1000, we should
end up with about 4 patches:

    img size: 1000, num_patches: 4
    img size: 2000, num_patches: 4
    img size: 3000, num_patches: 9
    img size: 4000, num_patches: 16

### Random Walks

Now that we have a sparse set of patches, we perform a large number of
[random walks](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L749-L765),
starting at the patches's origin, making random 1-pixel translations, and
a small random rotation, then evaluate a normalize score for the fit.

### Normalized fit scoring

Taking an array from the target image that we're aligning to, and a small patch
from the source image, we compute the [z-score](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L1199).

See Wikipedia's entry for [z-score](https://en.wikipedia.org/wiki/Standard_score).

I use this method to "normalize" the pixel values between the target and source
images, so we end up with arrays centered about zero, with pixel values that
are in units of standard deviation.  This approach removes any DC offset when
the mean is subtracted, and removes any relative scaling since the remainder is
divided by the standard deviation.  So we should expect most pixels are now
between +/- 3.0 or 3 units of standard deviation.

Next the
[cross correlation](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L1350)
is computed.  If the target and source patches are similar, their product is
large, if they're the opposite, their product is negative.

Finally, the product is divided by the number of valid pixels producing the
average cross correlation per pixel.  The range of this function is -1 to 1.

### Gradient Assent

After randomly walking a bit and collecting normalized fit score along the way,
next a number of
[gradient asset](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L452)
iterations is performed starting from the best score from the random walks.

This really isn't [gradient assent](https://en.wikipedia.org/wiki/Gradient_descent),
more like a genetic algorithm, making small adjustments in many directions, then
selecting the adjustment that maximizes the score and repeating.

Finally, we have candidate parameters for translation and rotation for each
patch.  Their parameters are added to a
[`param_cache`](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L801)
which stores the settings.

## Step 4. Aligning all the patches

Now that a large number of iterations was performed on a sparse subset of
patches, we go about aligning the dense set of patches, each of which overlaps
with their neighbors.  This is [done here](https://github.com/weegreenblobbie/nss/blob/b89623c8b93d547e127069cfffd90b95d0676c3b/align.py#L422-L443).

We use the nearest starting position for the patch in the `param_cache` as that
should be a good starting position, then perform a reasonable number of
iterations of "gradient assent". If the result of the gradient assent scored
high, we add those parameters to the `param_cache` to help out future patches
near by.

Finally, the aligned result is written out with a `-aligned.tiff` suffix.

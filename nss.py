#!/usr/bin/env python3
"""
nss.py - Nick Start Stacker
"""
import argparse
import datetime
import os
import glob
import subprocess
import sys
from multiprocessing import Pool

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-j",
        "--jobs",
        default = 4,
        type = int,
        help = "specifies the number of alignment jobs to run simultaneously"
    )

    parser.add_argument(
        "-m",
        "--mask",
        default = None,
        type = str,
        help = "specifes the mask image, pixels that are black are ignored during processing"
    )

    parser.add_argument(
        "-o",
        "--output",
        default = None,
        type = str,
        help = "specifes the aligned output image name, default is {prefix}-median.tiff"
    )

    parser.add_argument(
        "-t",
        "--target",
        default = None,
        type = str,
        help = "specifes the target image that all other images are alinged to"
    )

    parser.add_argument(
        "images",
        nargs = "+",
        type = str,
        help = "the set of source images to be aligned with the target"
    )

    args = parser.parse_args()

    images = args.images

    if not args.target:
        images = sorted(images)
        target = images[len(images)//2]
    else:
        target = args.target

    mask = args.mask if args.mask else ""

    # To support windows command line, process any globs.
    if '*' in target:
        target = glob.glob(target)[0]
    if '*' in mask:
        mask = glob.glob(mask)[0]
    if len(images) == 1 and '*' in images[0]:
        images = glob.glob(images[0])

    images = [x for x in images if '-aligned.tiff' not in x]

    images = set(images)
    images -= set([target])
    if mask: images -= set([mask])

    images = sorted(list(images))

    assert target not in images
    assert mask not in images

    num_images = len(images)

    pool = Pool(processes = args.jobs)

    with timeit():

        log("Launching {} threads to align {} images...\ntarget: {}".format(args.jobs, num_images, target))

        jobs = []
        for i, img in enumerate(images):
            jobs.append((target, mask, img, 'nss-log%02d.txt' % (i + 1)))

        results = pool.map(align_job, jobs)

        num_errors = 0

        for res, logfile in results:
            num_errors += res.returncode != 0
            if res.returncode:
                log(f"See {logfile} for the error message\n")

        log("\n{}/{} jobs failed\n".format(num_errors, num_images))


class timeit(object):


    def __enter__(self):
        self.d0 = datetime.datetime.now()

    def __exit__(self, *args):
        d1 = datetime.datetime.now()

        log("took %.2f seconds\n" % (d1 - self.d0).total_seconds())

def log(msg):

    sys.stdout.write(msg)
    sys.stdout.flush()


def align_job(args):

    target, mask, image, logfile = args

    args = [sys.executable, 'align.py', '--target', target]

    if mask:
        args.extend(['--mask', mask])

    args.append(image)

    log("Launching jobs: {}\n    logfile: {}\n".format(' '.join(args), logfile))

    with open(logfile, "w") as fd:

        p = subprocess.run(
            args,
            env = os.environ,
            stdout = fd,
            stderr = subprocess.STDOUT,
            bufsize=1,
        )

        return (p, logfile)


if __name__ == "__main__": main()

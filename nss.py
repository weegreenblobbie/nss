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

from nss import utils

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
    if len(images) == 1 and '*' in images[0]:
        images = sorted(glob.glob(images[0]))

    if not args.target:
        images = sorted(images)
        target = images[len(images) // 2]
    else:
        target = args.target

    # To support windows command line, process any globs.
    if '*' in target:
        target = glob.glob(target)[0]
    
    images = set(images)
    images -= set([target])
    images = sorted(list(images))

    assert target not in images

    num_images = len(images)

    pool = Pool(processes = args.jobs)

    with utils.timeit():

        utils.log("Launching {} threads to align {} images...\ntarget: {}".format(args.jobs, num_images, target))

        jobs = []
        for i, img in enumerate(images):
            jobs.append((target, img, 'nss-log%02d.txt' % (i + 1)))

        results = pool.map(align_job, jobs)

        num_errors = 0

        for res, logfile in results:
            num_errors += res.returncode != 0
            if res.returncode:
                utils.log(f"See {logfile} for the error message\n")

        utils.log("\n{}/{} jobs failed\n".format(num_errors, num_images))


def align_job(args):

    target, image, logfile = args

    args = [sys.executable, '-m', 'nss.align_moon', '--target', target, image]

    utils.log("Launching jobs: {}\n    logfile: {}\n".format(' '.join(args), logfile))

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

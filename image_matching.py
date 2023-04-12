import argparse
import os
import datetime

import cv2
import sift
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("image_matching")
    parser.add_argument(
        "--file_dir_1", type=str, default=None, help="input image 1"
    )
    parser.add_argument(
        "--file_dir_2", type=str, default=None, help="input image 2"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="output destination"
    )
    parser.add_argument(
        "-s", type=int, default=3, help="number of scales per octave"
    )
    parser.add_argument("-o", type=int, default=5, help="number of octaves")
    parser.add_argument(
        "-t", type=float, default=5e-2, help="threshold for detection"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="use for Gaussian blurring"
    )
    parser.add_argument(
        "--rescale",
        type=float,
        default=1.0,
        help="rescale images to make it faster",
    )

    return parser.parse_args()


def main(args):
    sift_sigma = args.sigma
    rescale_factor = args.rescale
    num_scales = args.s
    num_octaves = args.o
    t = args.t
    img_1_dir = args.file_dir_1
    img_2_dir = args.file_dir_2

    assert os.path.isfile(img_1_dir)
    assert os.path.isfile(img_2_dir)

    if rescale_factor < 0 or 1 - rescale_factor < 0:
        rescale_factor = 1

    img1 = cv2.imread(img_1_dir, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_2_dir, cv2.IMREAD_GRAYSCALE)

    s = datetime.datetime.now()

    key_locs, key_matches = sift.mysift_matching(
        [img1, img2],
        rescale_factor,
        sigma=sift_sigma,
        num_octaves=num_octaves,
        num_scales=num_scales,
        _threshold=t,
    )

    e = datetime.datetime.now()
    print(
        "Done!\n Time consuming: {} s".format((e - s).microseconds / 10**6)
    )

    plt.figure()

    dh = int(img2.shape[0] - img1.shape[0])
    top_padding = int(dh / 2)
    img1_padded = cv2.copyMakeBorder(
        img1, top_padding, dh - int(dh / 2), 0, 0, cv2.BORDER_CONSTANT, 0
    )
    plt.imshow(np.c_[img1_padded, img2], cmap="gray")

    for match in key_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1 = key_locs[0][img1_idx].pt[0] / rescale_factor
        y1 = (key_locs[0][img1_idx].pt[1]) / rescale_factor + top_padding
        x2 = (key_locs[1][img2_idx].pt[0]) / rescale_factor + img1.shape[1]
        y2 = key_locs[1][img2_idx].pt[1] / rescale_factor
        plt.plot(np.array([x1, x2]), np.array([y1, y2]), "o-")

    plt.tight_layout()
    out_dir = args.output_dir
    if out_dir is not None:
        plt.savefig(out_dir)
        print("Image matching result is saved at {}.".format(out_dir))

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

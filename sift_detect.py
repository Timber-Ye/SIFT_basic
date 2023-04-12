import argparse
import os
import datetime

import cv2
import sift


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("sift_detection")
    parser.add_argument(
        "--file_dir", type=str, default=None, help="input image"
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
    img_dir = args.file_dir

    assert os.path.isfile(img_dir)

    if rescale_factor < 0 or 1 - rescale_factor < 0:
        rescale_factor = 1

    src = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    img_scaled = cv2.resize(src, (0, 0), fx=rescale_factor, fy=rescale_factor)
    nomalized_img = cv2.normalize(
        img_scaled.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
    )

    s = datetime.datetime.now()

    pyramid = sift.DoGPyramid(num_scales, num_octaves, sift_sigma)
    keypoints, _ = pyramid.extract_keypoints(
        nomalized_img, t, sift.computeDescriptors
    )

    e = datetime.datetime.now()
    print(
        "Done!\n Time consuming: {} s".format((e - s).microseconds / 10**6)
    )

    dst = cv2.drawKeypoints(
        img_scaled,
        keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    out_dir = args.output_dir
    if out_dir is not None:
        cv2.imwrite(out_dir, dst)
        print(
            "Image labelled with SIFT keypoints is saved at {}.".format(
                out_dir
            )
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)

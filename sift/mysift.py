import cv2
import numpy as np
from pyramid import DoGPyramid
from descriptor import computeDescriptors


def mysift_matching(_images, rescale_factor, sigma=1.0, num_scales=3, num_octaves=5, _threshold=0.05):
    normalized = [cv2.normalize(cv2.resize(img, (0, 0), fx=rescale_factor, fy=rescale_factor).astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) for img in _images]

    dog_p = DoGPyramid(num_scales, num_octaves, sigma)
    keypoint_locations = []
    keypoint_descriptors = []

    for img in normalized:
        locs, desc = dog_p.extract_keypoints(img, _descriptor=computeDescriptors)

        # Store the information
        keypoint_locations.append(locs)
        keypoint_descriptors.append(desc)

    # OpenCV brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(keypoint_descriptors[0].astype(np.float32), keypoint_descriptors[1].astype(np.float32), 2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance or n.distance < 0.4 * m.distance:
            good.append(m)

    return keypoint_locations, good


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img1 = cv2.imread('../images/nvidia-3.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../images/nvidia-4.jpg', cv2.IMREAD_GRAYSCALE)
    rescale_factor = 0.3
    key_locs, key_matches = mysift_matching([img1, img2], rescale_factor)

    plt.figure()
    dh = int(img2.shape[0] - img1.shape[0])
    top_padding = int(dh / 2)
    img1_padded = cv2.copyMakeBorder(img1, top_padding, dh - int(dh / 2),
                                     0, 0, cv2.BORDER_CONSTANT, 0)
    plt.imshow(np.c_[img1_padded, img2], cmap="gray")

    for match in key_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1 = key_locs[0][img1_idx].pt[0] / rescale_factor
        y1 = (key_locs[0][img1_idx].pt[1]) / rescale_factor + top_padding
        x2 = (key_locs[1][img2_idx].pt[0]) / rescale_factor + img1.shape[1]
        y2 = key_locs[1][img2_idx].pt[1] / rescale_factor
        plt.plot(np.array([x1, x2]), np.array([y1, y2]), "o-")
    plt.show()

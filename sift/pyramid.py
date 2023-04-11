import cv2
import numpy as np
import scipy.ndimage


class DoGPyramid:
    def __init__(self, num_scales, num_octaves, base_sigma):
        self.num_scales = num_scales  # number of scales per octave
        self.num_octaves = num_octaves  # number of octaves
        self.sigma = base_sigma

        self.k = 2 ** (1 / self.num_scales)
        self.scaler = (
            np.vander([self.k], self.num_scales + 2, increasing=True).squeeze()
            * self.sigma
            * 1.414
        )

        self.t = None  # DoG threshold
        self.image = None

    def __compute_blurred_images(self):
        """
        Generate the Gaussian Pyramid
        :return: A list of octaves. num_scales + 3 blurred images per item(octave)
        """
        num_images_per_oct = self.num_scales + 3
        sigma_factor = np.sqrt(self.k**2 - 1)  # $\sqrt{k^2 - 1}$
        sigma = self.sigma

        filter_size = int(
            2 * np.ceil(3 * sigma) + 1.0
        )  # $w\approx 2\times\lceil 3\sigma\rceil + 1$
        last_image = cv2.GaussianBlur(
            self.image, (filter_size, filter_size), sigma
        )

        blurred_images = []
        for octave_idx in range(self.num_octaves):
            octave_stack = np.zeros(
                np.r_[last_image.shape, num_images_per_oct]
            )
            octave_stack[:, :, 0] = last_image

            for level_idx in range(1, num_images_per_oct):
                sigma = (
                    self.sigma * sigma_factor * (self.k ** (level_idx - 1))
                )  # $\sigma * \sqrt{k^2-1} * k^{l-1}$

                filter_size = int(2 * np.ceil(3 * sigma) + 1.0)
                octave_stack[:, :, level_idx] = cv2.GaussianBlur(
                    octave_stack[:, :, level_idx - 1],
                    (filter_size, filter_size),
                    sigma,
                )

            blurred_images.append(octave_stack)
            last_image = cv2.resize(
                octave_stack[:, :, self.num_scales], (0, 0), fx=0.5, fy=0.5
            )

        return blurred_images

    def __compute_difference_of_gaussians(self, blurred_images):
        """
        Generate the DoG Pyramid
        :param blurred_images: Gaussian Pyramid
        :return: A list of octaves. num_scales + 2 difference_of_gaussian per item(octave)
        """
        dogs = []

        for idx, _img in enumerate(blurred_images):
            dog = np.zeros(_img.shape - np.array([0, 0, 1]))
            num_dogs_per_octave = dog.shape[2]
            for dog_idx in range(num_dogs_per_octave):
                dog[:, :, dog_idx] = np.abs(
                    _img[:, :, dog_idx + 1] - _img[:, :, dog_idx]
                )

            dogs.append(dog)

        return dogs

    def __extrema_detection(self, diff_of_gaussians):
        """
        Detection of scale-space extrema
        :param diff_of_gaussians: DoG Pyramid
        :return: locations and corresponding scales of each scale-space extrema
        """
        keypoint_locations = []
        keypoint_scales = []

        for oct_idx, dog in enumerate(diff_of_gaussians):
            dog_max = scipy.ndimage.maximum_filter(dog, [3, 3, 3])
            is_keypoint = (dog == dog_max) & (
                dog >= self.t
            )  # non-maximum suppression
            is_keypoint[:, :, 0] = False
            is_keypoint[:, :, -1] = False
            locs = np.array(is_keypoint.nonzero()).T
            scales = self.scaler[locs[:, -1]]
            keypoint_locations.append(locs)
            keypoint_scales.append(scales)

        return keypoint_locations, keypoint_scales

    def extract_keypoints(self, _img, r_threshold=0.05, _descriptor=None):
        """
        Extract keypoints
        :return:
        """
        self.image = _img
        self.t = r_threshold
        blurred_images = self.__compute_blurred_images()
        dogs = self.__compute_difference_of_gaussians(blurred_images)
        kp_locs, kp_scales = self.__extrema_detection(dogs)

        if _descriptor is None:
            keypoints = []
            # Adapt keypoint location such that they correspond to the
            # originial image dimensions.

            kp_locs = np.array([i * 2**idx for idx, i in enumerate(kp_locs)])
            kp_scales = np.array(
                [i * 2 ** (idx + 1) for idx, i in enumerate(kp_scales)]
            )
            for _, (locs, sizes) in enumerate(zip(kp_locs, kp_scales)):
                for _, (loc, size) in enumerate(zip(locs, sizes)):
                    kp = cv2.KeyPoint(x=int(loc[1]), y=int(loc[0]), size=size)
                    keypoints.append(kp)
            return keypoints, None

        else:
            return _descriptor(blurred_images, kp_locs, kp_scales)


if __name__ == "__main__":
    from descriptor import computeDescriptors

    sift_sigma = 1.0  # sigma used for blurring
    rescale_factor = 0.3  # rescale images to make it faster
    num_scales = 3  # number of scales per octave
    num_octaves = 5  # number of octaves
    t_threshold = 0.05  # for feature detection

    img = cv2.imread("../images/hongluosi-2.jpg", cv2.IMREAD_GRAYSCALE)
    img_scaled = cv2.resize(img, (0, 0), fx=rescale_factor, fy=rescale_factor)
    nomalized_img = cv2.normalize(
        img_scaled.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
    )

    pyramid = DoGPyramid(num_scales, num_octaves, sift_sigma)
    keypoints, _ = pyramid.extract_keypoints(
        nomalized_img, t_threshold, computeDescriptors
    )

    im_with_keypoints = cv2.drawKeypoints(
        img_scaled,
        keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imwrite("../images/mysift-keypoints.jpg", im_with_keypoints)

    sift = cv2.SIFT_create(nOctaveLayers=1, sigma=1.0)
    kp = sift.detect(img_scaled, None)

    im_with_keypoints = cv2.drawKeypoints(
        img_scaled, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("../images/opencv-keypoints.jpg", im_with_keypoints)

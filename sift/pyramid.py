import cv2
import numpy as np
import scipy


class DoGPyramid:
    def __init__(self, img, num_scales, num_octaves, base_sigma, r_threshold):
        self.image = img
        self.num_scales = num_scales
        self.num_octaves = num_octaves
        self.sigma = base_sigma
        self.t = r_threshold

        k = 2 ** (1 / self.num_scales)
        self.scaler = np.vander([k], self.num_scales + 2, increasing=True) * self.sigma * 1.414

    def __compute_blurred_images(self):
        num_images_per_oct = self.num_scales + 3
        sigma_factor = np.sqrt(2 ** (2 / self.num_scales) - 1)
        sigma = self.sigma

        filter_size = int(2 * np.ceil(3 * sigma) + 1.0)
        last_image = cv2.GaussianBlur(self.image, (filter_size,filter_size), sigma)

        blurred_images = []
        for octave_idx in range(self.num_octaves):
            octave_stack = np.zeros(np.r_[last_image.shape, num_images_per_oct])
            octave_stack[:, :, 0] = last_image

            for level_idx in range(1, num_images_per_oct):
                sigma = sigma * sigma_factor
                filter_size = int(2 * np.ceil(3 * sigma) + 1.0)
                octave_stack[:, :, level_idx] = cv2.GaussianBlur(octave_stack[:, :, level_idx - 1], (filter_size,filter_size), sigma)

            blurred_images.append(octave_stack)
            last_image = cv2.resize(octave_stack[:, :, self.num_scales], (0,0), fx=0.5, fy=0.5)

        return blurred_images

    def __compute_difference_of_gaussians(self, blurred_images):
        dogs = []

        for idx, img in blurred_images:
            dog = np.zeros(img.shape - np.array([0, 0, 1]))
            num_dogs_per_octave = dog.shape[2]
            for dog_idx in range(num_dogs_per_octave):
                dog[:, :, dog_idx] = np.abs(img[:, :, dog_idx + 1] - img[:, :, dog_idx])

            dogs.append(dog)

        return dogs

    def __extract_keypoints(self, diff_of_gaussians):
        keypoint_locations = []
        keypoint_scales = []

        for oct_idx, dog in enumerate(diff_of_gaussians):
            dog_max = scipy.ndimage.maximum_filter(dog, [3, 3, 3])
            is_keypoint = (dog == dog_max) & (dog >= self.t)
            is_keypoint[:, :, 0] = False
            is_keypoint[:, :, -1] = False
            locs = np.array(is_keypoint.nonzero()).T
            scales = self.scaler[locs[:, -1]]
            keypoint_locations.append(locs * 2 ** oct_idx)
            keypoint_scales.append(scales)

        return keypoint_locations, keypoint_scales
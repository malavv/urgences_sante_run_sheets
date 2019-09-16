import cv2
import numpy as np
from operator import itemgetter

from segment.Helper import get_overlap


def rng_overlap(lhs, rhs):
    # start, end
    s1, e1 = lhs
    s2, e2 = rhs
    if e1 < s2 or e2 < s1:
        return 0
    return min(e1, e2) - max(s1, s2)


def x_overlap_px(lhs, rhs):
    # start, end
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    s1, e1 = (x1, x1 + w1)
    s2, e2 = (x2, x2 + w2)
    if e1 < s2 or e2 < s1:  # One totally before or after
        return 0
    return min(e1, e2) - max(s1, s2)  # Intersection


def add_rect(lhs, rhs):
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    xy_start = min(x1, x2), min(y1, y2)
    xy_end = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
    return xy_start[0], xy_start[1], xy_end[0] - xy_start[0], xy_end[1] - xy_start[1]


class NumImgs:
    def __init__(self):
        # Candidates are for each digit. They are considered only by their projection on the X axis.
        self.candidate = []

    def get_candidate(self, index):
        return None if index >= len(self.candidate) else self.candidate[index]

    def add_candidate(self, index, rect):
        prior = self.get_candidate(index)
        if prior is None:
            # First candidate
            self.candidate.append(rect)
        else:
            # Add the rectangles
            self.candidate[index] = add_rect(prior, rect)

    def find_digit_index(self, rect):
        for i in range(99):
            candidate = self.get_candidate(i)
            if candidate is None:  # First time using this candidate
                return i
            if x_overlap_px(rect, candidate) >= 2:  # more than 3 pixel overlap in horizontal direction
                return i
        # Could not find any valid idea
        return None

    def completely_in(self, arr, rect):
        for r in arr:
            if np.array_equal(r, rect):
                continue
            if x_overlap_px(r, rect) >= min(r[2], rect[2]):
                return True
        return False

    def find_fully_in(self, arr, rect):
        for i, r in enumerate(arr):
            if np.array_equal(r, rect):  # Skip same
                continue
            if x_overlap_px(r, rect) == min(r[2], rect[2]):  # Fully In
                return i
        return None

    def hstack(self):
        # Sort by 'x' position
        hstack = sorted(self.candidate, key=itemgetter(0))

        for r1 in hstack:
            i = self.find_fully_in(hstack, r1)
            while i is not None:
                hstack.pop(i)
                i = self.find_fully_in(hstack, r1)
        return hstack

    def add_img(self, rect):
        i = self.find_digit_index(rect)
        assert i is not None, "Incorrect digit guess"

        self.add_candidate(i, rect)


class DigitSegmenter:
    def __init__(self, show_debug_vis, overlap_perc_for_similar_threshold,
                 digit_padding_in_px, is_digit_shaped, show_candidate_rect):

        # Show debug visualizations
        self.show_debug_vis = show_debug_vis
        self.is_digit_shaped = is_digit_shaped
        self.is_similar_overlap_thresh = overlap_perc_for_similar_threshold
        self.digit_padding_in_px = digit_padding_in_px
        self.show_candidate_rect = show_candidate_rect

    def __repr__(self):
        return "DigitSegmenter()"

    def segment_with_mser(self, img):
        """
        Raw segmentation of the digits
        :param img:
        :return:

        * Possible future improvement might involve, trying erode/dilate to fill in small gap in digit.
        * Tried canny+contour but didn't work as well as this mser.
        """
        # So the thickening is an issue, since it scraps the result by either forcing a translation or including pieces
        # of the stuff around.
        # img_thick = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, (3, 3), iterations=3)
        # if self.show_debug_vis:
        #   cv2.imshow("2.5-Closing Morph. trans.", img)
        #   cv2.waitKey(0)

        # Normally here people are more interested in the raw regions than the crude bounding rectangle.
        # but let's start with something. The regions can be used to create form fitting "hulls" that are
        # more conservative than the bounding rectangles.
        mser = cv2.MSER_create()
        regions, bonding_rect = mser.detectRegions(img)

        # Draw MSER detected areas
        if self.show_debug_vis:
            vis = img.copy()  # because we will draw on it.
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            cv2.imshow('3-hulls', vis)
            cv2.waitKey(0)

        # Separate detected regions into distinct bounding rectangles (which are not garbage pixels around the border)
        num_imgs = NumImgs()
        for rect in bonding_rect:
            if self.is_digit_shaped(rect):
                x, y, w, h = rect
                if self.show_candidate_rect:
                    cv2.imshow('3-hulls', img[y:y + h, x:x + w])
                    cv2.waitKey(0)
                num_imgs.add_img(rect)

        digits = []
        for rect in num_imgs.hstack():
            x, y, w, h = rect
            digits.append(img[y:y + h, x:x + w].copy())

        return digits

    def make_similar_to_mnist(self, digits):
        # Inspired by the crash course AI digits learning thing.
        # These steps process the scanned images to be in the same format and have the same properties as the EMNIST images
        # They are described by the EMNIST authors in detail here: https://arxiv.org/abs/1702.05373v1
        processed_story = []

        for digit in digits:
            height, width = digit.shape[:2]

            # Squarify
            max_side = max(height, width) + self.digit_padding_in_px
            canvas = np.zeros((max_side, max_side), dtype=np.uint8)  # make square canvas
            canvas.fill(255)  # Make white

            # Place digit at center of square canvas
            off_x = int((max_side / 2) - (width / 2))
            off_y = int((max_side / 2) - (height / 2))
            canvas[off_y: off_y + height, off_x: off_x + width] = digit

            # step 1: Apply Gaussian blur filter
            img = cv2.GaussianBlur(canvas, (7, 7), 0)

            # step 4: Resize and resample to be 28 x 28 pixels
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

            # step 5: Normalize pixels and reshape before adding to the new story array
            img = img / float(255)
            img = img.reshape((28, 28))
            processed_story.append(img)

        # Stack Digits horizontally
        if len(processed_story) == 0:
            processed_story.append(np.zeros((0, 0), dtype=np.uint8))
        hstack = np.hstack(processed_story)

        # Denormalize into 8 bit image
        return 255 - (255 * hstack)

    def segment(self, img):
        # Load and show original image
        if self.show_debug_vis:
            cv2.imshow("1-original", img)
            cv2.waitKey(0)

        crude_digits = self.segment_with_mser(img.copy())  # not sure the copy is needed. test later.
        if self.show_debug_vis:
            for digit_img in crude_digits:
                cv2.imshow("4-digit", digit_img)
                cv2.waitKey(0)

        mnist_hstack = self.make_similar_to_mnist(crude_digits)
        if self.show_debug_vis:
            cv2.imshow("5-hstack", mnist_hstack)

        return mnist_hstack

import cv2
import numpy as np

from segment.Helper import get_overlap


def rng_overlap(lhs, rhs):
    # start, end
    s1, e1 = lhs
    s2, e2 = rhs
    if e1 < s2 or e2 < s1:
        return 0
    return min(e1, e2) - max(s1, s2)


class NumImgs:
    def __init__(self):
        # Rect for best candidate
        self.digit_candidate = [None, None, None, None]
        # Img for best candidate
        self.best_digits = [None, None, None, None]

    def find_digit_index(self, rect):
        x, y, w, h = rect
        for i, candidate in enumerate(self.digit_candidate):
            if candidate is None:  # First candidate
                return i
            x1, y1, w1, h1 = candidate
            overlap = rng_overlap((x, x + w), (x1, x1 + w1))
            if overlap > 3:  # more than 3 pixel overlap in horizontal direction
                return i
        return None

    def addImg(self, rect, img):
        x, y, w, h = rect
        i = self.find_digit_index(rect)

        assert i is not None, "Incorrect digit guess"

        prior = self.digit_candidate[i]

        if prior is None:  # First time set
            self.digit_candidate[i] = rect
            self.best_digits[i] = img
            return

        # Else check if taller (and replace)
        if h > prior[3]:
            self.digit_candidate[i] = rect
            self.best_digits[i] = img
            return

        # maybe also check if wider? but not 2 digit wide?


class DigitSegmenter:
    def __init__(self, mask_filename, show_debug_vis, overlap_perc_for_similar_threshold,
                 digit_padding_in_px, is_digit_shaped):

        # A mask for the hour mark in the image. The ":" part.
        self.hour_mask = cv2.bitwise_not(cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE))
        # Show debug visualizations
        self.show_debug_vis = show_debug_vis
        #
        self.is_digit_shaped = is_digit_shaped
        self.is_similar_overlap_thresh = overlap_perc_for_similar_threshold
        self.digit_padding_in_px = digit_padding_in_px

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

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (2, 2), iterations=5)
        if self.show_debug_vis:
            cv2.imshow("2.5-Closing Morph. trans.", img)
            cv2.waitKey(0)

        # Normally here people are more interested in the raw regions than the crude bounding rectangle.
        # but let's start with something. The regions can be used to create form fitting "hulls" that are
        # more conservative than the bounding rectangles.
        mser = cv2.MSER_create()
        regions, bonding_rect = mser.detectRegions(img)

        # Draw MSER detected areas
        vis = img.copy()  # because we will draw on it.
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        if self.show_debug_vis:
            cv2.imshow('3-hulls', vis)
            cv2.waitKey(0)

        # Separate detected regions into distinct bounding rectangles (which are not garbage pixels around the border)
#        valid_rects = []
        num_imgs = NumImgs()
        for rect in bonding_rect:
            x, y, w, h = rect

            if not self.is_digit_shaped(rect):
                continue

            num_imgs.addImg(rect, img[y:y + h, x:x + w].copy())

            #cv2.imshow('3-hulls', img[y:y + h, x:x + w])
            #cv2.waitKey(0)

            #if any(get_overlap(rect, valid_rect) > self.is_similar_overlap_thresh for valid_rect in valid_rects):
#                continue
#            valid_rects.append(rect.copy())

        # Extract Sub-Images (saving digit bounding rectangles as distinct images)
        #digits = []
        #for rect in valid_rects:
#            x, y, w, h = rect
#            digits.append(img[y:y + h, x:x + w].copy())

        return num_imgs.best_digits

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
            img = img / 255
            img = img.reshape((28, 28))
            processed_story.append(img)

        # Stack Digits horizontally
        hstack = np.hstack(processed_story)

        # Denormalize into 8 bit image
        return 255 - (255 * hstack)

    def segment(self, img):
        # Load and show original image
        if self.show_debug_vis:
            cv2.imshow("1-original", img)
            cv2.waitKey(0)

        # Show image no hour mark
        img = cv2.add(img, self.hour_mask)  # Remove the hour mark, by erasing the pixels
        if self.show_debug_vis:
            cv2.imshow("2-nomark", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        crude_digits = self.segment_with_mser(img.copy())  # not sure the copy is needed. test later.
        if self.show_debug_vis:
            for digit_img in crude_digits:
                cv2.imshow("4-digit", digit_img)
                cv2.waitKey(0)

        mnist_hstack = self.make_similar_to_mnist(crude_digits)
        if self.show_debug_vis:
            cv2.imshow("5-hstack", mnist_hstack)

        return mnist_hstack

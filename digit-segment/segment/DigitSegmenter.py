import cv2
import numpy as np


overlap_perc_for_similar_threshold = 0.8
digit_padding_in_px = 5

def is_digit_shaped(rect):
    x, y, w, h = rect
    return h > 3 and w > 3  # Complete heuristic to remove black around image.


def exists_similar(rect, vrects, area_overlap):
    return any(get_overlap(rect, valid_rect) > area_overlap for valid_rect in vrects)


def get_overlap(lhs, rhs):
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    # Some stackoverflow code to compute overlap of rectangle A and B. Seem to come from VB, ugly but meh.
    SA = w1 * h1
    SB = w2 * h2
    SI = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    SU = SA + SB - SI
    perc_overlap = SI / SU

    return perc_overlap


class DigitSegmenter:
    def __init__(self, mask_filename, show_debug_vis):
        # A mask for the hour mark in the image. The ":" part.
        self.hour_mask = cv2.bitwise_not(cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE))
        # Show debug
        self.show_debug_vis = show_debug_vis
        # MSER processor
        self.mser = cv2.MSER_create()

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

        # Normally here people are more interested in the raw regions than the crude bounding rectangle.
        # but let's start with something. The regions can be used to create form fitting "hulls" that are
        # more conservative than the bounding rectangles.
        regions, bonding_rect = self.mser.detectRegions(img)

        # Draw MSER detected areas
        vis = img.copy()  # because we will draw on it.
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        if self.show_debug_vis:
            cv2.imshow('3-hulls', vis)
            cv2.waitKey(0)

        # Separate detected regions into distinct bounding rectangles (which are not garbage pixels around the border)
        valid_rect = []
        for rect in bonding_rect:
            if not is_digit_shaped(rect):
                continue
            if exists_similar(rect, valid_rect, overlap_perc_for_similar_threshold):
                continue
            valid_rect.append(rect.copy())

        # Extract Sub-Images (saving digit bounding rectangles as distinct images)
        digits = []
        for rect in valid_rect:
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
            max_side = max(height, width) + digit_padding_in_px
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

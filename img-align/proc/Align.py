import cv2
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from tqdm import tqdm
import os

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.40
PERC_COMMON_THRESH = 0.40
KEYPOINT_COLOR_MATCH_FILE = (20, 200, 20)
MATCH_FILE_NAME = "matches.jpg"
COMPOSITE_FILE_NAME = "composite.jpg"


def prepare_img(filename):
    """
    Preparing a file from a complete path.
    :param filename: Path to the file
    :return: Grayscale Image
    """
    return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)


def add_composite(composite, new_image_dat):
    return Image.blend(composite, Image.fromarray(new_image_dat), alpha=0.5)


class Align:
    def __init__(self, ref_filename, imgs_filename, out_dir, compositor):
        self.ref = ref_filename
        self.inputs = imgs_filename
        self.out_dir = out_dir
        self.compositor = compositor

        self.writeHomography = False
        self.writeMatches = False

        print("Aligner initialized with %s images to process" % len(self.inputs))

    def __repr__(self):
        return "[%s, %s]" % (self.ref, self.inputs)

    @staticmethod
    def find_matches(orb, img, ref_descriptors, match_quality_threshold):
        # Detect ORB features and compute descriptors.
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors, ref_descriptors, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        num_match_to_keep = int(len(matches) * match_quality_threshold)
        matches = matches[:num_match_to_keep]

        return keypoints, matches

    @staticmethod
    def warp_img_by_matches(img, ref, ref_keypoints, keypoints, matches, idx_to_use):

        matches_to_use = [m for m in matches if m.trainIdx in idx_to_use]

        # Extract location of good matches
        points1 = np.zeros((len(matches_to_use), 2), dtype=np.float32)
        points2 = np.zeros((len(matches_to_use), 2), dtype=np.float32)

        for i, match in enumerate(matches_to_use):
            points1[i, :] = keypoints[match.queryIdx].pt
            points2[i, :] = ref_keypoints[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width = ref.shape

        # Wrap and return the data
        return cv2.warpPerspective(img, h, (width, height)), h

    def align(self):
        orb = cv2.ORB_create(MAX_FEATURES)

        # Load Reference Image
        ref = prepare_img(self.ref)
        ref_keypoints, ref_descriptors = orb.detectAndCompute(ref, None)

        # Results
        results = {}
        keypoints_dic = {}
        matches_dic = {}

        # A before composite is difficult to do here, because the images don't all have the same size

        # Processing
        print("Starting matching")
        start = timer()
        for filename in tqdm(self.inputs):
            # Load image
            img = prepare_img(filename)
            # Find Best Matches
            keypoints, matches = self.find_matches(orb, img, ref_descriptors, GOOD_MATCH_PERCENT)
            keypoints_dic[filename] = keypoints
            matches_dic[filename] = matches

        end = timer()
        print("Processed matches in %s sec" % round(end - start, 2))

        # Count the usage for each descriptor index.
        common_pts = {}
        for filename, matches in matches_dic.items():
            local_common = {}
            for match in matches:
                # increase view count for the index of this key point in the reference.
                local_common[match.trainIdx] = local_common.get(match.trainIdx, 0) + 1
            for idx, num_uses in local_common.items():
                if num_uses != 1:  # prevent descriptors with multiple matches to be counted.
                    continue
                common_pts[idx] = common_pts.get(idx, 0) + 1

        # find the commonest of common point (can't inverse dict because of non-unique lines.)
        # Keep only points for which matches were found in at least 30% of the images.
        threshold = PERC_COMMON_THRESH * len(self.inputs)
        idx_to_use = []
        for idx, num_uses in common_pts.items():
            if num_uses < threshold:
                continue
            else:
                idx_to_use.append(idx)

        to_draw = [kpt for i, kpt in enumerate(ref_keypoints) if i in idx_to_use]
        img2 = cv2.drawKeypoints(ref, to_draw, None, color=KEYPOINT_COLOR_MATCH_FILE, flags=0)
        cv2.imwrite(MATCH_FILE_NAME, img2)

        print("Starting warping")
        start = timer()
        #  composite = Image.fromarray(ref)
        for i, filename in enumerate(tqdm(self.inputs)):
            img = prepare_img(filename)
            # Warp image
            dat, hom = self.warp_img_by_matches(
                img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)

            # Add to composite
            self.compositor.compose(dat)
            #  composite = Image.blend(composite, Image.fromarray(dat), alpha=1/(i+1))
            #  composite = add_composite(composite, dat)

            # Write to file
            new_path = "%s/%s" % (self.out_dir, os.path.basename(filename))
            cv2.imwrite(new_path, dat)

        self.compositor.save(COMPOSITE_FILE_NAME)
        #  composite.save(COMPOSITE_FILE_NAME)
        end = timer()
        print("Processed warping in %s sec" % round(end - start, 2))

        return results

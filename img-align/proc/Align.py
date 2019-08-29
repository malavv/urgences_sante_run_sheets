import cv2
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from matplotlib import pyplot as plt

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.40


def prepare_img(filename):
    """
    Preparing a file from a complete path.
    :param filename: Path to the file
    :return: Grayscale Image
    """
    return cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)


class Align:
    def __init__(self, ref_filename, imgs_filename):
        self.ref = ref_filename
        self.inputs = imgs_filename

        self.writeHomography = False
        self.writeMatches = False

        print("Aligner initialized with with %s images to process" % len(self.inputs))

    def __repr__(self):
        return "[%s, %s]" % (self.ref, self.inputs)

    def find_matches(self, orb, img, ref_descriptors, match_quality_threshold):
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

    def warp_img_by_matches(self, img, ref, ref_keypoints, keypoints, matches, idx_to_use):

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
        img_dic = {}
        keypoints_dic = {}
        matches_dic = {}

        # Loading Images
        for filename in self.inputs:
            img_dic[filename] = prepare_img(filename)

        # A before composite is difficult to do here, because the images don't all have the same size

        # Processing
        print("Starting matching")
        start = timer()
        for filename, img in img_dic.items():
            # Find Best Matches
            keypoints, matches = self.find_matches(orb, img, ref_descriptors, GOOD_MATCH_PERCENT)
            keypoints_dic[filename] = keypoints
            matches_dic[filename] = matches
            img_dic[filename] = img

            # Draw top matches
            if self.writeMatches:
                cv2.imwrite("matches.jpg", cv2.drawMatches(filename, keypoints, ref, ref_keypoints, matches, None))
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
        threshold = 0.3 * len(self.inputs)
        idx_to_use = []
        for idx, num_uses in common_pts.items():
            if num_uses < threshold:
                continue
            else:
                idx_to_use.append(idx)

        to_draw = [kpt for i, kpt in enumerate(ref_keypoints) if i in idx_to_use]
        img2 = cv2.drawKeypoints(ref, to_draw, None, color=(20, 200, 20), flags=0)
        cv2.imwrite("matches.jpg", img2)

        print("Starting warping")
        start = timer()
        for filename in self.inputs:
            # Warp image
            dat, hom = self.warp_img_by_matches(
                img_dic[filename], ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)

            # Save warped image
            results[filename] = dat
        end = timer()
        print("Processed warping in %s sec" % round(end - start, 2))

        composite = Image.fromarray(ref)
        for filename, img in results.items():
            composite = Image.blend(composite, Image.fromarray(img), alpha=0.5)
        composite.save("after-composite.jpg")

        return results
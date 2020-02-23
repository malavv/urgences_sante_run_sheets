import cv2
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import os


class Align:
    KEYPOINT_COLOR_MATCH_FILE = (20, 200, 20)
    MATCH_FILE_NAME = "matches.jpg"
    COMPOSITE_FILE_NAME = "composite.jpg"

    def __init__(self, ref_filename, images_filename, out_dir, compositor):
        self.ref = cv2.imread(ref_filename, cv2.IMREAD_GRAYSCALE)
        self.inputs = images_filename
        self.out_dir = out_dir
        self.compositor = compositor

        print("Aligner initialized %s" % self)

    def __repr__(self):
        return "[num: %d, shape: %s, output: %s]" % (len(self.inputs), self.ref.shape, self.out_dir)

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

        # Transform IMG
        warped_img = cv2.warpPerspective(img, h, (width, height))

        return warped_img, h

    def warp_sheets(self, ref, ref_keypoints, keypoints_dic, matches_dic, idx_to_use, is_dry_run):
        homographic_dist = {}
        exception_file = {}
        for i, filename in enumerate(tqdm(self.inputs)):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            # Warp image
            try:
                dat, hom = self.warp_img_by_matches(img, ref, ref_keypoints, keypoints_dic[filename], matches_dic[filename], idx_to_use)
            except cv2.error as e:
                exception_file[filename] = format(e)
                continue

            # Add to composite
            if self.compositor is not None:
                self.compositor.compose(dat)

            # Add Homography to list
            homographic_dist[filename] = cv2.norm(hom)

            # Write to file
            new_path = "%s/%s" % (self.out_dir, os.path.basename(filename))
            if is_dry_run:
                print("write aligned image to %s" % new_path)
            else:
                cv2.imwrite(new_path, dat)
        return homographic_dist, exception_file

    def matches(self, orb, ref_descriptors, perc_quality_matches):
        keypoints_dic = {}
        matches_dic = {}
        for filename in tqdm(self.inputs):
            # Load image
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # Find Best Matches
            keypoints, matches = self.find_matches(orb, img, ref_descriptors, perc_quality_matches)
            # Image structure
            keypoints_dic[filename] = keypoints
            matches_dic[filename] = matches
        return keypoints_dic, matches_dic

    def get_common_matches(self, matches_dic, perc_common_matches):
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
        # Keep only points for which matches were found in at least X% of the images.
        threshold = perc_common_matches * len(self.inputs)
        idx_to_use = []
        for idx, num_uses in common_pts.items():
            if num_uses < threshold:
                continue
            else:
                idx_to_use.append(idx)

        return idx_to_use

    def align(self, num_features, perc_quality_matches, perc_common_matches, print_diagnostics, is_dry_run):
        # A 'before' composite is difficult to do here, because the images don't all have the same size

        # Load Reference Image
        orb = cv2.ORB_create(num_features)
        ref_keypoints, ref_descriptors = orb.detectAndCompute(self.ref, None)

        print("Starting matching")
        start = timer()
        keypoints_dic, matches_dic = self.matches(orb, ref_descriptors, perc_quality_matches)
        idx_to_use = self.get_common_matches(matches_dic, perc_common_matches)
        end = timer()
        print("Processed matches in %s sec" % round(end - start, 2))

        print("Starting warping")
        start = timer()
        homographic_dist, exception_file = self.warp_sheets(self.ref, ref_keypoints, keypoints_dic, matches_dic, idx_to_use, is_dry_run)
        end = timer()
        print("Processed warping in %s sec" % round(end - start, 2))

        if self.compositor is not None:
            self.compositor.save(Align.COMPOSITE_FILE_NAME)

        if print_diagnostics:
            to_draw = [kpt for i, kpt in enumerate(ref_keypoints) if i in idx_to_use]
            img2 = cv2.drawKeypoints(self.ref, to_draw, None, color=Align.KEYPOINT_COLOR_MATCH_FILE, flags=0)
            cv2.imwrite(Align.MATCH_FILE_NAME, img2)
            with open('dist.txt', 'w') as f:
                for k, v in homographic_dist.items():
                    f.write("%s\t%s\n" % (k, v))
            with open('except.txt', 'w') as f:
                for k, v in exception_file.items():
                    f.write("%s\t%s\n" % (k, v))


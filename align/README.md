# Image Alignment

The scanning process for RIPs introduced slight variations in the digitized forms. This alignment process uses one of
the RIP has a reference sheet, finds a set of common features between the reference sheet and all other sheets, and then
aligns all sheets based on these common features.

It works by finding sections of the reference RIP which are easy to locate and visible in most images. It uses
[ORB](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) 
(Oriented FAST and Rotated BRIEF) algorithm for keypoint detection.

From the article on: [learnopencv.com](https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)

## Features
- [x] Composite Generation (additive, averaging)
- [x] Reference RIP
- [x] Progress Tracking

## Parameters

| Param               | Description      |
|:--------------------|:-----------------|
| --ref FILE          | Reference RIP    |
| --img FILE_WILDCARD | Image Pattern    |
| --out DIR           | Output Directory |

## Optional

| Param                        | Description                             | Default |
|:-----------------------------|:----------------------------------------|:--------|
| --NUM_FEATURES number        | Number of features to find              | 5000    |
| --GOOD_MATCH_PERCENT percent | % of top matches to keep                | 0.4     |
| --PERC_COMMON_THRESH percent | Matches must be in at least % of sheets | 0.4     |
| --COMPOSITE type             | {ADD, MEAN, NONE}                       | NONE    |

## Example 
```{bash}
# Example call
python3 img-align/ \
    --ref "../data/us/forms/21543780.png" \
    --img "../data/us/forms/*.png" \
    --out "../data/us/aligned-forms"
```

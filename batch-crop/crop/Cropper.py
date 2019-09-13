import cv2
import os
from tqdm import tqdm


def get_name_template(filename):
    """
    Returns a filename template to be filled with the crop key.
    :param filename: Original image filename
    :return: filename template like "213123_%s.png"
    """
    base = os.path.basename(filename)
    sp = os.path.splitext(base)

    return "%s_%s%s" % (sp[0], "%s", sp[1])


def extract_conf(crop_config):
    """
    Uses a configuration line and deconstructs it into its components.
    :param crop_config: Crop Config like 100x50+200+200
    :return: Its component deconstructed (x, y, w, h)
    """
    tokens = crop_config.split("+")  # [0] = <width>x<height>, [1] = <x>, [2] = <y>
    wh = tokens[0].split("x")  # <width>x<height>
    return int(tokens[1]), int(tokens[2]), int(wh[0]), int(wh[1])  # X, Y, W, H


class Cropper:
    def __init__(self, images_filename, out_dir):
        self.inputs = images_filename
        self.out_dir = out_dir

    def __repr__(self):
        return "(%s, %s)" % (self.inputs, self.out_dir)

    def crop(self, crop_config):
        """
        Crops all images, using this cropping configuration and save it in the output dir.
        :param crop_config: The parse cropping configuration
        """
        crops = crop_config['crops'].items()

        for filename in tqdm(self.inputs):
            img = cv2.imread(filename)
            name_template = get_name_template(filename)

            for key, conf in crops:
                x, y, w, h = extract_conf(conf)
                crop_img = img[y:y + h, x:x + w]
                dst_image_filename = "%s/%s" % (self.out_dir, name_template % key)
                cv2.imwrite(dst_image_filename, crop_img)

from PIL import Image
import os
import numpy as np


def multiplicative_inverse_blend(acc, item, index):
    """
    Creates a running average of the images. (age biased above 255)

    If > 255 images, this will have an "age" bias. You won't see image past 255 represented in the composite.
    :param acc: Accumulator, here the self.composite
    :param item: The new image to add.
    :param index: Index of the additional image (always increasing)
    :return: Composite Image
    """
    return Image.blend(acc, item, alpha=1 / (index + 1))


def lossless_blend(acc, item, index):
    """
    Creates a running average of the images. (creates another composite if about to loose info)

    :param acc: Accumulator, here the self.composite
    :param item: The new image to add.
    :param index: Index of the additional image (always increasing)
    :return: Composite Image
    """
    # make composite an array
    acc = acc if isinstance(acc, list) else [acc]

    # Add new image at start of array
    if (index % 256) == 255:
        acc.index(item, 0)
    # Else update the composite at start of array
    else:
        acc[0] = Image.blend(acc[0], item, alpha=1 / (index + 1))

    return acc


def additive_blend(acc, item, index):
    """
    Creates a purely additive composite. Pixels are written and never taken back.

    :param acc: Accumulator, here the self.composite
    :param item: The new image to add.
    :param index: Index of the additional image (always increasing)
    :return: Composite Image
    """
    # Minimum here because black is 0 and white is 255
    return Image.fromarray(np.minimum(np.array(acc), np.array(item)))


class Compositor:
    def __init__(self, func):
        self.num = 0
        self.func = func
        self.composite = None

    def __repr__(self):
        return "compositor has seen %s images" % self.num

    @staticmethod
    def get_running_avg():
        return Compositor(multiplicative_inverse_blend)

    @staticmethod
    def get_lossless_running_avg():
        return Compositor(lossless_blend)

    @staticmethod
    def get_additive():
        return Compositor(additive_blend)

    def compose(self, img_dat):
        """Compose additional image"""
        img = Image.fromarray(img_dat)
        # The first composite is always the input image
        if self.composite is None:
            self.composite = img
            return self.composite
        # Change the composite builder
        self.composite = self.func(self.composite, img, self.num)
        # Index of the current composition
        self.num = self.num + 1

    def save(self, filename):
        """Save composite image"""

        split = os.path.splitext(filename)
        base = split[0]
        ext = split[1]

        composites = self.composite if isinstance(self.composite, list) else [self.composite]

        i = 0
        for composite in composites:
            composite.save("%s_%s%s" % (base, i, ext))
            i = i + 1

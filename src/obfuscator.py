"""
This module contains the ImageObfuscator class, which is used to obfuscate images based on a
set of masks and policies.
"""

import cupy as cp
import yaml
from cupyx.scipy import ndimage


class ImageObfuscator:
    def __init__(self, policies: dict):
        self.policies = policies
        with open(file="config.yml", mode="r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            self.available_policies = config["obfuscation_types"]

    def validate_inputs(self, masks: cp.ndarray, image: cp.ndarray):
        if (
            masks is not None
            and len(masks) > 0
            and (len(masks.shape) != 3 or masks.dtype != cp.bool_)
        ):
            raise ValueError(
                f"masks has shape {masks.shape} and dtype {masks.dtype}, expected shape (n, H, W) and dtype bool"
            )

        if len(image.shape) != 3 or image.shape[2] != 3 or image.dtype != cp.uint8:
            raise ValueError(
                f"image has shape {image.shape} and dtype {image.dtype}, expected shape (H, W, 3) and dtype uint8"
            )

        for policy in self.policies.values():
            if policy not in self.available_policies:
                raise ValueError(f"Unknown policy: {policy}")

    @staticmethod
    def apply_mask(img: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
        mask = cp.broadcast_to(mask[:, :, cp.newaxis], img.shape).astype(cp.uint8) * 255
        img[mask != 0] = 0
        return img

    @staticmethod
    def apply_blur(image: cp.ndarray, mask: cp.ndarray, sigma: int = 10) -> cp.ndarray:
        blurred_image = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
        image[mask != 0] = blurred_image[mask != 0, :3]
        return image

    @staticmethod
    def apply_pixelate(
        image: cp.ndarray, mask: cp.ndarray, square: int = 20
    ) -> cp.ndarray:
        image_cp = cp.asarray(image, dtype=cp.uint8)
        mask = mask[:, :, cp.newaxis].astype(cp.uint8) * 255
        # Downsampling followed by Upsampling to create the pixelated effect
        img_small = image_cp[::square, ::square, :]
        pixelated_img = cp.repeat(cp.repeat(img_small, square, axis=0), square, axis=1)
        # Crop the pixelated image to match the shape of the original image
        pixelated_img = pixelated_img[: image_cp.shape[0], : image_cp.shape[1], :]
        # Apply the pixelation effect only to the masked region
        final_image = cp.where(mask != 0, pixelated_img, image_cp)
        return final_image

    def obfuscate(self, masks: cp.ndarray, image: cp.ndarray, class_ids: list):
        self.validate_inputs(masks, image)

        # Create a copy of the original image
        image_copy = image.copy()

        for mask, class_id in zip(masks, class_ids):
            policy = self.policies.get(class_id)
            if policy == "masking":
                image_copy = self.apply_mask(image_copy, mask)
            elif policy == "blurring":
                image_copy = self.apply_blur(image_copy, mask)
            elif policy == "pixelation":
                image_copy = self.apply_pixelate(image_copy, mask)
            else:
                pass

        return cp.asarray(
            image_copy.get()
        )  # TODO: do we need to call get() to turn it into a numpy array in CPU?


class Colors:
    def __init__(self, num_categories: int):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.num = len(self.palette)
        self.colors_dict = {
            i: self.palette[i % self.num] for i in range(num_categories)
        }

    def __call__(self, i: int, bgr=False):
        """Converts hex color codes to RGB values."""
        channel: tuple[int, ...] = self.colors_dict[i]
        return (channel[2], channel[1], channel[0]) if bgr else channel

    def get_colors_dict(self):
        """Returns the colors dictionary."""
        return self.colors_dict

    @staticmethod
    def hex2rgb(hexa: str):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(hexa[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

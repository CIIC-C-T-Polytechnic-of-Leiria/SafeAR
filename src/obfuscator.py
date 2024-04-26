"""
This module contains the ImageObfuscator class, which is used to obfuscate
images based on a set of masks and policies.
"""

# TODO: Verify is GPu is being used in all operations
import cupy as cp
import yaml
from cucim.skimage import transform
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

    import cupy as cp

    @staticmethod
    def apply_mask(image: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
        # Assuming mask is already boolean
        cp.putmask(image, ~mask, 0)  # set masked area to 0 in-place
        return image

    # @staticmethod
    # def apply_mask(image: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
    #     """
    #     Applies a mask to an image, obfuscating the masked area.
    #
    #     Args:
    #         image (cp.ndarray): Image to be obfuscated
    #         mask (cp.ndarray): Mask to be applied to the image
    #
    #     Returns:
    #         cp.ndarray: Obfuscated image
    #     """
    #     # print(f"DEBUG: ap image type: {type(image)}, mask type: {type(mask)}")
    #     mask = mask[:, :, cp.newaxis].astype(cp.uint8) * 255
    #     inverse_mask = cp.logical_not(mask)
    #     # print(
    #     #     f"DEBUG: inverse_mask type: {type(inverse_mask)}, image type: {type(image)}"
    #     # )
    #     masked_image = cp.where(
    #         inverse_mask, image, cp.array(0)
    #     )  # set masked area to 0
    #     return cp.asarray(masked_image.get())

    @staticmethod
    def apply_blur(image: cp.ndarray, mask: cp.ndarray):
        mask = mask[:, :, cp.newaxis].astype(cp.uint8) * 255
        blurred_image = ndimage.median_filter(image, size=(51, 51, 1))
        mask_inv = cp.logical_not(mask)
        final_image = cp.where(mask_inv, image, blurred_image)
        return cp.asarray(final_image.get())

    @staticmethod
    def apply_pixelate(
        image: cp.ndarray, mask: cp.ndarray, rescale: int = 20
    ) -> cp.ndarray:

        image_cp = cp.asarray(image, dtype=cp.uint8)

        pixelated_img_small = transform.resize(
            image=image_cp,
            order=0,
            output_shape=(image_cp.shape[0] // rescale, image_cp.shape[1] // rescale),
            preserve_range=True,
            anti_aliasing=False,
        )
        pixelated_image = transform.resize(
            image=pixelated_img_small,
            order=0,
            output_shape=(image_cp.shape[0], image_cp.shape[1]),
            preserve_range=True,
            anti_aliasing=False,
        )

        mask = mask[:, :, cp.newaxis].astype(cp.uint8) * 255
        mask_inv = cp.logical_not(mask)
        final_image = cp.where(mask_inv, image_cp, pixelated_image)

        return final_image

    def obfuscate(self, masks: cp.ndarray, image: cp.ndarray, class_ids: list):

        self.validate_inputs(masks, image)

        # print(f"DEBUG: ob. masks type: {type(masks)}, image type: {type(image)}")

        for mask, class_id in zip(masks, class_ids):
            policy = self.policies.get(class_id)
            if policy == "masking":
                # print(f"DEBUG: image type at start of masking: {type(image)}")
                image = self.apply_mask(image, mask)
                # print(f"DEBUG: image type at end of masking: {type(image)}")
            elif policy == "blurring":
                image = self.apply_blur(image, mask)
            elif policy == "pixelation":
                image = self.apply_pixelate(image, mask)
            else:
                pass

        return cp.asarray(image.get())


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

import cv2
import numpy as np


class ImageObfuscator:
    """
    The ImageObfuscator class applies various obfuscation policies to an image.

    Attributes:
        policies (dict): A dictionary mapping class labels to obfuscation policies.
        available_policies (list): A list of available obfuscation policies.

    Methods:
        validate_inputs(masks, image): Validates the inputs to the obfuscate method.
        apply_mask(image, mask): Applies a mask to the image.
        apply_blur(image, mask): Applies a blur to the image.
        apply_pixelate(image, mask): Applies pixelation to the image.
        obfuscate(masks, image, classes): Applies the obfuscation policies to the image.
    """

    def __init__(self, policies: dict):
        self.policies = policies
        self.available_policies = ["masking", "bluring", "pixelation", "none"]

    def validate_inputs(self, masks, image):
        if masks is not None and len(masks) > 0:
            if len(masks.shape) != 3 or masks.dtype != np.bool_:
                raise ValueError(
                    f"masks has shape {masks.shape} and dtype {masks.dtype}, expected shape (n, H, W) and dtype bool"
                )

        if len(image.shape) != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"image has shape {image.shape} and dtype {image.dtype}, expected shape (H, W, 3) and dtype uint8"
            )

        for policy in self.policies.values():
            if policy not in self.available_policies:
                raise ValueError(f"Unknown policy: {policy}")

    def apply_mask(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        inverse_mask = cv2.bitwise_not(mask)
        masked_image = cv2.bitwise_and(image, image, mask=inverse_mask)
        return masked_image

    def apply_blur(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        blurred_image = cv2.medianBlur(image, ksize=51)
        mask_inv = cv2.bitwise_not(mask)
        final_image = cv2.bitwise_and(image, image, mask=mask_inv) + cv2.bitwise_and(
            blurred_image, blurred_image, mask=mask
        )
        return final_image

    def apply_pixelate(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        pixelated_image = cv2.resize(
            image, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST
        )
        pixelated_image = cv2.resize(
            pixelated_image,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_inv = cv2.bitwise_not(mask)
        final_image = cv2.bitwise_and(image, image, mask=mask_inv) + cv2.bitwise_and(
            pixelated_image, pixelated_image, mask=mask
        )
        return final_image

    def obfuscate(self, masks, image, class_ids):
        self.validate_inputs(masks, image)
        for mask, class_id in zip(masks, class_ids):
            policy = self.policies.get(class_id)
            if policy == "masking":
                image = self.apply_mask(image, mask)
            elif policy == "bluring":
                image = self.apply_blur(image, mask)
            elif policy == "pixelation":
                image = self.apply_pixelate(image, mask)
            else:
                pass
        return image


# class Colors:

#     def __init__(self):
#         """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
#         hexs = (
#             "FF3838",
#             "FF9D97",
#             "FF701F",
#             "FFB21D",
#             "CFD231",
#             "48F90A",
#             "92CC17",
#             "3DDB86",
#             "1A9334",
#             "00D4BB",
#             "2C99A8",
#             "00C2FF",
#             "344593",
#             "6473FF",
#             "0018EC",
#             "8438FF",
#             "520085",
#             "CB38FF",
#             "FF95C8",
#             "FF37C7",
#         )
#         self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
#         self.n = len(self.palette)
#         self.pose_palette = np.array(
#             [
#                 [255, 128, 0],
#                 [255, 153, 51],
#                 [255, 178, 102],
#                 [230, 230, 0],
#                 [255, 153, 255],
#                 [153, 204, 255],
#                 [255, 102, 255],
#                 [255, 51, 255],
#                 [102, 178, 255],
#                 [51, 153, 255],
#                 [255, 153, 153],
#                 [255, 102, 102],
#                 [255, 51, 51],
#                 [153, 255, 153],
#                 [102, 255, 102],
#                 [51, 255, 51],
#                 [0, 255, 0],
#                 [0, 0, 255],
#                 [255, 0, 0],
#                 [255, 255, 255],
#             ],
#             dtype=np.uint8,
#         )

#     def __call__(self, i, bgr=False):
#         """Converts hex color codes to RGB values."""
#         c = self.palette[int(i) % self.n]
#         return (c[2], c[1], c[0]) if bgr else c

#     @staticmethod
#     def hex2rgb(h):
#         """Converts hex color codes to RGB values (i.e. default PIL order)."""
#         return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


# colors = Colors()  # create instance for 'from utils.plots import colors'

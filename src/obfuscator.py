import cupy as cp
from cucim.skimage import transform


class ImageObfuscator:
    def __init__(self, policies: dict):
        self.policies = policies
        self.available_policies = ["masking", "bluring", "pixelation", "none"]

    def validate_inputs(self, masks: cp.ndarray, image: cp.ndarray):
        if masks is not None and len(masks) > 0:
            if len(masks.shape) != 3 or masks.dtype != cp.bool_:
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

    def apply_mask(self, image: cp.ndarray, mask: cp.ndarray):
        mask = mask.astype(cp.uint8) * 255
        inverse_mask = cp.logical_not(mask)
        masked_image = cp.where(inverse_mask, image, cp.array(0)) # set masked area to 0
        return masked_image.get()

    def apply_blur(self, image: cp.ndarray, mask: cp.ndarray):
        mask = mask.astype(cp.uint8) * 255
        blurred_image = cp.ndimage.median_filter(image, size=(51, 51, 1))
        mask_inv = cp.logical_not(mask)
        final_image = cp.where(mask_inv, image, blurred_image)
        return final_image.get()

    def apply_pixelate(
        self, image: cp.ndarray, mask: cp.ndarray, rescale: int = 30
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

        mask = mask.astype(cp.uint8) * 255
        mask = cp.repeat(mask[..., None], 3, axis=-1)
        final_image = cp.where(mask == 0, pixelated_image, image_cp)

        return final_image

    def obfuscate(self, masks: cp.ndarray, image: cp.ndarray, class_ids: list):
        
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

        return image.get()


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
        self.n = len(self.palette)
        self.colors_dict = {i: self.palette[i % self.n] for i in range(num_categories)}

    def __call__(self, i: int, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.colors_dict[i]
        return (c[2], c[1], c[0]) if bgr else c

    def get_colors_dict(self):
        """Returns the colors dictionary."""
        return self.colors_dict

    @staticmethod
    def hex2rgb(h: str):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

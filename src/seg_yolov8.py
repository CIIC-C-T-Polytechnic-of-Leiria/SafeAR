import cupy as cp
import onnxruntime as ort
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from cucim.skimage import transform


class Yolov8Seg:
    def __init__(
        self,
        model_path: str,
        target_size: Tuple[int, int] = (640, 640),
        interpolation: int = 11,
        output_path: str = None,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.55,
        num_masks: int = 32,
    ):
        self.model_path = model_path
        self.target_size = target_size
        self.interpolation = interpolation
        self.output_path = output_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_masks = num_masks

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        # self.session = ort.InferenceSession(
        #     self.model_path,
        #     providers=(
        #         ["CUDAExecutionProvider", "CPUExecutionProvider"]
        #         if ort.get_device() == "GPU"
        #         else ["CPUExecutionProvider"]
        #     ),
        # )

        print(f"Yolov8Seg: Model Using {ort.get_device()} for inference")

        self.ndtype = (
            cp.half  # cp.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else cp.single  # cp.float32
        )

        self.model_input_height, self.model_input_width = [
            x.shape for x in self.session.get_inputs()
        ][0][-2:]

        self.model_output_height, self.model_output_width = [
            x.shape for x in self.session.get_outputs()
        ][0][-2:]

    def __call__(self, img_in: cp.ndarray) -> cp.ndarray:
        image, ratio, (pad_w, pad_h) = self.preprocess_image(img_in)
        print(
            f"DEBUG: image shape: {image.shape}, self.session.get_inputs()[0].name: {self.session.get_inputs()[0].name}"
        )
        preds = self.session.run(None, {self.session.get_inputs()[0].name: image})
        return self.postprocess(
            preds,
            ratio,
            (pad_w, pad_h),
            self.num_masks,
            self.confidence_threshold,
            self.iou_threshold,
            img_in,
        )

    def postprocess(
        self,
        preds: cp.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        nm: int,
        conf_threshold: float,
        iou_threshold: float,
        img_0: cp.ndarray,
    ) -> cp.ndarray:

        instnc_prds, protos = (
            preds[0],
            preds[1],
        )  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        instnc_prds = cp.einsum("bcn->bnc", instnc_prds)

        # Predictions filtering by conf-threshold
        instnc_prds = instnc_prds[
            cp.amax(instnc_prds[..., 4:-nm], axis=-1) > conf_threshold
        ]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        instnc_prds = cp.c_[
            instnc_prds[..., :4],
            cp.amax(instnc_prds[..., 4:-nm], axis=-1),
            cp.argmax(instnc_prds[..., 4:-nm], axis=-1),
            instnc_prds[..., -nm:],
        ]

        # NMS filtering by iou-threshold;
        # Since cv2.dnn.NMSBoxes is not available in CuPy, you need to find an alternative method for non-maximum suppression.
        # One possible alternative is to use the 'cupyx.scipy.spatial.distance.cdist' function to calculate the IoU and perform NMS manually.

        # Each element in "instnc_prds" is a box [4 elements], score, classID, mask coefficients [32 elements]

        # Decode and return
        if len(instnc_prds) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            instnc_prds[..., [0, 1]] -= instnc_prds[..., [2, 3]] / 2
            instnc_prds[..., [2, 3]] += instnc_prds[..., [0, 1]]

            # Rescales bounding boxes from model shape to the shape of original image
            instnc_prds[..., :4] -= [padding[0], padding[1], padding[0], padding[1]]

            # Bounding boxes boundary clamp
            instnc_prds[..., [0, 2]] = instnc_prds[:, [0, 2]].clip(
                0, self.model_input_width
            )  # x1, x2
            instnc_prds[..., [1, 3]] = instnc_prds[:, [1, 3]].clip(
                0, self.model_input_height
            )

            masks = self.process_mask(
                protos=protos[0],  # (32, 160, 160)
                masks_coef=instnc_prds[..., 6:],  # 32 mask coefficients
                bboxes=instnc_prds[..., :4],  # boxes shape: (N, 4)
                img_0_shape=img_0.shape[0:2],  # (H, W) of original image
            )

            return instnc_prds[..., :6], masks  # boxes, masks
        else:
            return [], []

    def process_mask(
        self,
        protos: cp.ndarray,
        masks_coef: cp.ndarray,
        bboxes: cp.ndarray,
        img_0_shape: Tuple[int, int],
    ) -> cp.ndarray:
        c, mh, mw = protos.shape

        masks = (
            cp.matmul(masks_coef, protos.reshape((c, -1)))
            .reshape((-1, mh, mw))
            .transpose(1, 2, 0)
        )  # HWN

        masks = cp.ascontiguousarray(masks)

        masks = self.scale_mask(
            masks, img_0_shape
        )  # re-scale mask from P3 shape to original input image shape

        masks = cp.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return cp.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks: cp.ndarray, img_0_shape: Tuple[int, int], ratio_pad=None):
        img_1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from img_0_shape
            gain = cp.min(
                img_1_shape[0] / img_0_shape[0], img_1_shape[1] / img_0_shape[1]
            )  # gain  = old / new
            pad = (img_1_shape[1] - img_0_shape[1] * gain) / 2, (
                img_1_shape[0] - img_0_shape[0] * gain
            ) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(cp.round(pad[1] - 0.1)), int(cp.round(pad[0] - 0.1))  # y, x
        bottom, right = int(cp.round(img_1_shape[0] - pad[1] + 0.1)), int(
            cp.round(img_1_shape[1] - pad[0] + 0.1)
        )

        masks = masks[top:bottom, left:right]

        masks = transform.resize(
            image=masks, output_shape=img_0_shape, order=0, anti_aliasing=False
        )
        # # Since cv2.resize is not available in CuPy, you need to use 'cupyx.scipy.ndimage.zoom' function to resize the masks.

        masks = cp.ascontiguousarray(masks)

        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    @staticmethod
    def crop_mask(masks: cp.ndarray, boxes: cp.ndarray) -> cp.ndarray:
        n, h, w = masks.shape
        x1, y1, x2, y2 = cp.split(boxes[:, :, None], 4, 1)
        r = cp.arange(w, dtype=x1.dtype)[None, None, :]
        c = cp.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def preprocess_image(
        self, image: cp.ndarray
    ) -> Tuple[cp.ndarray, float, Tuple[int, int]]:

        resized_image, resize_ratio, padding = self.resize_and_pad(image)
        model_input = self.convert_to_model_input(resized_image)

        return model_input, resize_ratio, padding

    def resize_and_pad(
        self, img: cp.ndarray
    ) -> Tuple[cp.ndarray, float, Tuple[int, int]]:
        """
        This function adjusts an image to a specified size by resizing it while keeping its aspect ratio,
        then padding it as needed. It returns the modified image, the resize ratio, and the padding dimensions.

        Parameters:
            img (cp.ndarray): The original image to be resized and padded.

        Returns:
            p_img (cp.ndarray): The padded image.
            r_ratio (float): The resize ratio.
            (p_w, p_h) (Tuple[int, int]): The padding width and height.
        """
        o_shape = img.shape[:2]
        t_size = (self.model_input_height, self.model_input_width)

        r_ratio = cp.minimum(t_size[0] / o_shape[0], t_size[1] / o_shape[1])
        r_dim = int(cp.round(o_shape[1] * r_ratio)), int(cp.round(o_shape[0] * r_ratio))

        p_w, p_h = (t_size[1] - r_dim[0]) / 2, (t_size[0] - r_dim[1]) / 2

        r_img = transform.resize(img, (r_dim[1], r_dim[0]), anti_aliasing=True)

        p_t, p_b = int(cp.round(p_h - 0.1)), int(cp.round(p_h + 0.1))
        p_l, p_r = int(cp.round(p_w - 0.1)), int(cp.round(p_w + 0.1))
        p_img = cp.pad(
            r_img,
            ((p_t, p_b), (p_l, p_r), (0, 0)),
            mode="constant",
            constant_values=114,
        )

        return p_img, r_ratio, (p_w, p_h)

    def convert_to_model_input(self, image: cp.ndarray) -> cp.ndarray:
        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        image = (
            cp.ascontiguousarray(cp.einsum("HWC->CHW", image)[::-1], dtype=self.ndtype)
            / 255.0
        )
        img_process = image[None] if len(image.shape) == 3 else image
        return img_process

    @staticmethod
    def draw_bbox(
        img: cp.ndarray,
        bbox: cp.ndarray,
        class_name: str,
        policy: str,
        score: float,
        color: Tuple[int, int, int],
        thickness=1,
        font_scale=0.8,
        padding: cp.ndarray = None,
    ):
        # Convert the CuPy array to a PIL Image
        img_pil = Image.fromarray(cp.asnumpy(img))

        # Create an ImageDraw object
        draw = ImageDraw.Draw(img_pil)

        # Define the font (replace 'arial.ttf' with the path to your desired font file)
        font_path = "arial.ttf"
        font_size = int(font_scale * 16)
        font = ImageFont.truetype(font_path, font_size)

        # Reverse padding and aspect ratio
        if padding is not None:
            bbox += padding

        bbox = bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Draw the bounding box rectangle (using CuPy-based implementation)
        custom_draw_rectangle(img, bbox, color, thickness)

        # Draw the text using Pillow's ImageDraw
        text = f"{policy} | {class_name} : {score:.2f}"
        text_width, text_height = draw.textsize(text, font=font)
        draw.text(
            (x1 + 2, y1 + 11),
            text,
            fill=color,
            font=font,
        )

        # Convert the PIL Image back to a CuPy array
        img_cupy = cp.asarray(img_pil)

    # @staticmethod
    # def masks2segments(masks: np.ndarray) -> np.ndarray:
    #     segments = []
    #     for x in masks.astype("uint8"):
    #         c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
    #             0
    #         ]  # CHAIN_APPROX_SIMPLE
    #         if c:
    #             c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
    #         else:
    #             c = np.zeros((0, 2))  # no segments found
    #         segments.append(c.astype("float32"))
    #     return segments


def custom_draw_rectangle(
    img: cp.ndarray, bbox: cp.ndarray, color: Tuple[int, int, int], thickness: int
):
    x1, y1, x2, y2 = bbox
    img[y1 : y1 + thickness, x1:x2] = color
    img[y2 : y2 + thickness, x1:x2] = color
    img[y1:y2, x1 : x1 + thickness] = color
    img[y1:y2, x2 : x2 + thickness] = color
    return img

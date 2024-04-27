from time import time
from typing import Tuple

import cupy as cp
# Debugging
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from cucim.skimage import transform
from cupyx.scipy.special import expit


class Yolov8seg:
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
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_masks = num_masks

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 4
        options.enable_profiling = False

        exec_provider = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=exec_provider,
        )

        print(f"yolov8-seg using {ort.get_device()} for inference")

        self.dtype = (
            cp.half  # cp.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else cp.single  # cp.float32
        )

        self.model_in_height, self.model_in_width = [
            x.shape for x in self.session.get_inputs()
        ][0][-2:]

        self.model_output_height, self.model_output_width = [
            x.shape for x in self.session.get_outputs()
        ][0][-2:]

    def __call__(self, img_in: cp.ndarray) -> cp.ndarray:
        start_time = time()

        # Preprocess the image
        img, ratios, (pad_w, pad_h) = self.preproc_img(img=img_in)

        end_time1 = time()
        print(f"MOD: PREPROC TIME {((end_time1 - start_time) * 1000):.1f} milliseconds")

        # Run the model
        model_out = self.session.run(
            None, {self.session.get_inputs()[0].name: cp.asnumpy(img)}
        )

        end_time2 = time()
        print(f"MOD: INFER TIME {((end_time2 - end_time1) * 1000):.1f} milliseconds")

        # Postprocess the model output
        b_boxes_scr_id, masks = self.postproc_img(
            preds=model_out,
            ratios=ratios,
            pad=(pad_w, pad_h),
            nm=self.num_masks,
            conf_thresh=self.conf_threshold,
            iou_thresh=self.iou_threshold,
            img_0=img_in,
        )

        end_time3 = time()
        print(f"MOD: POSTPROC TIME {((end_time3 - end_time2) * 1000):.1f} milliseconds")

        return b_boxes_scr_id, masks

    def postproc_img(
        self,
        preds: list,
        ratios: Tuple[float, float],  # (ratio_x, ratio_y)
        pad: Tuple[int, int],
        nm: int,
        conf_thresh: float,
        iou_thresh: float,
        img_0: cp.ndarray,
    ) -> cp.ndarray:

        start_time = time()
        # preds[0] = boxes, scores, classes, masks coeffs, preds[1] = protos masks of dim. = (32, 160, 160)
        box_scr_cls_mskc, protos = (
            cp.asarray(preds[0]),
            cp.asarray(preds[1]),
        )
        # Transpose box_scr_cls_mskc: (B, xywhcc, N) -> (B, N, xywhcc)
        box_scr_cls_mskc = cp.einsum("bcn->bnc", box_scr_cls_mskc)

        end_time1 = time()
        print(
            f"POST: POSTPROC1 TIME {((end_time1 - start_time) * 1000):.1f} milliseconds"
        )

        # Apply NMS
        box_scr_cls_mskc = apply_nms(box_scr_cls_mskc, conf_thresh, nm, iou_thresh)

        end_time2 = time()
        print(f"POST: NMS TIME {((end_time2 - end_time1) * 1000):.1f} milliseconds")

        # Decode and return
        if len(box_scr_cls_mskc) > 0:
            # Rescales bounding boxes from model shape to the shape of original image
            box_scr_cls_mskc[..., :4] -= cp.array([pad[0], pad[1], pad[0], pad[1]])

            # Bounding boxes boundary clamp
            box_scr_cls_mskc[..., [0, 2]] = box_scr_cls_mskc[:, [0, 2]].clip(
                0, self.model_in_width
            )  # x1, x2
            box_scr_cls_mskc[..., [1, 3]] = box_scr_cls_mskc[:, [1, 3]].clip(
                0, self.model_in_height
            )  # y1, y2

            end_time3 = time()
            print(
                f"POST: PAD AND CLAMP {((end_time3 - end_time2) * 1000):.1f} milliseconds"
            )

            b_boxes = self.resize_bboxes(
                box_scr_cls_mskc[..., :4], ratios, pad, img_0.shape[0:2]
            )

            end_time4 = time()
            print(
                f"POST: RESIZE BBOXES {((end_time4 - end_time3) * 1000):.1f} milliseconds"
            )

            bool_masks = self.process_mask(
                protos_=protos[0],  # (32, 160, 160)
                masks_coef=box_scr_cls_mskc[..., 6:],  # 32 mask coefficients
                bboxes=b_boxes,  # boxes shape: (N, 4)
                img_0_shape=img_0.shape[0:2],  # (H, W) of original image
            )

            end_time5 = time()
            print(
                f"POST: PROCESS MASK {((end_time5 - end_time4) * 1000):.1f} milliseconds"
            )
            b_boxes_scr_id = cp.concatenate(
                [b_boxes, box_scr_cls_mskc[..., 4:6]], axis=1
            )

            end_time6 = time()
            print(
                f"POST: CONCATENATE_{((end_time6 - end_time5) * 1000):.1f} milliseconds"
            )

            return b_boxes_scr_id, bool_masks

        else:
            return [], []

    @staticmethod
    def scale_mask(masks: cp.ndarray, img_0_shape: Tuple[int, int], ratio_pad=None):
        img_1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from img_0_shape
            gain = cp.amin(
                cp.array(
                    [img_1_shape[0] / img_0_shape[0], img_1_shape[1] / img_0_shape[1]]
                )
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

    @staticmethod
    def resize_bboxes(
        bboxes: cp.ndarray,
        ratios: Tuple[float, float],
        pad: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> cp.ndarray:
        """
        Resize bounding boxes to the original image size.

        Parameters:
            bboxes (cp.ndarray): Bounding boxes in format x1, y1, x2, y2.
            ratios (Tuple[float, float]): Resize ratios.
            pad (Tuple[int, int]): Padding dimensions.
            img_size (Tuple[int, int]): Original image size (height, width).

        Returns:
            bboxes (cp.ndarray): Resized bounding boxes.
        """
        ratio = cp.minimum(ratios[0], ratios[1])
        bboxes /= ratio

        bboxes[..., :4] = cp.round(bboxes[..., :4])

        bboxes[..., [0, 2]] = cp.clip(bboxes[..., [0, 2]], 0, img_size[1] - 1)
        bboxes[..., [1, 3]] = cp.clip(bboxes[..., [1, 3]], 0, img_size[0] - 1)

        return bboxes

    def process_mask(
        self,
        protos_: cp.ndarray,
        masks_coef: cp.ndarray,
        bboxes: cp.ndarray,
        img_0_shape: Tuple[int, int],
        mask_thresh: float = 0.5,
    ) -> cp.ndarray:

        c, h, w = protos_.shape  # 32, 160, 160

        # Perform matrix mult. between mask coeffs and reshaped protos,
        # then reshape the result to the mask shape (Result shape: HW-N)
        masks = (
            cp.matmul(masks_coef, protos_.reshape((c, -1)))
            .reshape((-1, h, w))
            .transpose(1, 2, 0)
        )

        # Sigmoid activation to flatten the masks to [0, 1]
        masks = expit(masks)

        # Rescale masks to the original image size (img_0_shape)
        masks = self.scale_mask(masks, img_0_shape)
        masks = cp.einsum("HWN -> NHW", masks)

        # Crop the masks to the bounding boxes
        masks = self.crop_mask(masks, bboxes)

        if not masks.flags["C_CONTIGUOUS"]:
            masks = cp.ascontiguousarray(masks)

        # Apply thresholding to the masks
        return cp.greater(masks, mask_thresh)

    def preproc_img(
        self, img: cp.ndarray
    ) -> Tuple[cp.ndarray, Tuple[float, float], Tuple[int, int]]:

        resized_img, resize_ratios, padding = self.resize_and_pad(img)

        model_in_img = self.convert_to_yolov8_input(resized_img)

        return model_in_img, resize_ratios, padding

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
        t_size = (self.model_in_height, self.model_in_width)

        r_ratio_x, r_ratio_y = t_size[1] / o_shape[1], t_size[0] / o_shape[0]
        r_ratio = cp.minimum(r_ratio_x, r_ratio_y)
        r_dim = int(cp.round(o_shape[1] * r_ratio)), int(cp.round(o_shape[0] * r_ratio))

        p_w, p_h = (t_size[1] - r_dim[0]) / 2, (t_size[0] - r_dim[1]) / 2

        r_img = transform.resize(img, (r_dim[1], r_dim[0]), anti_aliasing=True)

        p_t, p_b = int(cp.round(p_h - 0.1)), int(cp.round(p_h + 0.1))
        p_l, p_r = int(cp.round(p_w - 0.1)), int(cp.round(p_w + 0.1))
        p_img = cp.pad(
            r_img,
            ((p_t, p_b), (p_l, p_r), (0, 0)),
            mode="constant",
            constant_values=0.1,
        )

        return p_img, (r_ratio_x, r_ratio_y), (p_w, p_h)

    def convert_to_yolov8_input(self, img: cp.ndarray) -> cp.ndarray:
        """
        Transforms: HWC to CHW -> div(255) -> contiguous -> CHW to BCHW
        """
        img = cp.einsum("HWC->CHW", img[:, :, ::-1])
        img = cp.ascontiguousarray(img, dtype=self.dtype)
        img = cp.expand_dims(img, axis=0) if len(img.shape) == 3 else img

        return img

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


def custom_draw_rectangle(
    img: cp.ndarray, bbox: cp.ndarray, color: Tuple[int, int, int], thickness: int
):
    x1, y1, x2, y2 = bbox
    img[y1 : y1 + thickness, x1:x2] = color
    img[y2 : y2 + thickness, x1:x2] = color
    img[y1:y2, x1 : x1 + thickness] = color
    img[y1:y2, x2 : x2 + thickness] = color
    return img


def iou(box1: cp.ndarray, box2: cp.ndarray) -> cp.ndarray:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    box1 (cp.ndarray): First bounding box (x1, y1, x2, y2)
    box2 (cp.ndarray): Second bounding box (x1, y1, x2, y2)

    Returns:
    iou (cp.ndarray): IoU value between the two boxes
    """
    x1, y1, x2, y2 = box1

    # Calculate intersection area
    xi1 = cp.maximum(x1, box2[:, 0])
    yi1 = cp.maximum(y1, box2[:, 1])
    xi2 = cp.minimum(x2, box2[:, 2])
    yi2 = cp.minimum(y2, box2[:, 3])

    intersection = cp.maximum(0, xi2 - xi1) * cp.maximum(0, yi2 - yi1)

    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    return intersection / union


def apply_nms(
    predictions: cp.ndarray, conf_threshold: float, nm: int, iou_threshold: float
) -> cp.ndarray:
    # Predictions filtering by conf-threshold
    predictions = predictions[cp.max(predictions[..., 4:-nm], axis=-1) > conf_threshold]

    predictions = cp.c_[
        predictions[..., :4],
        cp.amax(predictions[..., 4:-nm], axis=-1),
        cp.argmax(predictions[..., 4:-nm], axis=-1),
        predictions[..., -nm:],
    ]

    # Sort by confidence score (index 4) in descending order
    predictions = predictions[predictions[:, 4].argsort()[::-1]]

    # Bounding boxes format change: cxcywh -> xyxy
    center = predictions[..., [0, 1]].copy()
    predictions[..., [0, 1]] -= predictions[..., [2, 3]] / 2  # x1, y1
    predictions[..., [2, 3]] = center + predictions[..., [2, 3]] / 2  # x2, y2

    # Pre-allocate an array to store the filtered boxes
    filtered_boxes = cp.empty((0, predictions.shape[1]), dtype=predictions.dtype)

    while len(predictions) > 0:
        # Append the first box to the filtered boxes array
        filtered_boxes = cp.vstack((filtered_boxes, predictions[0:1]))

        # Remove the first box
        predictions = predictions[1:]

        # Calculate IoU for all remaining boxes at once
        ious = cp.broadcast_to(predictions[:, :4], (len(predictions), 4))
        ious = iou(filtered_boxes[-1, :4], ious)

        # Filter out boxes with IoU > threshold
        predictions = predictions[ious < iou_threshold]

    return filtered_boxes


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

import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple


class Yolov8Seg:

    def __init__(
        self,
        model_path: str,
        target_size: Tuple[int, int] = (640, 640),
        interpolation: int = cv2.INTER_AREA,
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
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ort.get_device() == "GPU"
                else ["CPUExecutionProvider"]
            ),
        )

        print(f"Yolov8Seg: Model Using {ort.get_device()} for inference")

        self.ndtype = (
            np.half  # np.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else np.single  # np.float32
        )

        self.model_input_height, self.model_input_width = [
            x.shape for x in self.session.get_inputs()
        ][0][-2:]

        self.model_output_height, self.model_output_width = [
            x.shape for x in self.session.get_outputs()
        ][0][-2:]

    def __call__(self, img_in: np.ndarray) -> np.ndarray:
        # print(f"Preprocessed image shape: {image.shape}")
        image, ratio, (pad_w, pad_h) = self.preprocess_image(img_in)
        # print(f"Preprocessed image shape: {image.shape}")
        # print(f"Preprocessed img_in shape: {img_in.shape}")  # HWC

        # img = np.squeeze(image)
        # img = np.transpose(img, (1, 2, 0)) * 255
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test_samples/preprocess_image.png", img)

        # print(f"Preprocessed image shape: {image.shape}")  # (1, 3, 640, 640)
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
        preds: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        nm: int,
        conf_threshold: float,
        iou_threshold: float,
        img_0: np.ndarray,
    ) -> np.ndarray:

        # print(f"Postprocess: ratio: {ratio}")

        instnc_prds, protos = (
            preds[0],
            preds[1],
        )  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        instnc_prds = np.einsum("bcn->bnc", instnc_prds)

        # Predictions filtering by conf-threshold
        instnc_prds = instnc_prds[
            np.amax(instnc_prds[..., 4:-nm], axis=-1) > conf_threshold
        ]

        # print(f"Number of detected objects: {len(instnc_prds)}")
        # Create a new matrix which merge these(box, score, cls, nm) into one
        instnc_prds = np.c_[
            instnc_prds[..., :4],
            np.amax(instnc_prds[..., 4:-nm], axis=-1),
            np.argmax(instnc_prds[..., 4:-nm], axis=-1),
            instnc_prds[..., -nm:],
        ]

        # NMS filtering by iou-threshold;
        instnc_prds = instnc_prds[
            cv2.dnn.NMSBoxes(
                instnc_prds[:, :4],
                instnc_prds[:, 4],
                conf_threshold,
                iou_threshold,
            )
        ]

        # each element in "instnc_prds" is a box [4 elements], score, classID, mask coefficients [32 elements]
        # print(f"Detected objects: {instnc_prds}, shape: {instnc_prds.shape}")

        # Decode and return
        if len(instnc_prds) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            instnc_prds[..., [0, 1]] -= instnc_prds[..., [2, 3]] / 2
            instnc_prds[..., [2, 3]] += instnc_prds[..., [0, 1]]

            # Rescales bounding boxes from model shape to the shape of original image
            instnc_prds[..., :4] -= [padding[0], padding[1], padding[0], padding[1]]
            # instnc_prds[..., :4] /= ratio

            # Bounding boxes boundary clamp
            instnc_prds[..., [0, 2]] = instnc_prds[:, [0, 2]].clip(
                0, self.model_input_width
            )  # x1, x2
            instnc_prds[..., [1, 3]] = instnc_prds[:, [1, 3]].clip(
                0, self.model_input_height
            )

            # print(
            #     f"model_output_width: {self.model_output_width}, model_output_height: {self.model_output_height}"
            # )

            # print(f"box_coord[..., :4]: {instnc_prds[..., :4]}")
            # print(f"protos.shape: {protos.shape}")  # (1, 32, 160, 160)

            masks = self.process_mask(
                protos=protos[0],  # (32, 160, 160)
                masks_coef=instnc_prds[..., 6:],  # 32 mask coefficients
                bboxes=instnc_prds[..., :4],  # boxes shape: (N, 4)
                img_0_shape=img_0.shape[0:2],  # (H, W) of original image
            )

            # segments = self.masks2segments(masks)  # Disabled segments for now

            return instnc_prds[..., :6], masks  # boxes, masks
        else:
            return [], []

    def process_mask(
        self,
        protos: np.ndarray,
        masks_coef: np.ndarray,
        bboxes: np.ndarray,
        img_0_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        protos: mask prototype for each class (C, H, W)
        masks_coef: mask coefficients for each detected object (N, 32)
        """
        c, mh, mw = protos.shape

        masks = (
            np.matmul(masks_coef, protos.reshape((c, -1)))
            .reshape((-1, mh, mw))
            .transpose(1, 2, 0)
        )  # HWN

        masks = np.ascontiguousarray(masks)

        masks = self.scale_mask(
            masks, img_0_shape
        )  # re-scale mask from P3 shape to original input image shape

        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks: np.ndarray, img_0_shape: Tuple[int, int], ratio_pad=None):
        img_1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from img_0_shape
            gain = min(
                img_1_shape[0] / img_0_shape[0], img_1_shape[1] / img_0_shape[1]
            )  # gain  = old / new
            pad = (img_1_shape[1] - img_0_shape[1] * gain) / 2, (
                img_1_shape[0] - img_0_shape[0] * gain
            ) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(img_1_shape[0] - pad[1] + 0.1)), int(
            round(img_1_shape[1] - pad[0] + 0.1)
        )
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (img_0_shape[1], img_0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    @staticmethod
    def masks2segments(masks: np.ndarray) -> np.ndarray:
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                0
            ]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def preprocess_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:

        resized_image, resize_ratio, padding = self.resize_and_pad(image)
        model_input = self.convert_to_model_input(resized_image)

        return model_input, resize_ratio, padding

    def resize_and_pad(
        self, original_image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes and adds padding to an image to fit the YOLOv8 model's input format.

        Parameters:
        - original_image (np.ndarray): The original image to be processed.

        Returns:
        - np.ndarray: The resized and padded image.
        - float: The resize ratio applied to the image.
        - Tuple[int, int]: The padding applied to the image (in pixels) in horizontal and vertical directions.
        """
        original_shape = original_image.shape[:2]
        target_size = (self.model_input_height, self.model_input_width)
        resize_ratio = min(
            target_size[0] / original_shape[0], target_size[1] / original_shape[1]
        )
        resized_dimensions = int(round(original_shape[1] * resize_ratio)), int(
            round(original_shape[0] * resize_ratio)
        )

        # Calculate the necessary padding for the resized image to fit the desired size
        padding_width, padding_height = (target_size[1] - resized_dimensions[0]) / 2, (
            target_size[0] - resized_dimensions[1]
        ) / 2

        # Resize the image while maintaining aspect ratio
        resized_image = cv2.resize(
            original_image, resized_dimensions, interpolation=cv2.INTER_LINEAR
        )

        # Add padding to the resized image
        padding_top, padding_bottom = int(round(padding_height - 0.1)), int(
            round(padding_height + 0.1)
        )
        padding_left, padding_right = int(round(padding_width - 0.1)), int(
            round(padding_width + 0.1)
        )
        padded_image = cv2.copyMakeBorder(
            resized_image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return padded_image, resize_ratio, (padding_width, padding_height)

    def convert_to_model_input(self, image: np.ndarray) -> np.ndarray:
        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        image = (
            np.ascontiguousarray(np.einsum("HWC->CHW", image)[::-1], dtype=self.ndtype)
            / 255.0
        )
        img_process = image[None] if len(image.shape) == 3 else image
        return img_process

    @staticmethod
    def draw_bbox(
        img: np.ndarray,
        bbox: np.ndarray,
        class_name: str,
        policy: str,
        score: float,
        color=(0, 150, 255),
        thickness=1,
        font=cv2.FONT_HERSHEY_PLAIN,
        font_scale=0.8,
        padding: np.ndarray = None,
    ):

        # Reverse padding and aspect ratio
        if padding is not None:
            bbox += padding

        bbox = bbox.astype(int)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f"{policy} | {class_name} : {score:.2f}"
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 + 11),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

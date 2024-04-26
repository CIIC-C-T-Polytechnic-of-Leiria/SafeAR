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
        self.confidence_threshold = confidence_threshold
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

        print(f"Yolov8seg: Model Using {ort.get_device()} for inference")

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
        img, ratios, (pad_w, pad_h) = self.preprocess_img(img_in)

        # print(
        #     f"DEBUG: img shape: {img.shape}, img type: {img.dtype}, max: {cp.max(img)}, min: {cp.min(img)}"
        # )

        preds = self.session.run(
            None, {self.session.get_inputs()[0].name: cp.asnumpy(img)}
        )

        # # ------ DEBUG: save img to disk ----------------------------
        # # Discard the batch dimension in img array (BCHW -> CHW)
        # img = cp.squeeze(img, axis=0)
        # # Invert CHW to HWC
        # img = cp.einsum("CHW->HWC", img)
        # # Convert to uint8 and 0-255 range
        # img = (img * 255.0).astype("uint8")
        # plt.imsave("img_after_processing.png", cp.asnumpy(img))
        # # print(f"DEBUG: preds type: {type(preds)}")
        # # ------------------------------------------------------------

        return self.postprocess_img(
            preds,
            ratios,
            (pad_w, pad_h),
            self.num_masks,
            self.confidence_threshold,
            self.iou_threshold,
            img_in,
        )

    def postprocess_img(
        self,
        preds: list,
        ratios: Tuple[float, float],  # (ratio_x, ratio_y)
        pad: Tuple[int, int],
        nm: int,
        conf_threshold: float,
        iou_threshold: float,
        img_0: cp.ndarray,
    ) -> cp.ndarray:

        # preds[0] = boxes, scores, classes, masks coefficients
        # preds[1] = protos masks of dim. = (32, 160, 160)
        instnc_prds, protos = (
            cp.asarray(preds[0]),
            cp.asarray(preds[1]),
        )

        # ------ DEBUG ----------------------------
        # print(
        #     f"DEBUG: instnc_prds shape: {instnc_prds.shape}, protos shape: {protos.shape}"
        # )
        # ----------------------------------------

        # Transpose preds: (Batch_sz, xywh_conf_cls_nm, Num_anchors) -> (Batch_sz, Num_anchors, xywh_conf_cls_nm)
        instnc_prds = cp.einsum("bcn->bnc", instnc_prds)

        # ------ DEBUG: Print the prediction array ----------------------------
        # DEBUG: Print the probability array
        # prob_array = cp.asnumpy(instnc_prds[..., 4:-nm])
        # print(f"prob array: {np.array2string(prob_array, precision=1)}")
        # ---------------------------------------------------------------------

        # Apply NMS
        instnc_prds = apply_nms(instnc_prds, conf_threshold, nm, iou_threshold)

        # ------ DEBUG: Print number of found boxes ----------------------------
        # print(f"len(instnc_prds): {len(instnc_prds)}")
        # ---------------------------------------------------------------------

        # Decode and return
        if len(instnc_prds) > 0:
            # Rescales bounding boxes from model shape to the shape of original image
            instnc_prds[..., :4] -= cp.array([pad[0], pad[1], pad[0], pad[1]])

            # Bounding boxes boundary clamp
            instnc_prds[..., [0, 2]] = instnc_prds[:, [0, 2]].clip(
                0, self.model_input_width
            )  # x1, x2
            instnc_prds[..., [1, 3]] = instnc_prds[:, [1, 3]].clip(
                0, self.model_input_height
            )  # y1, y2

            # print(
            #     f"DEBUG: protos shape: {protos.shape} instnc_prds[..., 6:] shape: {instnc_prds[..., 6:].shape}, "
            #     f"instnc_prds[..., :4] shape: {instnc_prds[..., :4].shape}, img_0 shape: {img_0.shape[0:2]}"
            # )

            b_boxes = self.resize_bboxes(
                instnc_prds[..., :4], ratios, pad, img_0.shape[0:2]
            )

            # print(
            #     f"DEBUG: b_boxes shape: {b_boxes.shape}, b_boxes: {b_boxes}, max: {cp.max(b_boxes)}, min: {cp.min(b_boxes)}"
            # )

            processed_masks = self.process_mask(
                protos_=protos[0],  # (32, 160, 160)
                masks_coef=instnc_prds[..., 6:],  # 32 mask coefficients
                bboxes=b_boxes,  # boxes shape: (N, 4)
                img_0_shape=img_0.shape[0:2],  # (H, W) of original image
            )
            b_boxes_scr_id = cp.concatenate([b_boxes, instnc_prds[..., 4:6]], axis=1)

            return b_boxes_scr_id, processed_masks

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
        # print(
        #     f"DEBUG crop_mask: masks shape: {masks.shape}, boxes shape: {boxes.shape}"
        # )
        x1, y1, x2, y2 = cp.split(boxes[:, :, None], 4, 1)
        # print(f"DEBUG crop_mask: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
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
    ) -> cp.ndarray:

        c, h, w = protos_.shape  # 32, 160, 160

        masks = (
            cp.matmul(masks_coef, protos_.reshape((c, -1)))
            .reshape((-1, h, w))
            .transpose(1, 2, 0)
        )  # HWN: (160, 160, Num of detected instances)

        # Sigmoid activation to flatten the masks to [0, 1]
        masks = expit(masks)

        # plot each mask
        # for i in range(masks.shape[-1]):
        #     plt.figure()
        #     plt.imshow(
        #         cp.asnumpy(masks[:, :, i]), cmap="gray"
        #     )  # Convert the CuPy array to a NumPy array
        #     plt.title(f"Mask {i + 1}")
        #     plt.axis("off")
        #     plt.savefig(f"mask_{i + 1}.png")

        # print("DEBUG: masks after matmul shape: ", masks.shape)
        masks = cp.ascontiguousarray(masks)

        masks = self.scale_mask(
            masks, img_0_shape
        )  # re-scale mask from P3 shape to original input image shape

        # save each mask
        # for i in range(masks.shape[-1]):
        #     plt.figure()
        #     plt.imshow(cp.asnumpy(masks[:, :, i]), cmap="gray")
        #     plt.savefig(f"mask_rescaled_{i + 1}.png")

        # print(
        #     f"DEBUG: masks after scaling shape: {masks.shape}, max: {cp.max(masks)}, min: {cp.min(masks)}"
        # )

        masks = cp.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)

        # plot cropped masks after scaling (need to convert to numpy array)
        # for i in range(masks.shape[0]):
        #     plt.figure()
        #     plt.imshow(cp.asnumpy(masks[i]), cmap="gray")
        #     plt.savefig(f"mask_cropped_{i + 1}.png")

        return cp.greater(
            masks, 0.5
        )  # TODO: this returns a boolean mask, should return a float mask

    def preprocess_img(
        self, img: cp.ndarray
    ) -> Tuple[cp.ndarray, Tuple[float, float], Tuple[int, int]]:

        resized_img, resize_ratios, padding = self.resize_and_pad(img)

        # ------ DEBUG: save resized_img to disk ----------------------------
        # print(
        #     f"DEBUG: resized_img shape: {resized_img.shape}, max: {cp.max(resized_img)}, min: {cp.min(resized_img)}"
        # )
        # img_to_save = cp.asnumpy(resized_img * 255.0).astype("uint8")
        # plt.imsave(f"resized_img{np.random.randint(10)}.png", img_to_save)
        # ---------------------------------------------------------------------

        model_in_img = self.convert_to_yolov8_input(resized_img)

        # ------ DEBUG: save model_in_img to disk ----------------------------
        # img_to_save = cp.einsum("CHW->HWC", model_in_img[0, :, :, :])
        # img_to_save = img_to_save[:, :, ::-1]  # Swap color channels (BGR -> RGB)
        # img_to_save = cp.asnumpy(img_to_save * 255.0).astype("uint8")
        # print(
        #     f"DEBUG: model_in_img shp: {img_to_save.shape} max: {cp.max(img_to_save)} min: {cp.min(img_to_save)}"
        # )
        # plt.imsave(f"model_in_img{np.random.randint(10)}.png", img_to_save)
        # ---------------------------------------------------------------------

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
        t_size = (self.model_input_height, self.model_input_width)

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
        img = cp.ascontiguousarray(img, dtype=self.ndtype)
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


def iou(box1: cp.ndarray, box2: cp.ndarray, epsilon=1e-9) -> cp.ndarray:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2 (cp.ndarray): Bounding boxes in format x1,y1,x2,y2.
    epsilon (float): Small constant to prevent division by zero.

    Returns:
    iou (cp.ndarray): IoU of box1 and box2.
    """
    x1, y1 = cp.maximum(box1[:2], box2[..., :2])
    x2, y2 = cp.minimum(box1[2:], box2[..., 2:])
    inter_area = cp.maximum(0, x2 - x1 + 1) * cp.maximum(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[..., 2] - box2[..., 0] + 1) * (box2[..., 3] - box2[..., 1] + 1)
    return inter_area / (box1_area + box2_area - inter_area + epsilon)


def apply_nms(
    instnc_prds: cp.ndarray, conf_threshold: float, nm: int, iou_threshold: float
) -> cp.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) and convert bounding box format.

    Parameters:
    instnc_prds (cp.ndarray): Instance predictions.
    conf_threshold (float): Confidence threshold for filtering.
    nm (int): Number of mask coefficients.
    iou_threshold (float): IoU threshold for NMS.

    Returns:
    instnc_prds (cp.ndarray): Filtered and NMS applied instance predictions.
    """
    # Predictions filtering by conf-threshold
    instnc_prds = instnc_prds[
        cp.amax(instnc_prds[..., 4:-nm], axis=-1) > conf_threshold
    ]

    instnc_prds = cp.c_[
        instnc_prds[..., :4],
        cp.amax(instnc_prds[..., 4:-nm], axis=-1),
        cp.argmax(instnc_prds[..., 4:-nm], axis=-1),
        instnc_prds[..., -nm:],
    ]

    # Sort by confidence score (index 4) in descending order
    instnc_prds = instnc_prds[instnc_prds[:, 4].argsort()[::-1]]

    # Bounding boxes format change: cxcywh -> xyxy
    center = instnc_prds[..., [0, 1]].copy()
    instnc_prds[..., [0, 1]] -= instnc_prds[..., [2, 3]] / 2  # x1, y1
    instnc_prds[..., [2, 3]] = center + instnc_prds[..., [2, 3]] / 2  # x2, y2

    boxes = []

    while len(instnc_prds) > 0:
        boxes.append(instnc_prds[0])
        if len(instnc_prds) == 1:
            break

        # remove the first box
        instnc_prds = instnc_prds[1:]

        # For all remaining boxes, calculate IoU with the first box
        ious = cp.array([iou(boxes[-1][:4], box) for box in instnc_prds[:, :4]])
        # print(f"DEBUG: ious: {ious}")

        # If IoU is greater than the threshold, remove the box
        instnc_prds = instnc_prds[ious < iou_threshold]

    if len(boxes) == 0:
        return cp.array([])

    return cp.stack(boxes)

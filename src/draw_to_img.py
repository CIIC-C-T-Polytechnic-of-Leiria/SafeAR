import cv2
import numpy as np


class ImageAnnotator:
    def __init__(self, img: np.ndarray):
        """
        Initialize the ImageAnnotator with an image.

        :param img: A NumPy array representing the image.
        """
        self.img = img.copy()

    def draw_bbox_and_info(
        self,
        bbox: list,
        class_name: str,
        score: float,
        color=(0, 255, 0),
        thickness=1,
        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale=0.3,
    ):
        """
        Draw a bounding box and information on the image.

        :param bbox: A list of [x1, y1, x2, y2] coordinates of the bounding box.
        :param class_name: The class name of the object.
        :param score: The confidence score of the detection.
        :param color: The color of the bounding box and text. Default is green.
        :param thickness: The thickness of the bounding box lines. Default is 2.
        :param font: The font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
        :param font_scale: The font scale factor. Default is 0.5.
        """
        x1, y1, x2, y2 = bbox
        
        print(f"x1: {int(x1)}, y1: {int(y1)}, x2: {int(x2)}, y2: {int(y2)}")
        
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2),), color, thickness)

        text = f"{class_name}: {score:.2f}"
        cv2.putText(
            self.img, text, (int(x1), int(y1 + 7)), font, font_scale, color, thickness
        )

    def get_image(self):
        return self.img

    def save(self, path: str):
        """
        Save the annotated image to a file.

        :param path: The file path to save the image.
        """
        cv2.imwrite(path, self.img)

import base64
import os
from io import BytesIO
from typing import Any

import cupy as cp
import imageio
import yaml

from src.obfuscator import ImageObfuscator
from src.seg_yolov8 import Yolov8seg


# # Reload the modules
# importlib.reload(seg_yolov8)
# importlib.reload(obfuscator)
# Import the classes from the reloaded modules


class SafeARService:

    def __init__(self):
        self.obfuscator = None
        self.model = None
        self.obfuscation_policies: dict[str, Any] = {}

    def configure(self, model_number: int, obfuscation_policies: dict):
        config_yml = self.load_config()
        model_name = list(config_yml["models"].keys())[model_number]
        model_path = config_yml["models"][model_name]["model_path"]

        self.model = Yolov8seg(model_path=model_path)
        self.obfuscation_policies = obfuscation_policies
        self.obfuscator = ImageObfuscator(policies=self.obfuscation_policies)

    def process_frame(self, image_base64: str) -> bytes:
        """
        Process a frame by detecting objects and applying obfuscation.
        Args:
            image_base64: str representation of an image, encoded in Base64 format

        Returns:
            safe_frame_bytes: the processed frame, encoded in bytes
        """
        # Decode the Base64 image string to bytes
        image_bytes = base64.b64decode(image_base64)
        print(f"Image bytes: {len(image_bytes)}")
        # Create a buffer (`BytesIO` object) from the image bytes
        buffer = BytesIO(image_bytes)

        # Read the image from the buffer using imageio
        img_array = imageio.v2.imread(buffer)
        # Convert the Numpy array to a cuPY array
        frame = cp.asarray(img_array)

        # DEBUG: save the input frame
        imageio.imwrite("outputs/img_in_flask_2.png", frame.get())

        boxes, masks = self.model(frame)
        safe_frame = self.obfuscator.obfuscate(
            image=frame, masks=masks, class_ids=[int(box[5]) for box in boxes]
        )

        safe_frame = safe_frame.astype(cp.uint8)
        safe_frame_bytes = safe_frame.tobytes()
        return safe_frame_bytes

    @staticmethod
    def read_base64_image(file_path):
        with open(file_path, "r") as f:
            image_base64 = f.read()
        image_data = base64.b64decode(image_base64)
        return image_data

    @staticmethod
    def save_processed_frame(frame_bytes, output_path):
        frame_array = cp.frombuffer(frame_bytes, dtype=cp.uint8)
        if len(frame_array) != 640 * 640 * 3:
            raise ValueError("Incorrect size of frame data")
        frame_array = frame_array.reshape((640, 640, 3))
        # Convert cupy array to numpy array
        frame_array = cp.asnumpy(frame_array)
        imageio.imwrite(output_path, frame_array)

    @staticmethod
    def load_config() -> dict:
        # Get the current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory
        parent_directory = os.path.dirname(current_directory)
        # Construct the path to the config.yml file
        config_file_path = os.path.join(parent_directory, "config.yml")

        with open(file=config_file_path, mode="r", encoding="utf-8") as file:
            config_yml = yaml.safe_load(file)
        return config_yml

    @staticmethod
    def list_models() -> list:
        config_yml = SafeARService.load_config()
        return list(config_yml["models"].keys())

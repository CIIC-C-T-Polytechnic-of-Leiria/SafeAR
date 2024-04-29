"""
 TODO:
    ! Implementar "warm-up" do modelo (5 ciclos de inferencia)- to do
    !! O que deve e tem de ir nos metadados do frame, e como pode ser codificado no base64?
    1.2) Criar variáviel de entrada e saída para o sistema de obfuscação - to do
    1.3) Testar output dos modelos Yolov9, Gelan e RTMDet e implementar class para adaptar a saída - 30% done
    2) Implementar frame obfuscatiom striding - implementar logica que está no Unity

    Usage:
    python main.py --model_number 0 --class_id_list 0 1 2 3  --img_source 0 --obfuscation_type_list blurring blurring blurring blurring
"""

import argparse
import importlib
from typing import Any

import yaml

import src.img_handle
import src.obfuscator
import src.seg_yolov8

# Reload the modules
importlib.reload(src.seg_yolov8)
importlib.reload(src.img_handle)
importlib.reload(src.obfuscator)

# Import the classes from the reloaded modules
from src.seg_yolov8 import Yolov8seg
from src.obfuscator import ImageObfuscator

# Testing/debugging imports
import cupy as cp
import base64
from io import BytesIO
import imageio.v2 as imageio


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
        image_bytes = base64.b64decode(image_base64)
        buffer = BytesIO(image_bytes)
        img_array = imageio.imread(buffer)
        frame = cp.asarray(img_array)

        # DEBUG: save the input frame
        # imageio.imwrite("outputs/img_in.png", frame.get())

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
        with open(file="config.yml", mode="r", encoding="utf-8") as file:
            config_yml = yaml.safe_load(file)
        return config_yml

    @staticmethod
    def list_models() -> list:
        config_yml = SafeARService.load_config()
        return list(config_yml["models"].keys())


def main(args):
    safeARservice = SafeARService()

    safeARservice.configure(
        model_number=args.model_number,
        obfuscation_policies=args.obfuscate_policies,
    )

    frame_bytes = safeARservice.process_frame(args.image_base64)

    return frame_bytes


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Obfuscation script")
    arg_parser.add_argument(
        "--model_number",
        type=int,
        default=0,
        help="Choose the number of the model to use. Use '0' for the default model.",
    )

    arg_parser.add_argument(
        "--class_id_list",
        nargs="+",
        type=int,
        help="Specify the list of class IDs to obfuscate. Separate IDs with spaces.",
    )

    arg_parser.add_argument(
        "--obfuscation_type_list",
        nargs="+",
        type=str,
        help="Specify the list of obfuscation types for each class ID. Separate types with spaces.",
    )

    arg_parser.add_argument(
        "--image_base64_file",
        type=str,
        help="Path to the file containing the Base64-encoded image string.",
    )

    arg_parser.add_argument(
        "--version",
        action="version",
        version="Obfuscation script 1.0",
    )

    arg_parser.add_argument(
        "--square",
        type=int,
        default=0,
        help="Size of the square for pixelation effect.",
    )

    arg_parser.add_argument(
        "--sigma",
        type=int,
        default=0,
        help="Sigma value for the blurring effect.",
    )

    args = arg_parser.parse_args()

    if not args.__dict__:
        arg_parser.print_help()
        exit()

    # Create the obfuscate_policies dictionary directly in the parser
    args.obfuscate_policies = dict(zip(args.class_id_list, args.obfuscation_type_list))

    # Read the Base64-encoded image string from the file
    with open(args.image_base64_file, "r") as f:
        args.image_base64 = f.read()

    return args


if __name__ == "__main__":
    main_args = parse_args()

    safeAR_frame_bytes = main(main_args)

    # # DEBUG: save the processed frame
    # safeAR_frame_array = cp.frombuffer(safeAR_frame_bytes, dtype=cp.uint8)
    # safeAR_frame_array = safeAR_frame_array.reshape((640, 640, 3))
    # imageio.imwrite("outputs/img_out2.png", safeAR_frame_array.get())

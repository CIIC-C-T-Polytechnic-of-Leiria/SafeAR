"""
 TODO:
    ! Implementar "warm-up" do modelo (5 ciclos de inferencia)- to do
    !! O que deve e tem de ir nos metadados do frame, e como pode ser codificado no base64?
    1.2) Criar variável de entrada e saída para o sistema de obfuscação - to do
    1.3) Testar output dos modelos Yolov9, Gelan e RTMDet e implementar class para adaptar a saída - 30% done
    2) Implementar frame obfuscatiom striding - implementar logica que está no Unity

    Usage:
    python main.py \
        --model_number 0 \
        --class_id_list 0 1 \
        --obfuscation_type_list "pixelation" "masking" \
        --image_base64_file "test_samples/images/img_640x640_base64.txt" \
        --square 10 \
        --sigma 5
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import argparse
import importlib

import src.seg_yolov8
from src.safear_service import SafeARService

# Reload the modules
importlib.reload(src.seg_yolov8)
importlib.reload(src.safear_service)


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

    # Print the absolute path of the file
    abs_path = os.path.abspath(args.image_base64_file)
    print(f"Absolute path of the file: {abs_path}")

    # Read the Base64-encoded image string from the file
    with open(args.image_base64_file, "r") as f:
        args.image_base64 = f.read()

    return args


if __name__ == "__main__":
    main_args = parse_args()

    safeAR_frame_bytes = main(main_args)

    # safeAR_image_base64 = base64.b64encode(safeAR_frame_bytes).decode("utf-8")

    #  DEBUG: save the processed frame
    import cupy as cp
    import imageio

    safeAR_frame_array = cp.frombuffer(safeAR_frame_bytes, dtype=cp.uint8)
    safeAR_frame_array = safeAR_frame_array.reshape((640, 640, 3))
    imageio.imwrite("outputs/OUTPUT_2.png", safeAR_frame_array.get())

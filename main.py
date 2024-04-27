"""
 TODO:
    1) Blurring está muito lento... - verificar se é possível acelerar
    1.1) Correr com GPU! - doing 
    1.2) Criar variáviel de entrada e saída para o sistema de obfuscação - to do
    1.3) Testar output dos modelos Yolov9, Gelan e RTMDet e implementar class para adaptar a saída - 30% done
    2) Implementar frame obfuscatiom striding - implementar logica que está no Unity

    Usage:
    python main.py --model_number 0 --class_id_list 0 1 2 3  --img_source 0 --obfuscation_type_list blurring blurring blurring blurring
"""

import argparse
import cProfile
import importlib
import io
import pstats
import time
from pstats import SortKey

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
from src.obfuscator import Colors

# Testing/debugging imports
import imageio
import cupy as cp


def load_config() -> dict:
    with open(file="config.yml", mode="r", encoding="utf-8") as file:
        config_yml = yaml.safe_load(file)
    return config_yml


def list_models(config_dict: dict):
    print("Available models:")
    models = list(config_dict["models"].keys())
    for i, model in enumerate(models, start=1):
        print(f"    [{i}] - {model}")


def read_image(image_path: str) -> cp.ndarray:
    image = imageio.v3.imread(image_path)
    return cp.asarray(image)


def main(
    model_number: int,
    obfuscate_policies: dict,
    display_fps: bool,
    display_boxes: bool,
    save_video: bool,
    source=0,
    config_yml=None,
):
    model_config = list(config_yml["models"].values())[model_number]

    # TODO: Instantiate the correct model based on the user's choice

    model = Yolov8seg(model_path=model_config["model_path"])

    # camera = Camera(source=source, display_fps=display_fps, save_video=save_video)

    frame = read_image("test_samples/images/students-lockers.jpg")

    colors = Colors(model_config["num_classes"])
    colors_dict = colors.get_colors_dict()

    # print(f"DEBUG: colors: {colors}")
    # print(f"DEBUG: policies dict: {policies}")

    obfuscator = ImageObfuscator(policies=obfuscate_policies)

    while True:
        while True:
            start_time = time.time()
            if frame is None:
                break

            boxes, masks = model(frame)
            end_time1 = time.time()
            print(
                f"MOD: TOTAL TIME {((end_time1 - start_time) * 1000):.1f} milliseconds"
            )

            safe_frame = obfuscator.obfuscate(
                masks=masks,
                image=cp.asarray(frame),
                class_ids=[int(box[5]) for box in boxes],
            )
            end_time2 = time.time()
            print(
                f"SYS: OBFUSC. TIME {((end_time2 - end_time1) * 1000):.1f} milliseconds"
            )

            #  save the processed frame
            safe_frame = safe_frame.astype(cp.uint8)
            imageio.imwrite("outputs/img2.jpg", safe_frame.get())
            end_time3 = time.time()
            print(f"SYS: SAVE TIME {((end_time3 - end_time2) * 1000):.1f} milliseconds")

            end_time = time.time()
            print(
                f"SYS: TOTAL TIME {((end_time - start_time) * 1000):.1f} milliseconds"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_number",
        type=int,
        default=0,
        help="Choose the number of the model to use. Use '0' for the default model.",
    )

    parser.add_argument(
        "--class_id_list",
        nargs="+",
        type=int,
        help="Specify the list of class IDs to obfuscate. Separate IDs with spaces.",
    )

    parser.add_argument(
        "--obfuscation_type_list",
        nargs="+",
        type=str,
        help="Specify the list of obfuscation types for each class ID. Separate types with spaces.",
    )

    parser.add_argument(
        "--img_source",
        type=lambda x: int(x) if x.isdigit() else x,
        default=0,
        help="Image source, can be a number or a path. Default is 0.",
    )

    parser.add_argument(
        "--show_fps",
        action="store_true",
        help="Choose to display the frames per second.",
    )

    parser.add_argument(
        "--show_boxes",
        action="store_true",
        help="Choose to display the bounding boxes.",
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Choose to save the processed video.",
    )

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit()

    if args.obfuscation_type_list is None:
        args.obfuscation_type_list = ["masking"] * len(args.class_id_list)
    elif args.class_id_list is not None and len(args.obfuscation_type_list) < len(
        args.class_id_list
    ):
        last_obfuscation_type = args.obfuscation_type_list[-1]
        args.obfuscation_type_list += [last_obfuscation_type] * (
            len(args.class_id_list) - len(args.obfuscation_type_list)
        )

    config = load_config()
    num_classes = config["models"]["yolov8"]["num_classes"]

    # Create a dictionary with all possible class IDs and None as the obfuscation policy
    policies = {i: "none" for i in range(num_classes)}

    # Update the dictionary with the actual obfuscation policies
    policies.update(
        (class_id, obfuscation_type)
        for class_id, obfuscation_type in zip(
            args.class_id_list, args.obfuscation_type_list
        )
    )
    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    main(
        model_number=args.model_number,
        obfuscate_policies=policies,
        source=args.img_source,
        display_fps=args.show_fps,
        display_boxes=args.show_boxes,
        save_video=args.save_video,
        config_yml=config,
    )
    # Stop profiling
    pr.disable()

    # Create a StringIO object to hold the profiling results
    s = io.StringIO()

    # Sort the results by cumulative time spent in the function
    sort = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sort)
    ps.print_stats()

    # Print the profiling results
    print(s.getvalue())

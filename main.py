"""
 TODO: 
    1) Possibiltar a escolha do modelo a ser utilizado
    2) Implementar frame obfuscatiom striding
    """

import importlib
import argparse
import yaml
import src.seg_yolov8
import src.img_handle
import src.obfuscator

# Reload the modules
importlib.reload(src.seg_yolov8)
importlib.reload(src.img_handle)
importlib.reload(src.obfuscator)

# Import the classes from the reloaded modules
from src.seg_yolov8 import Yolov8Seg
from src.img_handle import Camera
from src.obfuscator import ImageObfuscator


def load_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


def list_models(config):
    print("Available models:")
    models = list(config["models"].keys())
    for i, model in enumerate(models, start=1):
        print(f"    [{i}] - {model}")


def main(
    model_number: int,
    policies: dict,
    display_fps: bool,
    display_boxes: bool,
    save_video: bool,
    source=0,
    config=None,
):

    model_config = list(config["models"].values())[model_number]

    # TODO: Instantiate the correct model based on the user's choice

    model = Yolov8Seg(model_path=model_config["model_path"])

    camera = Camera(source=source, display_fps=display_fps, save_video=save_video)

    # print(f"DEBUG: policies dict: {policies}")

    obfuscator = ImageObfuscator(policies=policies)

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        boxes, masks = model(frame)

        frame = obfuscator.obfuscate(
            masks=masks,
            image=frame,
            class_ids=[int(box[5]) for box in boxes],
        )

        if display_boxes:
            for box in boxes:
                model.draw_bbox(
                    img=frame,
                    bbox=box[0:4],
                    class_name=model_config["class_names"][int(box[5])],
                    policy=policies[int(box[5])],
                    score=box[4],
                )

        camera.display_frame(frame, info="Press 'q' to quit.")

        if not camera.wait_key("q"):
            break

    camera.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        # description="\n\nUsage example:\n  python main.py --model_number 0 --class_id_list 0 1 2 --obfuscation_type_list bluring masking pixelation --img_source 0 --show_fps"
    )
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
        help="Specify the list of class IDs to obfuscate. Separate IDs with commas.",
    )

    parser.add_argument(
        "--obfuscation_type_list",
        nargs="+",
        type=str,
        help="Specify the list of obfuscation types for each class ID. Separate types with commas.",
    )

    parser.add_argument(
        "--img_source",
        default=0,
        help="Specify the source of the images to process. Default is 0.",
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

    # From config.yml, count the number of classes in the model
    config = load_config()
    num_classes = config["models"]["yolov8"]["num_classes"]

    # Create a dictionary with all possible class IDs and None as the obfuscation policy
    policies = {i: "none" for i in range(num_classes)}

    # Update the dictionary with the actual obfuscation policies
    policies.update(
        {
            class_id: obfuscation_type
            for class_id, obfuscation_type in zip(
                args.class_id_list, args.obfuscation_type_list
            )
        }
    )

    main(
        model_number=args.model_number,
        policies=policies,
        source=args.img_source,
        display_fps=args.show_fps,
        display_boxes=args.show_boxes,
        save_video=args.save_video,
        config=config,
    )

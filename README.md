<div style="display: flex; align-items: center; justify-content: center;">
  <img width="100" height="100" src="assets/safeAR_ipl_icon.png">
  <h1 style="margin-left: 20px;">SafeAR - Privacy in AR Contexts as a Service</h1>
</div>

<!-- <div align="center"> <img width="100" height="100" src="assets/safeAR_ipl_icon.png"> <h1>SafeAR - Privacy in AR Contexts as a Service</h1> </div> -->

### Overview

Welcome to SafeAR, a privacy-focused solution designed for augmented reality (AR) contexts. Our system processes input
from mobile device cameras and returns a sanitized version of the data, ensuring that sensitive information is obscured.

<p align="center"> <img src="assets/output_12_04_2024-ezgif.com-optimize.gif" width="700px" style="border:3px solid lightgray;"/> </p>
SafeAR comprises four modules - Preprocessing, Object Detection and Segmentation, Transformation (or Obfuscation), and Post-processing - working together to maintain privacy protection.

<p align="center"> <img src="assets/safeAR_layer_v0.png" width="750px" style="border:3px solid lightgray;"/> </p>

Upcoming Features
-----------------

We are continuously improving and adding new features to our system. Here's what you can expect in the near future:

- **Model selection**: SafeAR will allow users to select from a variety of pre-trained models for object detection and
  segmentation.
- **Metadata anonymization**: SafeAR will accept metadata from images or videos and anonymize it before returning it to
  the user, ensuring privacy.
- **Sensor data utilization**: Our system will be able to utilize sensor data from the mobile device to enhance
  performance and provide a better user experience.
- **Inpainting obfuscation**: Our Obfuscation module will offer inpainting as an obfuscation technique, providing even
  more options for securing sensitive information

Available Instance Segmentation Models
--------------------------------------

:construction:In development...

| Model       | Size (MB) | Training Data | Classes | Inference Time CPU (ms)\* | Inference Time GPU (ms)\* |
|-------------|-----------|---------------|---------|---------------------------|---------------------------|
| YOLOv5n-seg | 8.5       | COCO 2017     | 80      | -                         | -                         |
| YOLOv8n-seg | 13.8      | COCO 2017     | 80      |                           | ~20                       |
| YOLOv9c-seg | 111.1     | COCO 2017     | 80      | -                         | -                         |
| gelan-c-seg | 110.0     | COCO 2017     | 80      | -                         | -                         |
| RTMDet      | -         | COCO 2017     | 80      | -                         | -                         |

<small>\*Measured on: HP Victus, 32 GB of memory, Intel i5-12500Hx16 processor, Nvidia GeForceRTX 4060, Pop!\_OS 22.04
LTS operating system</small>

Repository Structure
--------------------

The repository is organized as follows:

```
safeAR-aaS/
│
├── 🏛️ assets/                   # Logos and other visual assets
├── 🚰 src/                      # Source code
├── 📁 seg_models/               # Pre-trained instance segmentation models (onnx format)
├── 🤷🏻‍♀️ .gitignore                # Git ignore file
├── 🛠️ config.yml                # Configuration file
├── 🐍 main.py                   # Main script to run the API
├── 📜 README.md                 # Readme file
└── 📜 requirements.txt          # Required packages
```

Installation
------------

Clone the repository:

```sh
# Clone the repository
git clone https://github.com/CIIC-C-T-Polytechnic-of-Leiria/SafeAR.git
cd SafeAR

# Create a new Conda environment (optional)
conda create -n safear_env python=3.10
conda activate safear_env

# Install the CUDA Toolkit and cuDNN
conda install cudatoolkit=11.8
conda install -c conda-forge cudnn=8.8.0.121

# Install the required packages
pip install -r requirements.txt

# Install ONNX Runtime with GPU support
pip install onnxruntime-gpu==1.17.0
```

:memo: Note
<small>

- SafeAR aaS was tested using `Python 3.10.12` environment.
- For Nvidia GPU computers, install `onnxruntime-gpu` package.
- For non-Nvidia GPU computers, use `onnxruntime` package.
- The versions of CUDA, cuDNN, and ONNX Runtime must be compatible with each other and with your GPU. Check
  the [official documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) to ensure
  compatibility.
  </small>

Model Download and Conversion
------------------------------ 

<details>
<summary> <b>Yolov5-seg</b> model </summary>
<br>

You may run this Colab [script](https://colab.research.google.com/drive/1BYFWd_h6ffWTa6SXqllYfYVxjxYj10tf?usp=sharing)
to download the model and convert them to ONNX format.

Afterwards, move the exported `onnx` model(s) to the `seg_models` directory.

</details>

<details>
<summary> <b>Yolov8-seg</b> model </summary>
<br>

You may download the model from the Ultralytics
repository: [Yolov8 Repository](https://docs.ultralytics.com/models/yolov8/#performance-metrics)

Afterwards, move the exported `onnx` model(s) to the `seg_models` directory.

</details>

<details>
<summary> <b>Yolov9-seg</b> and <b>Gelan</b> models </summary>
<br>

You may run this Colab [script](https://colab.research.google.com/drive/1Sv6cvCuAHWOOouXKy1dJ-G18RtMSk7dA?usp=sharing)
to download the models and convert them to ONNX format.

Afterwards, move the exported `onnx` model(s) to the `seg_models` directory.
</details>


<details>
<summary> <b>RTMDet</b> model </summary>
<br>
Under construction...

</details>

Usage
-----

The API can be used both as a command-line tool and as a Python library.

### Command-Line Usage

To use the API as a command-line tool, run the following command:

```bash
python main.py 
    --model_number MODEL_NUMBER 
    --class_id_list CLASS_ID_1 CLASS_ID_2 ... 
    --obfuscation_type_list OBFS_TYPE_1 OBFS_TYPE_2 ... 
    --img_source IMG_SOURCE 
    --show_fps
    --show_boxes
    --save_video
```

where:

* `MODEL_NUMBER` is the number of the model to use (0-based index).
* `CLASS_ID_1 CLASS_ID_2 ...` is a list of class IDs to obfuscate. If model is trained in COCO dataset, see the
  mapping [here](seg_models/mscoco_classID_labels.txt)
* `OBFS_TYPE_1 OBFS_TYPE_2 ...` available obfuscation types are `blurring`, `masking`, and `pixelation`.
* `IMG_SOURCE` is the source of the images to process. This can be a file path, a URL, or a camera index.
* `--show_fps` is an optional flag to show the frames per second.
* `--show_boxes` is an optional flag to save the bounding boxes to a file.
* `--save_video` is an optional flag to save the processed video to a file.

For example:

```bash
python main.py \
    --model_number 0 \
    --class_id_list 0 1 2 \
    --obfuscation_type_list blurring masking pixelation \
    --img_source 0 --show_fps \
```

This will use the first available model to obfuscate objects with class IDs 0, 1, and 2 in the video stream from the
default camera, using the `blurring`, `masking`, and `pixelation` obfuscation types, and showing the frames per second
on
the screen.

### Python Library Usage

:construction: To be implemented...

Acknowledgements
----------------

This work is funded by FCT - Fundação para a Ciência e a Tecnologia, I.P., through project with reference
2022.09235.PTDC.

<!-- Contributing
------------

TO BE DONE... -->

License
-------

:construction:To be determined...

<p align="center">
<img src="assets/CIIC_logo_v2.png" width="750px"/>
</p>



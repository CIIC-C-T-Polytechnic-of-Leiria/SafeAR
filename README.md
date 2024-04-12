<!-- <div style="display: flex; align-items: flex-end;">
<img width="100" height="80" src="assets/safeAR_ipl_v1.png">
<h1 style="margin-left: 20px;">SafeAR SaaS - Privacy in AR contexts as a service</h1>
</div> -->

<img align="left" width="100" height="100" src="assets\safeAR_ipl_icon.png">

# SafeAR - Privacy in AR contexts aaS

<br/><br/>

Introducing SafeAR aaS - the ultimate privacy solution for AR contexts! Our system takes input from mobile device cameras and returns a sanitazed version of the data with sensitive information obscured. With four modules - `Preprocessing`, `Object Detection` and `Segmentation`, `Transformation` (or Obfuscation), and `Post-processing` - SafeAR aaS ensures privacy protection.

<p align="center">
<img src="assets/safeAR_layer_v0.png" width="750px"/>
</p>

We're constantly improving and adding new features to our system. Here's what's coming up next:

- Model selection: SafeAR SaaS will allow users to select from a variety of pre-trained models for object detection and segmentation.
- Metadata anonymization: SafeAR SaaS will accept metadata from images or videos and anonymize it before returning it to the user, ensuring privacy.
- Sensor data utilization: Our system will be able to utilize sensor data from the mobile device to enhance performance and provide  better user experience.
- Inpainting obfuscation: Our Obfuscation module will offer inpainting as an obfuscation technique, providing even more options for securing sensitive information.


Available Instance Segmentation Models
--------------------------------------

In development...


| Model | Size (MB) | Training Data | Classes | Inference Time CPU (ms)\* | Inference Time GPU (ms)\* |
| --- | --- | --- | --- | --- |
| YOLOv5n-seg | - | COCO 2017 | 80 | - | - |
| YOLOv8n-seg | - | COCO 2017 | 80 | - | - |

\*Measured on a HP Victus, 32 GB of memory, Intel i5-12500Hx16 processor, with Nvidia GeForceRTX 4600 and Pop!\_OS 22.04 LTS operating system.

Repository Structure
--------------------

The repository is organized as follows:

```
safeAR-aaS/
│
├── 📁 assets/                   # Logos and other visual assets
├── 📁 seg_models/               # Pre-trained instance segmentation models (onnx format)
├── 📁 src/                      # Source code
├── 📜 .gitignore                # Git ignore file
├── 📜 config.yml                # Configuration file
├── 📜 main.py                   # Main script to run the API
├── 📹 output.mp4                # Sample output video
├── 📜 README.md                 # Readme file
└── 📜 requirements.txt          # Required packages

```

Installation
------------

Clone the repository:
```bash
git clone https://github.com/yourusername/safeAR-saas.git
```
Install the required packages:
```
pip install -r requirements.txt
```
Note: SafeAR SaaS was tested using Python 3.10.12.


Model Download and Conversion
------------------------------  

For setup instructions, please follow the links to the respective repositories:

<!-- - **RT-DETR**: [RT-DETR Repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) -->
- **Yolov5-seg**: [Yolov5 Repository](https://gitcode.net/openmodel/yolov5-seg)
- **Yolov8-seg**: [Yolov8 Repository](https://docs.ultralytics.com/models/yolov8/#performance-metrics) 


Note: The models should be converted to ONNX format and placed in the `seg_models` directory.

Usage
-----

The API can be used both as a command-line tool and as a Python library.

### Command-Line Usage

To use the API as a command-line tool, run the following command:
```css
python main.py --model_number MODEL_NUMBER --class_id_list CLASS_ID_1 CLASS_ID_2 ... --obfuscation_type_list OBFS_TYPE_1 OBFS_TYPE_2 ... --img_source IMG_SOURCE --show_fps
```
where:

* `MODEL_NUMBER` is the number of the model to use (0-based index).
* `CLASS_ID_1 CLASS_ID_2 ...` is a list of class IDs to obfuscate, according to the model's classes.
* `OBFS_TYPE_1 OBFS_TYPE_2 ...` available obfuscation types are `bluring`, `masking`, and `pixelation`.
* `IMG_SOURCE` is the source of the images to process. This can be a file path, a URL, or a camera index.
* `--show_fps` is an optional flag to show the frames per second.
* `--save_boxes` is an optional flag to save the bounding boxes to a file.
* `--save_video` is an optional flag to save the processed video to a file.

For example:
```bash
python main.py --model_number 0 --class_id_list 0 1 2 --obfuscation_type_list bluring masking pixelation --img_source 0 --show_fps
```
This will use the first available model to obfuscate objects with class IDs 0, 1, and 2 in the video stream from the default camera, using the `bluring`, `masking`, and `pixelation` obfuscation types, and showing the frames per second on the screen.

### Python Library Usage

To be implemented...

Acknowledgements
----------------

This work is funded by FCT - Fundação para a Ciência e a Tecnologia, I.P., through project with reference 2022.09235.PTDC.

<!-- Contributing
------------

TO BE DONE... -->

License
-------

To be determined...

<p align="center">
<img src="assets/CIIC_logo_v2.png" width="750px"/>
</p>
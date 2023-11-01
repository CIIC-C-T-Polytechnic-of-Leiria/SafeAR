## YOLOv5 with Obfuscation
This repository is a modified version of the original YOLOv5 repository. It includes additional functionality for obfuscating certain classes of objects in images or videos.

## Modified Files
The following files in the YOLOv5 repository have been modified to support obfuscation:

```
yolov5
├── segment
│   ├── predict.py (this file was changed/adpated to support obsfucation)
│   └── safe_layer.py (this file contains obsfucation functions)
└── ...

```
## Usage
You can run the obfuscation using the `predict.py` script in the segment directory. The script takes several command-line arguments:

`--weights`: Path to the weights file for the model.  
`--source`: Path to the input data (image or video).  
`--view-img`: Include this flag to display the image/video window during inference.  
`--classes`: List of class (number `0` to `79`) indices to detect and obfuscate (see this [file](yolov5\data\coco128-seg.yaml)).  
`--policy`: Obfuscation policy (`'mask'`, `'blur'`, or `'pixelate'`) to apply to the detected objects. 

 
## Example
Here's an example command that runs the `'mask'` obfuscation policy on class `0`` (i.e., "person" class) in the video "path/to/video.mp4:


```bash
python segment/predict.py --weights yolov5n-seg.pt --source path/to/video.mp4 --view-img --classes 0 --policy "mask"
```
This command will display the video with the specified obfuscation applied.

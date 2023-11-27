# YOLOv5 with Obfuscation 🕵️‍♂️🔒



(*Privacy, Confidentiality and other Ethical implications of image/video obfuscation*)

This repository is a modified version of the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5). It includes additional functionality for obfuscating certain classes of objects in images or videos.

## To Do List 📝
- [x] Implement obfuscation functionality with different policies (masking, blur, pixelation) 
- [x] Implement different obfuscation policies for different classes
- [ ] Measure segmentation accuracy and inferencing time
- [ ] Evaluate performance under adversarial attacks
- [ ] Implement a GUI to select the obfuscation policy and classes to obfuscate?

<details>
<summary><h2>Possible Attacks! 🤔 </h2></summary>

#### White Box Attacks:
- [ ] [FGSM - Fast Gradient Sign Method](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=pt-br)
- [ ] [Certified Radius-Guided Attack](https://arxiv.org/abs/2304.02693)
- [ ] [PGD - Projected Gradient Descent Method](https://angms.science/doc/CVX/CVX_PGD.pdf)
- [ ] [DeepFool](https://arxiv.org/pdf/1511.04599.pdf)

#### Black Box Attacks:

##### Gradient Estimation Attacks:
- [ ] [ZOO - Zeroth Order Optimization](https://arxiv.org/pdf/1708.03999.pdf)
- [ ] [NES - Natural Evolution Strategies](https://arxiv.org/pdf/1703.03864.pdf)

##### Transfer-Based Attacks:
- [ ] [SimBA](https://arxiv.org/pdf/1905.07121.pdf)
- [ ] [Square Attack](https://arxiv.org/pdf/1912.00049.pdf)

##### Bandit Gradient Descent (BGD) and Certified Radius-Guided Black-Box Attack:
- [ ] [Bandit Gradient Descent (BGD)](https://arxiv.org/abs/2304.02693)
- [ ] [Certified Radius-Guided Black-Box Attack](https://deepai.org/publication/a-certified-radius-guided-attack-framework-to-image-segmentation-models)

#### Adversarial Segmentation Attacks:
- [ ] [Adaptive Segmentation Mask Attack (ASMA)](https://github.com/utkuozbulak/adaptive-segmentation-mask-attack)
- [ ] [Model Obfuscation Attacks](https://arxiv.org/abs/2306.06112)
- [ ] [Self-Obfuscation Attack](https://deepai.org/publication/hiding-behind-backdoors-self-obfuscation-against-generative-models)
- [ ] [Random Segmentation Attack](https://arxiv.org/abs/2309.05941)

#### Attacks to Organize in the Future:
- [ ] [CW](https://arxiv.org/pdf/1608.04644.pdf)
- [ ] [JSMA](https://arxiv.org/pdf/1511.07528.pdf)
- [ ] [One Pixel Attack](https://arxiv.org/pdf/1710.08864.pdf)
- [ ] [Local Search Attack](https://arxiv.org/pdf/1801.10578.pdf)
- [ ] [Boundary Attack](https://arxiv.org/pdf/1712.04248.pdf)
- [ ] [HopSkipJumpAttack](https://arxiv.org/pdf/1904.02144.pdf)
- [ ] [Spatial Attack](https://arxiv.org/pdf/2004.04635.pdf)
- [ ] [GenAttack](https://arxiv.org/pdf/2007.06680.pdf)
- [ ] [AutoZOOM](https://arxiv.org/pdf/2008.09677.pdf)
- [ ] [SignHunter](https://arxiv.org/pdf/1907.07171.pdf)

</details>

## Visual Examples and Videos 📷

(*Before and after obfuscation*)

## Modified Files 📂
The following files in the YOLOv5 repository have been modified to support obfuscation:

```plaintext
yolov5
├── segment
│   ├── predict.py (this file was changed/adpated to support obsfucation)
│   └── safe_layer.py (this file contains obsfucation functions)
└── ...

```
## Dependencies 📦
(*to complete*)

## Usage (not tested on a different machine yet...) 📋

To use this repository, you must first clone it using the following command 
```bash
git clone https://github.com/CIIC-C-T-Polytechnic-of-Leiria/SafeAR
```

To use this repository, you must first install the dependencies listed in the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

You can run the obfuscation using the `predict.py` script in the segment directory. The script takes several command-line arguments:

`--weights`: Path to the weights file for the model.  
`--source`: Path to the input data (image or video).  
`--view-img`: Include this flag to display the image/video window during inference.  
`--classes`: List of class (number `0` to `79`) indices to detect and obfuscate (refer to this [file](data/coco128-seg.yaml)).  
`--policy`: Obfuscation policy (`'mask'`, `'blur'`, or `'pixelate'`) to apply to the detected objects. 

 
## Example of usage 🚀
Here's an example command that runs the `'mask'` obfuscation policy on class `0` (i.e., "person" class) in the video "path/to/video.mp4:


```bash
python segment/predict.py --weights yolov5n-seg.pt --source path/to/video.mp4 --view-img --classes 0 --policy "mask"

python segment/predict.py --weights yolov5n-seg.pt --source 1 --view-img --classes 0 --policy "mask"
```
This command will display the video with the specified obfuscation applied.

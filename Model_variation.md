
**Model Comparison**
| **Model** | **Processing Location** | **Data Sent to Server** | **Obfuscation Location** | **Latency Concerns** | **Accuracy Potential** |
| --- | --- | --- | --- | --- | --- |
| **Cloud** | Server | Video Frames | Server | High (not suitable) | High |
| **Edge** | Device | None | Device | Low (suitable) | Medium (limited by device compute) |
| **Hybrid** | Both (Device & Server) | Reduced Image | Server | Medium (may not meet real-time req.) | High (leveraging server compute) |
| **BoxProxy** | Both (Device & Server) | Bounding Boxes | Server | Medium (may not meet real-time req.) | Medium (dependent on box accuracy) |

**System Overview**
The proposed system, dubbed "ARShield," consists of the following components:

1. **Edge Device**: A high-performance, low-power edge device (e.g., smartphone, AR glasses) responsible for capturing video frames, performing object detection, and sending relevant data to the server.
2. **Server**: A powerful server that receives data from the edge device, performs segmentation and obfuscation, and returns the processed results to the edge device.
3. **Communication Protocol**: A custom-designed protocol that ensures efficient, secure, and low-latency communication between the edge device and server.

**ARShield Architecture**

```mermaid
graph LR;
  EdgeDevice((Edge Device))
  ObjectDetection((Object Detection))
  SFP((Selective Frame Processing (SFP)))
  WebRTC((WebRTC-based Communication Protocol))
  Server((Server))
  Segmentation((Segmentation (Mask R-CNN, U-Net)))
  SOR((Secure Object Representation (SOR)))
  Caching((Caching Mechanism))
  ObfuscatedObjectReturn((Obfuscated Object Return))
  ARApplication((AR Application))
  RenderObfuscatedObjects((Render Obfuscated Objects))

  EdgeDevice --> ObjectDetection;
  ObjectDetection --> SFP;
  SFP --> WebRTC;
  WebRTC --> Server;
  Server --> Segmentation;
  Segmentation --> SOR
  SOR --> Caching;
  Caching --> ObfuscatedObjectReturn;
  ObfuscatedObjectReturn --> ARApplication;
  ARApplication --> RenderObfuscatedObjects;
```


**Metrics**



| **Metric** | **Description** | **Unit** |
| --- | --- | --- |
| **mAP** | Mean Average Precision | % |
| **Precision** | Ratio of true positives to sum of true positives and false positives | % |
| **Recall** | Ratio of true positives to sum of true positives and false negatives | % |
| **F1-score** | Harmonic mean of precision and recall | % |
| **IoU** | Intersection over Union (segmentation quality) | % |
| **Dice Coefficient** | Similar to IoU, but provides a more nuanced measure of segmentation quality | % |
| **Boundary F1-score** | Evaluates the model's ability to accurately predict object boundaries | % |
| **Obfuscation Rate** | Proportion of successfully obfuscated objects | % |
| **Obfuscation Quality** | Measures the quality of the obfuscated objects (e.g., PSNR or SSIM) | dB or unitless |
| **End-to-End Latency** | Time it takes to perform object detection, segmentation, and obfuscation | ms |
| **Throughput** | Frames per second processed by the system | fps |
| **Power Consumption** | Energy efficiency of the system | Watt-hours (Wh) |
| **False Positive Rate** | Ratio of false positives to sum of true positives and false positives | % |
| **False Negative Rate** | Ratio of false negatives to sum of true positives and false negatives | % |
| **Detection Time** | Time it takes to detect objects in a frame | ms |
| **Boundary IoU** | Evaluates the overlap between predicted segmentation mask boundary and ground truth mask boundary | % |
| **Region-based Metrics** | Precision, recall, and F1-score for region-based segmentation tasks | % |
| **Obfuscation Strength** | Measures the effectiveness of obfuscation in protecting sensitive information | unitless |
| **Obfuscation Robustness** | Evaluates the robustness of obfuscation against various attacks or manipulations | unitless |
| **Frame Drop Rate** | Rate at which frames are dropped due to processing limitations or latency issues | % |
| **System Resource Utilization** | Monitors the usage of system resources such as CPU, GPU, memory, and bandwidth | % or MB/s |
| **AR Protection Index (API)** | Composite metric evaluating overall performance of AR system in detecting, segmenting, and obfuscating sensitive objects while maintaining a good user experience | unitless |
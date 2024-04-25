
**Proposed Models Comparison**

| **Model** | **Processing Location** | **Data Sent to Server** | **Obfuscation Location** | **Accuracy Potential** |
| --- | --- | --- | --- | --- | 
| **Cloud** | Server | Video Frames | Server  | High |
| **Edge** | Device | None | Device  | Medium (limited by device compute) |
| **Hybrid** | Both (Device & Server) | Reduced Image | Server |  High (leveraging server compute) |


*Note*: A box proxy  may be considered as an option in systems where speed is a priority over accuracy. This model simplifies the object representation of obfucation reigions to bounding boxes, resulting in faster processing times but potentially lower accuracy.

**System Overview**

The proposed system, dubbed "SafeAR Layer," consists of the following components:

1. **Edge Device**: A high-performance, low-power edge device (e.g., smartphone, AR glasses) responsible for capturing video frames, performing object detection, and sending relevant data to the server.
2. **Server**: A powerful server that receives data from the edge device, performs segmentation and obfuscation, and returns the processed results to the edge device.
3. **Communication Protocol**: A custom-designed protocol that ensures efficient, secure, and low-latency communication between the edge device and server.
**ARShield Architecture**

<span style="color: #87CEEB">**Edge Device**</span>
↪ <span style="color: #FFC107">**Object Detection**</span>
  ↪ <span style="color: #F7DC6F">**Selective Frame Processing (SFP)**</span>
    ↪ <span style="color: #8BC34A">**WebRTC-based Communication Protocol**</span>
      ↪ <span style="color: #9C27B0">**Server**</span>
        ↪ <span style="color: #66D9EF">**Segmentation**</span>
          ↪ <span style="color: #B3E5FC">**Secure Object Representation (SOR)**</span>
            ↪ <span style="color: #E5E5EA">**Caching Mechanism**</span>
              ↪ <span style="color: #9E9E9E">**Obfuscated Object Return**</span>
                ↩ <span style="color: #87CEEB">**Edge Device**</span>
                  ↪ <span style="color: #FFC107">**AR Application**</span>
                    ↪ <span style="color: #F7DC6F">**Render Obfuscated Objects**</span>




**1. Performance Metrics:**

**1.1 Segmentation (Cloud and Device)**

* mAP: Mean Average Precision (% )
* IoU: Intersection over Union (segmentation quality) (% )
* Dice Coefficient: 


**1.2. Latency and Speed (Cloud and Device)**

* End-to-end latency: Time it takes to perform object detection, segmentation, and obfuscation (ms)
* Throughput: Frames per second processed by the system (fps)
* Network latency: Time it takes for data to travel from the device to the server and back (ms) (if applicable)


**1.3. Network Throughput (Cloud only)**

* Upload Speed: The rate at which data is transferred from the device to the cloud (Mbps)
* Download Speed: The rate at which data is transferred from the cloud to the device (Mbps)
* Network Latency: The time it takes for data to travel from the device to the cloud and back (ms)

**2. Subjective User Experience Metrics (Device only)**

* User Satisfaction: Measures the user's satisfaction with the AR system (e.g., on a scale of 1 to 5)
* Usability: Measures the ease of use of the AR system (e.g., using the System Usability Scale)

**3.Security Metrics:**

* Obfuscation Strength: Measures the effectiveness of obfuscation in protecting sensitive information ?
* Obfuscation Robustness: Evaluates the robustness of obfuscation against various attacks or manipulations ?
* Data Privacy: Measures the extent to which the AR system protects user data and maintains privacy ?


**4.System Metrics:**

* Power Consumption
* Estimated Cost
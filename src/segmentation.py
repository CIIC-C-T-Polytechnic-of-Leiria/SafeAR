# """
# This module contains the class definition for the SegModel class.
# """
#
# import cv2
# import numpy as np
# import onnxruntime as ort
# from typing import Tuple, Union
#
#
# class SegModel:
#     def __init__(
#         self,
#         model_path: str,
#         target_size: Tuple[int, int] = (640, 640),
#         interpolation: int = cv2.INTER_AREA,
#         output_format: str = "numpy",
#         output_path: str = None,
#         confidence_threshold: float = 0.7,
#         iou_threshold: float = 0.5,
#         num_masks: int = 32,
#     ):
#         """
#
#         """
#         self.model_path = model_path
#         self.target_size = target_size
#         self.interpolation = interpolation
#         self.output_format = output_format
#         self.output_path = output_path
#         self.confidence_threshold = confidence_threshold
#         self.iou_threshold = iou_threshold
#         self.num_masks = num_masks
#
#         self.model = init_model()
#
#
#     def init_model(self):
#         """
#         Initialize the model.
#         """
#         # Load the model
#         self.model = ort.InferenceSession(self.model_path,
#                                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
#                                           if ort.get_device() == "GPU"
#                                           else ["CPUExecutionProvider"]
#                                           )
#
#         # Get the model metadata
#         model_inputs = self.model.get_inputs()
#         model_outputs = self.model.get_outputs()
#         self.input_names = [input.name for input in model_inputs]
#         self.output_names = [output.name for output in model_outputs]
#         self.input_shapes = [input.shape for input in model_inputs]
#         self.output_shapes = [output.shape for output in model_outputs]
#
#     def preprocess_image(self, image: np.ndarray) -> Union[None, np.ndarray]:
#
#         try:
#             resized_img = cv2.resize(
#                 image, (self.target_size), interpolation=self.interpolation
#             )
#             return resized_img
#
#         except Exception as e:
#             print(f"Error processing image: {e}")
#             return None
#
#     def predict(self, image: np.ndarray) -> Union[None, np.ndarray]:
#         """
#
#         """
#         try:
#             preprocessed_img = self.preprocess_image(image)
#
#             if preprocessed_img is None:
#                 return None
#
#             output = self.infer(preprocessed_img)
#
#         except Exception as e:
#             print(f"Error predicting image: {e}")
#             return None
#
#
#     def infer(self, image: np.ndarray) -> Union[None, np.ndarray]:
#         """
#
#         """
#         try:
#             # Preprocess the image
#             preprocessed_img = self.preprocess_image(image)
#
#             # Check if the image was preprocessed successfully
#             if preprocessed_img is None:
#                 return None
#
#             # Perform inference
#             input_data = np.expand_dims(preprocessed_img, axis=0)
#             input_data = input_data.astype(np.float32)
#             input_data = np.transpose(input_data, [0, 3, 1, 2])
#             outputs = self.model.run(None, {self.input_names[0]: input_data})
#
#             # Postprocess the output
#             # (code for postprocessing goes here)
#         except Exception as e:
#             print(f"Error performing inference: {e}")
#             return None
#

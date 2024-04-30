# import ffmpegcv
# import cupy as cp
# import time
#
#
# class Camera:
#     def __init__(
#         self, source, display_fps=False, save_video=False, output_file="output.mp4"
#     ):
#         self.display_fps = display_fps
#         self.frame_count = 0
#         self.start_time = time.time()
#
#         self.save_video = save_video
#         # if self.save_video:
#         #     # Assuming you have a way to get video dimensions and FPS
#         #     # For simplicity, we'll use placeholders here
#         #     width = 640
#         #     height = 480
#         #     fps = 30
#
#         #     # Define the codec and create a VideoWriter object
#         #     # Note: ffmpegcv does not directly support VideoWriter
#         #     # You might need to use a workaround or another library for this
#         #     self.out = ffmpegcv.VideoWriter(output_file, width, height, fps)
#
#         # Initialize ffmpegcv for video capture
#         self.cap = ffmpegcv.VideoCapture(
#             source, codec="h264_cuvid"
#         )  ## TODO: Se source for camera
#
#     def get_frame(self, process_frame=None):
#         ret, frame = self.cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             return None
#         if process_frame:
#             frame = process_frame(frame)
#         self.frame_count += 1
#         # convert frame to CuPy array
#         frame = cp.asarray(frame)
#         return frame
#
#     def display_frame(self, frame, info=None):
#         # Note: Displaying frames directly with CuPy or ffmpegcv is not straightforward
#         # You might need to convert the frame to a format that can be displayed
#         # For example, converting to a NumPy array and using OpenCV for display
#         if self.display_fps:
#             fps = self.get_fps()
#             # Display FPS and info using OpenCV or another library
#             pass
#         if self.save_video:
#             # Write the frame to the output file
#             # Note: You might need to convert the frame to a format compatible with ffmpegcv
#             self.out.write(frame)
#
#     def release(self):
#         self.cap.release()
#         if self.save_video:
#             self.out.release()
#
#     def wait_key(self, key_to_quit="q"):
#         # Note: ffmpegcv does not support waitKey
#         # You might need to implement a different mechanism for key press detection
#         pass
#
#     def get_fps(self):
#         elapsed_time = time.time() - self.start_time
#         return self.frame_count / elapsed_time
#
#     def set_fps(self, fps):
#         # Note: Setting FPS might not be directly supported by ffmpegcv
#         pass
#
#     def resize_frame(self, frame, width, height):
#         # Convert frame to CuPy array if not already
#         frame_cp = cp.asarray(frame)
#         # Resize the frame using CuPy
#         resized_frame = cp.resize(frame_cp, (height, width))
#         # Convert back to a format compatible with ffmpegcv or display
#         return cp.asnumpy(resized_frame)

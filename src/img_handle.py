import cv2
import time


class Camera:
    def __init__(
        self, source, display_fps=False, save_video=False, output_file="output.mp4"
    ):
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        elif isinstance(source, str):
            self.cap = cv2.VideoCapture(source)
        else:
            raise ValueError(
                "source must be either an integer (camera index) or a string (file path)."
            )

        if not self.cap.isOpened():
            raise ValueError("Could not open video source.")

        self.display_fps = display_fps
        self.frame_count = 0
        self.start_time = time.time()

        self.save_video = save_video
        if self.save_video:
            # Get the video dimensions
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
            self.out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    def get_frame(self, process_frame=None):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        if process_frame:
            frame = process_frame(frame)
        self.frame_count += 1
        return frame

    def display_frame(self, frame, info=None):
        if self.display_fps:
            fps = self.get_fps()
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.9,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                info,
                (10, 20),
                cv2.FONT_HERSHEY_PLAIN,
                0.9,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("Processed Frame", frame)
        if self.save_video:
            self.out.write(frame)

    def release(self):
        self.cap.release()
        if self.save_video:
            self.out.release()
        cv2.destroyAllWindows()

    def wait_key(self, key_to_quit="q"):
        key = cv2.waitKey(1) & 0xFF
        if key == ord(key_to_quit):
            self.release()
            return False
        return True

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time

    def set_fps(self, fps):
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def resize_frame(self, frame, width, height):
        return cv2.resize(frame, (width, height))

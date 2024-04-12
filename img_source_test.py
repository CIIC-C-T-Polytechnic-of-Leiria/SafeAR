import cv2

def test_webcam(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        return
    ret, frame = cap.read()
    if not ret:
        print(f"Can't receive frame from camera {index}")
        return
    cv2.imshow(f'Camera {index}', frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# Test the webcams
for i in range(2):  # Adjust the range if you have more webcams
    test_webcam(i)
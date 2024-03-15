# Still Trying

import cv2
import os
import threading
import shutil
from ultralytics import YOLO

class CSI_Camera:
    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

# function for CSI Camera initialization
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def motion_and_detect():
    # Clear previous prediction
    if os.path.exists("runs"):
        shutil.rmtree("./runs")

    # Model
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="PsVRPB3iBvz752pG3MrN")
    # project = rf.workspace().project("smart-cart-2-fvadu")
    model = YOLO('best_old.pt')
    names = model.names

    # Initialize the camera top
    cam_top = cv2.VideoCapture(0)  # Use 0 for default camera

    # Initialize variables
    prev_frame = None
    #motion_direction = None

    while True:
        motion_detected = False

        ret, frame = cam_top.read()

        if not ret:
            break

        # Convert frame to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)  # Apply Gaussian blur to reduce noise

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # Calculate the absolute difference between the current frame and the previous frame
        frame_delta = cv2.absdiff(prev_frame, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:  # Adjust this threshold as needed
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if y > prev_frame.shape[0] / 2:
                #motion_direction = "Top to Bottom"
                motion_detected = True
            else:
                #motion_direction = "Bottom to Top"
                motion_detected = True

        # Display the frames and motion direction
        # cv2.putText(frame, f"Motion Direction: {motion_direction}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow("Camera Feed", frame)

        prev_frame = gray_frame

        # Bottom Cam
        if motion_detected:
            cv2.imwrite("./images/prediction.jpg", frame)

            print("\nMotion is detected.")

            # infer on a local image
            prediction = model("./images/prediction.jpg", conf=0.5, save=True)
            for p in prediction:
                for c in p.boxes.cls:
                    print("\nPrediction:", names[int(c)])
        else:
            print("\nNo motion is detected.")

            # visualize your prediction
            #model.predict("./images/after.jpg", confidence=40, overlap=30).save("./images/prediction.jpg")

        # source_image_path = './images/after.jpg'
        # destination_image_path = './images/before.jpg'
        # shutil.copyfile(source_image_path, destination_image_path)

        # Exit by pressing 'Q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if keyboard.is_pressed('q'):
            print("Exiting the program.")
            break

    cam_top.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_and_detect()
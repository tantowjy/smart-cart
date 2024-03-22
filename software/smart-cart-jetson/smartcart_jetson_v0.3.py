import cv2
import firebase_admin
import time
from firebase_admin import credentials, db
from ultralytics import YOLO

# Initialize the Firebase app
cred = credentials.Certificate('smart-cart-f45d1-firebase.json')
firebase_admin.initialize_app(cred, {
	'databaseURL' : 'https://smart-cart-f45d1-default-rtdb.firebaseio.com'
})

id_customer = str(int(time.time()))
ref = db.reference(f'detect/{id_customer}/')


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
    sensor_id=0
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    
def tubi_detect():
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    # video_capture = cv2.VideoCapture(0)
    prev_frame = None

    if video_capture.isOpened():
        try:
            start_time = time.time()
            object_counts = {}
            
            while True:
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Convert frame to grayscale for object detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Load a model
                model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

                # Run batched inference on a list of images
                # results = model.predict(source=frame, save_txt=False, project='Object Detection', conf=0.5, iou=0.5)  # return a list of Results objects
                results = model(frame)

                
                # Process the results
                #object_counts = {}
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.data[0][-1])
                        object_name = model.names[class_id]
                        # print(object_name)
                        if object_name in object_counts:
                            object_counts[object_name] += 1
                        else:
                            object_counts[object_name] = 1
                        
                
                for object_name, count in object_counts.items():
                    print(f"{object_name}: {count}")
                    ref.child(object_name).set({'count': count})

                    
                time.sleep(1)

                # Time for exit
                elapsed_time = time.time() - start_time
                if elapsed_time >= 120:  # 2 minutes
                    print("Program closed after 2 minutes")
                    break

                # Exit by pressing 'Q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            video_capture.release()

    else:
        print("Unable to open camera")


if __name__ == "__main__":
    tubi_detect()

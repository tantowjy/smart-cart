import cv2
from ultralytics import YOLO

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
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
    window_title = "Tubi Shop Smart Cart"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    prev_frame = None
    motion_direction = None
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret, frame = video_capture.read()

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
                        # Load a model
                        cv2.imwrite('temp.jpg', frame) 
                        model = YOLO('productmodel_231015.pt')  # pretrained YOLOv8n model
                        
                        # Run batched inference on a list of images
                        model.predict(source=0, save_txt=True, project='Top to Bottom', conf=0.5, iou=0.5)  # return a list of Results objects
                        # Exit by pressing 'Q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    else:
                        cv2.imwrite('temp.jpg', frame) 
                        model = YOLO('productmodel_231015.pt')  # pretrained YOLOv8n model

                        # Run batched inference on a list of images
                        model.predict(source=0, save_txt=True, project='Bottom to Top', conf=0.5, iou=0.5)# return a list of Results objects
                        # Exit by pressing 'Q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                # Display the frames and motion direction
                cv2.putText(frame, "Motion Direction:{}".format(motion_direction), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_title, frame)

                prev_frame = gray_frame

                # Exit by pressing 'Q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    tubi_detect()

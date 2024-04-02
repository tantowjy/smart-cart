import cv2
import firebase_admin
import time
import json
from firebase_admin import credentials, db, storage, firestore
from ultralytics import YOLO

# Initialize the Firebase app
cred = credentials.Certificate("tubescc2023-firebase-adminsdk-a4hvb-b62ff9d985.json")#path to key
firebase_admin.initialize_app(cred, {"storageBucket": "tubescc2023.appspot.com"})# initialize storage bucket
bucket = storage.bucket()
db = firestore.client()
json_file_path = 'object_counts.json'
document_name = 'list barang' 

#pipeline for camere number 1 (object detection)
def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor_id=0 ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=True"
        % flip_method
    )
#pipeline for camera number 2 (motion detection)
def gstreamer_pipeline_2(capture_width=1280, capture_height=720, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor_id=1 ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=True"
        % flip_method
    )
#save json to local data for handling
def save_json_to_file(json_data, file_path):
    # Write JSON data to file
    with open(file_path, 'w') as file:
        file.write(json_data)
    print(f"JSON data saved to {file_path}") 
#extracting names from json data
def extract_names_from_json(json_file):
    # Read JSON data from file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Extract names based on counts
    names = []
    for item in json_data['objects']:
        name = item['name']
        count = item['count']
        names.extend([name] * count)
    
    return names
#retrieve active user right now, for developer mode only 1 can be active for tesing
def get_activeuser():
    # Access Firestore database
    db = firestore.client()

    # Define the path to the document you want to retrieve
    doc_ref = db.collection('currentUser').document('active')
    
    try:
    # Get the document snapshot
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            # Retrieve all data from the document as a dictionary
            data = doc.to_dict()
            # Now, 'data' contains all fields from the document
            # print("Retrieved data:", data)
            # Save the data to a local JSON file
            with open('active_user_data.json', 'w') as json_file:
                json.dump(data, json_file)
                print("Data saved to 'active_user_data.json' successfully.")
        else:
            print("Document does not exist")
    except Exception as e:
        print("Error:", e)

    with open('active_user_data.json', 'r') as json_file:
        data = json.load(json_file)
    if 'user' in data:
        user_email = data['user']
        # print("Retrieved user email:", user_email)
        return user_email
    else:
        print("Key 'user' not found in the JSON data.")
        return None
#save names to local file
def save_names_to_file(names, file_path):
    # Write names to file
    with open(file_path, 'w') as file:
        file.write('\n'.join(names))
    print(f"Names saved to {file_path}")
#Push names of detected objects in strings to firestorage of specific active user
def save_names_to_firestore(user_id, document_name, names):#parent_folder, collection_name, document_name, names):
    # Reference the user's document
    doc_id="ProdukDimasukan"
    user_ref = db.collection('pengguna').document(user_id)
    doc_ref = user_ref.collection(document_name).document(doc_id)
    doc_ref.set({'names': names})
    print(f"Names saved to Firestore collection {user_id}/{document_name}/{doc_id}")
#Object detection based on pre trained model
def tubi_detect():
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    #video_capture = cv2.VideoCapture(0) #used for PC testing
    prev_frame = None
    user_id=get_activeuser()
    if video_capture.isOpened():
        try:
            start_time = time.time()
            
            while True:
                ret, frame = video_capture.read()

                if not ret:
                    break

                # Convert frame to grayscale for object detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Load a model
                model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

                # Run batched inference on a list of images
                # results = model.predict(source=frame, save_txt=True, project='Object Detection', conf=0.5, iou=0.5)  # return a list of Results objects
                results = model(frame)

                
                # Process the results
                object_counts = {}
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.data[0][-1])
                        object_name = model.names[class_id]
                        # print(object_name)
                        if object_name in object_counts:
                            object_counts[object_name] += 1
                        else:
                            object_counts[object_name] = 1
                        
                
                # Create a dictionary with object names and counts
                output_data = {"objects": []}
                for object_name, count in object_counts.items():
                    output_data["objects"].append({"name": object_name, "count": count})

                # Convert the dictionary to JSON string
                json_output = json.dumps(output_data)

                # Print the JSON output
                print(json_output)
                # result_blob = bucket.blob("results/result.json")

                # Upload JSON data to Firebase Storage
                blob = bucket.blob("object_counts.json")
                blob.upload_from_string(json_output)
                save_json_to_file(json_output, 'object_counts.json')

                print("JSON data uploaded to Firebase Storage")
                time.sleep(1)
                # Time for exit
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10:  # 5 seconds only get one read
                    print("Object detection done after 10 seconds")
                    break

                # Exit by pressing 'Q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            video_capture.release()
            names = extract_names_from_json(json_file_path)

            # Save names to a local file
            save_names_to_file(names, 'names.txt')
            

            # Save names to Firestore for the active user
            save_names_to_firestore(user_id, document_name, names)

    else:
        print("Unable to open camera")

def detect_motion():
    video_capture=cv2.VideoCapture(gstreamer_pipeline_2(), cv2.CAP_GSTREAMER)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=350, detectShadows=False)
    start_time = time.time()
    # Initialize direction variables
    direction = "None"
    last_direction = "None"
    temp_direction="None"

    # Minimum contour area threshold
    min_contour_area = 600  

    # Maximum distance threshold (in pixels) for objects to be considered nearer than 30 cm
    max_distance_pixels = 200  

    while True:
        ret, frame = video_capture.read()
        last_direction = temp_direction
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if valid_contours:
            # Sort valid contours by area (largest first) and select the largest contour
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:1]

            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                centroid_x = x + w // 2
                centroid_y = y + h // 2

                distance_pixels = max(w, h)

                if distance_pixels < max_distance_pixels:
                    if centroid_y < frame.shape[0] // 2:
                        direction = "Up"
                    else:
                        direction = "Down"
                else:
                    direction = "None"
        else:
            direction = "None"  # Set direction to "None" when no valid contours are detected

        # Update last direction if the current direction is not "None"
        
        temp_direction = direction

        # Display current and last detected directions on the frame
        text = f"Current Direction: {direction}, Last Direction: {last_direction}"
        #cv2.putText(frame, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.imshow('Motion Detection', frame)
        print(text)
        time.sleep(0.25)
        if direction == "Down" and last_direction != "Down":
                        tubi_detect()
        elapsed_time = time.time() - start_time
        if elapsed_time >= 120:  # 2 minutes
                print("Program closed after 120 seconds")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    #cv2.destroyAllWindows()
if __name__ == "__main__":
    detect_motion()
    


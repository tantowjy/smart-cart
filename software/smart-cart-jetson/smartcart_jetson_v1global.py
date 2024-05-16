import cv2
import firebase_admin
import time
import json
from firebase_admin import credentials, db, storage, firestore
from ultralytics import YOLO

# Initialize the Firebase app
cred = credentials.Certificate("smart_cart.json")#path to key
firebase_admin.initialize_app(cred, {"storageBucket": "smart-cart-4a2f9.appspot.com"})# initialize storage bucket
bucket = storage.bucket()
db = firestore.client()
json_file_path = 'object_counts.json'
document_name = 'listBarang' 

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

def get_activeuser():
    db = firestore.client()
    doc_ref = db.collection('currentUser').document('active')
    
    try:
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
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
        return user_email
    else:
        print("Key 'user' not found in the JSON data.")
        return None
#save names to local file
def save_names_to_file(names, file_path):
    with open(file_path, 'w') as file:
        file.write('\n'.join(names))
    print(f"Names saved to {file_path}")

def save_names_to_firestore(user_id, document_name, names):
    doc_id = "produkDimasukkan"
    user_ref = db.collection('pengguna').document(user_id)
    doc_ref = user_ref.collection(document_name).document(doc_id)
    doc_ref.set({'names': names})
    print(f"Names saved to Firestore collection {user_id}/{document_name}/{doc_id}")

def tubi_detect():
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    prev_frame = None
    user_id=get_activeuser()
    if video_capture.isOpened():
        try:
            start_time = time.time()
            
            while True:
                ret, frame = video_capture.read()

                if not ret:
                    break

                model = YOLO('spc_10products_v4s.pt') 
                results = model(frame)

                object_counts = {}
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.data[0][-1])
                        object_name = model.names[class_id]
                        if object_name in object_counts:
                            object_counts[object_name] += 1
                        else:
                            object_counts[object_name] = 1
                        
                
                output_data = {"objects": []}
                for object_name, count in object_counts.items():
                    output_data["objects"].append({"name": object_name, "count": count})
                json_output = json.dumps(output_data)
                print(json_output)
                blob = bucket.blob("object_counts.json")
                blob.upload_from_string(json_output)
                save_json_to_file(json_output, 'object_counts.json')

                print("JSON data uploaded to Firebase Storage")
                time.sleep(1)

                elapsed_time = time.time() - start_time
                if elapsed_time >= 7:  
                    print("Object detection done")
                    break

                # Exit by pressing 'Q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            video_capture.release()
            names = extract_names_from_json(json_file_path)
            save_names_to_file(names, 'names.txt')
            save_names_to_firestore(user_id, document_name, names)

    else:
        print("Unable to open camera")

def detect_motion():
    video_capture=cv2.VideoCapture(gstreamer_pipeline_2(), cv2.CAP_GSTREAMER)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=250, detectShadows=False)
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
        
        temp_direction = direction
        text = f"Current Direction: {direction}, Last Direction: {last_direction}"
        print(text)
        time.sleep(0.25)
        if direction == "Down" and last_direction != "Down":
                        tubi_detect()
        elapsed_time = time.time() - start_time


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
if __name__ == "__main__":
    detect_motion()
    


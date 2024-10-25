import cv2 #real time video and image capture
import mediapipe as mp #live and recorded video analysis.
import numpy as np #handling multi-dimensional arrays,
import tensorflow as tf # loading the pre-trained model
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model.h5')

# Define mediapipe Face detector
face_detection = mp.solutions.face_detection.FaceDetection()

# Detection function
def get_detection(frame):
    height, width, channel = frame.shape

    # Convert frame BGR to RGB colorspace
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect results from the frame
    result = face_detection.process(imgRGB)

    try:
        for count, detection in enumerate(result.detections):
            # Extract bounding box information 
            box = detection.location_data.relative_bounding_box
            x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width), int(box.height*height)
            
    # If detection is not available then pass 
    except:
        return None

    return x, y, w, h

CATEGORIES = ['no_mask', 'mask']

# Start video capture
cap = cv2.VideoCapture(0)

try:
    while True:
        _, frame = cap.read()  # Capture each frame from the video feed
        img = frame.copy()  # Make a copy of the current frame

        detection_result = get_detection(frame)  # Call the face detection function
        
        if detection_result is not None:
            x, y, w, h = detection_result

            crop_img = img[y:y+h, x:x+w]

            crop_img = cv2.resize(crop_img, (100, 100))

            crop_img = np.expand_dims(crop_img, axis=0)

            # Get the prediction from the model
            prediction = model.predict(crop_img)
            print(prediction)
            index = np.argmax(prediction)
            res = CATEGORIES[index]

            # Set color based on result
            if index == 0:
                color = (0, 0, 255)  # Red for "no_mask"
            else:
                color = (0, 255, 0)  # Green for "mask"

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("frame", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Program is stop.")

finally:
    # Release resources
    cap.release()   # Release the webcam
    cv2.destroyAllWindows()   # Close the display window
    print("Resources released and program terminated cleanly.")

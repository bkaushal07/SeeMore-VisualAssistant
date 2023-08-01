# from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2
#
# model = YOLO('yolov8l-seg.pt') #pretrained on COCO
# model.predict(source="0", show=True, conf=0.5)  # accepts all formats
import cv2
import pyttsx3
from ultralytics import YOLO
from collections import Counter
import pandas as pd
# Initialize YOLOv8 model
model = YOLO('yolov8l-seg.pt')

# Open primary camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

while cap.isOpened():
    # Read frame from the camera
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break
    input_text=""
    # Perform object detection on the frame
    results = model.predict(frame,save=True)
    boxes = results[0].boxes.numpy()
    res_plotted = results[0].plot()
    cv2.imshow("result", res_plotted)
    for box in boxes:
        input_text = input_text+results[0].names[int(box.cls[0])]+","
    # Split the string into a list of words
    words = input_text.split(',')

    # Count the occurrences of each word
    word_counts = Counter(words)
    output=""
    # Generate the output string
    for word, count in word_counts.items():
        if word !="":
            if count > 1:
                output = output+' '+str(count)+' '+word+'s'
            else:
                output = output+' '+str(count)+' '+word

    # Print the output
    print(output)
    engine = pyttsx3.init()

    # Set the properties of the speech
    engine.setProperty('voice', 'en-us')
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)

    # Convert text to speech
    engine.say(output)
    engine.runAndWait()
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
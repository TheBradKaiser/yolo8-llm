from ultralytics import YOLO
import cv2
import time

# Load YOLO model
# model = YOLO("./yolov8n.pt")
model = YOLO("./yolo11n.pt")

# Set video source
video_path = '/dev/video2'
cap = cv2.VideoCapture(2)

# Check if the video stream opened successfully
if not cap.isOpened():
    print("Failed to open video stream.")
    exit()

# Set optional camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.startWindowThread()
cv2.namedWindow("YOLO Inference")
# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        print("Failed to read frame. Exiting...")
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    summary = results[0].summary()
    for x in summary:
        if (x["name"] == 'person' and x["confidence"] > .5):
            # save the next 1200 annotated_frames of the video to a .mp4 file
            pass
    cv2.imshow("YOLO Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

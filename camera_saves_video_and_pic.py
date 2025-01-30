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

# Define video writer parameters
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
output_fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.startWindowThread()
cv2.namedWindow("YOLO Inference")

# Loop through the video frames
saving_frames = False
frames_to_save = 0
video_writer = None

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
    # print(summary)
    for x in summary:
        if x["name"] == 'person' and x["confidence"] > 0.5:
            if not saving_frames:
                # Initialize video writer
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_image = f'./saves/images/detected_person_{timestamp}.png'
                cv2.imwrite(output_image,annotated_frame)
                output_file = f"./saves/videos/detected_person_{timestamp}.mp4"
                video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (frame_width, frame_height))
                saving_frames = True
                frames_to_save = output_fps * 10  # 10 seconds worth of frames

    # If saving_frames is True, write the current frame
    if saving_frames and frames_to_save > 0:
        # print("saving frames...")
        video_writer.write(annotated_frame)
        frames_to_save -= 1

        # Stop saving if frames_to_save reaches 0
        if frames_to_save == 0:
            saving_frames = False
            video_writer.release()
            video_writer = None

    cv2.imshow("YOLO Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
if video_writer:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()


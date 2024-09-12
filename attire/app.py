import cv2
import torch

# Use DirectShow as the backend
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)
# D:\IDK(random)\one\yolov5\runs\
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the YOLO model to detect objects in the frame
    results = model(img_rgb)

    # Extract detected objects (including attire)
    detections = results.pandas().xyxy[0]  # pandas DataFrame containing bounding boxes, confidence, etc.

    # Draw bounding boxes and labels on the frame
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, confidence, class_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        
        # Filter for attire-related items (e.g., 'person', 'bag', etc.)
        if class_name in ['person','formal-shirt','pant','tie','blazer','formal-shoes']:  # Extend this to other attire-related classes
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            match_percentage = confidence * 100
            print(match_percentage)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Webcam Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('Road_Video.mp4')

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

def get_density(count):
    if count < 10:
        return "Low"
    elif count < 20:
        return "Medium"
    else:
        return "High"

def signal_time(density):
    if density == "Low":
        return 20
    elif density == "Medium":
        return 40
    else:
        return 60

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:
                vehicle_count += 1

    density = get_density(vehicle_count)
    green_time = signal_time(density)

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Density: {density}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Green Signal: {green_time} sec", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Traffic Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Vehicles:", vehicle_count)
    print("Density:", density)
    print("Green Signal ON for", green_time, "seconds")
    time.sleep(2)

cap.release()
cv2.destroyAllWindows()

print("System Closed Successfully.")
import cv2
import time
import pyttsx3
from ultralytics import YOLO


def main():
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    important_classes = {"person", "car", "bicycle", "motorbike", "bus", "truck"}
    last_spoken_time = 0
    speak_gap = 1.5
    last_spoken_msg = ""

    print("[INFO] Press 'q' on the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't receive frame, exiting...")
            break

        h, w, _ = frame.shape

        results = model(frame, verbose=False)
        r = results[0]

        messages = []

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            name = r.names[cls_id]
            if name not in important_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            if cx < w / 3:
                pos_word = "on your left"
            elif cx > 2 * w / 3:
                pos_word = "on your right"
            else:
                pos_word = "ahead of you"

            box_height = y2 - y1
            if box_height > 0.6 * h:
                dist_label = "very close"
            elif box_height > 0.3 * h:
                dist_label = "near"
            else:
                dist_label = "far"

            if dist_label not in ["near", "very close"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue

            spoken_name = "bike" if name == "motorbike" else name

            if spoken_name == "person" and dist_label == "very close":
                msg = f"Danger! Someone is extremely close {pos_word}. Move away immediately."
            elif spoken_name in ["car", "bus", "truck", "bike"] and dist_label == "very close":
                msg = f"Danger! {spoken_name.capitalize()} is very close {pos_word}. Be very careful."
            elif spoken_name == "person" and dist_label == "near":
                msg = f"Warning, a person is near {pos_word}."
            elif spoken_name in ["car", "bus", "truck", "bike"] and dist_label == "near":
                msg = f"Caution, a {spoken_name} is near {pos_word}."
            else:
                msg = f"{spoken_name} {dist_label} {pos_word}."

            messages.append(msg)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Smart Navigation - Press 'q' to quit", frame)

        current_time = time.time()
        if messages:
            unique_msgs = []
            for m in messages:
                if m not in unique_msgs:
                    unique_msgs.append(m)
                if len(unique_msgs) == 2:
                    break

            msg = " ".join(unique_msgs)

            if (current_time - last_spoken_time > speak_gap) and msg != last_spoken_msg:
                print("[VOICE]:", msg)
                engine.say(msg)
                engine.runAndWait()
                last_spoken_time = current_time
                last_spoken_msg = msg

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2
import time
import pyttsx3 
import threading


# --- LOAD YOLO MODEL ---
model = YOLO("C:/Users/rakaj/Documents/Kuliah/Semeter 3/DeepLearning Lec/Project/AOL/model_ppe.pt")




# --- TIMER UNTUK NOTIFIKASI ---
last_warning_time = {
    "NO-Hardhat": 0,
    "NO-Mask": 0,
    "NO-Safety Vest": 0
}
WARNING_INTERVAL = 20

# --- VIDEO CAPTURE ---
cap = cv2.VideoCapture(0)


def speak_warning(text):
    engine = pyttsx3.init()
    engine.setProperty("rate",150)
    engine.say(text)
    engine.runAndWait()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize untuk performa
    small_frame = cv2.resize(frame, (640, 480))

    results = model(small_frame, verbose=False)
    missing_items = set()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0) if "NO" not in label else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            if "NO" in label:
                missing_items.add(label)

    # --- WARNING MESSAGE (IF ELSE) ---
    current_time = time.time()
    active_warnings = []

    for item in missing_items:
        # Check if the interval has passed for this specific item
        if current_time - last_warning_time.get(item, 0) > WARNING_INTERVAL:
            if item == "NO-Hardhat": # Ensure this matches your model's exact class name
                active_warnings.append("Risk of head injury")
            elif item == "NO-Mask":
                active_warnings.append("Risk of respiratory injury")
            elif item == "NO-Safety Vest":
                active_warnings.append("Risk of low visibility")
            
            # Update the last warning time for this specific item
            last_warning_time[item] = current_time

    # If there are any new warnings to speak
    if active_warnings:
        # Join warnings: "Risk of head injury and Risk of respiratory injury"
        full_warning_text = "WARNING: " + " and ".join(active_warnings)
        
        print(full_warning_text)
        
        # Speak the combined text
        threading.Thread(target=speak_warning, args=(full_warning_text,), daemon=True).start()

        # Optional: Display the combined text on the frame
        cv2.putText(frame, full_warning_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    cv2.imshow("PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEAN EXIT ---
cap.release()
cv2.destroyAllWindows()

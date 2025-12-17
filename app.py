from flask import Flask, Response, render_template
import cv2, time, threading, os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

app = Flask(__name__)

# LOAD MODEL
model = YOLO("model_ppe.pt")

camera_on = False
cap = None

last_warning_time = {
    "NO-Hardhat": 0,
    "NO-Mask": 0,
    "NO-Safety Vest": 0
}
WARNING_INTERVAL = 20

def speak_warning(text):
    filename = "warning.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def gen_frames():
    global cap, camera_on

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (640, 480))
        results = model(small, verbose=False)

        people_count = 0
        missing_items = set()
        current_time = time.time()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label.lower() == "person":
                    people_count += 1

                color = (0,255,0) if "NO" not in label else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

                if "NO" in label:
                    missing_items.add(label)

        warnings = []
        apparel = []

        for item in missing_items:
            if current_time - last_warning_time[item] > WARNING_INTERVAL:
                if item == "NO-Hardhat":
                    warnings.append("risk of head injury")
                    apparel.append("hardhat")
                elif item == "NO-Mask":
                    warnings.append("risk of respiratory injury")
                    apparel.append("mask")
                elif item == "NO-Safety Vest":
                    warnings.append("risk of low visibility")
                    apparel.append("safety vest")
                last_warning_time[item] = current_time

        if warnings:
            text = "Warning. " + " and ".join(warnings) + ". Please wear " + " and ".join(apparel)
            threading.Thread(target=speak_warning, args=(text,), daemon=True).start()
            cv2.putText(frame,text,(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.putText(frame,f"People Count: {people_count}",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    global camera_on, cap
    camera_on = True
    cap = cv2.VideoCapture(1)
    return "started"

@app.route("/stop")
def stop():
    global camera_on, cap
    camera_on = False
    if cap:
        cap.release()
    return "stopped"

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)

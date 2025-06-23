import cv2
from deepface import DeepFace

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw camera-style tracking box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Analyze only the face region
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if result:
                dominant_emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][dominant_emotion]

                if confidence > 50:
                    text = f"{dominant_emotion.upper()} ({int(confidence)}%)"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        except Exception as e:
            print("Error analyzing face:", e)

    # Show the frame
    cv2.imshow("Emotion Detector (Face Tracking)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
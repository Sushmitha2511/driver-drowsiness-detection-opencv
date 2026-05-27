import cv2
import dlib
from imutils import face_utils
from utils.ear import eye_aspect_ratio
from utils.mar import mouth_aspect_ratio

# Real-time driver monitoring system
# Applies behavioral anomaly detection (similar to cybersecurity IDS systems)

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.7

# Initialize counters
blink_counter = 0
alarm_on = False

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get facial landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # Mouth landmarks
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (255, 0, 0), 1)

        # Drowsiness detection (eyes)
        if ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_counter = 0

        # Yawning detection
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display EAR & MAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

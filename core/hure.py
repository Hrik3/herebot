import cv2
from fer.fer import FER

# Initialize the emotion detector
detector = FER()

# Start webcam
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and emotions in the frame
    result = detector.detect_emotions(frame)

    if result:
        # Take the first detected face (simplest case)
        face = result[0]
        x, y, w, h = face['box']
        emotions = face['emotions']

        # Find the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)

        # Map emotion to state
        if dominant_emotion == 'happy':
            state = 'positive'
        elif dominant_emotion == 'neutral':
            state = 'neutral'
        else:   # sad, angry, disgust, fear, surprise
            state = 'negative'

        # Draw bounding box and state on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow('Face State Detection', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
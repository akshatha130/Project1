import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hand_tracker = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Video file path
video_path = r'C:\Aksatha_College\python\Python Program\Io\video2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize the canvas for trails
canvas = None

# Neon colors for effects
neon_colors = [(255, 20, 147), (0, 255, 255), (0, 255, 127), (0, 191, 255), (255, 165, 0)]
color_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)

    # Overlay for trails and effects
    overlay = np.zeros_like(frame)

    # Check if any pose landmarks are detected
    if pose_results.pose_landmarks:
        # Process landmarks for a single person
        landmarks = pose_results.pose_landmarks.landmark

        # Draw landmarks (e.g., legs) based on pose landmarks
        for landmark in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]:
            point = landmarks[landmark]
            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])
            color = neon_colors[color_index % len(neon_colors)]
            cv2.circle(overlay, (x, y), 12, color, -1)
            color_index += 1

        # Define the bounding box around the detected person
        x_min = int(min([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x]) * frame.shape[1]) - 50
        x_max = int(max([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x]) * frame.shape[1]) + 50
        y_min = int(min([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]) * frame.shape[0]) - 50
        y_max = int(max([landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]) * frame.shape[0]) + 50

        # Extract the region of interest (ROI) for hand tracking
        person_roi = frame_rgb[y_min:y_max, x_min:x_max]

        # Run hand detection on the ROI
        hand_results = hand_tracker.process(person_roi)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for fingertip in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                    x = int(hand_landmarks.landmark[fingertip].x * person_roi.shape[1]) + x_min
                    y = int(hand_landmarks.landmark[fingertip].y * person_roi.shape[0]) + y_min
                    color = neon_colors[color_index % len(neon_colors)]
                    cv2.circle(overlay, (x, y), 10, color, -1)
                    color_index += 1

    # Combine overlay with canvas for trail effect
    canvas = cv2.addWeighted(canvas, 0.85, overlay, 0.6, 0)
    blurred_canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
    combined = cv2.addWeighted(frame, 0.6, blurred_canvas, 0.4, 0)

    # Display the output
    cv2.imshow("Neon Glow Effect", combined)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
hand_tracker.close()

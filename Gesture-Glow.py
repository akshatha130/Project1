import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Pose, Hands, and FaceMesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hand_tracker = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

# Video file path
video_path = r'C:\Aksatha_College\python\Python Program\Io\video4.mp4'
cap = cv2.VideoCapture(video_path)

canvas = None
color_palette = [(255, 182, 193), (255, 105, 180), (0, 255, 255), (255, 20, 147), (240, 128, 128)]
color_index = 0

# Define Face Outline Landmarks
FACE_OUTLINE_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152
]

# Timer for Artistic Effects
artistic_effect_time = 0.3  # seconds for time interval effect duration

# Artistic Effect Variables
sparkle_start_time = time.time()
sparkle_duration = 0.2  # seconds
sparkle_size = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    overlay = np.zeros_like(frame)

    def random_color():
        return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # Define bounding box for person ROI
        x_min = int(min([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x]) * frame.shape[1]) - 50
        x_max = int(max([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x]) * frame.shape[1]) + 50
        y_min = int(min([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]) * frame.shape[0]) - 50
        y_max = int(max([landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]) * frame.shape[0]) + 50

        # Artistic Effect: Fluid Curved Lines on Legs
        for landmark in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]:
            point = landmarks[landmark]
            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])
            color = random.choice(color_palette)
            curve_radius = random.randint(30, 70)

            # Create a smooth curve that follows the landmark position
            curve_points = [(x + random.randint(-10, 10), y + random.randint(-10, 10)) for _ in range(5)]
            for i in range(1, len(curve_points)):
                cv2.line(overlay, curve_points[i-1], curve_points[i], color, 3)

        # Extract region of interest for hand tracking
        person_roi = frame_rgb[y_min:y_max, x_min:x_max]

        # Run hand detection on the ROI
        hand_results = hand_tracker.process(person_roi)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for fingertip in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                    x = int(hand_landmarks.landmark[fingertip].x * person_roi.shape[1]) + x_min
                    y = int(hand_landmarks.landmark[fingertip].y * person_roi.shape[0]) + y_min
                    color = random.choice(color_palette)
                    cv2.circle(overlay, (x, y), 10, color, -1)

        # Draw glowing geometric patterns
        if random.random() < 0.05:  # Add patterns at random intervals
            shape_center = (random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0]))
            size = random.randint(20, 100)
            color = random.choice(color_palette)
            thickness = random.randint(2, 5)

            # Create circles or polygons
            if random.random() < 0.5:
                cv2.circle(overlay, shape_center, size, color, thickness)
            else:
                vertices = []
                for i in range(6):  # Hexagon shape
                    angle = np.deg2rad(i * 60)
                    x = int(shape_center[0] + size * np.cos(angle))
                    y = int(shape_center[1] + size * np.sin(angle))
                    vertices.append((x, y))
                cv2.polylines(overlay, [np.array(vertices)], isClosed=True, color=color, thickness=thickness)

        # Fun: Sparkles on screen
        if time.time() - sparkle_start_time > sparkle_duration:
            sparkle_start_time = time.time()
            sparkle_x = random.randint(0, frame.shape[1])
            sparkle_y = random.randint(0, frame.shape[0])
            sparkle_color = random.choice(color_palette)
            sparkle_radius = random.randint(3, 8)
            cv2.circle(overlay, (sparkle_x, sparkle_y), sparkle_radius, sparkle_color, -1)

    # Combine overlay with canvas for trail effect
    canvas = cv2.addWeighted(canvas, 0.85, overlay, 0.6, 0)
    blurred_canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
    combined = cv2.addWeighted(frame, 0.6, blurred_canvas, 0.4, 0)

    # Display the output with artistic effects
    cv2.imshow("Artistic Effects", combined)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
hand_tracker.close()
face_mesh.close()

import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_path = "AlexThrow1.mp4"
cap = cv2.VideoCapture(video_path)


def angle_3pt(point_a, point_b, point_c):
    """
    Computes the angle ABC (in degrees).
    A, B, C are (x, y) tuples in pixel space.
    """
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    BA = (ax - bx, ay - by)
    BC = (cx - bx, cy - by)

    dot = BA[0]*BC[0] + BA[1]*BC[1]
    mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)
    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    if mag_BA * mag_BC == 0:
        return None

    cos_angle = dot / (mag_BA * mag_BC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    # ==============================
    # HISTORICAL STATE VARIABLES
    # ==============================

    prev_wrist_3d = None          # previous RIGHT_WRIST 3D position (x,y,z)
    prev_timestamp = None         # previous frame timestamp

    # Shoulder calibration (meters per mediapipe-3d-unit)
    shoulder_scale_3d = None
    REAL_SHOULDER_WIDTH_M = 0.44  # your real shoulder width in meters

    # Event detection
    reachback_time = None
    power_pocket_time = None
    hit_time = None

    max_reachback_distance = -9999
    peak_arm_speed_mps = -9999

    # Bracing detection
    previous_knee_x = None
    knee_stop_time = None
    bracing_good = None

    # 3D smoothing buffer for wrist
    wrist_smoothing_buffer = []
    SMOOTHING_WINDOW = 7


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Time in seconds
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_height, frame_width, _ = frame.shape

            # ==============================
            # EXTRACT KEYPOINTS (3D)
            # ==============================

            left_shoulder  = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            right_elbow    = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist    = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            right_knee     = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            # ==============================
            # SMOOTH THE 3D WRIST POSITION
            # ==============================

            wrist_smoothing_buffer.append((right_wrist.x, right_wrist.y, right_wrist.z))

            if len(wrist_smoothing_buffer) > SMOOTHING_WINDOW:
                wrist_smoothing_buffer.pop(0)

            smoothed_wrist_x = sum(p[0] for p in wrist_smoothing_buffer) / len(wrist_smoothing_buffer)
            smoothed_wrist_y = sum(p[1] for p in wrist_smoothing_buffer) / len(wrist_smoothing_buffer)
            smoothed_wrist_z = sum(p[2] for p in wrist_smoothing_buffer) / len(wrist_smoothing_buffer)

            smoothed_wrist_3d = (smoothed_wrist_x, smoothed_wrist_y, smoothed_wrist_z)

            # ==============================
            # PIXEL COORDS FOR ANGLES
            # ==============================

            shoulder_px     = (right_shoulder.x * frame_width, right_shoulder.y * frame_height)
            elbow_px        = (right_elbow.x    * frame_width, right_elbow.y    * frame_height)
            wrist_px        = (right_wrist.x    * frame_width, right_wrist.y    * frame_height)
            knee_px         = (right_knee.x     * frame_width, right_knee.y     * frame_height)

            # ==============================
            # 3D SHOULDER WIDTH CALIBRATION
            # ==============================

            if shoulder_scale_3d is None:
                dx = left_shoulder.x - right_shoulder.x
                dy = left_shoulder.y - right_shoulder.y
                dz = left_shoulder.z - right_shoulder.z

                shoulder_distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

                if shoulder_distance_3d > 0:
                    shoulder_scale_3d = REAL_SHOULDER_WIDTH_M / shoulder_distance_3d
                    print(f"[Calibration] scale = {shoulder_scale_3d:.6f} m/unit")

            # ==============================
            # COMPUTE 3D WRIST VELOCITY
            # ==============================

            wrist_velocity_3d = None

            if prev_wrist_3d is not None and prev_timestamp is not None:
                dt = timestamp - prev_timestamp
                if dt > 0:
                    dx = smoothed_wrist_x - prev_wrist_3d[0]
                    dy = smoothed_wrist_y - prev_wrist_3d[1]
                    dz = smoothed_wrist_z - prev_wrist_3d[2]

                    raw_velocity = math.sqrt(dx*dx + dy*dy + dz*dz) / dt

                    # Spike suppression
                    if raw_velocity > 0.12:
                        raw_velocity *= 0.5

                    if shoulder_scale_3d is not None:
                        wrist_velocity_3d = raw_velocity * shoulder_scale_3d

                        if wrist_velocity_3d > peak_arm_speed_mps:
                            peak_arm_speed_mps = wrist_velocity_3d
                            hit_time = timestamp

            # ==============================
            # REACHBACK DETECTION
            # (Wrist farthest behind shoulder horizontally)
            # ==============================

            wrist_behind_distance = shoulder_px[0] - wrist_px[0]

            if wrist_behind_distance > max_reachback_distance:
                max_reachback_distance = wrist_behind_distance
                reachback_time = timestamp

            # ==============================
            # POWER POCKET DETECTION
            # ==============================

            elbow_angle_deg = angle_3pt(shoulder_px, elbow_px, wrist_px)

            if power_pocket_time is None:
                if elbow_angle_deg is not None and elbow_angle_deg < 100 and wrist_px[0] > elbow_px[0]:
                    power_pocket_time = timestamp

            # ==============================
            # BRACING DETECTION
            # (Knee stops before hit)
            # ==============================

            if previous_knee_x is not None:
                if abs(knee_px[0] - previous_knee_x) < 0.5 and knee_stop_time is None:
                    knee_stop_time = timestamp

            if knee_stop_time and hit_time and bracing_good is None:
                bracing_good = knee_stop_time < hit_time

            previous_knee_x = knee_px[0]

            # Save previous frame values
            prev_wrist_3d = smoothed_wrist_3d
            prev_timestamp = timestamp

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Disc Golf Analyzer (3D Smoothed + Clean Names)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ==============================
    # OUTPUT RESULTS
    # ==============================

    print("\n==========================")
    print("     THROW BREAKDOWN")
    print("==========================")
    print(f"Reach-Back Time       : {reachback_time:.3f} sec")
    print(f"Power Pocket Time     : {power_pocket_time:.3f} sec")
    print(f"Hit Time              : {hit_time:.3f} sec")
    print(f"Peak Arm Speed        : {peak_arm_speed_mps*2.23694:.2f} mph")

    if bracing_good is None:
        print("Bracing               : Undetermined")
    else:
        print("Bracing               :", "GOOD" if bracing_good else "LATE")

    cap.release()
    cv2.destroyAllWindows()

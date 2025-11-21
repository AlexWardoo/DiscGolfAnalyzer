import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_path = "AlexThrow1.mp4"
cap = cv2.VideoCapture(video_path)


def angle_3pt(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    BA = (ax - bx, ay - by)
    BC = (cx - bx, ay - by)

    dot = BA[0]*BC[0] + BA[1]*BC[1]
    magBA = math.sqrt(BA[0]**2 + BA[1]**2)
    magBC = math.sqrt(BC[0]**2 + BC[1]**2)

    if magBA * magBC == 0:
        return None

    cos_angle = dot / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


with mp_pose.Pose(static_image_mode=False,
                  model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    prev_rw = None      # previous RIGHT_WRIST 3D coords (x,y,z)
    prev_time = None

    # Calibration using REAL SHOULDER WIDTH (meters)
    SCALE_3D = None
    REAL_SHOULDER_WIDTH = 0.44   # meters (17.3 inches)

    # State Machine values
    reachback_time = None
    power_pocket_time = None
    hit_time = None

    max_reachback_dist = -9999
    max_forward_vel_mps = -9999

    prev_knee_x = None
    knee_stop_time = None
    brace_good = None

    wrist_buffer = []
    BUFFER_SIZE = 7


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # timestamp in seconds
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape

            # MAIN KEYPOINTS (3D)
            ls = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            re = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            rk = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            # add current 3D wrist to buffer
            wrist_buffer.append((rw.x, rw.y, rw.z))
            if len(wrist_buffer) > BUFFER_SIZE:
                wrist_buffer.pop(0)

            # smoothed wrist position
            avg_x = sum(p[0] for p in wrist_buffer) / len(wrist_buffer)
            avg_y = sum(p[1] for p in wrist_buffer) / len(wrist_buffer)
            avg_z = sum(p[2] for p in wrist_buffer) / len(wrist_buffer)


            # pixel coords (for drawing & angles)
            s = (rs.x * w, rs.y * h)
            e = (re.x * w, re.y * h)
            r = (rw.x * w, rw.y * h)
            knee = (rk.x * w, rk.y * h)

            # -------------------------
            # 1) ONE-TIME 3D SHOULDER CALIBRATION
            # -------------------------
            if SCALE_3D is None:
                dx = ls.x - rs.x
                dy = ls.y - rs.y
                dz = ls.z - rs.z     # MediaPipe z is proportional to x units

                shoulder_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

                if shoulder_3d > 0:
                    SCALE_3D = REAL_SHOULDER_WIDTH / shoulder_3d
                    print(f"[3D Calibration] shoulder_3d={shoulder_3d:.5f}, SCALE_3D={SCALE_3D:.5f} meters/unit")

            # -------------------------
            # 2) 3D WRIST VELOCITY
            # -------------------------
            wrist_velocity_3d = None
            if prev_rw is not None:
                dt = t - prev_time
                if dt > 0:
                    dx3 = rw.x - prev_rw.x
                    dy3 = rw.y - prev_rw.y
                    dz3 = rw.z - prev_rw.z

                    wrist_velocity_3d = math.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3) / dt

            # Convert to real-world meters/sec and mph
            if wrist_velocity_3d is not None and SCALE_3D is not None:
                mps = wrist_velocity_3d * SCALE_3D
                mph = mps * 2.23694

                if mps > max_forward_vel_mps:
                    max_forward_vel_mps = mps
                    hit_time = t

            # -------------------------
            # 3) Reachback Detection
            # -------------------------
            backward_dist = s[0] - r[0]
            if backward_dist > max_reachback_dist:
                max_reachback_dist = backward_dist
                reachback_time = t

            # -------------------------
            # 4) Power Pocket Detection
            # -------------------------
            elbow_angle = angle_3pt(s, e, r)
            if power_pocket_time is None:
                if elbow_angle is not None and elbow_angle < 100 and r[0] > e[0]:
                    power_pocket_time = t

            # -------------------------
            # 5) Bracing Detection
            # -------------------------
            if prev_knee_x is not None:
                if abs(knee[0] - prev_knee_x) < 0.5 and knee_stop_time is None:
                    knee_stop_time = t

            if knee_stop_time is not None and hit_time is not None and brace_good is None:
                brace_good = knee_stop_time < hit_time

            prev_knee_x = knee[0]

            prev_rw = rw
            prev_time = t

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Disc Golf Analyzer (3D Corrected)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------------------------
    # FINAL SUMMARY
    # -------------------------
    print("\n=========================")
    print("     THROW BREAKDOWN")
    print("=========================")
    print(f"Reach-Back Time       : {reachback_time:.3f} sec")
    print(f"Power Pocket Time     : {power_pocket_time:.3f} sec")
    print(f"Hit Time (max speed)  : {hit_time:.3f} sec")
    print(f"Peak Arm Speed        : {max_forward_vel_mps*2.23694:.2f} mph")

    if brace_good is None:
        print("Bracing               : Undetermined")
    else:
        print("Bracing               :", "GOOD" if brace_good else "LATE")

    cap.release()
    cv2.destroyAllWindows()

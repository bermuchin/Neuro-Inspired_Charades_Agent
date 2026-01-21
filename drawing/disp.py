import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# === 1. 모델 파일 경로 ===
POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

# === 2. 인덱스 및 연결선 정의 ===
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]

# 입술 바깥쪽과 안쪽 정의
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# === 3. 전역 결과 변수 및 콜백 함수 ===
pose_result = None
hand_result = None
face_result = None

def pose_callback(result, output_image, timestamp_ms):
    global pose_result
    pose_result = result

def hand_callback(result, output_image, timestamp_ms):
    global hand_result
    hand_result = result

def face_callback(result, output_image, timestamp_ms):
    global face_result
    face_result = result

# === 4. 그리기 유틸 함수 ===
def draw_contour(canvas, landmarks, indices, color, thickness, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, thickness)

# === 5. 옵션 설정 ===
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=pose_callback)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=6, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=hand_callback)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1, min_face_detection_confidence=0.1, min_tracking_confidence=0.1,
    result_callback=face_callback)

# === 6. 메인 실행 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
     vision.FaceLandmarker.create_from_options(face_options) as face_lm:

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int(time.time() * 1000)

        pose_lm.detect_async(mp_image, ts)
        hand_lm.detect_async(mp_image, ts)
        face_lm.detect_async(mp_image, ts)

        black_board = np.zeros((h, w, 3), dtype=np.uint8)

        # [얼굴 그리기]
        if face_result and face_result.face_landmarks:
            for landmarks in face_result.face_landmarks:
                draw_contour(black_board, landmarks, FACE_OVAL, (100, 100, 100), 1, w, h)
                draw_contour(black_board, landmarks, LEFT_EYE, (0, 255, 255), 2, w, h)
                draw_contour(black_board, landmarks, RIGHT_EYE, (0, 255, 255), 2, w, h)
                
                # 입술: 바깥쪽과 안쪽 모두 그리기
                draw_contour(black_board, landmarks, LIPS_OUTER, (0, 0, 255), 2, w, h)
                draw_contour(black_board, landmarks, LIPS_INNER, (0, 0, 255), 1, w, h) # 안쪽은 조금 더 얇게
                
                draw_contour(black_board, landmarks, LEFT_EYEBROW, (255, 255, 255), 2, w, h)
                draw_contour(black_board, landmarks, RIGHT_EYEBROW, (255, 255, 255), 2, w, h)

        # [포즈 그리기] (얼굴 부위 제외 필터링)
        if pose_result and pose_result.pose_landmarks:
            for landmarks in pose_result.pose_landmarks:
                for conn in POSE_CONNECTIONS:
                    if conn[0] >= 11 and conn[1] >= 11:
                        p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                        cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 200, 0), 2)
                for i, lm in enumerate(landmarks):
                    if i >= 11:
                        cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)

        # [손 그리기]
        if hand_result and hand_result.hand_landmarks:
            for i, landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[i][0].category_name
                color = (100, 100, 255) if handedness == "Right" else (255, 100, 100)
                for conn in HAND_CONNECTIONS:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color, 2)
                for lm in landmarks:
                    cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 3, color, -1)

        cv2.imshow('Retina Skeleton View', black_board)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
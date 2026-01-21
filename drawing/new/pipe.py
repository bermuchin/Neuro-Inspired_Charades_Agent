import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# === 1. 모델 파일 자동 다운로드 설정 ===
def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

download_model("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", POSE_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", FACE_MODEL)

# === 2. 시각화 연결선 정의 (Face Mesh용) ===
mp_face_mesh = mp.solutions.face_mesh
FACEMESH_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
FACEMESH_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS

# 강조 부위 인덱스
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

# === 3. 전역 결과 변수 및 비동기 콜백 함수 ===
latest_pose = None
latest_hand = None
latest_face = None

def pose_callback(result, output_image, timestamp_ms):
    global latest_pose
    latest_pose = result

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand
    latest_hand = result

def face_callback(result, output_image, timestamp_ms):
    global latest_face
    latest_face = result

# === 4. 그리기 유틸리티 함수 ===
def draw_line_by_indices(canvas, landmarks, indices, color, thickness, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, thickness)

# === 5. Landmarker 옵션 설정 (LIVE_STREAM 모드) ===
base_options = lambda path: python.BaseOptions(model_asset_path=path)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options(POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1, min_pose_detection_confidence=0.5, result_callback=pose_callback)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options(HAND_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=6, min_hand_detection_confidence=0.5, result_callback=hand_callback)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options(FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1, min_face_detection_confidence=0.5, result_callback=face_callback)

# === 6. 메인 실행 루프 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
     vision.FaceLandmarker.create_from_options(face_options) as face_lm:

    print("MediaPipe 0.10.31 Running... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)

        # 비동기 감지 요청
        pose_lm.detect_async(mp_image, timestamp_ms)
        hand_lm.detect_async(mp_image, timestamp_ms)
        face_lm.detect_async(mp_image, timestamp_ms)

        # 결과 출력을 위한 검은 배경
        black_board = np.zeros((h, w, 3), dtype=np.uint8)

        # [1. FACE MESH 시각화]
        if latest_face and latest_face.face_landmarks:
            for landmarks in latest_face.face_landmarks:
                # 1-1. 전체 그물망 (Surface 느낌 - 아주 얇은 회색)
                for conn in FACEMESH_TESSELATION:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (50, 50, 50), 1)
                
                # 1-2. 주요 부위 강조 (눈, 입, 눈썹)
                draw_line_by_indices(black_board, landmarks, LEFT_EYE, (0, 255, 255), 2, w, h)
                draw_line_by_indices(black_board, landmarks, RIGHT_EYE, (0, 255, 255), 2, w, h)
                draw_line_by_indices(black_board, landmarks, LIPS_OUTER, (0, 0, 255), 2, w, h)
                draw_line_by_indices(black_board, landmarks, LIPS_INNER, (0, 0, 255), 1, w, h)
                draw_line_by_indices(black_board, landmarks, LEFT_EYEBROW, (255, 255, 255), 2, w, h)
                draw_line_by_indices(black_board, landmarks, RIGHT_EYEBROW, (255, 255, 255), 2, w, h)

        # [2. POSE SKELETON 시각화]
        if latest_pose and latest_pose.pose_landmarks:
            for landmarks in latest_pose.pose_landmarks:
                # 얼굴 부위(0-10) 제외하고 그리기
                for conn in mp.solutions.pose.POSE_CONNECTIONS:
                    if conn[0] >= 11 and conn[1] >= 11:
                        p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                        if p1.presence > 0.5 and p2.presence > 0.5:
                            cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 0), 2)
                for i, lm in enumerate(landmarks):
                    if i >= 11 and lm.presence > 0.5:
                        cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)

        # [3. HANDS 시각화]
        if latest_hand and latest_hand.hand_landmarks:
            for i, landmarks in enumerate(latest_hand.hand_landmarks):
                # 오른손/왼손 판별 및 색상 지정
                label = latest_hand.handedness[i][0].category_name
                color = (255, 100, 100) if label == "Left" else (100, 100, 255) # 왼손 빨강, 오른손 파랑
                
                for conn in mp.solutions.hands.HAND_CONNECTIONS:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color, 2)
                for lm in landmarks:
                    cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 3, color, -1)

        # 화면 출력
        cv2.imshow('Multi-Landmarker LiveStream (Tasks API)', black_board)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import easyocr
from PIL import ImageFont, ImageDraw, Image
import math



reader = easyocr.Reader(['ko'], gpu=False) 


MOUTH_OPEN_THRESHOLD = 15 
# ==========================================

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

draw_color = (255, 255, 255)
brush_thickness = 12
xp, yp = 0, 0
img_canvas = None

# 한글 렌더링 함수
def put_korean_text(img, text, pos, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("AppleGothic", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

ocr_result = "..."

while True:
    success, img = cap.read()
    if not success:
        break

    # 1. 전처리
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    if img_canvas is None:
        img_canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # === [수정된 부분] 변수 초기화 (얼굴 인식 실패 대비) ===
    is_mouth_open = False
    status_color = (0, 0, 255) # 기본값: 빨간색
    # ====================================================

    # 2. 얼굴 인식 (입 벌림 감지)
    face_result = face_mesh.process(img_rgb)
    
    if face_result.multi_face_landmarks:
        for face_lms in face_result.multi_face_landmarks:
            upper_lip = face_lms.landmark[13]
            lower_lip = face_lms.landmark[14]
            
            ux, uy = int(upper_lip.x * w), int(upper_lip.y * h)
            lx, ly = int(lower_lip.x * w), int(lower_lip.y * h)
            
            lip_distance = math.hypot(lx - ux, ly - uy)
            
            cv2.circle(img, (ux, uy), 2, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (lx, ly), 2, (0, 255, 255), cv2.FILLED)
            
            if lip_distance > MOUTH_OPEN_THRESHOLD:
                is_mouth_open = True
                status_color = (0, 255, 0) # 초록색
            else:
                is_mouth_open = False
                status_color = (0, 0, 255) # 빨간색

    # 3. 손 인식 (그리기 좌표)
    hand_result = hands.process(img_rgb)
    
    if hand_result.multi_hand_landmarks:
        for hand_lms in hand_result.multi_hand_landmarks:
            lm = hand_lms.landmark[8]
            x1, y1 = int(lm.x * w), int(lm.y * h)

            # 입 벌림 여부에 따라 동작 결정
            if is_mouth_open:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)

    # 4. 이미지 합성
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, img, mask=img_inv)
    img = cv2.bitwise_or(img, img_canvas)


    img = put_korean_text(img, f"result: {ocr_result}", (10, 10), 20, (0, 255, 0))

    cv2.imshow("Mouth Control Canvas", img)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        img_canvas = np.zeros((h, w, 3), np.uint8)
        ocr_result = "초기화됨"
    elif key == 32: # Spacebar
        print("인식 중...")
        ocr_img = cv2.bitwise_not(img_canvas)
        try:
            result_list = reader.readtext(ocr_img, detail=0)
            text = " ".join(result_list)
            ocr_result = text if text.strip() else "인식 실패"
            print(f"인식된 텍스트: {text}")
        except Exception as e:
            ocr_result = f"e"

cap.release()
cv2.destroyAllWindows()
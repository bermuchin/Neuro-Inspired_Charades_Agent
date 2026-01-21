import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

draw_color = (255, 255, 255) 
brush_thickness = 12
xp, yp = 0, 0
img_canvas = None

MOUTH_OPEN_THRESHOLD = 15 

# minimum window
status_window_name = "Controller (Press Space to View, C to Clear, Q to Quit)"
cv2.namedWindow(status_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(status_window_name, 400, 50)
status_img = np.zeros((50, 400, 3), np.uint8)

print("running... Press 'q' to quit, 'c' to clear canvas, SPACE to view drawing.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    if img_canvas is None:
        img_canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    is_mouth_open = False


    face_result = face_mesh.process(img_rgb)
    if face_result.multi_face_landmarks:
        for face_lms in face_result.multi_face_landmarks:
            upper_lip = face_lms.landmark[13]
            lower_lip = face_lms.landmark[14]
            ux, uy = int(upper_lip.x * w), int(upper_lip.y * h)
            lx, ly = int(lower_lip.x * w), int(lower_lip.y * h)
            lip_distance = math.hypot(lx - ux, ly - uy)
            
            if lip_distance > MOUTH_OPEN_THRESHOLD:
                is_mouth_open = True


    hand_result = hands.process(img_rgb)
    if hand_result.multi_hand_landmarks:
        for hand_lms in hand_result.multi_hand_landmarks:
            lm = hand_lms.landmark[8]
            x1, y1 = int(lm.x * w), int(lm.y * h)

            if is_mouth_open:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    # 3. 화면에 아무것도 띄우지 않음 (최소한의 상태창만 노출)
    cv2.imshow(status_window_name, status_img)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        img_canvas = np.zeros((h, w, 3), np.uint8)
        print("initialized.")
    elif key == 32: # Spacebar
        # 결과 이미지(캔버스)만 새 창에 띄우기
        plt_img = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)
        #plt.figure(figsize=(10, 6))
        manager = plt.get_current_fig_manager()

    # Toggle full screen mode
        manager.full_screen_toggle()
        plt.imshow(plt_img)
        plt.title("Current Canvas Drawing")
        plt.axis('off')
        print("displaying...")
        plt.show()

cap.release()
cv2.destroyAllWindows()
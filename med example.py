import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3 , 640)
cap.set(4 , 480)

canvas = np.zeros((480 , 640 , 3), dtype = np.uint8)
color = (0 , 0 , 255)
prev_point = None


def fingers_up(hand_landmarks):
    tips_ids = [4 , 8 , 12 , 16 , 20]
    fingers = []

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success , frame = cap.read()
    frame = cv2.flip(frame , 1)
    img_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h , w , _ = frame.shape
            lm = handLms.landmark
            Cx = int(lm[8].x *w)
            Cy = int(lm[8].y *h)

            fingers = fingers_up(handLms)

            if fingers[1]==1 and sum(fingers[2:])==0 :
                cv2.circle(frame , (Cx , Cy) , 10 , color , -1)

                if prev_point is not None:
                    cv2.line(canvas , prev_point , (Cx , Cy) , color , 5)

                prev_point = (Cx , Cy)
            else:
                prev_point = None

            mp_draw.draw_landmarks(frame , handLms , mp_hands.HAND_CONNECTIONS)

    else:
        prev_point = None


    combo = cv2.add(frame , canvas)
    cv2.imshow('Drawing' , combo)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('r'):
        color = (0 , 0 , 255)
    elif key == ord('g'):
        color = (0 , 255 , 0)
    elif key == ord('b'):
        color = (255 , 0 , 0)

cap.release()
cv2.destroyAllWindows()
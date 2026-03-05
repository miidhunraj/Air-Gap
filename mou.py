import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
from pynput import keyboard

# --- PERFORMANCE OPTIMIZATIONS ---
pyautogui.PAUSE = 0               # Removes the default 0.1s delay after every move/click
pyautogui.FAILSAFE = True         # Move mouse to corner to abort if needed
PROCESS_EVERY_N_FRAME = 2         # Process every 2nd frame to save CPU
FRAME_COUNT = 0

# --- CONFIGURATION ---
SMOOTHENING = 5  
PLOC_X, PLOC_Y = 0, 0
CLOC_X, CLOC_Y = 0, 0
FRAME_R = 100         

# --- STATE VARIABLES ---
clicked = False
right_clicked = False
dragging = False
last_click_time = 0
thumb_index_hold_start = 0

# --- DYNAMIC SENSITIVITY TOGGLE ---
def on_press(key):
    global SMOOTHENING
    try:
        if key == keyboard.Key.up:
            SMOOTHENING = max(1, SMOOTHENING - 1)
            print(f"Sensitivity Increased (Smoothening: {SMOOTHENING})")
        elif key == keyboard.Key.down:
            SMOOTHENING += 1
            print(f"Sensitivity Decreased (Smoothening: {SMOOTHENING})")
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
# Setting model_complexity=0 can further improve speed on lower-end PCs
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # 1. OPTIMIZATION: Frame Skipping
    FRAME_COUNT += 1
    if FRAME_COUNT % PROCESS_EVERY_N_FRAME != 0:
        continue

    # 2. OPTIMIZATION: Resize image for faster processing
    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = hand_lms.landmark
            
            # GET LANDMARKS
            itip = landmarks[8]  # Index
            ttip = landmarks[4]  # Thumb
            mtip = landmarks[12] # Middle

            # COORDINATE MAPPING
            x_raw = np.interp(itip.x * w, (FRAME_R, w - FRAME_R), (0, screen_w))
            y_raw = np.interp(itip.y * h, (FRAME_R, h - FRAME_R), (0, screen_h))

            # SMOOTHING
            CLOC_X = PLOC_X + (x_raw - PLOC_X) / SMOOTHENING
            CLOC_Y = PLOC_Y + (y_raw - PLOC_Y) / SMOOTHENING

            # DISTANCE CALCULATIONS
            dist_thumb_index = math.hypot(itip.x - ttip.x, itip.y - ttip.y)
            dist_thumb_middle = math.hypot(mtip.x - ttip.x, mtip.y - ttip.y)
            dist_index_middle = math.hypot(itip.x - mtip.x, itip.y - mtip.y)

            # 1. SCROLL LOGIC
            if dist_index_middle < 0.03:
                scroll_amount = (PLOC_Y - CLOC_Y) * 2 
                pyautogui.scroll(int(scroll_amount))
            else:
                # 2. MOUSE MOVEMENT
                pyautogui.moveTo(CLOC_X, CLOC_Y, _pause=False)

            # 3. LEFT CLICK / DOUBLE CLICK / DRAG
            if dist_thumb_index < 0.05:
                if not clicked:
                    curr_time = time.time()
                    if curr_time - last_click_time < 0.3:
                        pyautogui.doubleClick()
                        clicked = True 
                    else:
                        thumb_index_hold_start = curr_time
                        clicked = True
                    last_click_time = curr_time
                
                if not dragging and (time.time() - thumb_index_hold_start > 1.0):
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                elif clicked:
                    if time.time() - thumb_index_hold_start < 1.0:
                        pyautogui.click()
                clicked = False

            # 4. RIGHT CLICK
            if dist_thumb_middle < 0.05:
                if not right_clicked:
                    pyautogui.rightClick()
                    right_clicked = True
            else:
                right_clicked = False

            PLOC_X, PLOC_Y = CLOC_X, CLOC_Y
            
            # Visuals
            cv2.rectangle(img, (FRAME_R, FRAME_R), (w - FRAME_R, h - FRAME_R), (255, 0, 255), 2)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, f"Smooth: {SMOOTHENING}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("AuraControl - Optimized", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
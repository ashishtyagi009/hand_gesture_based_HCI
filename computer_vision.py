import cv2
import mediapipe as mp
import time
import webbrowser
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import psutil
import pyautogui
import math
import numpy as np

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Setup system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

def get_volume():
    return int(volume.GetMasterVolumeLevelScalar() * 100)

def increase_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = min(current_volume + 0.1, 1.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Volume increased to: {int(new_volume * 100)}%")

def decrease_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(current_volume - 0.1, 0.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Volume decreased to: {int(new_volume * 100)}%")

def is_index_finger_up_and_others_down(hand_landmarks):
    finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    finger_dips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]]
    index_up = finger_tips[0].y < finger_dips[0].y
    others_down = all(tip.y > dip.y for tip, dip in zip(finger_tips[1:], finger_dips[1:]))
    return index_up and others_down

def is_middle_finger_up_and_others_down(hand_landmarks):
    finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    finger_dips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]]
    middle_up = finger_tips[1].y < finger_dips[1].y
    others_down = all(tip.y > dip.y for tip, dip in zip(finger_tips[0:1] + finger_tips[2:], finger_dips[0:1] + finger_dips[2:]))
    return middle_up and others_down

def is_pinky_finger_up_and_index_up(hand_landmarks):
    finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    finger_dips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]]
    index_up = finger_tips[0].y < finger_dips[0].y
    pinky_up = finger_tips[3].y < finger_dips[3].y
    others_down = all(tip.y > dip.y for tip, dip in zip(finger_tips[1:3], finger_dips[1:3]))
    return index_up and pinky_up and others_down

def is_pinky_finger_up_and_others_down(hand_landmarks):
    finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    finger_dips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]]
    pinky_up = finger_tips[3].y < finger_dips[3].y
    others_down = all(tip.y > dip.y for tip, dip in zip(finger_tips[0:3], finger_dips[0:3]))
    return pinky_up and others_down

def is_browser_running():
    browser_names = ["chrome"]
    for proc in psutil.process_iter(['name']):
        if any(browser in proc.info['name'].lower() for browser in browser_names):
            return True
    return False

def is_thumbs_up(hand_landmarks):
    """Detect if the hand shows a thumbs-up gesture."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Thumb should be above the hand, and other fingers should be curled
    thumb_up = thumb_tip.y < thumb_ip.y
    other_fingers_down = (
        index_tip.y > thumb_ip.y and
        middle_tip.y > thumb_ip.y and
        ring_tip.y > thumb_ip.y and
        pinky_tip.y > thumb_ip.y
    )
    return thumb_up and other_fingers_down

def is_mouse_gesture(hand_landmarks):
    """Detect if index and middle finger tips are touching, and others are down."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    finger_dips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]]
    middle_up = finger_tips[1].y < finger_dips[1].y
    index_up = finger_tips[0].y < finger_dips[0].y
    touching = abs(index_tip.x - middle_tip.x) < 0.05 and abs(index_tip.y - middle_tip.y) < 0.05
    others_down = all(tip.y > dip.y for tip, dip in zip(finger_tips[2:], finger_dips[2:]))
    return touching and others_down

def show_timer_on_screen(frame, message, timer_value):
    """Display a timer on the screen."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{message}: {timer_value:.1f}s", (50, 50),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def switch_tab():
    pyautogui.hotkey('ctrl', 'tab')

def press_key(key):
    pyautogui.press(key)

def close_window():
    pyautogui.hotkey('alt', 'f4')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Mouse control functions

def move_mouse(x, y):
    # Set the screen resolution to a 16:9 aspect ratio
    screen_width = 1920
    screen_height = 1080

    # Apply smoothing by gradually adjusting mouse movement
    cursor_x = x * screen_width
    cursor_y = y * screen_height

    # Smoothing factor (higher value means smoother movement)
    smoothing_factor = 1

    # Get the current position of the mouse
    current_x, current_y = pyautogui.position()

    # Calculate the difference
    delta_x = cursor_x - current_x
    delta_y = cursor_y - current_y

    # Apply smoothing by interpolating the movement
    smoothed_x = current_x + delta_x * smoothing_factor
    smoothed_y = current_y + delta_y * smoothing_factor

    # Move the mouse to the smoothed position
    pyautogui.moveTo(smoothed_x, smoothed_y)



def left_click():
    pyautogui.click()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    gesture_control_enabled = False
    gesture_start_time = None

    mouse_mode = False
    mouse_mode_start_time = None

    volume_control_on = False
    last_action_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label

                if not gesture_control_enabled and handedness == "Right" and is_thumbs_up(hand_landmarks):
                    if gesture_start_time is None:
                        gesture_start_time = time.time()
                    elapsed_time = time.time() - gesture_start_time
                    show_timer_on_screen(frame, "Enable Gesture Control", elapsed_time)

                    if elapsed_time >= 5:
                        gesture_control_enabled = True
                        print("Gesture control enabled!")
                        gesture_start_time = None
                        time.sleep(1)  # Avoid rapid toggling
                    continue
                elif not is_thumbs_up(hand_landmarks):
                    gesture_start_time = None

                if not mouse_mode and gesture_control_enabled and handedness == "Left" and is_thumbs_up(hand_landmarks):
                    if gesture_start_time is None:
                        gesture_start_time = time.time()
                    elapsed_time = time.time() - gesture_start_time
                    show_timer_on_screen(frame, "Disable Gesture Control", elapsed_time)

                    if elapsed_time >= 5:
                        gesture_control_enabled = False
                        print("Gesture control disabled!")
                        gesture_start_time = None
                        time.sleep(1)  # Avoid rapid toggling
                    continue


                if gesture_control_enabled and time.time() - last_action_time > 0.3:  # Further reduced delay to 0.3s
                    if not mouse_mode and is_index_finger_up_and_others_down(hand_landmarks):
                        volume_control_on = True
                    else:
                        volume_control_on = False

                    if volume_control_on:
                        if handedness == "Right":
                            increase_volume()
                        else:
                            decrease_volume()

                    if handedness == "Left" and is_middle_finger_up_and_others_down(hand_landmarks):
                        #if not is_browser_running():
                            print("Opening browser...")
                            webbrowser.open("https://www.google.com")

                    if handedness == "Right" and is_middle_finger_up_and_others_down(hand_landmarks):
                        print("Closing window...")
                        close_window()

                    if handedness == "Left" and is_pinky_finger_up_and_index_up(hand_landmarks):
                        print("Pressing up arrow key...")
                        press_key('up')
                    elif handedness == "Right" and is_pinky_finger_up_and_index_up(hand_landmarks):
                        print("Pressing down arrow key...")
                        press_key('down')

                    if handedness == "Left" and is_pinky_finger_up_and_others_down(hand_landmarks):
                        print("Pressing left arrow key...")
                        press_key('left')
                    elif handedness == "Right" and is_pinky_finger_up_and_others_down(hand_landmarks):
                        print("Pressing right arrow key...")
                        press_key('right')

                    last_action_time = time.time()

                if not mouse_mode and handedness == "Right" and is_mouse_gesture(hand_landmarks):
                    if mouse_mode_start_time is None:
                        mouse_mode_start_time = time.time()
                    elapsed_time = time.time() - mouse_mode_start_time
                    show_timer_on_screen(frame, "Enable Mouse Control", elapsed_time)

                    if elapsed_time >= 5:
                        mouse_mode = True
                        print("Mouse mode enabled!")
                        mouse_mode_start_time = None
                        time.sleep(1)  # Avoid rapid toggling
                    continue

                if mouse_mode and handedness == "Left" and is_thumbs_up(hand_landmarks):
                    if mouse_mode_start_time is None:
                        mouse_mode_start_time = time.time()
                    elapsed_time = time.time() - mouse_mode_start_time
                    show_timer_on_screen(frame, "Disable mouse Control", elapsed_time)

                    if elapsed_time >= 5:
                        mouse_mode = False
                        print("mouse control disabled!")
                        mouse_mode_start_time = None
                        time.sleep(1)  # Avoid rapid toggling
                    continue
                elif not is_thumbs_up(hand_landmarks):
                    mouse_mode_start_time = None

                if mouse_mode:
                    volume_control_on = False
                    if is_mouse_gesture(hand_landmarks):
                        left_click()
                        time.sleep(0.3)  # Prevent rapid clicks
                    elif is_index_finger_up_and_others_down(hand_landmarks):
                        cursor_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                        cursor_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        move_mouse(cursor_x, cursor_y)
                    continue

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Real-Time Hand Tracking', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Real-Time Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()  
cv2.destroyAllWindows()

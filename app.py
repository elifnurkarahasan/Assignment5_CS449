import cv2
import mediapipe as mp
import math
from flask import Flask, render_template, Response, request, jsonify
import pyautogui

app = Flask(__name__)

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

vertical_intensity_mode = False
horizontal_intensity_mode = False

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def move_virtual_cursor(index_tip):
    screen_width, screen_height = pyautogui.size()
    cursor_x = (1 - index_tip.x) * screen_width
    cursor_y = index_tip.y * screen_height
    pyautogui.moveTo(cursor_x, cursor_y)

def simulate_click():
    pyautogui.click()

def trigger_scroll(movement_x, movement_y):
    global vertical_intensity_mode, horizontal_intensity_mode

    # If vertical_intensity_mode is on and vertical movement dominates, adjust brightness via up/down
    if vertical_intensity_mode and abs(movement_y) > abs(movement_x) and abs(movement_y) > 0.005:
        if movement_y < 0:
            pyautogui.press('up')
        else:
            pyautogui.press('down')
        return

    # If horizontal_intensity_mode is on and horizontal movement dominates, adjust brightness via left/right
    if horizontal_intensity_mode and abs(movement_x) > abs(movement_y) and abs(movement_x) > 0.005:
        if movement_x < 0:
            pyautogui.press('left')
        else:
            pyautogui.press('right')
        return

    # If no intensity mode or the movement doesn't match the mode, do normal scrolling:
    if abs(movement_x) > abs(movement_y) and abs(movement_x) > 0.005:
        # Horizontal scrolling
        scroll_amount = movement_x * 200
        pyautogui.hscroll(int(scroll_amount))
    elif abs(movement_y) > abs(movement_x) and abs(movement_y) > 0.005:
        # Vertical scrolling
        scroll_amount = -movement_y * 200
        pyautogui.scroll(int(scroll_amount))

def gen_frames():
    cap = cv2.VideoCapture(0)
    gesture_stable_frames = 0
    stability_threshold = 1
    last_index_x, last_index_y = None, None
    last_middle_x, last_middle_y = None, None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = landmarks.landmark[4]
                index_tip = landmarks.landmark[8]
                middle_tip = landmarks.landmark[12]
                ring_tip = landmarks.landmark[16]
                pinky_tip = landmarks.landmark[20]
                index_base = landmarks.landmark[5]
                middle_base = landmarks.landmark[9]
                ring_base = landmarks.landmark[13]
                pinky_base = landmarks.landmark[17]
                thumb_base = landmarks.landmark[2]

                is_index_raised = index_tip.y < index_base.y
                is_middle_raised = middle_tip.y < middle_base.y
                is_ring_down = ring_tip.y >= ring_base.y
                is_pinky_down = pinky_tip.y >= pinky_base.y
                is_thumb_up = thumb_tip.y < thumb_base.y
                is_middle_down = middle_tip.y >= middle_base.y

                # Cursor gesture
                if is_index_raised and is_thumb_up and is_ring_down and is_pinky_down and is_middle_down:
                    move_virtual_cursor(index_tip)

                # Pinch gesture (click)
                thumb_index_distance = calculate_distance(thumb_tip, index_tip)
                if thumb_index_distance < 0.04 and is_thumb_up and is_index_raised and is_middle_down and is_ring_down and is_pinky_down:
                    simulate_click()

                # Scrolling gesture
                if calculate_distance(index_tip, middle_tip) < 0.045 and is_ring_down and is_pinky_down:
                    if is_index_raised and is_middle_raised:
                        gesture_stable_frames += 1
                        if gesture_stable_frames >= stability_threshold:
                            if last_index_x is not None and last_middle_x is not None:
                                movement_x = ((index_tip.x - last_index_x) + (middle_tip.x - last_middle_x)) / 2
                                movement_y = ((index_tip.y - last_index_y) + (middle_tip.y - last_middle_y)) / 2
                                trigger_scroll(movement_x, movement_y)
                            last_index_x, last_index_y = index_tip.x, index_tip.y
                            last_middle_x, last_middle_y = middle_tip.x, middle_tip.y
                    else:
                        gesture_stable_frames = 0
                        last_index_x, last_index_y = None, None
                        last_middle_x, last_middle_y = None, None

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_vertical_mode', methods=['POST'])
def toggle_vertical_mode():
    global vertical_intensity_mode
    vertical_intensity_mode = not vertical_intensity_mode
    return jsonify({"vertical_intensity_mode": vertical_intensity_mode})

@app.route('/toggle_horizontal_mode', methods=['POST'])
def toggle_horizontal_mode():
    global horizontal_intensity_mode
    horizontal_intensity_mode = not horizontal_intensity_mode
    return jsonify({"horizontal_intensity_mode": horizontal_intensity_mode})

if __name__ == "__main__":
    app.run(debug=True)

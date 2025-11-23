import cv2
import mediapipe as mp
import numpy as np
import sys

# Add src to path
sys.path.insert(0, 'src')

from game_manager import GameManager

def main():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Get window size
    window_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    window_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Game Manager
    game_manager = GameManager(window_width=window_width, window_height=window_height)
    
    # Hand tracking variables
    prev_y = {"index": None, "middle": None, "ring": None, "pinky": None}
    THRESHOLD = 25
    
    # Lane mapping (Finger -> Lane ID)
    # 0: Index (Lane 0)
    # 1: Middle (Lane 1)
    # 2: Ring (Lane 2)
    # 3: Pinky (Lane 3)
    finger_to_lane = {
        "index": 0,
        "middle": 1,
        "ring": 2,
        "pinky": 3
    }

    print("\n🎹 AirBeats - Touchless Piano Tiles")
    print("Controls:")
    print("  [SPACE] Start Game")
    print("  [P]     Pause/Resume")
    print("  [R]     Retry (Game Over)")
    print("  [Q]     Quit")
    print("  [ESC]   Quit")

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Draw Hand Landmarks & Detect Taps
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger coordinates
                    h, w, _ = frame.shape
                    landmarks = handLms.landmark
                    
                    fingers = {
                        "index": landmarks[8],
                        "middle": landmarks[12],
                        "ring": landmarks[16],
                        "pinky": landmarks[20]
                    }
                    
                    coords = {}
                    for name, lm in fingers.items():
                        coords[name] = (int(lm.x * w), int(lm.y * h))

                    # Detect Taps
                    for name, (x, y) in coords.items():
                        # Smoothing
                        if prev_y[name] is not None:
                            # Simple low-pass filter
                            y_smoothed = int(0.7 * prev_y[name] + 0.3 * y)
                            diff = prev_y[name] - y_smoothed
                            
                            # Check for downward movement (Tap)
                            if diff < -THRESHOLD:
                                # Visual feedback for tap
                                cv2.circle(frame, (x, y), 20, (0, 255, 255), cv2.FILLED)
                                cv2.putText(frame, "TAP!", (x-30, y-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                                
                                # Send input to Game Manager
                                lane_id = finger_to_lane.get(name)
                                if lane_id is not None:
                                    game_manager.handle_input(lane_id)
                                    
                            # Update prev_y with smoothed value to avoid jitter
                            prev_y[name] = y_smoothed
                        else:
                            prev_y[name] = y

                        # Draw finger markers
                        color = (0, 255, 0) # Default green
                        cv2.circle(frame, (x, y), 10, color, cv2.FILLED)
                        cv2.putText(frame, name[0].upper(), (x-10, y-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Update Game Logic
            game_manager.update()
            
            # Draw Game Elements
            game_manager.draw(frame)

            # Show Frame
            cv2.imshow("AirBeats - Week 2 Prototype", frame)

            # Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or Q to quit
                break
            elif key == ord(' '): # Space to start
                if game_manager.state_machine.is_menu():
                    game_manager.start_game()
            elif key == ord('p'): # P to pause/resume
                if game_manager.state_machine.is_playing():
                    game_manager.pause_game()
                elif game_manager.state_machine.is_paused():
                    game_manager.resume_game()
            elif key == ord('r'): # R to retry
                if game_manager.state_machine.is_game_over():
                    game_manager.start_game()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    game_manager.cleanup()

if __name__ == "__main__":
    main()
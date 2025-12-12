import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys

# Add src to path
sys.path.insert(0, 'src')

from game_manager import GameManager

def main():
    # Initialize Pygame
    pygame.init()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Get window size from camera
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set window size 
    window_width = 1280
    window_height = 720
    
    # Create Pygame window
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("AirBeats - Touchless Piano Tiles")
    clock = pygame.time.Clock()

    # Initialize Game Manager
    game_manager = GameManager(window_width=window_width, window_height=window_height)
    
    # Hand tracking variables
    prev_y = {"index": None, "middle": None, "ring": None, "pinky": None}
    THRESHOLD = 20
    
    # Lane mapping (Finger -> Lane ID)
    finger_to_lane = {
        "index": 0,
        "middle": 1,
        "ring": 2,
        "pinky": 3
    }

    print("\nAirBeats - Touchless Piano Tiles")
    print("Controls:")
    print("  [MOUSE] Navigate Menu")
    print("  [ESC]   Back / Pause")
    
    running = True
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while running:
            # 1. Capture Camera Frame
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # 2. Process Hand Landmarks & Detect Taps
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    # Draw landmarks on OpenCV frame 
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
                            y_smoothed = int(0.7 * prev_y[name] + 0.3 * y)
                            diff = prev_y[name] - y_smoothed
                            
                            # Check for downward movement (Tap)
                            if diff < -THRESHOLD:
                                # Visual feedback on OpenCV frame
                                cv2.circle(frame, (x, y), 20, (0, 255, 255), cv2.FILLED)
                                cv2.putText(frame, "TAP!", (x-30, y-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                                
                                # Send input to Game Manager
                                lane_id = finger_to_lane.get(name)
                                if lane_id is not None:
                                    game_manager.handle_input(lane_id)
                                    
                            prev_y[name] = y_smoothed
                        else:
                            prev_y[name] = y
                            
                        # Draw finger markers on OpenCV frame
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)

            # Resize frame to fit window if needed, or center it
            frame_resized = cv2.resize(frame, (window_width, window_height))
            
            # Convert BGR (OpenCV) to RGB (Pygame)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)            
            frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            
            # Update GameManager with current frame
            game_manager.current_frame_surface = frame_surface

            # 4. Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Pass event to Game Manager
                game_manager.handle_event(event)

                # Check if game manager requested exit
                if game_manager.should_exit:
                    running = False

            # 5. Update Game Logic
            game_manager.update()
            
            # 6. Draw Game
            game_manager.draw(screen)
            
            # 7. Update Display
            pygame.display.flip()
            clock.tick(60)

    # Cleanup
    cap.release()
    pygame.quit()
    game_manager.cleanup()
    sys.exit()

if __name__ == "__main__":
    main()
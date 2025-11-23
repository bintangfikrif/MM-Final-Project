"""
Game Manager Module

"""

import sys
sys.path.insert(0, 'src')

from game_state_machine import GameState
from score_manager import ScoreManager
from combo_counter import ComboCounter
from timer import Timer
from tile_manager import TileManager
from audio_manager import AudioManager

class GameManager:
    """
    Main manager yang menggabungkan:
    - GameState: State management (MENU, PLAYING, PAUSED, GAME_OVER)
    - ScoreManager: Score tracking dengan combo multiplier
    - ComboCounter: Combo management
    - Timer: Game duration tracking
    - TileManager: Spawning and moving tiles
    - AudioManager: Playing sounds
    
    Koordinasi seluruh game flow dan logic.
    """
    
    def __init__(self, game_duration=60, max_misses=3, window_width=640, window_height=480):
        """
        Initialize Game Manager dengan semua subsystems.
        """
        print("="*60)
        print("🎮 INITIALIZING GAME MANAGER")
        print("="*60)
        
        # Initialize semua subsystems
        self.state_machine = GameState()
        self.score_manager = ScoreManager(max_misses=max_misses)
        self.combo_counter = ComboCounter()
        self.timer = Timer(duration=game_duration, mode=Timer.STOPWATCH)
        self.tile_manager = TileManager(window_width, window_height)
        self.audio_manager = AudioManager()
        
        # Game configuration
        self.game_duration = game_duration
        self.max_misses = max_misses
        
        # Game session data
        self.final_score = 0
        self.final_max_combo = 0
        self.game_session_active = False
        
        print("✅ All subsystems initialized!")
        print(f"   Game Duration: {game_duration}s")
        print(f"   Max Misses: {max_misses}")
        print("="*60 + "\n")
    
    # ============ GAME FLOW CONTROL ============
    
    def start_game(self):
        """Mulai game dari MENU ke PLAYING."""
        if not self.state_machine.is_menu() and not self.state_machine.is_game_over():
             # Allow restart from game over
            if not self.state_machine.is_game_over():
                print("⚠️  Game tidak dalam state MENU atau GAME_OVER!")
                return False
        
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        # Reset managers
        self.score_manager.reset()
        self.combo_counter.reset()
        self.timer.reset()
        self.tile_manager = TileManager(self.tile_manager.window_width, self.tile_manager.window_height) # Reset tiles
        
        self.timer.start()
        self.game_session_active = True
        
        print("🎮 GAME STARTED!")
        return True
    
    def pause_game(self):
        """Pause game."""
        if not self.state_machine.is_playing():
            return False
        
        if not self.state_machine.transition_to(GameState.PAUSED):
            return False
        
        self.timer.pause()
        print("⏸️  GAME PAUSED!")
        return True
    
    def resume_game(self):
        """Resume game."""
        if not self.state_machine.is_paused():
            return False
        
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        self.timer.start()
        print("▶️  GAME RESUMED!")
        return True
    
    def end_game(self):
        """End game."""
        if not (self.state_machine.is_playing() or self.state_machine.is_paused()):
            return False
        
        if not self.state_machine.transition_to(GameState.GAME_OVER):
            return False
        
        self.timer.stop()
        self.game_session_active = False
        
        self.final_score = self.score_manager.total_score
        self.final_max_combo = self.combo_counter.max_combo
        
        print("💀 GAME OVER!")
        print(f"   Final Score: {self.final_score}")
        return True
    
    def return_to_menu(self):
        """Return ke MENU."""
        if not self.state_machine.is_game_over():
            return False
        
        if not self.state_machine.transition_to(GameState.MENU):
            return False
        
        print("🏠 BACK TO MENU!")
        return True
    
    # ============ GAME LOOP ============

    def update(self):
        """Main update loop called every frame."""
        if not self.state_machine.is_playing():
            return

        # Update Timer
        if self.timer.is_time_up():
            self.end_game()
            return

        # Update Tiles
        self.tile_manager.update()

        # Check Misses
        missed_tiles = self.tile_manager.check_misses()
        for tile in missed_tiles:
            self.on_tile_miss()

    def draw(self, frame):
        """Main draw loop called every frame."""
        # Draw Tiles
        self.tile_manager.draw(frame)
        
        # Draw UI Overlay (Score, Combo, Timer)
        # TODO: Move this to a dedicated UI Manager later
        import cv2
        
        # Status Text
        status_text = f"Score: {self.score_manager.total_score} | Combo: {self.combo_counter.current_combo}x"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timer_text = f"Time: {self.timer.get_display_time()}"
        cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.state_machine.is_game_over():
            cv2.putText(frame, "GAME OVER", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {self.final_score}", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'R' to Retry or 'Q' to Quit", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def handle_input(self, lane):
        """Handle player input (tap in a lane)."""
        if not self.state_machine.is_playing():
            return

        hit_tile = self.tile_manager.check_hit(lane)
        if hit_tile:
            self.on_tile_hit()
            # Play sound based on lane (simplified mapping)
            notes = ['C', 'E', 'G', 'C_high']
            note = notes[lane % 4]
            self.audio_manager.play_note(note)
        else:
            # Optional: Penalty for tapping empty lane?
            pass

    # ============ GAME ACTIONS ============
    
    def on_tile_hit(self, base_points=10):
        """Handle tile hit."""
        self.combo_counter.add_hit()
        self.score_manager.add_hit(base_points)
    
    def on_tile_miss(self):
        """Handle tile miss."""
        self.combo_counter.add_miss()
        game_over = self.score_manager.add_miss()
        if game_over:
            self.end_game()

    def cleanup(self):
        self.audio_manager.cleanup()


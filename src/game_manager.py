"""
Game Manager Module

Orchestrates all game systems and manages game flow.
"""

import sys
sys.path.insert(0, 'src')

from game_state_machine import GameState
from score_manager import ScoreManager
from combo_counter import ComboCounter
from timer import Timer
from tile_manager import TileManager
from audio_manager import AudioManager
from songs import SongManager
from difficulty_manager import DifficultyManager

# Import UI Screens
from ui.menu_screen import MenuScreen
from ui.game_screen import GameScreen
from ui.pause_screen import PauseScreen
from ui.game_over_screen import GameOverScreen
from ui.settings_screen import SettingsScreen

class GameManager:
    """
    Main manager yang menggabungkan semua subsystems.
    """
    
    def __init__(self, game_duration=60, max_misses=99, window_width=640, window_height=480):
        """Initialize Game Manager"""
        print("="*60)
        print("🎮 INITIALIZING GAME MANAGER")
        print("="*60)
        
        # Initialize subsystems
        self.state_machine = GameState()
        self.score_manager = ScoreManager(max_misses=max_misses)
        self.combo_counter = ComboCounter()
        self.timer = Timer(duration=game_duration, mode=Timer.STOPWATCH)
        self.tile_manager = TileManager(window_width, window_height)
        self.audio_manager = AudioManager()
        self.song_manager = SongManager()
        self.difficulty_manager = DifficultyManager(initial_difficulty='MEDIUM')
        
        # Game configuration
        self.game_duration = game_duration
        self.max_misses = max_misses
        self.window_width = window_width
        self.window_height = window_height
        
        # Game session data
        self.final_score = 0
        self.final_max_combo = 0
        self.game_session_active = False
        
        # Camera frame storage
        self.current_frame_surface = None
        
        # Initialize Screens
        self.screens = {
            GameState.MENU: MenuScreen(window_width, window_height, self),
            GameState.PLAYING: GameScreen(window_width, window_height, self),
            GameState.PAUSED: PauseScreen(window_width, window_height, self),
            GameState.GAME_OVER: GameOverScreen(window_width, window_height, self._get_game_stats()),
            "SETTINGS": SettingsScreen(window_width, window_height, self)
        }
        
        # Set initial screen
        self.screens[GameState.MENU].on_enter()
        
        print("✅ All subsystems initialized!")
        print(f"   Game Duration: {game_duration}s")
        print(f"   Max Misses: {max_misses}")
        print("="*60 + "\n")
    
    def _get_game_stats(self):
        """Helper to get current game stats for Game Over screen."""
        return {
            'final_score': self.score_manager.total_score,
            'max_combo': self.combo_counter.max_combo,
            'total_hits': self.score_manager.total_hits,
            'total_misses': self.score_manager.miss_count,
            'accuracy': self.score_manager.get_accuracy(),
            'game_duration': self.timer.get_elapsed_time(),
            'game_duration_display': self.timer.get_display_time()
        }

    # ============ GAME FLOW CONTROL ============
    
    def start_game(self):
        """Start game with selected song and difficulty"""
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        # Reset managers
        self.score_manager.reset()
        self.combo_counter.reset()
        self.timer.reset()
        
        # Get song tiles with current difficulty
        song_tiles = self.song_manager.get_current_song_tiles()
        song_duration = self.song_manager.get_song_duration()
        
        # Get difficulty speed
        difficulty_speed = self.difficulty_manager.get_tile_speed()
        
        # Create tile manager with song data
        self.tile_manager = TileManager(
            self.window_width, 
            self.window_height,
            speed=difficulty_speed
        )
        self.tile_manager.load_song_tiles(song_tiles)
        self.tile_manager.start_song()
        
        # Update timer to song duration
        self.timer.duration = song_duration
        self.timer.start()
        self.game_session_active = True
        
        # Update screens
        self.screens[GameState.MENU].on_exit()
        self.screens[GameState.PLAYING].on_enter()
        
        current_song = self.song_manager.songs[self.song_manager.current_song].name
        current_diff = self.difficulty_manager.current_difficulty
        print(f"🎮 GAME STARTED!")
        print(f"   Song: {current_song}")
        print(f"   Difficulty: {current_diff}")
        print(f"   Duration: {song_duration:.1f}s")
        return True
    
    def pause_game(self):
        """Pause game."""
        if not self.state_machine.transition_to(GameState.PAUSED):
            return False
        
        self.timer.pause()
        self.screens[GameState.PLAYING].on_exit()
        self.screens[GameState.PAUSED].on_enter()
        print("⏸️  GAME PAUSED!")
        return True
    
    def resume_game(self):
        """Resume game."""
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        self.timer.start()
        self.screens[GameState.PAUSED].on_exit()
        self.screens[GameState.PLAYING].on_enter()
        print("▶️  GAME RESUMED!")
        return True
    
    def end_game(self):
        """End game."""
        if not self.state_machine.transition_to(GameState.GAME_OVER):
            return False
        
        self.timer.stop()
        self.game_session_active = False
        
        self.final_score = self.score_manager.total_score
        self.final_max_combo = self.combo_counter.max_combo
        
        # Update Game Over Screen with final stats
        game_over_screen = self.screens[GameState.GAME_OVER]
        game_over_screen.final_stats = self._get_game_stats()
        game_over_screen.rank_data = game_over_screen._calculate_rank()
        
        self.screens[GameState.PLAYING].on_exit()
        self.screens[GameState.GAME_OVER].on_enter()
        
        print("💀 GAME OVER!")
        print(f"   Final Score: {self.final_score}")
        return True
    
    def return_to_menu(self):
        """Return to MENU."""
        if not self.state_machine.transition_to(GameState.MENU):
            return False
        
        self.screens[GameState.GAME_OVER].on_exit()
        self.screens[GameState.MENU].on_enter()
        print("🏠 BACK TO MENU!")
        return True

    # ============ GAME LOOP ============

    def update(self):
        """Main update loop called every frame."""
        
        # Update active screen
        current_state = self.state_machine.current_state
        
        if current_state in self.screens:
            self.screens[current_state].update()

        # Update Game Logic (only if PLAYING)
        if self.state_machine.is_playing():
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

    def draw(self, surface):
        """Main draw loop called every frame."""
        current_state = self.state_machine.current_state
        if current_state in self.screens:
            self.screens[current_state].draw(surface)

    def handle_event(self, event):
        """Handle pygame events passed from main loop."""
        current_state = self.state_machine.current_state
        
        # Delegate to active screen
        if current_state in self.screens:
            next_screen_name = self.screens[current_state].handle_event(event)
            
            if next_screen_name:
                self._handle_screen_transition(next_screen_name)

    def _handle_screen_transition(self, next_screen_name):
        """Handle transition request from screens."""
        if next_screen_name == "GAME":
            if self.state_machine.is_menu() or self.state_machine.is_game_over():
                self.start_game()
            elif self.state_machine.is_paused():
                self.resume_game()
                
        elif next_screen_name == "MENU":
            if self.state_machine.is_game_over() or self.state_machine.is_paused():
                if self.state_machine.is_paused():
                    self.state_machine.transition_to(GameState.MENU)
                    self.screens[GameState.PAUSED].on_exit()
                    self.screens[GameState.MENU].on_enter()
                else:
                    self.return_to_menu()
                    
        elif next_screen_name == "SETTINGS":
            print("⚙️ Settings screen requested")
            pass

    def handle_input(self, lane):
        """Handle player input (tap in a lane)."""
        if not self.state_machine.is_playing():
            return

        hit_tile = self.tile_manager.check_hit(lane)
        if hit_tile:
            self.on_tile_hit()
            # Play sound based on tile's note
            self.audio_manager.play_note(hit_tile.note)
        else:
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
        """Cleanup resources."""
        self.audio_manager.cleanup()
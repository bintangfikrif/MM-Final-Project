"""
Game Manager Module

"""

import sys
sys.path.insert(0, 'src')

from game_state_machine import GameState
from score_manager import ScoreManager
from combo_counter import ComboCounter
from timer import Timer


class GameManager:
    """
    Main manager yang menggabungkan:
    - GameState: State management (MENU, PLAYING, PAUSED, GAME_OVER)
    - ScoreManager: Score tracking dengan combo multiplier
    - ComboCounter: Combo management
    - Timer: Game duration tracking
    
    Koordinasi seluruh game flow dan logic.
    """
    
    def __init__(self, game_duration=60, max_misses=3):
        """
        Initialize Game Manager dengan semua subsystems.
        
        Args:
            game_duration (int): Durasi game dalam detik (default 60s)
            max_misses (int): Max misses sebelum game over (default 3)
        """
        print("="*60)
        print("🎮 INITIALIZING GAME MANAGER")
        print("="*60)
        
        # Initialize semua subsystems
        self.state_machine = GameState()
        self.score_manager = ScoreManager(max_misses=max_misses)
        self.combo_counter = ComboCounter()
        self.timer = Timer(duration=game_duration, mode=Timer.STOPWATCH)
        
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
        """
        Mulai game dari MENU ke PLAYING.
        
        Returns:
            bool: True jika berhasil start, False jika sudah playing
        """
        # Check state
        if not self.state_machine.is_menu():
            print("⚠️  Game tidak dalam state MENU!")
            return False
        
        # Transition ke PLAYING
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        # Reset semua managers untuk game baru
        self.score_manager.reset()
        self.combo_counter.reset()
        self.timer.reset()
        
        # Start timer
        self.timer.start()
        self.game_session_active = True
        
        print("🎮 GAME STARTED!")
        print(f"   State: {self.state_machine.get_state()}")
        print(f"   Timer: {self.timer.get_display_time()}")
        return True
    
    def pause_game(self):
        """
        Pause game dari PLAYING ke PAUSED.
        
        Returns:
            bool: True jika berhasil pause
        """
        if not self.state_machine.is_playing():
            print("⚠️  Game harus dalam state PLAYING untuk pause!")
            return False
        
        if not self.state_machine.transition_to(GameState.PAUSED):
            return False
        
        # Pause timer
        self.timer.pause()
        
        print("⏸️  GAME PAUSED!")
        return True
    
    def resume_game(self):
        """
        Resume game dari PAUSED ke PLAYING.
        
        Returns:
            bool: True jika berhasil resume
        """
        if not self.state_machine.is_paused():
            print("⚠️  Game harus dalam state PAUSED untuk resume!")
            return False
        
        if not self.state_machine.transition_to(GameState.PLAYING):
            return False
        
        # Resume timer
        self.timer.start()
        
        print("▶️  GAME RESUMED!")
        return True
    
    def end_game(self):
        """
        End game dari PLAYING/PAUSED ke GAME_OVER.
        
        Returns:
            bool: True jika berhasil end
        """
        if not (self.state_machine.is_playing() or self.state_machine.is_paused()):
            print("⚠️  Game harus PLAYING atau PAUSED untuk end!")
            return False
        
        if not self.state_machine.transition_to(GameState.GAME_OVER):
            return False
        
        # Stop timer
        self.timer.stop()
        self.game_session_active = False
        
        # Simpan final stats
        self.final_score = self.score_manager.total_score
        self.final_max_combo = self.combo_counter.max_combo
        
        print("💀 GAME OVER!")
        print(f"   Final Score: {self.final_score}")
        print(f"   Final Combo: {self.final_max_combo}x")
        return True
    
    def return_to_menu(self):
        """
        Return ke MENU dari GAME_OVER.
        
        Returns:
            bool: True jika berhasil return ke menu
        """
        if not self.state_machine.is_game_over():
            print("⚠️  Game harus dalam state GAME_OVER untuk return ke menu!")
            return False
        
        if not self.state_machine.transition_to(GameState.MENU):
            return False
        
        print("🏠 BACK TO MENU!")
        return True
    
    # ============ GAME ACTIONS ============
    
    def on_tile_hit(self, base_points=10):
        """
        Handle saat player berhasil hit tile.
        
        Args:
            base_points (int): Base points untuk tile ini (default 10)
            
        Returns:
            dict: Hit result dengan score, combo, multiplier
        """
        if not self.state_machine.is_playing():
            print("⚠️  Game tidak sedang bermain!")
            return None
        
        # Tambah combo
        self.combo_counter.add_hit()
        
        # Hitung score dengan multiplier combo
        multiplier = self.combo_counter.get_multiplier()
        hit_points = self.score_manager.add_hit(base_points)
        
        # Check milestone combo
        milestone = self.combo_counter.should_show_milestone_popup()
        
        result = {
            'hit_points': hit_points,
            'total_score': self.score_manager.total_score,
            'combo': self.combo_counter.current_combo,
            'multiplier': multiplier,
            'is_milestone': milestone
        }
        
        return result
    
    def on_tile_miss(self):
        """
        Handle saat player miss tile (tidak di-tap tepat waktu).
        
        Returns:
            dict: Miss result dengan game over status
        """
        if not self.state_machine.is_playing():
            print("⚠️  Game tidak sedang bermain!")
            return None
        
        # Reset combo
        self.combo_counter.add_miss()
        
        # Track miss di score manager
        game_over = self.score_manager.add_miss()
        
        result = {
            'miss_count': self.score_manager.miss_count,
            'max_misses': self.score_manager.max_misses,
            'game_over': game_over,
            'combo_broken': True
        }
        
        # Jika game over, transition to GAME_OVER
        if game_over:
            self.end_game()
        
        return result
    
    def check_time_up(self):
        """
        Check apakah waktu game sudah habis.
        
        Returns:
            bool: True jika time up (untuk COUNTDOWN mode)
        """
        if not self.state_machine.is_playing():
            return False
        
        return self.timer.is_time_up()
    
    # ============ STATUS & DISPLAY ============
    
    def get_game_status(self):
        """
        Hitung status lengkap game saat ini.
        
        Returns:
            dict: Lengkap game status untuk UI
        """
        return {
            # State
            'state': self.state_machine.get_state(),
            'is_playing': self.state_machine.is_playing(),
            'is_paused': self.state_machine.is_paused(),
            'is_menu': self.state_machine.is_menu(),
            'is_game_over': self.state_machine.is_game_over(),
            
            # Score
            'score': self.score_manager.total_score,
            'combo': self.combo_counter.current_combo,
            'max_combo': self.combo_counter.max_combo,
            'multiplier': self.combo_counter.get_multiplier(),
            
            # Misses
            'misses': self.score_manager.miss_count,
            'max_misses': self.score_manager.max_misses,
            'game_over_by_misses': self.score_manager.is_game_over(),
            
            # Timer
            'elapsed_time': self.timer.get_elapsed_time(),
            'time_display': self.timer.get_display_time(),
            'time_percentage': self.timer.get_time_percentage(),
            
            # Stats
            'total_hits': self.score_manager.total_hits,
            'accuracy': (self.score_manager.total_hits / (self.score_manager.total_hits + self.score_manager.miss_count) * 100) if (self.score_manager.total_hits + self.score_manager.miss_count) > 0 else 0.0
        }
    
    def print_game_status(self):
        """Print status game untuk debugging."""
        status = self.get_game_status()
        
        print("\n" + "="*60)
        print("GAME STATUS")
        print("="*60)
        print(f"State: {status['state']}")
        print(f"\nScore: {status['score']}")
        print(f"Combo: {status['combo']}x (max: {status['max_combo']}x)")
        print(f"Multiplier: {status['multiplier']:.1f}x")
        print(f"\nMisses: {status['misses']}/{status['max_misses']}")
        print(f"Hits: {status['total_hits']}")
        print(f"Accuracy: {status['accuracy']:.1f}%")
        print(f"\nTime: {status['time_display']} ({status['time_percentage']:.1f}%)")
        print("="*60 + "\n")
    
    def get_final_stats(self):
        """
        Hitung final stats untuk game over screen.
        
        Returns:
            dict: Final game statistics
        """
        status = self.get_game_status()
        
        return {
            'final_score': status['score'],
            'max_combo': status['max_combo'],
            'total_hits': status['total_hits'],
            'total_misses': status['misses'],
            'accuracy': status['accuracy'],
            'game_duration': self.timer.get_elapsed_time(),
            'game_duration_display': self.timer.format_time(self.timer.get_elapsed_time())
        }
    
    def print_final_stats(self):
        """Print final stats untuk game over screen."""
        stats = self.get_final_stats()
        
        print("\n" + "="*60)
        print("FINAL STATS")
        print("="*60)
        print(f"Final Score: {stats['final_score']}")
        print(f"Max Combo: {stats['max_combo']}x")
        print(f"Total Hits: {stats['total_hits']}")
        print(f"Total Misses: {stats['total_misses']}")
        print(f"Accuracy: {stats['accuracy']:.1f}%")
        print(f"Game Duration: {stats['game_duration_display']}")
        print("="*60 + "\n")
    
    # ============ DEBUG ============
    
    def __str__(self):
        """String representation."""
        status = self.get_game_status()
        return f"Game: {status['state']} | Score: {status['score']} | Combo: {status['combo']}x"

"""
Game State Machine Module

"""


class GameState:
    """
    State Machine untuk manage game states dan transitions.
    
    Valid states: MENU, PLAYING, PAUSED, GAME_OVER
    
    State flow:
        MENU ──→ PLAYING ──→ PAUSED
                   ↓
                GAME_OVER ──→ MENU
    
    Semua transisi yang tidak listed adalah INVALID dan akan di-reject.
    """
    
    # ============ STATE CONSTANTS ============
    MENU = "MENU"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    GAME_OVER = "GAME_OVER"
    
    # ============ VALID TRANSITIONS ============
    # Dictionary yang mendefinisikan transisi yang diperbolehkan
    # Key: current state, Value: list of valid next states
    VALID_TRANSITIONS = {
        MENU: [PLAYING],                    # Dari MENU hanya bisa ke PLAYING
        PLAYING: [PAUSED, GAME_OVER],       # Dari PLAYING bisa ke PAUSED atau GAME_OVER
        PAUSED: [PLAYING, GAME_OVER],       # Dari PAUSED bisa ke PLAYING atau GAME_OVER
        GAME_OVER: [MENU]                   # Dari GAME_OVER hanya bisa ke MENU
    }
    
    def __init__(self):
        """
        Initialize game state machine.
        State dimulai dari MENU (main menu screen).
        """
        self.current_state = self.MENU
        self.previous_state = None
        
        print("✅ GameState initialized")
        print(f"   Current state: {self.current_state}")
    
    def transition_to(self, new_state):
        """
        Attempt untuk transition ke state baru.
        
        Melakukan validation:
        1. Check apakah new_state adalah valid state
        2. Check apakah transisi dari current_state ke new_state dibolehkan
        
        Args:
            new_state (str): Target state untuk transition
            
        Returns:
            bool: True jika transition berhasil, False jika invalid
            
        Example:
            >>> state = GameState()
            >>> state.transition_to(GameState.PLAYING)  # MENU → PLAYING
            True
            >>> state.transition_to(GameState.MENU)     # PLAYING → MENU (INVALID)
            False
        """
        
        # ===== STEP 1: Validate new_state adalah valid state =====
        all_valid_states = [self.MENU, self.PLAYING, self.PAUSED, self.GAME_OVER]
        
        if new_state not in all_valid_states:
            print(f"❌ ERROR: '{new_state}' is not a valid state!")
            print(f"   Valid states are: {all_valid_states}")
            return False
        
        # ===== STEP 2: Check apakah transisi dibolehkan =====
        allowed_transitions = self.VALID_TRANSITIONS.get(self.current_state, [])
        
        if new_state not in allowed_transitions:
            print(f"❌ ERROR: Cannot transition from {self.current_state} to {new_state}")
            print(f"   From {self.current_state}, valid transitions are: {allowed_transitions}")
            return False
        
        # ===== STEP 3: Perform transition (VALID) =====
        self.previous_state = self.current_state
        self.current_state = new_state
        
        print(f"🔄 Transition: {self.previous_state} → {self.current_state}")
        return True
    
    # ============ HELPER METHODS - CHECK CURRENT STATE ============
    
    def is_menu(self):
        """
        Check apakah current state adalah MENU.
        
        Returns:
            bool: True jika state = MENU, False otherwise
        """
        return self.current_state == self.MENU
    
    def is_playing(self):
        """
        Check apakah current state adalah PLAYING.
        
        Returns:
            bool: True jika state = PLAYING, False otherwise
        """
        return self.current_state == self.PLAYING
    
    def is_paused(self):
        """
        Check apakah current state adalah PAUSED.
        
        Returns:
            bool: True jika state = PAUSED, False otherwise
        """
        return self.current_state == self.PAUSED
    
    def is_game_over(self):
        """
        Check apakah current state adalah GAME_OVER.
        
        Returns:
            bool: True jika state = GAME_OVER, False otherwise
        """
        return self.current_state == self.GAME_OVER
    
    # ============ STATUS METHODS ============
    
    def get_state(self):
        """
        Get current state string.
        
        Returns:
            str: Current state (MENU, PLAYING, PAUSED, atau GAME_OVER)
        """
        return self.current_state
    
    def get_status(self):
        """
        Get complete status information tentang state machine.
        
        Returns:
            dict: Dictionary berisi:
                - current (str): Current state
                - previous (str): Previous state (None jika belum ada transition)
                - valid_next_states (list): List of valid next states
                
        Example:
            >>> state = GameState()
            >>> state.transition_to(GameState.PLAYING)
            >>> status = state.get_status()
            >>> print(status)
            {
                'current': 'PLAYING',
                'previous': 'MENU',
                'valid_next_states': ['PAUSED', 'GAME_OVER']
            }
        """
        valid_next_states = self.VALID_TRANSITIONS.get(self.current_state, [])
        
        return {
            'current': self.current_state,
            'previous': self.previous_state,
            'valid_next_states': valid_next_states
        }
    
    def reset(self):
        """
        Reset state machine ke initial state (MENU).
        
        Digunakan ketika:
        - Game dimulai pertama kali
        - Player return ke main menu setelah game over
        """
        self.previous_state = self.current_state
        self.current_state = self.MENU
        print(f"🔄 State reset to {self.MENU}")
    
    # ============ DEBUG / DISPLAY ============
    
    def __str__(self):
        """String representation untuk debugging."""
        return f"State: {self.current_state} (previous: {self.previous_state})"
    
    def print_status(self):
        """Print status lengkap (untuk debugging)."""
        status = self.get_status()
        print(f"Current State: {status['current']}")
        print(f"Previous State: {status['previous']}")
        print(f"Valid Next States: {status['valid_next_states']}")

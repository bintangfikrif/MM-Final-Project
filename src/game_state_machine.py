class GameState:
    # STATE CONSTANTS
    MENU = "MENU"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    GAME_OVER = "GAME_OVER"
    
    # VALID TRANSITIONS
    VALID_TRANSITIONS = {
        MENU: [PLAYING],                    
        PLAYING: [PAUSED, GAME_OVER],       
        PAUSED: [PLAYING, GAME_OVER, MENU], 
        GAME_OVER: [MENU]                   
    }
    
    def __init__(self):
        self.current_state = self.MENU
        self.previous_state = None
        
        print("GameState initialized")
        print(f"Current state: {self.current_state}")
    
    def transition_to(self, new_state):
        
        # Validate new_state adalah valid state 
        all_valid_states = [self.MENU, self.PLAYING, self.PAUSED, self.GAME_OVER]
        
        if new_state not in all_valid_states:
            print(f"ERROR: '{new_state}' is not a valid state!")
            print(f"Valid states are: {all_valid_states}")
            return False
        
        # Check apakah transisi dibolehkan 
        allowed_transitions = self.VALID_TRANSITIONS.get(self.current_state, [])
        
        if new_state not in allowed_transitions:
            print(f"ERROR: Cannot transition from {self.current_state} to {new_state}")
            print(f"From {self.current_state}, valid transitions are: {allowed_transitions}")
            return False
        
        # Perform transition (VALID) 
        self.previous_state = self.current_state
        self.current_state = new_state
        
        print(f"Transition: {self.previous_state} → {self.current_state}")
        return True
    
    def is_menu(self):
        return self.current_state == self.MENU
    
    def is_playing(self):
        return self.current_state == self.PLAYING
    
    def is_paused(self):
        return self.current_state == self.PAUSED
    
    def is_game_over(self):
        return self.current_state == self.GAME_OVER
        
    def get_state(self):
        return self.current_state
    
    def get_status(self):
        valid_next_states = self.VALID_TRANSITIONS.get(self.current_state, [])
        
        return {
            'current': self.current_state,
            'previous': self.previous_state,
            'valid_next_states': valid_next_states
        }
    
    def reset(self):
        self.previous_state = self.current_state
        self.current_state = self.MENU
        print(f"State reset to {self.MENU}")
    
    # DEBUG / DISPLAY 
    
    def __str__(self):
        return f"State: {self.current_state} (previous: {self.previous_state})"
    
    def print_status(self):
        status = self.get_status()
        print(f"Current State: {status['current']}")
        print(f"Previous State: {status['previous']}")
        print(f"Valid Next States: {status['valid_next_states']}")

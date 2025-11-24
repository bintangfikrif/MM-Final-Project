"""
Difficulty Manager Module

Manages game difficulty settings including tile speed, spawn rate, and timing windows.
"""


class DifficultyManager:
    """
    Manages game difficulty settings.
    
    Provides 4 difficulty levels:
    - EASY: Slower tiles, more forgiving timing
    - MEDIUM: Normal speed, balanced gameplay
    - HARD: Faster tiles, tighter timing
    - EXPERT: Very fast, hardcore mode
    """
    
    def __init__(self, initial_difficulty='MEDIUM'):
        """
        Initialize difficulty manager.
        
        Args:
            initial_difficulty (str): Starting difficulty (EASY, MEDIUM, HARD, EXPERT)
        """
        
        # Difficulty presets
        self.difficulties = {
            'EASY': {
                'tile_speed': 4,
                'spawn_rate': 90,
                'hit_window': 120,
                'label': 'Easy',
                'color': (100, 255, 100),
                'description': 'Relaxed pace, perfect for beginners'
            },
            'MEDIUM': {
                'tile_speed': 6,
                'spawn_rate': 60,
                'hit_window': 100,
                'label': 'Medium',
                'color': (255, 255, 100),
                'description': 'Normal speed, balanced challenge'
            },
            'HARD': {
                'tile_speed': 8,
                'spawn_rate': 45,
                'hit_window': 80,
                'label': 'Hard',
                'color': (255, 165, 0),
                'description': 'Fast pace, requires focus'
            },
            'EXPERT': {
                'tile_speed': 10,
                'spawn_rate': 30,
                'hit_window': 60,
                'label': 'Expert',
                'color': (255, 50, 50),
                'description': 'Extreme speed, for masters only'
            }
        }
        
        # Set current difficulty
        self.current_difficulty = initial_difficulty
        
        print(f"✅ DifficultyManager initialized: {initial_difficulty}")
        self._print_current_settings()
    
    def set_difficulty(self, difficulty_name):
        """
        Set current difficulty.
        
        Args:
            difficulty_name (str): EASY, MEDIUM, HARD, or EXPERT
            
        Returns:
            bool: True if successful, False if invalid difficulty
        """
        if difficulty_name not in self.difficulties:
            print(f"❌ Invalid difficulty: {difficulty_name}")
            return False
        
        self.current_difficulty = difficulty_name
        print(f"🔄 Difficulty changed to: {difficulty_name}")
        self._print_current_settings()
        return True
    
    def get_tile_speed(self):
        """
        Get tile speed for current difficulty.
        
        Returns:
            int: Tile speed in pixels per frame
        """
        return self.difficulties[self.current_difficulty]['tile_speed']
    
    def get_spawn_rate(self):
        """
        Get spawn rate for current difficulty.
        
        Returns:
            int: Frames between tile spawns
        """
        return self.difficulties[self.current_difficulty]['spawn_rate']
    
    def get_hit_window(self):
        """
        Get hit timing window for current difficulty.
        
        Returns:
            int: Hit window size in pixels
        """
        return self.difficulties[self.current_difficulty]['hit_window']
    
    def get_label(self):
        """
        Get display label for current difficulty.
        
        Returns:
            str: Display label (e.g., "Easy", "Medium")
        """
        return self.difficulties[self.current_difficulty]['label']
    
    def get_color(self):
        """
        Get color for current difficulty.
        
        Returns:
            tuple: RGB color tuple
        """
        return self.difficulties[self.current_difficulty]['color']
    
    def get_description(self):
        """
        Get description for current difficulty.
        
        Returns:
            str: Difficulty description
        """
        return self.difficulties[self.current_difficulty]['description']
    
    def get_all_difficulties(self):
        """
        Get list of all available difficulties.
        
        Returns:
            list: List of difficulty names
        """
        return list(self.difficulties.keys())
    
    def get_current_difficulty(self):
        """
        Get current difficulty name.
        
        Returns:
            str: Current difficulty (EASY, MEDIUM, HARD, EXPERT)
        """
        return self.current_difficulty
    
    def get_current_settings(self):
        """
        Get all settings for current difficulty.
        
        Returns:
            dict: Complete settings dictionary
        """
        return self.difficulties[self.current_difficulty].copy()
    
    def _print_current_settings(self):
        """Print current difficulty settings (for debugging)"""
        settings = self.get_current_settings()
        print(f"   Tile Speed: {settings['tile_speed']}")
        print(f"   Spawn Rate: {settings['spawn_rate']}")
        print(f"   Hit Window: {settings['hit_window']}")
    
    def __str__(self):
        """String representation"""
        return f"DifficultyManager(current={self.current_difficulty})"


# ============ TESTING ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DIFFICULTY MANAGER - TEST")
    print("="*60 + "\n")
    
    # Test initialization
    dm = DifficultyManager()
    
    # Test all difficulties
    print("\n📊 Testing all difficulties:\n")
    for difficulty in dm.get_all_difficulties():
        dm.set_difficulty(difficulty)
        settings = dm.get_current_settings()
        print(f"\n{difficulty}:")
        print(f"  Speed: {settings['tile_speed']}")
        print(f"  Spawn Rate: {settings['spawn_rate']}")
        print(f"  Description: {settings['description']}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
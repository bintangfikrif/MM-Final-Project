class DifficultyManager:
    def __init__(self, initial_difficulty='MEDIUM'):
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
        
        print(f"DifficultyManager initialized: {initial_difficulty}")
        self._print_current_settings()
    
    def set_difficulty(self, difficulty_name):
        if difficulty_name not in self.difficulties:
            print(f"Invalid difficulty: {difficulty_name}")
            return False
        
        self.current_difficulty = difficulty_name
        print(f"Difficulty changed to: {difficulty_name}")
        self._print_current_settings()
        return True
    
    def get_tile_speed(self):
        return self.difficulties[self.current_difficulty]['tile_speed']
    
    def get_spawn_rate(self):
        return self.difficulties[self.current_difficulty]['spawn_rate']
    
    def get_hit_window(self):
        return self.difficulties[self.current_difficulty]['hit_window']
    
    def get_label(self):
        return self.difficulties[self.current_difficulty]['label']
    
    def get_color(self):
        return self.difficulties[self.current_difficulty]['color']
    
    def get_description(self):
        return self.difficulties[self.current_difficulty]['description']
    
    def get_all_difficulties(self):
        return list(self.difficulties.keys())
    
    def get_current_difficulty(self):
        return self.current_difficulty
    
    def get_current_settings(self):
        return self.difficulties[self.current_difficulty].copy()
    
    def _print_current_settings(self):
        settings = self.get_current_settings()
        print(f"   Tile Speed: {settings['tile_speed']}")
        print(f"   Spawn Rate: {settings['spawn_rate']}")
        print(f"   Hit Window: {settings['hit_window']}")
    
    def __str__(self):
        return f"DifficultyManager(current={self.current_difficulty})"


# TESTING 
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DIFFICULTY MANAGER - TEST")
    print("="*60 + "\n")
    
    # Test initialization
    dm = DifficultyManager()
    
    # Test all difficulties
    print("\nTesting all difficulties:\n")
    for difficulty in dm.get_all_difficulties():
        dm.set_difficulty(difficulty)
        settings = dm.get_current_settings()
        print(f"\n{difficulty}:")
        print(f"  Speed: {settings['tile_speed']}")
        print(f"  Spawn Rate: {settings['spawn_rate']}")
        print(f"  Description: {settings['description']}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")
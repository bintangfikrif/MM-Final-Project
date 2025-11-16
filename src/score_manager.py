"""
Score Manager Module

"""


class ScoreManager:
    """
    Manages score, combo, and miss tracking for the game.
    
    Scoring system menggunakan combo multiplier:
    - Base points: 10 per hit
    - Combo multiplier: 1 + (combo_count × 0.1)
    - Formula: final_score = base_points × (1 + combo_count × 0.1)
    
    Example:
        Hit 1: 10 × 1.0 = 10 poin
        Hit 2: 10 × 1.1 = 11 poin
        Hit 3: 10 × 1.2 = 12 poin
    
    Game ends ketika player mencapai max_misses (default: 3)
    """
    
    def __init__(self, max_misses=3):
        """
        Initialize score manager.
        
        Args:
            max_misses (int): Maximum misses allowed before game over. Default: 3
        """
        self.total_score = 0          # Total score accumulated
        self.current_combo = 0         # Current combo count (hits berturut-turut)
        self.max_combo = 0             # Maximum combo achieved in this game
        self.total_hits = 0            # Total successful hits
        self.miss_count = 0            # Current miss count
        self.max_misses = max_misses   # Maximum misses before game over
        
        print("✅ ScoreManager initialized")
        print(f"   Max misses allowed: {self.max_misses}")
    
    def add_hit(self, base_points=10):
        """
        Register a successful tile hit and update score.
        
        Calculates score dengan combo multiplier:
        final_score = base_points × (1 + combo × 0.1)
        
        Args:
            base_points (int): Base points for this hit. Default: 10
            
        Returns:
            int: Points earned in this hit (including multiplier)
            
        Example:
            >>> score_mgr = ScoreManager()
            >>> score_mgr.add_hit(10)  # Returns 10, Score = 10, Combo = 1
            >>> score_mgr.add_hit(10)  # Returns 11, Score = 21, Combo = 2
        """
        
        # ===== STEP 1: Calculate multiplier =====
        # Multiplier: 1.0x untuk first hit, 1.1x untuk second, 1.2x untuk third, dst
        multiplier = 1.0 + (self.current_combo * 0.1)
        
        # ===== STEP 2: Calculate earned points =====
        hit_score = int(base_points * multiplier)
        
        # ===== STEP 3: Update counters =====
        self.total_score += hit_score
        self.current_combo += 1
        self.total_hits += 1
        
        # ===== STEP 4: Update max combo if needed =====
        if self.current_combo > self.max_combo:
            self.max_combo = self.current_combo
        
        # ===== STEP 5: Print feedback =====
        print(f"✅ HIT! +{hit_score}pts | Score: {self.total_score} | Combo: {self.current_combo}x")
        
        return hit_score
    
    def add_miss(self):
        """
        Register a miss (tile not hit in time).
        
        Ketika miss:
        1. Increment miss count
        2. Reset combo ke 0 (combo break)
        3. Check apakah game over (miss_count >= max_misses)
        
        Returns:
            bool: True if max misses reached (game over), False otherwise
        """
        
        # ===== STEP 1: Increment miss count =====
        self.miss_count += 1
        
        # ===== STEP 2: Reset combo =====
        if self.current_combo > 0:
            print(f"⚠️  Combo broken! Was at {self.current_combo}x")
        self.current_combo = 0
        
        # ===== STEP 3: Check if game over =====
        print(f"❌ MISS! Misses: {self.miss_count}/{self.max_misses}")
        
        # ===== STEP 4: Return game over status =====
        is_game_over = self.miss_count >= self.max_misses
        if is_game_over:
            print(f"🏁 GAME OVER! Max misses reached!")
        
        return is_game_over
    
    def reset_combo(self):
        """
        Reset combo counter when player misses.
        Called when a tile passes the hit zone without being tapped.
        
        This is called internally by add_miss(), so you usually don't call this directly.
        """
        if self.current_combo > 0:
            print(f"⚠️  Combo broken! Was at {self.current_combo}x")
        self.current_combo = 0
    
    def get_status(self):
        """
        Get current scoring status.
        
        Returns:
            dict: Dictionary containing:
                - score (int): Current total score
                - combo (int): Current combo count
                - max_combo (int): Maximum combo achieved
                - misses (int): Current miss count
                - hits_total (int): Total hits in game
                - accuracy (float): Hit accuracy percentage
                
        Example:
            >>> score_mgr.get_status()
            {
                'score': 150,
                'combo': 5,
                'max_combo': 8,
                'misses': 1,
                'hits_total': 10,
                'accuracy': 90.9
            }
        """
        # Calculate accuracy: hits / (hits + misses) × 100
        total_actions = self.total_hits + self.miss_count
        if total_actions > 0:
            accuracy = (self.total_hits / total_actions) * 100
        else:
            accuracy = 0.0
        
        return {
            'score': self.total_score,
            'combo': self.current_combo,
            'max_combo': self.max_combo,
            'misses': self.miss_count,
            'hits_total': self.total_hits,
            'accuracy': accuracy
        }
    
    def get_combo_multiplier(self):
        """
        Get current combo multiplier value.
        
        Returns:
            float: Current multiplier (e.g., 1.0x, 1.5x, 2.0x)
            
        Example:
            >>> score_mgr.current_combo = 0
            >>> score_mgr.get_combo_multiplier()  # Returns 1.0
            >>> score_mgr.current_combo = 5
            >>> score_mgr.get_combo_multiplier()  # Returns 1.5
        """
        return 1.0 + (self.current_combo * 0.1)
    
    def is_game_over(self):
        """
        Check if game should be over due to too many misses.
        
        Returns:
            bool: True if game over, False otherwise
        """
        return self.miss_count >= self.max_misses
    
    def reset(self):
        """
        Reset all counters for a new game.
        Called when starting a new game session.
        """
        self.total_score = 0
        self.current_combo = 0
        self.max_combo = 0
        self.miss_count = 0
        self.total_hits = 0
        print("🔄 Score Manager reset for new game")
    
    def __str__(self):
        """String representation for debugging"""
        return f"Score: {self.total_score} | Combo: {self.current_combo}x | Misses: {self.miss_count}/{self.max_misses}"
    
    def print_status(self):
        """Print status lengkap (untuk debugging)."""
        status = self.get_status()
        print(f"Score: {status['score']}")
        print(f"Combo: {status['combo']}x (max: {status['max_combo']}x)")
        print(f"Hits: {status['hits_total']}")
        print(f"Misses: {status['misses']}/{self.max_misses}")
        print(f"Accuracy: {status['accuracy']:.1f}%")
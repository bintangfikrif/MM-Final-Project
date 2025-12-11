class ScoreManager:    
    
    def __init__(self, max_misses=3):
        self.total_score = 0          
        self.current_combo = 0         
        self.max_combo = 0             
        self.total_hits = 0            
        self.miss_count = 0            
        self.max_misses = max_misses   
        self.combo_history = []
        
        print("ScoreManager initialized")
        print(f"Max misses allowed: {self.max_misses}")
    
    def add_hit(self, base_points=10):
        
        # Calculate multiplier 
        multiplier = 1.0 + (self.current_combo * 0.1)
        
        # Calculate earned points 
        hit_score = int(base_points * multiplier)
        
        # Update counters 
        self.total_score += hit_score
        self.current_combo += 1
        self.total_hits += 1
        
        # Update max combo if needed 
        if self.current_combo > self.max_combo:
            self.max_combo = self.current_combo
            
        # Record history
        self._record_combo_change("hit")
        
        # Print feedback 
        print(f"HIT! +{hit_score}pts | Score: {self.total_score} | Combo: {self.current_combo}x")
        
        return hit_score
    
    def add_miss(self):  
        # Increment miss count 
        self.miss_count += 1
        
        # Reset combo 
        if self.current_combo > 0:
            print(f"Combo broken! Was at {self.current_combo}x")
            self._record_combo_change("miss")
            
        self.current_combo = 0
        
        # Check if game over 
        print(f"MISS! Misses: {self.miss_count}/{self.max_misses}")
        
        # Return game over status 
        is_game_over = self.miss_count >= self.max_misses
        if is_game_over:
            print(f"GAME OVER! Max misses reached!")
        
        return is_game_over
    
    def reset_combo(self):
        if self.current_combo > 0:
            print(f"Combo broken! Was at {self.current_combo}x")
            self._record_combo_change("miss")
        self.current_combo = 0

    def get_accuracy(self):
        total_actions = self.total_hits + self.miss_count
        if total_actions > 0:
            return (self.total_hits / total_actions) * 100
        return 0.0
    
    def get_status(self):
        return {
            'score': self.total_score,
            'combo': self.current_combo,
            'max_combo': self.max_combo,
            'misses': self.miss_count,
            'hits_total': self.total_hits,
            'accuracy': self.get_accuracy(),
            'history_count': len(self.combo_history)
        }
    
    def get_combo_multiplier(self):
        return 1.0 + (self.current_combo * 0.1)
    
    def is_game_over(self):
        return self.miss_count >= self.max_misses
    
    def reset(self):
        self.total_score = 0
        self.current_combo = 0
        self.max_combo = 0
        self.miss_count = 0
        self.total_hits = 0
        self.combo_history = []
        print("Score Manager reset for new game")
    
    # COMBO HELPER METHODS (Merged from ComboCounter)
    
    def get_combo_milestone(self):
        if self.current_combo == 0:
            return 0
        return (self.current_combo // 5) * 5
    
    def should_show_milestone_popup(self):
        if self.current_combo == 0:
            return False
        return self.current_combo % 5 == 0

    def _record_combo_change(self, event_type):
        self.combo_history.append({
            'type': event_type,
            'combo_after': self.current_combo
        })
        
    def get_history(self):
        return self.combo_history

    def __str__(self):
        return f"Score: {self.total_score} | Combo: {self.current_combo}x | Misses: {self.miss_count}/{self.max_misses}"
    
    def print_status(self):
        status = self.get_status()
        print(f"Score: {status['score']}")
        print(f"Combo: {status['combo']}x (max: {status['max_combo']}x)")
        print(f"Hits: {status['hits_total']}")
        print(f"Misses: {status['misses']}/{self.max_misses}")
        print(f"Accuracy: {status['accuracy']:.1f}%")
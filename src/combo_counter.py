class ComboCounter:
    def __init__(self):
        self.current_combo = 0      
        self.max_combo = 0          
        self.combo_history = []     
        
        print("ComboCounter initialized")
        print(f"Current combo: {self.current_combo}")
    
    # COMBO MANIPULATION 
    
    def add_hit(self):
        self.current_combo += 1
        
        # Update max combo jika diperlukan
        if self.current_combo > self.max_combo:
            self.max_combo = self.current_combo
        
        multiplier = self.get_multiplier(self.current_combo - 1)
        print(f"Combo +1 | Current: {self.current_combo}x | Multiplier: {multiplier:.1f}x")
        
        # Simpan ke history
        self._record_combo_change("hit")
        
        return self.current_combo
    
    def add_miss(self):
        if self.current_combo > 0:
            print(f"Combo broken! Was at {self.current_combo}x")
            self._record_combo_change("miss")
        
        self.current_combo = 0
        print(f"MISS! Combo reset to 0")
        
        return self.current_combo
    
    def reset(self):
        self.current_combo = 0
        self.max_combo = 0
        self.combo_history = []
        print("ComboCounter reset")
    
    # MULTIPLIER CALCULATION 
    
    def get_multiplier(self, combo=None):
        if combo is None:
            # Gunakan current combo - 1 (karena 0-based)
            combo = max(0, self.current_combo - 1)
        
        multiplier = 1.0 + (combo * 0.1)
        return multiplier
    
    # STATUS METHODS 
    
    def get_current_combo(self):
        return self.current_combo
    
    def get_max_combo(self):
        return self.max_combo
    
    def get_status(self):
        return {
            'current': self.current_combo,
            'max': self.max_combo,
            'multiplier': self.get_multiplier(),
            'history_count': len(self.combo_history)
        }
    
    # HELPER METHODS 
    
    def is_combo_active(self):
        return self.current_combo > 0
    
    def get_combo_milestone(self):
        if self.current_combo == 0:
            return 0
        
        # Hitung milestone terdekat (kelipatan 5)
        milestone = (self.current_combo // 5) * 5
        return milestone
    
    def should_show_milestone_popup(self):
        if self.current_combo == 0:
            return False
        
        return self.current_combo % 5 == 0
    
    # DEBUG / HISTORY 
    
    def _record_combo_change(self, event_type):
        self.combo_history.append({
            'type': event_type,
            'combo_after': self.current_combo
        })
    
    def get_history(self):
        return self.combo_history
    
    def print_history(self):
        print("\nCOMBO HISTORY")
        for i, event in enumerate(self.combo_history, 1):
            event_type = "HIT" if event['type'] == "hit" else "MISS"
            print(f"{i}. {event_type} → combo={event['combo_after']}")
        print(f"Final: {self.current_combo}x (max: {self.max_combo}x)\n")
    
    def __str__(self):
        return f"Combo: {self.current_combo}x (max: {self.max_combo}x, mult: {self.get_multiplier():.1f}x)"
    
    def print_status(self):
        status = self.get_status()
        print(f"Current Combo: {status['current']}x")
        print(f"Max Combo: {status['max']}x")
        print(f"Multiplier: {status['multiplier']:.1f}x")
        print(f"Total Events: {status['history_count']}")

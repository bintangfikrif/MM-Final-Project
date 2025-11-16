"""
Combo Counter Module

"""


class ComboCounter:
    """
    Mengelola combo count dan combo multiplier.
    
    Combo System:
    - Setiap hit yang berhasil: combo +1
    - Setiap miss: combo reset ke 0
    - Multiplier: 1.0 + (combo * 0.1)
    
    Contoh:
        Hit 1: combo=1, multiplier=1.0x
        Hit 2: combo=2, multiplier=1.1x
        Hit 3: combo=3, multiplier=1.2x
        Miss: combo=0, multiplier=1.0x (reset)
    """
    
    def __init__(self):
        """
        Initialize Combo Counter.
        """
        self.current_combo = 0      # Combo count sekarang
        self.max_combo = 0          # Combo max yang pernah dicapai
        self.combo_history = []     # Riwayat combo (untuk debug)
        
        print("✅ ComboCounter initialized")
        print(f"   Current combo: {self.current_combo}")
    
    # ============ COMBO MANIPULATION ============
    
    def add_hit(self):
        """
        Tambah combo saat hit berhasil.
        
        Returns:
            int: Combo count setelah ditambah
            
        Example:
            >>> combo = ComboCounter()
            >>> combo.add_hit()  # Returns 1
            >>> combo.add_hit()  # Returns 2
        """
        self.current_combo += 1
        
        # Update max combo jika diperlukan
        if self.current_combo > self.max_combo:
            self.max_combo = self.current_combo
        
        # Multiplier dihitung dari combo SEBELUM ditambah (0-based)
        # Hit ke-1: combo=1, tapi multiplier dari combo 0 = 1.0x
        # Hit ke-2: combo=2, tapi multiplier dari combo 1 = 1.1x
        multiplier = self.get_multiplier(self.current_combo - 1)
        print(f"✅ Combo +1 | Current: {self.current_combo}x | Multiplier: {multiplier:.1f}x")
        
        # Simpan ke history
        self._record_combo_change("hit")
        
        return self.current_combo
    
    def add_miss(self):
        """
        Reset combo saat miss.
        
        Returns:
            int: Combo count setelah reset (selalu 0)
        """
        if self.current_combo > 0:
            print(f"⚠️  Combo broken! Was at {self.current_combo}x")
            self._record_combo_change("miss")
        
        self.current_combo = 0
        print(f"❌ MISS! Combo reset to 0")
        
        return self.current_combo
    
    def reset(self):
        """
        Reset combo counter untuk game baru.
        """
        self.current_combo = 0
        self.max_combo = 0
        self.combo_history = []
        print("🔄 ComboCounter reset")
    
    # ============ MULTIPLIER CALCULATION ============
    
    def get_multiplier(self, combo=None):
        """
        Hitung multiplier berdasarkan combo count.
        
        Formula: multiplier = 1.0 + (combo * 0.1)
        
        Args:
            combo (int): Combo count yang akan dihitung. 
                        Jika None, gunakan current combo (0-based).
        
        Returns:
            float: Multiplier value (e.g., 1.0x, 1.5x, 2.0x)
            
        Example:
            >>> combo = ComboCounter()
            >>> combo.current_combo = 5
            >>> combo.get_multiplier()  # Returns 1.4 (5-1)*0.1 + 1.0
            >>> combo.get_multiplier(10)  # Returns 1.9 (10-1)*0.1 + 1.0
        """
        if combo is None:
            # Gunakan current combo - 1 (karena 0-based)
            combo = max(0, self.current_combo - 1)
        
        multiplier = 1.0 + (combo * 0.1)
        return multiplier
    
    # ============ STATUS METHODS ============
    
    def get_current_combo(self):
        """
        Get combo count sekarang.
        
        Returns:
            int: Current combo count
        """
        return self.current_combo
    
    def get_max_combo(self):
        """
        Get max combo yang pernah dicapai.
        
        Returns:
            int: Maximum combo achieved
        """
        return self.max_combo
    
    def get_status(self):
        """
        Get status lengkap combo counter.
        
        Returns:
            dict: Dictionary berisi:
                - current (int): Combo sekarang
                - max (int): Max combo pernah
                - multiplier (float): Multiplier sekarang
                - history_count (int): Total events
        """
        return {
            'current': self.current_combo,
            'max': self.max_combo,
            'multiplier': self.get_multiplier(),
            'history_count': len(self.combo_history)
        }
    
    # ============ HELPER METHODS ============
    
    def is_combo_active(self):
        """
        Check apakah combo aktif (> 0).
        
        Returns:
            bool: True jika combo > 0
        """
        return self.current_combo > 0
    
    def get_combo_milestone(self):
        """
        Get milestone combo terdekat yang sudah dicapai.
        
        Milestones: 5, 10, 15, 20, 25, dst
        
        Returns:
            int: Milestone combo (0 jika belum ada)
            
        Example:
            >>> combo = ComboCounter()
            >>> combo.current_combo = 7
            >>> combo.get_combo_milestone()  # Returns 5
            >>> combo.current_combo = 15
            >>> combo.get_combo_milestone()  # Returns 15
        """
        if self.current_combo == 0:
            return 0
        
        # Hitung milestone terdekat (kelipatan 5)
        milestone = (self.current_combo // 5) * 5
        return milestone
    
    def should_show_milestone_popup(self):
        """
        Check apakah harus tampilkan popup milestone.
        
        Returns:
            bool: True jika current combo adalah milestone (5, 10, 15, dst)
        """
        if self.current_combo == 0:
            return False
        
        return self.current_combo % 5 == 0
    
    # ============ DEBUG / HISTORY ============
    
    def _record_combo_change(self, event_type):
        """
        Catat combo change ke history (internal use).
        
        Args:
            event_type (str): "hit" atau "miss"
        """
        self.combo_history.append({
            'type': event_type,
            'combo_after': self.current_combo
        })
    
    def get_history(self):
        """
        Get combo history untuk debug.
        
        Returns:
            list: List of combo changes
        """
        return self.combo_history
    
    def print_history(self):
        """
        Print combo history untuk debugging.
        """
        print("\n=== COMBO HISTORY ===")
        for i, event in enumerate(self.combo_history, 1):
            event_type = "HIT" if event['type'] == "hit" else "MISS"
            print(f"{i}. {event_type} → combo={event['combo_after']}")
        print(f"Final: {self.current_combo}x (max: {self.max_combo}x)\n")
    
    def __str__(self):
        """String representation untuk debugging."""
        return f"Combo: {self.current_combo}x (max: {self.max_combo}x, mult: {self.get_multiplier():.1f}x)"
    
    def print_status(self):
        """Print status lengkap."""
        status = self.get_status()
        print(f"Current Combo: {status['current']}x")
        print(f"Max Combo: {status['max']}x")
        print(f"Multiplier: {status['multiplier']:.1f}x")
        print(f"Total Events: {status['history_count']}")

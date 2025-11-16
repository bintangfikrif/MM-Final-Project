"""
Miss Detector Module

"""


class MissDetector:
    """
    Mendeteksi dan track tiles yang missed (lewat tanpa di-tap).
    
    Sistem:
    1. Track tiles yang active (dalam hit zone)
    2. Saat tile keluar hit zone tanpa di-tap → missed
    3. Callback ke game manager untuk handle miss
    
    Tile lifecycle:
    - SPAWNED: Tile baru muncul (di atas layar)
    - ACTIVE: Tile dalam hit zone (bisa di-tap)
    - MISSED: Tile lewat hit zone tanpa di-tap
    - HIT: Tile berhasil di-tap
    - DESTROYED: Tile dihapus dari game
    """
    
    # ============ TILE STATES ============
    SPAWNED = "SPAWNED"      # Baru muncul
    ACTIVE = "ACTIVE"        # Dalam hit zone
    MISSED = "MISSED"        # Lewat tanpa di-tap
    HIT = "HIT"              # Berhasil di-tap
    DESTROYED = "DESTROYED"  # Dihapus
    
    def __init__(self):
        """
        Initialize Miss Detector.
        """
        self.tracked_tiles = {}      # {tile_id: tile_info}
        self.missed_tiles = []        # List of missed tile ids
        self.total_tiles_spawned = 0  # Counter untuk tile id
        self.total_misses = 0         # Total misses terdeteksi
        self.total_hits = 0           # Total tiles yang di-hit (untuk accuracy)
        
        print("✅ MissDetector initialized")
    
    # ============ TILE TRACKING ============
    
    def spawn_tile(self, tile_id=None, lane=None, base_points=10):
        """
        Register tile baru yang spawn.
        
        Args:
            tile_id (int): Unique tile ID. Jika None, auto-generate.
            lane (int): Lane number (0-3 untuk 4 lanes)
            base_points (int): Base points untuk tile ini (default 10)
            
        Returns:
            dict: Tile info
        """
        # Auto-generate tile_id jika tidak diberikan
        if tile_id is None:
            self.total_tiles_spawned += 1
            tile_id = self.total_tiles_spawned
        
        tile_info = {
            'id': tile_id,
            'lane': lane,
            'base_points': base_points,
            'state': self.SPAWNED,
            'was_in_hit_zone': False,  # Pernah masuk hit zone?
            'spawned_at': None,  # Timestamp saat spawn (opsional)
        }
        
        self.tracked_tiles[tile_id] = tile_info
        print(f"✅ Tile spawned: id={tile_id}, lane={lane}")
        
        return tile_info
    
    def enter_hit_zone(self, tile_id):
        """
        Tile memasuki hit zone (bisa di-tap sekarang).
        
        Args:
            tile_id (int): Tile ID yang memasuki hit zone
            
        Returns:
            bool: True jika berhasil, False jika tile tidak ada
        """
        if tile_id not in self.tracked_tiles:
            print(f"⚠️  Tile {tile_id} tidak ditemukan!")
            return False
        
        tile = self.tracked_tiles[tile_id]
        tile['state'] = self.ACTIVE
        tile['was_in_hit_zone'] = True
        
        print(f"📍 Tile {tile_id} entered hit zone")
        return True
    
    def on_tile_hit(self, tile_id):
        """
        Tile berhasil di-tap.
        
        Args:
            tile_id (int): Tile ID yang di-tap
            
        Returns:
            bool: True jika berhasil hit, False jika sudah missed/tidak ada
        """
        if tile_id not in self.tracked_tiles:
            print(f"⚠️  Tile {tile_id} tidak ditemukan!")
            return False
        
        tile = self.tracked_tiles[tile_id]
        
        # Cek apakah tile dalam state yang bisa di-hit
        if tile['state'] not in [self.SPAWNED, self.ACTIVE]:
            print(f"⚠️  Tile {tile_id} tidak bisa di-hit (state: {tile['state']})")
            return False
        
        tile['state'] = self.HIT
        self.total_hits += 1  # Track hit
        print(f"✅ Tile {tile_id} HIT!")
        
        return True
    
    def exit_hit_zone(self, tile_id):
        """
        Tile keluar dari hit zone (lewat tanpa di-tap).
        
        Args:
            tile_id (int): Tile ID yang keluar hit zone
            
        Returns:
            dict: Miss info (dengan tile details), atau None jika tidak applicable
        """
        if tile_id not in self.tracked_tiles:
            print(f"⚠️  Tile {tile_id} tidak ditemukan!")
            return None
        
        tile = self.tracked_tiles[tile_id]
        
        # Hanya miss jika tile pernah dalam hit zone dan belum di-hit
        if tile['was_in_hit_zone'] and tile['state'] != self.HIT:
            tile['state'] = self.MISSED
            self.missed_tiles.append(tile_id)
            self.total_misses += 1
            
            miss_info = {
                'tile_id': tile_id,
                'lane': tile['lane'],
                'base_points': tile['base_points'],
                'is_miss': True
            }
            
            print(f"❌ Tile {tile_id} MISSED! (Lewat tanpa di-tap)")
            return miss_info
        else:
            # Tile belum masuk hit zone atau sudah di-hit, bukan miss
            print(f"⏭️  Tile {tile_id} passed (tidak dihitung miss)")
            return None
    
    def destroy_tile(self, tile_id):
        """
        Hapus tile dari tracking.
        
        Args:
            tile_id (int): Tile ID yang dihapus
            
        Returns:
            bool: True jika berhasil destroy
        """
        if tile_id not in self.tracked_tiles:
            print(f"⚠️  Tile {tile_id} tidak ditemukan!")
            return False
        
        tile = self.tracked_tiles[tile_id]
        tile['state'] = self.DESTROYED
        del self.tracked_tiles[tile_id]
        
        print(f"🗑️  Tile {tile_id} destroyed")
        return True
    
    # ============ STATUS & QUERY ============
    
    def get_active_tiles(self):
        """
        Get list of tiles yang sedang active (dalam hit zone).
        
        Returns:
            list: List of active tile IDs
        """
        return [tile_id for tile_id, tile in self.tracked_tiles.items() 
                if tile['state'] == self.ACTIVE]
    
    def get_tile_info(self, tile_id):
        """
        Get info lengkap tentang tile tertentu.
        
        Args:
            tile_id (int): Tile ID
            
        Returns:
            dict: Tile info, atau None jika tile tidak ada
        """
        if tile_id not in self.tracked_tiles:
            return None
        
        return self.tracked_tiles[tile_id].copy()
    
    def get_status(self):
        """
        Get status detector lengkap.
        
        Returns:
            dict: Dictionary berisi:
                - total_spawned (int): Total tiles yang pernah spawn
                - total_active (int): Tiles yang sedang active
                - total_missed (int): Total tiles yang missed
                - total_hit (int): Total tiles yang berhasil di-hit
                - miss_rate (float): Persentase miss (0-100%)
                - accuracy (float): Persentase hit (0-100%)
        """
        total_completed = self.total_misses + self.total_hits
        
        accuracy = 0.0
        miss_rate = 0.0
        
        if total_completed > 0:
            accuracy = (self.total_hits / total_completed) * 100
            miss_rate = (self.total_misses / total_completed) * 100
        
        return {
            'total_spawned': self.total_tiles_spawned,
            'total_active': len(self.get_active_tiles()),
            'total_missed': self.total_misses,
            'total_completed': total_completed,
            'miss_rate': miss_rate,
            'accuracy': accuracy
        }
    
    def get_missed_tiles(self):
        """
        Get list of tiles yang missed.
        
        Returns:
            list: List of missed tile IDs
        """
        return self.missed_tiles.copy()
    
    def reset(self):
        """
        Reset detector untuk game baru.
        """
        self.tracked_tiles = {}
        self.missed_tiles = []
        self.total_tiles_spawned = 0
        self.total_misses = 0
        self.total_hits = 0
        
        print("🔄 MissDetector reset")
    
    # ============ DEBUG / DISPLAY ============
    
    def __str__(self):
        """String representation untuk debugging."""
        status = self.get_status()
        return f"MissDetector: Spawned={status['total_spawned']}, Active={status['total_active']}, Missed={status['total_missed']}, Accuracy={status['accuracy']:.1f}%"
    
    def print_status(self):
        """Print status lengkap."""
        status = self.get_status()
        print("\n" + "="*60)
        print("MISS DETECTOR STATUS")
        print("="*60)
        print(f"Total Spawned: {status['total_spawned']}")
        print(f"Total Active: {status['total_active']}")
        print(f"Total Missed: {status['total_missed']}")
        print(f"Total Completed: {status['total_completed']}")
        print(f"Miss Rate: {status['miss_rate']:.1f}%")
        print(f"Accuracy: {status['accuracy']:.1f}%")
        print("="*60 + "\n")
    
    def print_all_tiles(self):
        """Print semua tiles yang tracked."""
        print("\n" + "="*60)
        print("ALL TRACKED TILES")
        print("="*60)
        for tile_id, tile in self.tracked_tiles.items():
            print(f"Tile {tile_id}: state={tile['state']}, lane={tile['lane']}")
        print("="*60 + "\n")

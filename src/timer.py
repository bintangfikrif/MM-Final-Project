"""
Timer Module

"""

import time


class Timer:
    """
    Mengelola game time dengan dua mode:
    1. COUNTDOWN: Mulai dari X detik, hitung mundur ke 0 (game over saat time up)
    2. STOPWATCH: Mulai dari 0, hitung naik (track durasi bermain)
    
    Support pause/resume functionality.
    """
    
    # ============ TIMER MODES ============
    COUNTDOWN = "COUNTDOWN"  # Hitung mundur ke 0
    STOPWATCH = "STOPWATCH"  # Hitung naik dari 0
    
    def __init__(self, duration=60, mode=STOPWATCH):
        """
        Initialize timer.
        
        Args:
            duration (int): Durasi timer dalam detik
                           - COUNTDOWN: total detik yang tersedia
                           - STOPWATCH: tidak digunakan
            mode (str): COUNTDOWN atau STOPWATCH
        """
        self.duration = duration           # Total durasi dalam detik
        self.mode = mode                   # Tipe timer
        
        self.start_time = None             # Waktu timer dimulai
        self.pause_time = None             # Waktu timer di-pause
        self.total_paused = 0              # Total waktu yang di-pause
        
        self.is_running = False            # Timer sedang berjalan?
        self.is_paused = False             # Timer sedang di-pause?
        self.is_finished = False           # Timer sudah selesai?
        
        print("✅ Timer initialized")
        print(f"   Mode: {self.mode}")
        print(f"   Duration: {self.duration}s")
    
    # ============ TIMER CONTROL ============
    
    def start(self):
        """
        Mulai timer.
        
        Returns:
            bool: True jika berhasil start, False jika sudah running
        """
        if self.is_running and not self.is_paused:
            print("⚠️  Timer sudah berjalan!")
            return False
        
        if self.is_paused:
            # Resume dari pause
            self.pause_time = None
            self.is_paused = False
            print("▶️  Timer resumed")
            return True
        
        # Start fresh
        self.start_time = time.time()
        self.total_paused = 0
        self.is_running = True
        self.is_finished = False
        
        print(f"▶️  Timer started ({self.mode})")
        return True
    
    def pause(self):
        """
        Pause timer.
        
        Returns:
            bool: True jika berhasil pause, False jika tidak running
        """
        if not self.is_running or self.is_paused:
            print("⚠️  Tidak bisa pause - timer tidak berjalan")
            return False
        
        self.pause_time = time.time()
        self.is_paused = True
        
        elapsed = self.get_elapsed_time()
        print(f"⏸️  Timer paused at {elapsed:.1f}s")
        return True
    
    def stop(self):
        """
        Stop timer sepenuhnya.
        
        Returns:
            dict: Final timer status
        """
        self.is_running = False
        self.is_paused = False
        
        elapsed = self.get_elapsed_time()
        print(f"⏹️  Timer stopped at {elapsed:.1f}s")
        
        return self.get_status()
    
    # ============ TIME TRACKING ============
    
    def get_elapsed_time(self):
        """
        Hitung waktu yang sudah berlalu sejak timer dimulai.
        Mempertimbangkan waktu pause.
        
        Returns:
            float: Waktu yang berlalu dalam detik (>= 0)
        """
        if not self.is_running:
            return 0.0
        
        current_time = time.time()
        
        # Jika sedang pause, gunakan pause_time
        if self.is_paused:
            current_time = self.pause_time
        
        # Hitung elapsed (dikurangi total pause time)
        elapsed = (current_time - self.start_time) - self.total_paused
        
        return max(0.0, elapsed)
    
    def get_remaining_time(self):
        """
        Hitung waktu tersisa (untuk mode COUNTDOWN).
        
        Returns:
            float: Waktu tersisa dalam detik
                   - Positif: masih ada waktu
                   - 0: waktu habis (game over)
        """
        if self.mode != self.COUNTDOWN:
            return None
        
        elapsed = self.get_elapsed_time()
        remaining = self.duration - elapsed
        
        return max(0.0, remaining)
    
    def is_time_up(self):
        """
        Cek apakah waktu sudah habis (untuk COUNTDOWN mode).
        
        Returns:
            bool: True jika remaining time <= 0
        """
        if self.mode != self.COUNTDOWN:
            return False
        
        remaining = self.get_remaining_time()
        return remaining <= 0
    
    def get_time_percentage(self):
        """
        Hitung progress dalam persentase (0-100%).
        
        Returns:
            float: Persentase (0.0 - 100.0)
        """
        if self.mode == self.COUNTDOWN:
            remaining = self.get_remaining_time()
            percentage = (remaining / self.duration) * 100
        else:  # STOPWATCH
            elapsed = self.get_elapsed_time()
            percentage = (elapsed / self.duration) * 100
        
        return max(0.0, min(100.0, percentage))
    
    # ============ FORMAT DISPLAY ============
    
    def format_time(self, seconds=None):
        """
        Format detik menjadi string MM:SS.
        
        Args:
            seconds (float): Detik yang diformat. Jika None, gunakan current time.
            
        Returns:
            str: Format MM:SS (e.g., "01:30", "00:45")
        """
        if seconds is None:
            if self.mode == self.COUNTDOWN:
                seconds = self.get_remaining_time()
            else:
                seconds = self.get_elapsed_time()
        
        seconds = max(0, int(seconds))
        minutes = seconds // 60
        secs = seconds % 60
        
        return f"{minutes:02d}:{secs:02d}"
    
    def get_display_time(self):
        """
        Hitung waktu untuk ditampilkan di UI dalam format MM:SS.
        
        Returns:
            str: Format MM:SS
        """
        if self.mode == self.COUNTDOWN:
            return self.format_time(self.get_remaining_time())
        else:
            return self.format_time(self.get_elapsed_time())
    
    # ============ STATUS METHODS ============
    
    def get_status(self):
        """
        Hitung status lengkap timer.
        
        Returns:
            dict: Dictionary berisi:
                - mode (str): COUNTDOWN atau STOPWATCH
                - is_running (bool): Timer sedang berjalan
                - is_paused (bool): Timer sedang pause
                - is_finished (bool): Timer sudah selesai
                - elapsed (float): Waktu yang telah berlalu
                - remaining (float): Waktu tersisa (COUNTDOWN only)
                - percentage (float): Progress persentase
                - display_time (str): Format MM:SS untuk UI
        """
        return {
            'mode': self.mode,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'is_finished': self.is_finished,
            'elapsed': self.get_elapsed_time(),
            'remaining': self.get_remaining_time() if self.mode == self.COUNTDOWN else None,
            'percentage': self.get_time_percentage(),
            'display_time': self.get_display_time()
        }
    
    def reset(self):
        """
        Reset timer ke state awal.
        """
        self.start_time = None
        self.pause_time = None
        self.total_paused = 0
        self.is_running = False
        self.is_paused = False
        self.is_finished = False
        
        print(f"🔄 Timer reset ({self.mode})")
    
    # ============ DEBUG / DISPLAY ============
    
    def __str__(self):
        """String representation untuk debugging."""
        return f"Timer: {self.get_display_time()} ({self.mode})"
    
    def print_status(self):
        """Print status lengkap."""
        status = self.get_status()
        print(f"Mode: {status['mode']}")
        print(f"Running: {status['is_running']} | Paused: {status['is_paused']}")
        print(f"Display: {status['display_time']}")
        print(f"Progress: {status['percentage']:.1f}%")
        if status['remaining'] is not None:
            print(f"Remaining: {status['remaining']:.1f}s")

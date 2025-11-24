import random
class Song:
    """Represents a song with note sequence"""
    
    def __init__(self, name, bpm, notes_sequence):
        self.name = name
        self.bpm = bpm
        self.notes_sequence = notes_sequence
    
    def get_notes_with_timing(self, bpm_multiplier=1.0):
        """
        Convert beat numbers to actual time with BPM multiplier.
        Smart lane assignment: Random but avoid consecutive same lanes.
        """
        adjusted_bpm = self.bpm * bpm_multiplier
        seconds_per_beat = 60.0 / adjusted_bpm
        
        tiles = []
        previous_lane = -1  # Track last lane used
        
        for note, beat in self.notes_sequence:
            time = beat * seconds_per_beat
            
            # ✅ SMART LANE ASSIGNMENT
            # Avoid using same lane consecutively for better gameplay
            available_lanes = [0, 1, 2, 3]
            
            # Remove previous lane from options (if exists)
            if previous_lane != -1:
                available_lanes.remove(previous_lane)
            
            # Choose random lane from remaining options
            lane = random.choice(available_lanes)
            previous_lane = lane
            
            tiles.append({
                'note': note,
                'time': time,
                'lane': lane
            })
        
        return tiles
# ============ SONG DATABASE ============

# Twinkle Twinkle Little Star (C Major, 4/4 time)
TWINKLE_TWINKLE = Song(
    name="Twinkle Twinkle Little Star",
    bpm=120, 
    notes_sequence=[
        # Intro delay (4 beats = ~2 seconds at 120 BPM)
        
        # First verse
        ('C', 4),
        ('C', 6),
        ('G', 8),
        ('G', 10),
        ('A', 12),
        ('A', 14),
        ('G', 16),
        
        # Second phrase
        ('F', 20),
        ('F', 22),
        ('E', 24),
        ('E', 26),
        ('D', 28),
        ('D', 30),
        ('C', 32),
        
        # Third phrase
        ('G', 36),
        ('G', 38),
        ('F', 40),
        ('F', 42),
        ('E', 44),
        ('E', 46),
        ('D', 48),
        
        # Fourth phrase
        ('G', 52),
        ('G', 54),
        ('F', 56),
        ('F', 58),
        ('E', 60),
        ('E', 62),
        ('D', 64),
        
        # Repeat first verse
        ('C', 68),
        ('C', 70),
        ('G', 72),
        ('G', 74),
        ('A', 76),
        ('A', 78),
        ('G', 80),
        
        # Final phrase
        ('F', 84),
        ('F', 86),
        ('E', 88),
        ('E', 90),
        ('D', 92),
        ('D', 94),
        ('C', 96),
    ]
)


# Mary Had a Little Lamb (simpler, good for EASY mode)
MARY_HAD_A_LITTLE_LAMB = Song(
    name="Mary Had a Little Lamb",
    bpm=100,
    notes_sequence=[
        # "Mary had a little lamb"
        ('E', 0),
        ('D', 1),
        ('C', 2),
        ('D', 3),
        ('E', 4),
        ('E', 5),
        ('E', 6),
        
        # "Little lamb, little lamb"
        ('D', 8),
        ('D', 9),
        ('D', 10),
        ('E', 12),
        ('G', 13),
        ('G', 14),
        
        # "Mary had a little lamb"
        ('E', 16),
        ('D', 17),
        ('C', 18),
        ('D', 19),
        ('E', 20),
        ('E', 21),
        ('E', 22),
        ('E', 23),
        
        # "Its fleece was white as snow"
        ('D', 24),
        ('D', 25),
        ('E', 26),
        ('D', 27),
        ('C', 28),
    ]
)


# Happy Birthday (popular, medium difficulty)
HAPPY_BIRTHDAY = Song(
    name="Happy Birthday",
    bpm=120,
    notes_sequence=[
        # "Happy birthday to you"
        ('C', 0),
        ('C', 1),
        ('D', 2),
        ('C', 3),
        ('F', 4),
        ('E', 6),
        
        # "Happy birthday to you"
        ('C', 8),
        ('C', 9),
        ('D', 10),
        ('C', 11),
        ('G', 12),
        ('F', 14),
        
        # "Happy birthday dear [name]"
        ('C', 16),
        ('C', 17),
        ('C_high', 18),
        ('A', 19),
        ('F', 20),
        ('E', 21),
        ('D', 22),
        
        # "Happy birthday to you"
        ('B', 24),
        ('B', 25),
        ('A', 26),
        ('F', 27),
        ('G', 28),
        ('F', 30),
    ]
)


# Jingle Bells (faster tempo, good for HARD/EXPERT)
JINGLE_BELLS = Song(
    name="Jingle Bells",
    bpm=140,
    notes_sequence=[
        # "Jingle bells, jingle bells"
        ('E', 0),
        ('E', 1),
        ('E', 2),
        ('E', 4),
        ('E', 5),
        ('E', 6),
        
        # "Jingle all the way"
        ('E', 8),
        ('G', 9),
        ('C', 10),
        ('D', 11),
        ('E', 12),
        
        # "Oh what fun it is to ride"
        ('F', 16),
        ('F', 17),
        ('F', 18),
        ('F', 19),
        ('F', 20),
        ('E', 21),
        ('E', 22),
        ('E', 23),
        
        # "In a one horse open sleigh"
        ('E', 24),
        ('D', 25),
        ('D', 26),
        ('E', 27),
        ('D', 28),
        ('G', 30),
    ]
)


# ============ SONG MANAGER ============

class SongManager:
    """Manages available songs and difficulty settings"""
    
    def __init__(self):
        """Initialize with all available songs"""
        self.songs = {
            'twinkle': TWINKLE_TWINKLE,
            'mary': MARY_HAD_A_LITTLE_LAMB,
            'birthday': HAPPY_BIRTHDAY,
            'jingle': JINGLE_BELLS
        }
        
        # Difficulty → BPM multipliers
        self.difficulty_bpm_multipliers = {
            'EASY': 0.9,      # 90% speed
            'MEDIUM': 1.0,    # 100% original speed
            'HARD': 1.1,      # 110% speed
            'EXPERT': 1.3     # 130% speed
        }
        
        self.current_song = 'twinkle'
        self.current_difficulty = 'MEDIUM'
        
        print("✅ SongManager initialized")
        print(f"   Available songs: {len(self.songs)}")
    
    def get_song(self, song_id):
        """Get song by ID"""
        return self.songs.get(song_id)
    
    def get_all_songs(self):
        """Get list of all song IDs and names"""
        return [(song_id, song.name) for song_id, song in self.songs.items()]
    
    def set_current_song(self, song_id):
        """Set current song"""
        if song_id in self.songs:
            self.current_song = song_id
            print(f"🎵 Song selected: {self.songs[song_id].name}")
            return True
        return False
    
    def set_difficulty(self, difficulty):
        """Set difficulty"""
        if difficulty in self.difficulty_bpm_multipliers:
            self.current_difficulty = difficulty
            print(f"🎚️  Difficulty: {difficulty}")
            return True
        return False
    
    def get_current_song_tiles(self):
        """
        Get tiles for current song with current difficulty.
        
        Returns:
            list: List of tile data with note, time, lane
        """
        song = self.songs[self.current_song]
        bpm_multiplier = self.difficulty_bpm_multipliers[self.current_difficulty]
        
        tiles = song.get_notes_with_timing(bpm_multiplier)
        
        print(f"📋 Generated {len(tiles)} tiles for {song.name} ({self.current_difficulty})")
        return tiles
    
    def get_song_duration(self):
        """Get duration of current song in seconds"""
        tiles = self.get_current_song_tiles()
        if tiles:
            return tiles[-1]['time'] + 2.0  # Add 2s buffer
        return 60.0
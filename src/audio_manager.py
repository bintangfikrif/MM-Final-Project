import pygame
import os

class AudioManager:
    """
    Audio Manager for AirBeats - Interactive Piano Tiles Game
    
    Features:
    - Load and play 8 piano notes (C, D, E, F, G, A, B, C_high)
    - Volume control for piano notes
    - Low-latency audio playback for rhythm game
    """
    
    def __init__(self):
        """Initialize pygame mixer with low-latency settings"""
        # Low-latency audio configuration
        pygame.mixer.pre_init(
            frequency=44100,    # CD quality sample rate
            size=-16,           # 16-bit audio
            channels=2,         # Stereo
            buffer=128          # Small buffer for minimal latency
        )
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)  # Allow multiple simultaneous sounds
        
        # Storage for piano notes
        self.notes = {}
        self.notes_volume = 1.0  # Default maximum volume
        
        # Load all piano notes
        self.load_notes()
        print("✅ Audio Manager initialized!")
    
    def load_notes(self):
        """Load all piano note sound files"""
        notes_path = "assets/sounds/piano"
        note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C_high']
        
        print("\n📁 Loading piano notes...")
        
        for note in note_names:
            filepath = os.path.join(notes_path, f"{note}.wav")
            if os.path.exists(filepath):
                self.notes[note] = pygame.mixer.Sound(filepath)
                self.notes[note].set_volume(self.notes_volume)
                print(f"  ✅ {note}.wav")
            else:
                print(f"  ❌ {filepath} not found!")
        
        print(f"\n✅ Loaded {len(self.notes)}/8 piano notes\n")
    
    def play_note(self, note_name):
        """
        Play a piano note by name.
        This is called when player successfully hits a tile.
        
        Args:
            note_name (str): Note to play (C, D, E, F, G, A, B, C_high)
        
        Example:
            audio.play_note('C')  # Play note C
            audio.play_note('G')  # Play note G
        """
        if note_name in self.notes:
            # Stop any previous instance to avoid overlap
            self.notes[note_name].stop()
            # Play the note (auto-stop after 2 seconds)
            self.notes[note_name].play(maxtime=2000)
        else:
            print(f"⚠️  Note '{note_name}' not found!")
    
    def set_notes_volume(self, volume):
        """
        Set piano notes volume.
        
        Args:
            volume (float): Volume level from 0.0 (silent) to 1.0 (maximum)
        
        Example:
            audio.set_notes_volume(0.8)  # Set to 80%
            audio.set_notes_volume(1.0)  # Set to maximum
        """
        # Clamp volume between 0.0 and 1.0
        self.notes_volume = max(0.0, min(1.0, volume))
        
        # Apply to all loaded notes
        for note in self.notes.values():
            note.set_volume(self.notes_volume)
    
    def get_notes_volume(self):
        """
        Get current piano notes volume.
        
        Returns:
            float: Current volume level (0.0 to 1.0)
        """
        return self.notes_volume
    
    def cleanup(self):
        """Clean up audio resources"""
        pygame.mixer.quit()
        print("🔇 Audio Manager cleaned up")
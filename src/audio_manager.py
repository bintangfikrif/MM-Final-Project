import pygame
import os

class AudioManager:
    """
    Manages all audio for AirBeats game.
    Handles piano notes, background music, and sound effects.
    """
    
    def __init__(self):
        """Initialize pygame mixer and load all sounds"""
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        self.notes = {}
        self.sfx = {}
        self.bgm_volume = 0.5
        self.sfx_volume = 0.7
        self.notes_volume = 0.8
        
        self.load_notes()
        print("✅ Audio Manager initialized!")
    
    def load_notes(self):
        """Load all piano note sound files"""
        notes_path = "assets/sounds/piano"
        note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C_high']
        
        for note in note_names:
            filepath = os.path.join(notes_path, f"{note}.wav")
            if os.path.exists(filepath):
                self.notes[note] = pygame.mixer.Sound(filepath)
                self.notes[note].set_volume(self.notes_volume)
                print(f"  Loaded: {note}.wav")
            else:
                print(f"  ⚠️  Missing: {filepath}")
        
        print(f"✅ Loaded {len(self.notes)} piano notes")
    
    def play_note(self, note_name):
        """
        Play a piano note by name.
        
        Args:
            note_name (str): Name of note to play (C, D, E, F, G, A, B, C_high)
        """
        if note_name in self.notes:
            self.notes[note_name].play()
        else:
            print(f"⚠️  Note '{note_name}' not found!")
    
    def play_bgm(self, music_file):
        """
        Play background music on loop.
        
        Args:
            music_file (str): Path to music file
        """
        if os.path.exists(music_file):
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.set_volume(self.bgm_volume)
            pygame.mixer.music.play(-1)  # Loop forever
            print(f"🎵 Playing BGM: {music_file}")
        else:
            print(f"⚠️  Music file not found: {music_file}")
    
    def stop_bgm(self):
        """Stop background music"""
        pygame.mixer.music.stop()
    
    def set_bgm_volume(self, volume):
        """
        Set background music volume.
        
        Args:
            volume (float): Volume level (0.0 to 1.0)
        """
        self.bgm_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.bgm_volume)
    
    def set_notes_volume(self, volume):
        """
        Set piano notes volume.
        
        Args:
            volume (float): Volume level (0.0 to 1.0)
        """
        self.notes_volume = max(0.0, min(1.0, volume))
        for note in self.notes.values():
            note.set_volume(self.notes_volume)
    
    def cleanup(self):
        """Clean up audio resources"""
        pygame.mixer.quit()
        print("🔇 Audio Manager cleaned up")
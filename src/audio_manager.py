import pygame
import os

class AudioManager:
    def __init__(self):
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
        print("Audio Manager initialized!")
    
    def load_notes(self):
        notes_path = "assets/sounds/piano"
        note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C_high']
        
        print("\nLoading piano notes...")
        
        for note in note_names:
            filepath = os.path.join(notes_path, f"{note}.wav")
            if os.path.exists(filepath):
                self.notes[note] = pygame.mixer.Sound(filepath)
                self.notes[note].set_volume(self.notes_volume)
                print(f"  {note}.wav")
            else:
                print(f"  {filepath} not found!")
        
        print(f"\nLoaded {len(self.notes)}/8 piano notes\n")
    
    def play_note(self, note_name):
        if note_name in self.notes:
            # Stop any previous instance to avoid overlap
            self.notes[note_name].stop()
            # Play the note (auto-stop after 2 seconds)
            self.notes[note_name].play(maxtime=2000)
        else:
            print(f"⚠️  Note '{note_name}' not found!")
    
    def set_notes_volume(self, volume):
        # Clamp volume between 0.0 and 1.0
        self.notes_volume = max(0.0, min(1.0, volume))
        
        # Apply to all loaded notes
        for note in self.notes.values():
            note.set_volume(self.notes_volume)
    
    def get_notes_volume(self):
        return self.notes_volume
    
    def play_music(self, song_name):
        # Try to load music file
        music_path = f"assets/sounds/music/{song_name}.mp3"
        if not os.path.exists(music_path):
            music_path = f"assets/sounds/music/{song_name}.wav"
        
        if os.path.exists(music_path):
            try:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play(-1) # Loop indefinitely
                pygame.mixer.music.set_volume(0.5) # 50% volume for BGM
                print(f"Playing music: {song_name}")
            except Exception as e:
                print(f"Error playing music: {e}")
        else:
            print(f"Music file not found: {song_name} (checked .mp3 and .wav in assets/sounds/music/)")

    def stop_music(self):
        pygame.mixer.music.stop()
        print("Music stopped")

    def cleanup(self):
        pygame.mixer.quit()
        print("Audio Manager cleaned up")
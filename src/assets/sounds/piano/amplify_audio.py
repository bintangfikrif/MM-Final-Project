from pydub import AudioSegment
import os

def amplify_audio_file(input_file, gain_db=6.0):
    try:
        audio = AudioSegment.from_wav(input_file)
        
        original_db = audio.dBFS
        
        amplified = audio + gain_db
        
        new_db = amplified.dBFS
        
        amplified.export(input_file, format="wav")
        
        return True, original_db, new_db
        
    except Exception as e:
        print(f"Error: {e}")
        return False, 0, 0

def amplify_all_piano_notes(gain_db=6.0):
    piano_folder = "assets/sounds/piano"
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C_high']
    
    print(f"Amplifying piano notes by {gain_db} dB...")
    print("="*60)
    print(f"Note: 3dB ≈ 40% louder | 6dB ≈ 2x louder | 9dB ≈ 3x louder\n")
    
    success_count = 0
    
    for note in notes:
        audio_file = os.path.join(piano_folder, f"{note}.wav")
        
        if not os.path.exists(audio_file):
            print(f"{note}.wav not found, skipping...")
            continue
        
        print(f"{note}.wav:")
        
        success, original_db, new_db = amplify_audio_file(audio_file, gain_db)
        
        if success:
            print(f"Original: {original_db:.1f} dBFS")
            print(f"New:      {new_db:.1f} dBFS")
            print(f"Boost:    +{gain_db} dB")
            print(f"Amplified!\n")
            success_count += 1
        else:
            print(f"Failed!\n")
    
    print("="*60)
    print(f"Amplification complete! {success_count}/{len(notes)} files amplified")
    print("\nNow test with test_audio.py - notes should be much louder!")

if __name__ == "__main__":
    amplify_all_piano_notes(gain_db=6.0)  
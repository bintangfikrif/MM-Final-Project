import pygame
from src.audio_manager import AudioManager

def main():
    pygame.init()
    audio = AudioManager()
    
    print("\nControls:")
    print("  1-8: Play notes")
    print("  UP/DOWN: Volume")
    print("  ESC: Quit\n")
    
    screen = pygame.display.set_mode((800, 500))
    pygame.display.set_caption("AirBeats Audio Test")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.Font(None, 80)
    font_medium = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    running = True
    last_note = ""
    flash_timer = 0
    
    while running:
        dt = clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    audio.play_note('C')
                    last_note = "C"
                    flash_timer = 200
                elif event.key == pygame.K_2:
                    audio.play_note('D')
                    last_note = "D"
                    flash_timer = 200
                elif event.key == pygame.K_3:
                    audio.play_note('E')
                    last_note = "E"
                    flash_timer = 200
                elif event.key == pygame.K_4:
                    audio.play_note('F')
                    last_note = "F"
                    flash_timer = 200
                elif event.key == pygame.K_5:
                    audio.play_note('G')
                    last_note = "G"
                    flash_timer = 200
                elif event.key == pygame.K_6:
                    audio.play_note('A')
                    last_note = "A"
                    flash_timer = 200
                elif event.key == pygame.K_7:
                    audio.play_note('B')
                    last_note = "B"
                    flash_timer = 200
                elif event.key == pygame.K_8:
                    audio.play_note('C_high')
                    last_note = "C (high)"
                    flash_timer = 200
                elif event.key == pygame.K_UP:
                    new_vol = min(1.0, audio.get_notes_volume() + 0.1)
                    audio.set_notes_volume(new_vol)
                    last_note = f"Vol: {int(new_vol*100)}%"
                    flash_timer = 200
                elif event.key == pygame.K_DOWN:
                    new_vol = max(0.0, audio.get_notes_volume() - 0.1)
                    audio.set_notes_volume(new_vol)
                    last_note = f"Vol: {int(new_vol*100)}%"
                    flash_timer = 200
        
        if flash_timer > 0:
            flash_timer -= dt
        
        if flash_timer > 0:
            screen.fill((70, 80, 100))
        else:
            screen.fill((25, 30, 40))
        
        note_text = font_large.render(last_note if last_note else "Press 1-8", True, (255, 255, 255))
        note_rect = note_text.get_rect(center=(400, 220))
        screen.blit(note_text, note_rect)
        
        volume_pct = int(audio.get_notes_volume() * 100)
        volume_text = font_medium.render(f"Volume: {volume_pct}%", True, (150, 200, 150))
        volume_rect = volume_text.get_rect(center=(400, 320))
        screen.blit(volume_text, volume_rect)
        
        controls_text = font_small.render("1-8: Notes | UP/DOWN: Volume | ESC: Quit", True, (100, 100, 100))
        screen.blit(controls_text, (20, 20))
        
        mapping = "1=C  2=D  3=E  4=F  5=G  6=A  7=B  8=C(high)"
        mapping_text = font_small.render(mapping, True, (80, 80, 80))
        mapping_rect = mapping_text.get_rect(center=(400, 460))
        screen.blit(mapping_text, mapping_rect)
        
        pygame.display.flip()
    
    audio.cleanup()
    pygame.quit()
    print("\nTest complete!")

if __name__ == "__main__":
    main()
import pygame
from src.audio_manager import AudioManager

def main():
    """Test Audio Manager"""
    print("=== Testing Audio Manager ===\n")
    

    pygame.init()
    

    audio = AudioManager()
    
    print("\n=== Testing Piano Notes ===")
    print("Press keys to play notes:")
    print("  1 = C")
    print("  2 = D")
    print("  3 = E")
    print("  4 = F")
    print("  5 = G")
    print("  6 = A")
    print("  7 = B")
    print("  8 = C (high)")
    print("  ESC = Quit")
    
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Audio Manager Test")
    clock = pygame.time.Clock()
    
    font = pygame.font.Font(None, 36)
    
    running = True
    last_note = "Press a key (1-8)"
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    audio.play_note('C')
                    last_note = "Playing: C"
                elif event.key == pygame.K_2:
                    audio.play_note('D')
                    last_note = "Playing: D"
                elif event.key == pygame.K_3:
                    audio.play_note('E')
                    last_note = "Playing: E"
                elif event.key == pygame.K_4:
                    audio.play_note('F')
                    last_note = "Playing: F"
                elif event.key == pygame.K_5:
                    audio.play_note('G')
                    last_note = "Playing: G"
                elif event.key == pygame.K_6:
                    audio.play_note('A')
                    last_note = "Playing: A"
                elif event.key == pygame.K_7:
                    audio.play_note('B')
                    last_note = "Playing: B"
                elif event.key == pygame.K_8:
                    audio.play_note('C_high')
                    last_note = "Playing: C (high)"
        
        screen.fill((30, 30, 30))
        text = font.render(last_note, True, (255, 255, 255))
        text_rect = text.get_rect(center=(200, 100))
        screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    

    audio.cleanup()
    pygame.quit()
    print("\nTest complete!")

if __name__ == "__main__":
    main()
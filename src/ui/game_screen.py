"""
Game Screen Module

Screen untuk gameplay utama (Piano Tiles).
Menampilkan:
- Camera feed background
- Falling tiles
- Score & Combo HUD
- Visual feedback (taps)

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
try:
    from ui.base_screen import BaseScreen
    from components.color import TEXT_WHITE, ACCENT_ORANGE
except ImportError:
    from base_screen import BaseScreen
    # Fallback colors
    TEXT_WHITE = (255, 255, 255)
    ACCENT_ORANGE = (255, 165, 0)

class GameScreen(BaseScreen):
    """
    Screen untuk gameplay utama.
    """
    
    def __init__(self, width, height, game_manager=None):
        super().__init__(width, height, game_manager)
        
        # Fonts
        self.font_score = pygame.font.Font(None, 48)
        self.font_combo = pygame.font.Font(None, 64)
        self.font_timer = pygame.font.Font(None, 36)
        
        print("✅ GameScreen initialized")

    def handle_event(self, event):
        """
        Handle gameplay events.
        """
        if event.type == pygame.KEYDOWN:
            # Pause game
            if event.key == pygame.K_p or event.key == pygame.K_ESCAPE:
                if self.game_manager:
                    self.game_manager.pause_game()
                return "PAUSED"
                
        return None

    def update(self):
        """
        Update gameplay logic.
        Note: Logic utama ada di GameManager.update(), screen ini lebih ke visual.
        """
        pass

    def draw(self, surface):
        """
        Draw gameplay ke surface.
        """
        # 1. Draw Camera Feed Background
        if self.game_manager and hasattr(self.game_manager, 'current_frame_surface'):
            if self.game_manager.current_frame_surface:
                # Scale frame to fit screen if needed
                # Assuming frame is already correct size or we blit directly
                surface.blit(self.game_manager.current_frame_surface, (0, 0))
            else:
                surface.fill((0, 0, 0)) # Fallback black
        else:
            surface.fill((20, 20, 20)) # Fallback dark gray

        # 2. Draw Tiles
        if self.game_manager and self.game_manager.tile_manager:
            self.game_manager.tile_manager.draw(surface)

        # 3. Draw HUD
        self._draw_hud(surface)
        
        # 4. Draw Visual Feedback (Taps)
        # Ini bisa diambil dari game_manager.latest_taps atau sejenisnya
        # Untuk sekarang kita skip dulu, nanti diintegrasikan

    def _draw_hud(self, surface):
        """Draw Score, Combo, Timer."""
        if not self.game_manager:
            return

        # Score (Top Left)
        score_text = f"Score: {self.game_manager.score_manager.total_score}"
        score_surf = self.font_score.render(score_text, True, TEXT_WHITE)
        surface.blit(score_surf, (20, 20))

        # Timer (Top Right)
        timer_text = self.game_manager.timer.get_display_time()
        timer_surf = self.font_timer.render(timer_text, True, TEXT_WHITE)
        timer_rect = timer_surf.get_rect(topright=(self.width - 20, 20))
        surface.blit(timer_surf, timer_rect)

        # Combo (Center, if active)
        combo = self.game_manager.combo_counter.current_combo
        if combo > 0:
            combo_text = f"{combo}x"
            # Scale combo text based on combo size (simple effect)
            scale = min(1.5, 1.0 + (combo / 50.0))
            
            # Render text
            combo_surf = self.font_combo.render(combo_text, True, ACCENT_ORANGE)
            
            # Calculate center position
            center_x = self.width // 2
            center_y = self.height // 2 - 50
            
            combo_rect = combo_surf.get_rect(center=(center_x, center_y))
            surface.blit(combo_surf, combo_rect)
            
            # "COMBO" label below number
            label_surf = self.font_timer.render("COMBO", True, TEXT_WHITE)
            label_rect = label_surf.get_rect(center=(center_x, center_y + 40))
            surface.blit(label_surf, label_rect)

    def on_enter(self):
        super().on_enter()
        print("🎮 Entered GameScreen")

    def on_exit(self):
        super().on_exit()
        print("🎮 Exited GameScreen")

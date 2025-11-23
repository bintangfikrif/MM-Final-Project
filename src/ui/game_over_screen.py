"""
Game Over Screen Module

Menampilkan final statistics dan ranking setelah game berakhir.
Screen ini menampilkan:
- Final score
- Statistics (combo, hits, misses, accuracy, duration)
- Ranking system
- RETRY dan MENU buttons

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
import sys
sys.path.insert(0, 'src')

try:
    from ui.base_screen import BaseScreen
except ImportError:
    from base_screen import BaseScreen

# ============ COLOR CONSTANTS ============
BG_DARK = (30, 30, 30)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (150, 150, 150)
TEXT_LIGHT_GRAY = (200, 200, 200)
HUD_SCORE = (255, 220, 0)           # Gold untuk score
ACCENT_GREEN = (0, 200, 100)        # S Rank
ACCENT_YELLOW = (255, 220, 0)       # A Rank
ACCENT_ORANGE = (255, 165, 0)       # Title highlight
ACCENT_RED = (200, 50, 50)          # B-D Rank
BUTTON_NORMAL = (70, 130, 180)      # Normal state
BUTTON_HOVER = (100, 160, 220)      # Hover state
BUTTON_CLICK = (50, 100, 150)       # Click state


class GameOverButton:
    """
    Simple button class untuk GameOver screen.
    
    Attributes:
        rect: Pygame rect untuk button area
        text: Button text
        is_hovered: Button sedang di-hover?
        is_clicked: Button baru di-click?
    """
    
    def __init__(self, x, y, width, height, text):
        """
        Initialize button.
        
        Args:
            x, y: Position (top-left)
            width, height: Button dimensions
            text: Button label
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        self.is_clicked = False
        try:
            self.font = pygame.font.Font(None, 28)
        except pygame.error:
            # Fallback jika pygame.font belum initialized
            self.font = None
    
    def update(self, mouse_pos):
        """
        Update button state berdasarkan mouse position.
        
        Args:
            mouse_pos: Tuple (x, y) dari mouse position
        """
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked_at(self, mouse_pos):
        """
        Check apakah button di-click.
        
        Args:
            mouse_pos: Tuple (x, y) dari click position
            
        Returns:
            bool: True jika click di dalam button area
        """
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, surface):
        """
        Draw button ke surface.
        
        Args:
            surface: Pygame surface untuk draw
        """
        # Initialize font jika belum
        if self.font is None:
            try:
                self.font = pygame.font.Font(None, 28)
            except:
                return
        
        # Choose color berdasarkan state
        color = BUTTON_HOVER if self.is_hovered else BUTTON_NORMAL
        
        # Draw button rectangle
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, TEXT_WHITE, self.rect, 2)  # Border
        
        # Draw text
        text_surface = self.font.render(self.text, True, TEXT_WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class GameOverScreen(BaseScreen):
    """
    Game Over Screen - Menampilkan final stats dan ranking.
    
    Attributes:
        final_stats: Dict berisi final game statistics
        rank_data: Dict berisi ranking info
        buttons: Dict berisi button objects
        button_enabled_time: Time untuk enable buttons (delay feature)
        current_time: Current time untuk tracking delay
    """
    
    def __init__(self, width=1280, height=720, final_stats=None):
        """
        Initialize Game Over Screen.
        
        Args:
            width: Screen width (default 1280)
            height: Screen height (default 720)
            final_stats: Dict dengan game statistics:
                - final_score (int): Total score
                - max_combo (int): Highest combo
                - total_hits (int): Total successful hits
                - total_misses (int): Total misses
                - accuracy (float): Hit accuracy percentage
                - game_duration (float): Duration in seconds
                - game_duration_display (str): MM:SS format
        """
        super().__init__(width, height)
        
        # Game statistics
        self.final_stats = final_stats or self._get_default_stats()
        
        # Calculate ranking
        self.rank_data = self._calculate_rank()
        
        # Initialize buttons
        self._init_buttons()
        
        # Button delay feature (disable buttons untuk 1 detik)
        self.button_enabled_time = 1.0  # Seconds before buttons are clickable
        self.current_time = 0.0
        
        print("✅ GameOverScreen initialized")
        print(f"   Final Score: {self.final_stats['final_score']}")
        print(f"   Rank: {self.rank_data['rank']}")
    
    # ============ INITIALIZATION ============
    
    def _get_default_stats(self):
        """
        Get default stats untuk testing.
        
        Returns:
            dict: Default statistics
        """
        return {
            'final_score': 0,
            'max_combo': 0,
            'total_hits': 0,
            'total_misses': 0,
            'accuracy': 0.0,
            'game_duration': 0.0,
            'game_duration_display': '00:00'
        }
    
    def _calculate_rank(self):
        """
        Calculate ranking berdasarkan score dan accuracy.
        
        Ranking System:
        - S: score >= 1000 AND accuracy >= 95%
        - A: score >= 800  AND accuracy >= 90%
        - B: score >= 600  AND accuracy >= 80%
        - C: score >= 400  AND accuracy >= 70%
        - D: score < 400
        
        Returns:
            dict: Ranking data dengan:
                - rank (str): 'S', 'A', 'B', 'C', atau 'D'
                - title (str): Deskripsi ranking
                - color: RGB color untuk ranking
        """
        score = self.final_stats['final_score']
        accuracy = self.final_stats['accuracy']
        
        # S Rank: Excellent!
        if score >= 1000 and accuracy >= 95:
            return {
                'rank': 'S',
                'title': '★★★ Excellent!',
                'color': ACCENT_GREEN
            }
        
        # A Rank: Great!
        if score >= 800 and accuracy >= 90:
            return {
                'rank': 'A',
                'title': '★★ Great!',
                'color': ACCENT_YELLOW
            }
        
        # B Rank: Good
        if score >= 600 and accuracy >= 80:
            return {
                'rank': 'B',
                'title': '★ Good',
                'color': ACCENT_ORANGE
            }
        
        # C Rank: OK
        if score >= 400 and accuracy >= 70:
            return {
                'rank': 'C',
                'title': '◆ OK',
                'color': ACCENT_YELLOW
            }
        
        # D Rank: Practice More
        return {
            'rank': 'D',
            'title': '◆ Practice More',
            'color': ACCENT_RED
        }
    
    def _init_buttons(self):
        """
        Initialize buttons untuk GameOver screen.
        """
        button_width = 200
        button_height = 50
        button_y = self.height - 120
        
        # Button spacing: center horizontally
        spacing = 50  # Space antara buttons
        total_width = (button_width * 2) + spacing
        start_x = (self.width - total_width) / 2
        
        self.buttons = {
            'RETRY': GameOverButton(
                x=int(start_x),
                y=button_y,
                width=button_width,
                height=button_height,
                text='RETRY'
            ),
            'MENU': GameOverButton(
                x=int(start_x + button_width + spacing),
                y=button_y,
                width=button_width,
                height=button_height,
                text='MENU'
            )
        }
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            str: Next screen name, atau None jika stay di screen ini
        """
        # MOUSEMOTION: Update button hover state
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            for button in self.buttons.values():
                button.update(mouse_pos)
        
        # MOUSEBUTTONDOWN: Check button click
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                
                # Check RETRY button
                if self.buttons['RETRY'].is_clicked_at(mouse_pos):
                    if self._are_buttons_enabled():
                        print("🔄 RETRY clicked")
                        return 'GAME'
                
                # Check MENU button
                if self.buttons['MENU'].is_clicked_at(mouse_pos):
                    if self._are_buttons_enabled():
                        print("🏠 MENU clicked")
                        return 'MENU'
        
        # KEYDOWN: Keyboard shortcuts
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                # ENTER/SPACE: RETRY
                if self._are_buttons_enabled():
                    print("🔄 RETRY (keyboard)")
                    return 'GAME'
            
            elif event.key == pygame.K_m:
                # M: MENU
                if self._are_buttons_enabled():
                    print("🏠 MENU (keyboard)")
                    return 'MENU'
            
            elif event.key == pygame.K_ESCAPE:
                # ESC: MENU
                if self._are_buttons_enabled():
                    print("🏠 MENU (ESC)")
                    return 'MENU'
        
        return None
    
    # ============ UPDATE / ANIMATION ============
    
    def update(self, dt=0.016):
        """
        Update screen state.
        
        Args:
            dt: Delta time sejak last update (default 16ms = 60 FPS)
        """
        # Track elapsed time untuk button delay feature
        self.current_time += dt
    
    def _are_buttons_enabled(self):
        """
        Check apakah buttons sudah bisa di-click.
        
        Returns:
            bool: True jika current_time >= button_enabled_time
        """
        return self.current_time >= self.button_enabled_time
    
    # ============ DRAWING ============
    
    def draw(self, surface):
        """
        Draw GameOver screen ke surface.
        
        Args:
            surface: Pygame surface untuk draw
        """
        # Fill background
        surface.fill(BG_DARK)
        
        # Draw title
        self._draw_title(surface)
        
        # Draw final score (big and prominent)
        self._draw_final_score(surface)
        
        # Draw statistics box
        self._draw_statistics_box(surface)
        
        # Draw ranking
        self._draw_ranking(surface)
        
        # Draw buttons
        self._draw_buttons(surface)
        
        # Draw button delay message jika buttons belum enabled
        if not self._are_buttons_enabled():
            self._draw_wait_message(surface)
    
    def _draw_title(self, surface):
        """
        Draw "GAME OVER" title.
        
        Args:
            surface: Pygame surface
        """
        font_large = pygame.font.Font(None, 80)
        text_surface = font_large.render("GAME OVER", True, ACCENT_ORANGE)
        text_rect = text_surface.get_rect(center=(self.width // 2, 50))
        surface.blit(text_surface, text_rect)
    
    def _draw_final_score(self, surface):
        """
        Draw final score (big, prominent).
        
        Args:
            surface: Pygame surface
        """
        # "Final Score:"
        font_label = pygame.font.Font(None, 36)
        label_surface = font_label.render("Final Score:", True, TEXT_LIGHT_GRAY)
        label_rect = label_surface.get_rect(center=(self.width // 2, 140))
        surface.blit(label_surface, label_rect)
        
        # Score number (big and gold)
        font_score = pygame.font.Font(None, 72)
        score_text = f"{self.final_stats['final_score']:,}"  # Format dengan comma
        score_surface = font_score.render(score_text, True, HUD_SCORE)
        score_rect = score_surface.get_rect(center=(self.width // 2, 220))
        surface.blit(score_surface, score_rect)
    
    def _draw_statistics_box(self, surface):
        """
        Draw statistics box dengan detailed info.
        
        Args:
            surface: Pygame surface
        """
        # Box dimensions
        box_width = 400
        box_height = 180
        box_x = (self.width - box_width) // 2
        box_y = 310
        
        # Draw box background
        pygame.draw.rect(surface, (50, 50, 50), 
                        (box_x, box_y, box_width, box_height))
        pygame.draw.rect(surface, TEXT_GRAY, 
                        (box_x, box_y, box_width, box_height), 2)
        
        # Statistics text
        font_stat = pygame.font.Font(None, 24)
        stats = [
            f"Max Combo:      {self.final_stats['max_combo']}x",
            f"Total Hits:     {self.final_stats['total_hits']}",
            f"Total Misses:   {self.final_stats['total_misses']}",
            f"Accuracy:       {self.final_stats['accuracy']:.1f}%",
            f"Duration:       {self.final_stats['game_duration_display']}"
        ]
        
        y_offset = box_y + 20
        for stat in stats:
            stat_surface = font_stat.render(stat, True, TEXT_LIGHT_GRAY)
            surface.blit(stat_surface, (box_x + 30, y_offset))
            y_offset += 30
    
    def _draw_ranking(self, surface):
        """
        Draw ranking berdasarkan score dan accuracy.
        
        Args:
            surface: Pygame surface
        """
        font_rank = pygame.font.Font(None, 48)
        rank_title = self.rank_data['title']
        rank_color = self.rank_data['color']
        
        rank_surface = font_rank.render(f"RANK: {rank_title}", True, rank_color)
        rank_rect = rank_surface.get_rect(center=(self.width // 2, 540))
        surface.blit(rank_surface, rank_rect)
    
    def _draw_buttons(self, surface):
        """
        Draw RETRY dan MENU buttons.
        
        Args:
            surface: Pygame surface
        """
        # Fade out buttons jika belum enabled
        if not self._are_buttons_enabled():
            # Draw buttons dengan reduced alpha
            for button in self.buttons.values():
                # Draw semi-transparent button
                temp_surface = pygame.Surface((button.rect.width, button.rect.height))
                temp_surface.fill(BUTTON_NORMAL)
                temp_surface.set_alpha(128)
                surface.blit(temp_surface, button.rect)
                
                # Draw text
                font = pygame.font.Font(None, 28)
                text_surface = font.render(button.text, True, TEXT_GRAY)
                text_rect = text_surface.get_rect(center=button.rect.center)
                surface.blit(text_surface, text_rect)
        else:
            # Draw normal buttons
            for button in self.buttons.values():
                button.draw(surface)
    
    def _draw_wait_message(self, surface):
        """
        Draw "Press any key to continue" message.
        
        Args:
            surface: Pygame surface
        """
        elapsed = self.button_enabled_time - self.current_time
        font_wait = pygame.font.Font(None, 20)
        wait_text = f"Ready in {elapsed:.1f}s..."
        wait_surface = font_wait.render(wait_text, True, TEXT_GRAY)
        wait_rect = wait_surface.get_rect(center=(self.width // 2, 650))
        surface.blit(wait_surface, wait_rect)
    
    # ============ LIFECYCLE METHODS ============
    
    def on_enter(self):
        """
        Called when screen is entered.
        Reset timer untuk button delay.
        """
        self.current_time = 0.0
        print("📊 Entered GameOverScreen")
    
    def on_exit(self):
        """
        Called when screen is exited.
        """
        print("📊 Exited GameOverScreen")
    
    # ============ STATUS & DISPLAY ============
    
    def get_status(self):
        """
        Get screen status.
        
        Returns:
            dict: Status information
        """
        return {
            'screen': 'GAME_OVER',
            'final_score': self.final_stats['final_score'],
            'rank': self.rank_data['rank'],
            'buttons_enabled': self._are_buttons_enabled(),
            'elapsed_time': self.current_time
        }
    
    def print_status(self):
        """Print status untuk debugging."""
        status = self.get_status()
        print("\n" + "="*60)
        print("GAME OVER SCREEN STATUS")
        print("="*60)
        print(f"Final Score: {status['final_score']}")
        print(f"Rank: {status['rank']}")
        print(f"Buttons Enabled: {status['buttons_enabled']}")
        print(f"Elapsed Time: {status['elapsed_time']:.1f}s")
        print("="*60 + "\n")


if __name__ == "__main__":
    """
    Simple test dengan pygame display.
    """
    print("\n" + "="*60)
    print("GAME OVER SCREEN - VISUAL TEST")
    print("="*60 + "\n")
    
    pygame.init()
    
    # Create screen
    width, height = 1280, 720
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AirBeats - Game Over Screen")
    clock = pygame.time.Clock()
    
    # Create GameOverScreen dengan sample data
    sample_stats = {
        'final_score': 1250,
        'max_combo': 15,
        'total_hits': 45,
        'total_misses': 3,
        'accuracy': 93.8,
        'game_duration': 92.5,
        'game_duration_display': '01:32'
    }
    
    screen = GameOverScreen(width=width, height=height, final_stats=sample_stats)
    screen.on_enter()
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Convert to seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            next_screen = screen.handle_event(event)
            if next_screen:
                print(f"\n→ Transitioning to: {next_screen}")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
        
        screen.update(dt)
        screen.draw(surface)
        
        pygame.display.flip()
    
    pygame.quit()
    print("\nVisual test ended.")
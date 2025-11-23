"""
Pause Screen Module
Pause overlay screen untuk AirBeats game

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
try:
    from ui.base_screen import BaseScreen
except ImportError:
    from base_screen import BaseScreen

try:
    from components.button import Button
    from components.color import BG_DARK, TEXT_WHITE, ACCENT_ORANGE, BUTTON_NORMAL, BUTTON_HOVER
except ImportError:
    # Fallback colors dan Button class
    BG_DARK = (30, 30, 30)
    TEXT_WHITE = (255, 255, 255)
    ACCENT_ORANGE = (255, 165, 0)
    BUTTON_NORMAL = (70, 130, 180)
    BUTTON_HOVER = (100, 160, 220)
    
    # Fallback Button class
    class Button:
        def __init__(self, x, y, width, height, text, font, callback=None):
            self.rect = pygame.Rect(x, y, width, height)
            self.text = text
            self.font = font
            self.callback = callback
            self.is_hovered = False
            self.is_pressed = False
            self.is_enabled = True
            self.is_focused = False
        
        def handle_event(self, event, mouse_pos=None):
            if not self.is_enabled:
                return False
            if mouse_pos is None:
                mouse_pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEMOTION:
                self.is_hovered = self.rect.collidepoint(mouse_pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect.collidepoint(mouse_pos):
                    self.is_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                if self.is_pressed and self.rect.collidepoint(mouse_pos):
                    self.is_pressed = False
                    self._on_click()
                    return True
                self.is_pressed = False
            return False
        
        def _on_click(self):
            if self.callback:
                self.callback()
        
        def set_focused(self, focused):
            self.is_focused = focused
        
        def draw(self, surface):
            color = BUTTON_HOVER if (self.is_hovered or self.is_focused) else BUTTON_NORMAL
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
            text_surface = self.font.render(self.text, True, TEXT_WHITE)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)


class PauseScreen(BaseScreen):
    """
    Pause overlay screen.
    
    Features:
    - Semi-transparent overlay
    - "PAUSED" title
    - Buttons: RESUME, SETTINGS, MENU
    - Keyboard navigation (UP/DOWN/ENTER)
    - Mouse support
    - Back to game dengan ESC atau RESUME button
    """
    
    def __init__(self, width, height, game_manager=None):
        """
        Initialize pause screen.
        
        Args:
            width (int): Screen width
            height (int): Screen height
            game_manager (GameManager): Reference ke GameManager
        """
        super().__init__(width, height, game_manager)
        
        # Fonts
        self.font_title = pygame.font.Font(None, 80)
        self.font_button = pygame.font.Font(None, 48)
        
        # Calculate positions
        center_x = width // 2
        title_y = 150
        buttons_start_y = 280
        button_spacing = 100
        button_width = 300
        button_height = 60
        
        # Buttons
        self.buttons = {
            'resume': Button(
                center_x - button_width // 2,
                buttons_start_y,
                button_width,
                button_height,
                "RESUME",
                self.font_button,
                callback=self.on_resume_clicked
            ),
            'settings': Button(
                center_x - button_width // 2,
                buttons_start_y + button_spacing,
                button_width,
                button_height,
                "SETTINGS",
                self.font_button,
                callback=self.on_settings_clicked
            ),
            'menu': Button(
                center_x - button_width // 2,
                buttons_start_y + button_spacing * 2,
                button_width,
                button_height,
                "MENU",
                self.font_button,
                callback=self.on_menu_clicked
            )
        }
        
        # Button order untuk keyboard navigation
        self.button_order = ['resume', 'settings', 'menu']
        self.selected_button_index = 0
        self.buttons['resume'].set_focused(True)
        
        # Overlay
        self.overlay_alpha = 128  # Semi-transparent
        
        print(f"✅ PauseScreen initialized")
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event):
        """Handle pause screen events."""
        
        # ===== KEYBOARD NAVIGATION =====
        if event.type == pygame.KEYDOWN:
            # UP arrow
            if event.key == pygame.K_UP:
                self._previous_button()
            
            # DOWN arrow
            elif event.key == pygame.K_DOWN:
                self._next_button()
            
            # ENTER
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                self._click_selected_button()
            
            # ESC (resume)
            elif event.key == pygame.K_ESCAPE:
                self.set_next_screen("GAME")
                return "GAME"
        
        # ===== MOUSE EVENTS =====
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            # Update hover states
            for button in self.buttons.values():
                button.handle_event(event, mouse_pos)
        
        if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
            mouse_pos = pygame.mouse.get_pos()
            # Handle clicks
            for button_name, button in self.buttons.items():
                if button.handle_event(event, mouse_pos):
                    # Button was clicked
                    return self._get_transition_for_button(button_name)
        
        return None
    
    # ============ BUTTON NAVIGATION ============
    
    def _next_button(self):
        """Move to next button."""
        self.buttons[self.button_order[self.selected_button_index]].set_focused(False)
        self.selected_button_index = (self.selected_button_index + 1) % len(self.button_order)
        self.buttons[self.button_order[self.selected_button_index]].set_focused(True)
        print(f"⬇️  Selected: {self.button_order[self.selected_button_index]}")
    
    def _previous_button(self):
        """Move to previous button."""
        self.buttons[self.button_order[self.selected_button_index]].set_focused(False)
        self.selected_button_index = (self.selected_button_index - 1) % len(self.button_order)
        self.buttons[self.button_order[self.selected_button_index]].set_focused(True)
        print(f"⬆️  Selected: {self.button_order[self.selected_button_index]}")
    
    def _click_selected_button(self):
        """Click selected button."""
        button_name = self.button_order[self.selected_button_index]
        button = self.buttons[button_name]
        button._on_click()
    
    # ============ BUTTON CALLBACKS ============
    
    def on_resume_clicked(self):
        """Handle RESUME button click."""
        self.set_next_screen("GAME")
    
    def on_settings_clicked(self):
        """Handle SETTINGS button click."""
        self.set_next_screen("SETTINGS")
    
    def on_menu_clicked(self):
        """Handle MENU button click."""
        self.set_next_screen("MENU")
    
    def _get_transition_for_button(self, button_name):
        """Get transition for button click."""
        if button_name == 'resume':
            self.set_next_screen("GAME")
            return "GAME"
        elif button_name == 'settings':
            self.set_next_screen("SETTINGS")
            return "SETTINGS"
        elif button_name == 'menu':
            self.set_next_screen("MENU")
            return "MENU"
        return None
    
    # ============ UPDATE ============
    
    def update(self):
        """Update pause logic."""
        pass
    
    # ============ RENDERING ============
    
    def draw(self, surface):
        """Draw pause screen with overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(self.overlay_alpha)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, 0))
        
        # Title
        title_surface = self.font_title.render("PAUSED", True, ACCENT_ORANGE)
        title_rect = title_surface.get_rect(center=(self.width // 2, 150))
        surface.blit(title_surface, title_rect)
        
        # Buttons
        for button in self.buttons.values():
            button.draw(surface)
        
        # Hint text
        font_small = pygame.font.Font(None, 32)
        hint_text = font_small.render("Press ESC or RESUME to continue", True, (150, 150, 150))
        hint_rect = hint_text.get_rect(center=(self.width // 2, self.height - 50))
        surface.blit(hint_text, hint_rect)
    
    # ============ LIFECYCLE ============
    
    def on_enter(self):
        """Called when entering pause screen."""
        super().on_enter()
        print("⏸️  Entered PauseScreen")
        # Reset selected button
        self.selected_button_index = 0
        self.buttons['resume'].set_focused(True)
    
    def on_exit(self):
        """Called when exiting pause screen."""
        super().on_exit()
        print("⏸️  Exited PauseScreen")
    
    # ============ STATUS ============
    
    def get_selected_button(self):
        """Get currently selected button name."""
        return self.button_order[self.selected_button_index]
    
    def __str__(self):
        """String representation."""
        return f"PauseScreen (selected: {self.get_selected_button()})"
"""
Button Component Module
Reusable button untuk menu, settings, dll

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
from color import BUTTON_NORMAL, BUTTON_HOVER, BUTTON_CLICK, TEXT_WHITE, BUTTON_DISABLED


class Button:
    """
    Interactive button component untuk UI.
    
    Features:
    - Mouse hover detection
    - Click detection
    - Keyboard focus
    - Text rendering
    - State management (normal, hover, click, disabled)
    """
    
    def __init__(self, x, y, width, height, text, font, callback=None):
        """
        Initialize button.
        
        Args:
            x (int): X position
            y (int): Y position
            width (int): Button width
            height (int): Button height
            text (str): Button text
            font (pygame.font.Font): Font untuk text
            callback (function): Function yang di-call saat button clicked
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.callback = callback
        
        # State
        self.is_hovered = False
        self.is_pressed = False
        self.is_enabled = True
        self.is_focused = False
        
        print(f"✅ Button created: {text} at ({x}, {y})")
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event, mouse_pos=None):
        """
        Handle mouse dan keyboard events.
        
        Args:
            event (pygame.event.Event): Event untuk di-handle
            mouse_pos (tuple): Mouse position (x, y), jika None gunakan pygame.mouse.get_pos()
            
        Returns:
            bool: True jika button di-click
        """
        if not self.is_enabled:
            return False
        
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        
        # ===== MOUSE MOVEMENT =====
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(mouse_pos)
            return False
        
        # ===== MOUSE CLICK =====
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                self.is_pressed = True
                return False
        
        if event.type == pygame.MOUSEBUTTONUP:
            if self.is_pressed and self.rect.collidepoint(mouse_pos):
                self.is_pressed = False
                self._on_click()
                return True  # Button was clicked
            self.is_pressed = False
            return False
        
        # ===== KEYBOARD =====
        if event.type == pygame.KEYDOWN:
            if self.is_focused:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self._on_click()
                    return True
        
        return False
    
    def _on_click(self):
        """Handle button click."""
        print(f"✅ Button clicked: {self.text}")
        if self.callback:
            self.callback()
    
    # ============ STATE MANAGEMENT ============
    
    def set_enabled(self, enabled):
        """
        Enable/disable button.
        
        Args:
            enabled (bool): True untuk enable, False untuk disable
        """
        self.is_enabled = enabled
        if not enabled:
            self.is_hovered = False
            self.is_pressed = False
    
    def set_focused(self, focused):
        """
        Set keyboard focus.
        
        Args:
            focused (bool): True untuk fokus, False untuk unfocus
        """
        self.is_focused = focused
    
    def set_text(self, text):
        """
        Update button text.
        
        Args:
            text (str): New text
        """
        self.text = text
    
    # ============ RENDERING ============
    
    def get_color(self):
        """
        Get current button color berdasarkan state.
        
        Returns:
            tuple: RGB color
        """
        if not self.is_enabled:
            return BUTTON_DISABLED
        elif self.is_pressed:
            return BUTTON_CLICK
        elif self.is_hovered or self.is_focused:
            return BUTTON_HOVER
        else:
            return BUTTON_NORMAL
    
    def draw(self, surface):
        """
        Draw button ke pygame surface.
        
        Args:
            surface (pygame.Surface): Target surface
        """
        # Draw button background
        color = self.get_color()
        pygame.draw.rect(surface, color, self.rect)
        
        # Draw border jika focused
        if self.is_focused:
            pygame.draw.rect(surface, (255, 255, 255), self.rect, 3)
        else:
            pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Render text
        text_surface = self.font.render(self.text, True, TEXT_WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    # ============ UTILITY ============
    
    def set_position(self, x, y):
        """
        Update button position.
        
        Args:
            x (int): New X position
            y (int): New Y position
        """
        self.rect.x = x
        self.rect.y = y
    
    def get_rect(self):
        """
        Get button rect.
        
        Returns:
            pygame.Rect: Button rectangle
        """
        return self.rect
    
    def __str__(self):
        """String representation."""
        return f"Button({self.text}, enabled={self.is_enabled}, hovered={self.is_hovered})"
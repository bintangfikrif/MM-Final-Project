"""
Slider Component Module
Reusable slider untuk volume, difficulty, dll

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
from color import BUTTON_NORMAL, BUTTON_HOVER, ACCENT_ORANGE, TEXT_WHITE, TEXT_GRAY


class Slider:
    """
    Interactive slider component untuk mengubah nilai.
    
    Features:
    - Mouse drag
    - Keyboard navigation (LEFT/RIGHT)
    - Value range clamping
    - Visual feedback
    """
    
    def __init__(self, x, y, width, min_val, max_val, current_val, label=""):
        """
        Initialize slider.
        
        Args:
            x (int): X position
            y (int): Y position
            width (int): Slider width
            min_val (int/float): Minimum value
            max_val (int/float): Maximum value
            current_val (int/float): Current value
            label (str): Label untuk slider (opsional)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        
        self.min_val = min_val
        self.max_val = max_val
        self.value = current_val
        self.label = label
        
        # Track area
        self.track_rect = pygame.Rect(x, y + 5, width, 10)
        
        # Thumb (slider handle)
        self.thumb_width = 15
        self.thumb_height = 20
        
        # State
        self.is_dragging = False
        self.is_hovered = False
        self.is_focused = False
        
        print(f"✅ Slider created: {label} ({min_val}-{max_val}, current={current_val})")
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event, mouse_pos=None):
        """
        Handle mouse dan keyboard events.
        
        Args:
            event (pygame.event.Event): Event untuk di-handle
            mouse_pos (tuple): Mouse position (x, y)
        """
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        
        # ===== MOUSE MOVEMENT =====
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self._get_thumb_rect().collidepoint(mouse_pos)
            
            if self.is_dragging:
                self._update_value_from_mouse(mouse_pos[0])
        
        # ===== MOUSE BUTTON =====
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self._get_thumb_rect().collidepoint(mouse_pos):
                self.is_dragging = True
        
        if event.type == pygame.MOUSEBUTTONUP:
            self.is_dragging = False
        
        # ===== KEYBOARD =====
        if event.type == pygame.KEYDOWN:
            if self.is_focused:
                step = (self.max_val - self.min_val) / 20  # 20 steps
                
                if event.key == pygame.K_LEFT:
                    self.value = max(self.min_val, self.value - step)
                    print(f"⬅️  Slider decreased: {self.value:.1f}")
                
                elif event.key == pygame.K_RIGHT:
                    self.value = min(self.max_val, self.value + step)
                    print(f"➡️  Slider increased: {self.value:.1f}")
    
    def _get_thumb_rect(self):
        """Get thumb rectangle."""
        thumb_x = self._value_to_x(self.value)
        return pygame.Rect(
            thumb_x - self.thumb_width // 2,
            self.y,
            self.thumb_width,
            self.thumb_height
        )
    
    def _value_to_x(self, value):
        """Convert value ke x position."""
        # Normalize value to 0-1
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        # Convert to x position
        return self.x + (normalized * self.width)
    
    def _x_to_value(self, x):
        """Convert x position ke value."""
        # Clamp x to track bounds
        x = max(self.x, min(self.x + self.width, x))
        # Normalize to 0-1
        normalized = (x - self.x) / self.width
        # Convert to value
        value = self.min_val + (normalized * (self.max_val - self.min_val))
        return value
    
    def _update_value_from_mouse(self, mouse_x):
        """Update value dari mouse position."""
        self.value = self._x_to_value(mouse_x)
    
    # ============ STATE MANAGEMENT ============
    
    def set_focused(self, focused):
        """Set keyboard focus."""
        self.is_focused = focused
    
    def set_value(self, value):
        """
        Set slider value.
        
        Args:
            value (int/float): New value
        """
        self.value = max(self.min_val, min(self.max_val, value))
    
    def get_value(self):
        """
        Get current slider value.
        
        Returns:
            float: Current value
        """
        return self.value
    
    def get_normalized_value(self):
        """
        Get value as 0-1 (normalized).
        
        Returns:
            float: Normalized value (0.0 - 1.0)
        """
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        return max(0.0, min(1.0, normalized))
    
    # ============ RENDERING ============
    
    def draw(self, surface, font=None):
        """
        Draw slider ke pygame surface.
        
        Args:
            surface (pygame.Surface): Target surface
            font (pygame.font.Font): Font untuk label (opsional)
        """
        # Draw label
        if self.label and font:
            label_text = font.render(self.label, True, TEXT_WHITE)
            surface.blit(label_text, (self.x, self.y - 25))
        
        # Draw track background
        pygame.draw.rect(surface, (50, 50, 50), self.track_rect)
        
        # Draw filled track (progress)
        filled_width = (self.value - self.min_val) / (self.max_val - self.min_val) * self.width
        filled_rect = pygame.Rect(self.x, self.y + 5, filled_width, 10)
        pygame.draw.rect(surface, ACCENT_ORANGE, filled_rect)
        
        # Draw thumb (slider handle)
        thumb_rect = self._get_thumb_rect()
        thumb_color = BUTTON_HOVER if (self.is_hovered or self.is_focused) else BUTTON_NORMAL
        pygame.draw.rect(surface, thumb_color, thumb_rect)
        
        # Draw border
        pygame.draw.rect(surface, (0, 0, 0), self.track_rect, 2)
        pygame.draw.rect(surface, (0, 0, 0), thumb_rect, 2)
        
        # Draw value text
        if font:
            value_text = font.render(f"{self.value:.1f}", True, TEXT_GRAY)
            surface.blit(value_text, (self.x + self.width + 15, self.y - 5))
    
    # ============ UTILITY ============
    
    def set_position(self, x, y):
        """Update slider position."""
        self.x = x
        self.y = y
        self.track_rect = pygame.Rect(x, y + 5, self.width, 10)
    
    def __str__(self):
        """String representation."""
        return f"Slider({self.label}, value={self.value:.1f}, range={self.min_val}-{self.max_val})"
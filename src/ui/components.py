import pygame

# ==========================================
# COLORS (from color.py)
# ==========================================

# BACKGROUND COLORS
BG_DARK = (30, 30, 30)           # Primary background
BG_LIGHT = (60, 60, 60)          # Secondary background / Hover
BG_OVERLAY = (0, 0, 0, 128)      # Semi-transparent overlay (dengan alpha)

# TEXT COLORS
TEXT_WHITE = (255, 255, 255)     # Primary text
TEXT_GRAY = (150, 150, 150)      # Secondary text / Disabled
TEXT_LIGHT_GRAY = (200, 200, 200) # Tertiary text

# BUTTON COLORS
BUTTON_NORMAL = (70, 130, 180)   # Normal state (Steel Blue)
BUTTON_HOVER = (100, 160, 220)   # Hover state (Lighter Blue)
BUTTON_CLICK = (50, 100, 150)    # Click state (Darker Blue)
BUTTON_DISABLED = (100, 100, 100) # Disabled state (Gray)

# ACCENT COLORS
ACCENT_ORANGE = (255, 165, 0)    # Accent color
ACCENT_GREEN = (0, 200, 100)     # Success / Hit
ACCENT_RED = (200, 50, 50)       # Error / Miss
ACCENT_YELLOW = (255, 220, 0)    # Warning / Combo

# GAME HUD COLORS
HUD_SCORE = (255, 220, 0)        # Score text (Gold)
HUD_COMBO = (0, 200, 100)        # Combo text (Green)
HUD_TIMER = (255, 100, 100)      # Timer text (Red when low)
HUD_ACCURACY = (100, 200, 255)   # Accuracy text (Light Blue)

# COLOR DICTIONARY 
COLORS = {
    # Background
    'bg_dark': BG_DARK,
    'bg_light': BG_LIGHT,
    
    # Text
    'text_white': TEXT_WHITE,
    'text_gray': TEXT_GRAY,
    'text_light_gray': TEXT_LIGHT_GRAY,
    
    # Button
    'button_normal': BUTTON_NORMAL,
    'button_hover': BUTTON_HOVER,
    'button_click': BUTTON_CLICK,
    'button_disabled': BUTTON_DISABLED,
    
    # Accent
    'accent_orange': ACCENT_ORANGE,
    'accent_green': ACCENT_GREEN,
    'accent_red': ACCENT_RED,
    'accent_yellow': ACCENT_YELLOW,
    
    # HUD
    'hud_score': HUD_SCORE,
    'hud_combo': HUD_COMBO,
    'hud_timer': HUD_TIMER,
    'hud_accuracy': HUD_ACCURACY,
}

def get_color(name):
    return COLORS.get(name, TEXT_WHITE)


# ==========================================
# TEXT (from text.py)
# ==========================================

class Text:
    # Alignment options
    ALIGN_LEFT = "left"
    ALIGN_CENTER = "center"
    ALIGN_RIGHT = "right"
    
    def __init__(self, x, y, text, font, color=TEXT_WHITE, align=ALIGN_LEFT, shadow=False):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.align = align
        self.shadow = shadow
        
        print(f"Text created: '{text}' at ({x}, {y})")
    
    def set_text(self, text):
        self.text = str(text)
    
    def set_color(self, color):
        self.color = color
    
    def set_position(self, x, y):
        self.x = x
        self.y = y
    
    def get_rect(self):
        text_surface = self.font.render(self.text, True, self.color)
        rect = text_surface.get_rect()
        
        if self.align == self.ALIGN_CENTER:
            rect.center = (self.x, self.y)
        elif self.align == self.ALIGN_RIGHT:
            rect.topright = (self.x, self.y)
        else:  
            rect.topleft = (self.x, self.y)
        
        return rect
    
    def draw(self, surface):
        text_surface = self.font.render(self.text, True, self.color)
        rect = self.get_rect()
        
        # Draw shadow
        if self.shadow:
            shadow_surface = self.font.render(self.text, True, (0, 0, 0))
            surface.blit(shadow_surface, (rect.x + 2, rect.y + 2))
        
        # Draw text
        surface.blit(text_surface, rect)


class ScoreDisplay(Text):
    def __init__(self, x, y, font, score=0):
        super().__init__(x, y, f"Score: {score}", font, color=HUD_SCORE, shadow=True)
        self.score = score
    
    def set_score(self, score):
        self.score = score
        self.set_text(f"Score: {score}")


class ComboDisplay(Text):
    def __init__(self, x, y, font, combo=0):
        super().__init__(x, y, f"Combo: {combo}x", font, color=HUD_COMBO, shadow=True)
        self.combo = combo
    
    def set_combo(self, combo):
        self.combo = combo
        if combo > 0:
            self.set_text(f"Combo: {combo}x")
        else:
            self.set_text("Combo: 0x")
            self.set_color(TEXT_GRAY)


class TimerDisplay(Text):
    def __init__(self, x, y, font, time_str="00:00"):
        super().__init__(x, y, time_str, font, color=HUD_TIMER, shadow=True)
        self.time_str = time_str
    
    def set_time(self, time_str):
        self.time_str = time_str
        self.set_text(time_str)


class AccuracyDisplay(Text):
    def __init__(self, x, y, font, accuracy=0.0):
        super().__init__(x, y, f"Accuracy: {accuracy:.1f}%", font, color=HUD_ACCURACY, shadow=True)
        self.accuracy = accuracy
    
    def set_accuracy(self, accuracy):
        self.accuracy = accuracy
        self.set_text(f"Accuracy: {accuracy:.1f}%")


# ==========================================
# BUTTON (from button.py)
# ==========================================

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
        
        print(f"Button created: {text} at ({x}, {y})")
    
    # EVENT HANDLING 
    
    def handle_event(self, event, mouse_pos=None):
        if not self.is_enabled:
            return False
        
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        
        # MOUSE MOVEMENT
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(mouse_pos)
            return False
        
        # MOUSE CLICK
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                self.is_pressed = True
                return False
        
        if event.type == pygame.MOUSEBUTTONUP:
            if self.is_pressed and self.rect.collidepoint(mouse_pos):
                self.is_pressed = False
                self._on_click()
                return True  
            self.is_pressed = False
            return False
        
        # KEYBOARD
        if event.type == pygame.KEYDOWN:
            if self.is_focused:
                if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self._on_click()
                    return True
        
        return False
    
    def _on_click(self):
        print(f"Button clicked: {self.text}")
        if self.callback:
            self.callback()
    
    # STATE MANAGEMENT
    
    def set_enabled(self, enabled):
        self.is_enabled = enabled
        if not enabled:
            self.is_hovered = False
            self.is_pressed = False
    
    def set_focused(self, focused):
        self.is_focused = focused
    
    def set_text(self, text):
        self.text = text
    
    # RENDERING 
    
    def get_color(self):
        if not self.is_enabled:
            return BUTTON_DISABLED
        elif self.is_pressed:
            return BUTTON_CLICK
        elif self.is_hovered or self.is_focused:
            return BUTTON_HOVER
        else:
            return BUTTON_NORMAL
    
    def draw(self, surface):
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
    
    # UTILITY 
    
    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
    
    def get_rect(self):
        return self.rect
    
    def __str__(self):
        return f"Button({self.text}, enabled={self.is_enabled}, hovered={self.is_hovered})"


# ==========================================
# SLIDER (from slider.py)
# ==========================================

class Slider:
    def __init__(self, x, y, width, min_val, max_val, current_val, label=""):
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
        
        print(f"Slider created: {label} ({min_val}-{max_val}, current={current_val})")
    
    # EVENT HANDLING
    
    def handle_event(self, event, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        
        # MOUSE MOVEMENT
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self._get_thumb_rect().collidepoint(mouse_pos)
            
            if self.is_dragging:
                self._update_value_from_mouse(mouse_pos[0])
        
        # MOUSE BUTTON
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self._get_thumb_rect().collidepoint(mouse_pos):
                self.is_dragging = True
        
        if event.type == pygame.MOUSEBUTTONUP:
            self.is_dragging = False
        
        # KEYBOARD
        if event.type == pygame.KEYDOWN:
            if self.is_focused:
                step = (self.max_val - self.min_val) / 20  
                
                if event.key == pygame.K_LEFT:
                    self.value = max(self.min_val, self.value - step)
                    print(f"Slider decreased: {self.value:.1f}")
                
                elif event.key == pygame.K_RIGHT:
                    self.value = min(self.max_val, self.value + step)
                    print(f"Slider increased: {self.value:.1f}")
    
    def _get_thumb_rect(self):
        thumb_x = self._value_to_x(self.value)
        return pygame.Rect(
            thumb_x - self.thumb_width // 2,
            self.y,
            self.thumb_width,
            self.thumb_height
        )
    
    def _value_to_x(self, value):
        # Normalize value to 0-1
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        # Convert to x position
        return self.x + (normalized * self.width)
    
    def _x_to_value(self, x):
        # Clamp x to track bounds
        x = max(self.x, min(self.x + self.width, x))
        # Normalize to 0-1
        normalized = (x - self.x) / self.width
        # Convert to value
        value = self.min_val + (normalized * (self.max_val - self.min_val))
        return value
    
    def _update_value_from_mouse(self, mouse_x):
        self.value = self._x_to_value(mouse_x)
    
    # STATE MANAGEMENT
    def set_focused(self, focused):
        self.is_focused = focused
    
    def set_value(self, value):
        self.value = max(self.min_val, min(self.max_val, value))
    
    def get_value(self):
        return self.value
    
    def get_normalized_value(self):
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        return max(0.0, min(1.0, normalized))
    
    # RENDERING
    
    def draw(self, surface, font=None):
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
    
    # UTILITY
    
    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.track_rect = pygame.Rect(x, y + 5, self.width, 10)
    
    def __str__(self):
        return f"Slider({self.label}, value={self.value:.1f}, range={self.min_val}-{self.max_val})"

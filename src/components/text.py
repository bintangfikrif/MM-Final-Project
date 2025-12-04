import pygame
from color import TEXT_WHITE, TEXT_GRAY, HUD_SCORE, HUD_COMBO, HUD_TIMER, HUD_ACCURACY


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
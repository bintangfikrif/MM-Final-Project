import pygame
import random
import math

class Particle:
    """A single particle for visual effects."""
    def __init__(self, x, y, color, velocity, life=1.0, size=5):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.life = life
        self.max_life = life
        self.size = size
        self.gravity = 0.2

    def update(self):
        """Update particle position and life."""
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= 0.02
        
    def draw(self, surface):
        """Draw particle with fading alpha."""
        if self.life > 0:
            alpha = int((self.life / self.max_life) * 255)
            
            # Create a temporary surface for alpha blending
            radius = int(self.size)
            if radius < 1: radius = 1
            
            # Surface size needs to be 2x radius
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (radius, radius), radius)
            
            surface.blit(s, (int(self.x) - radius, int(self.y) - radius))

class ScorePopup:
    """Floating text for score feedback."""
    def __init__(self, x, y, text, color, life=1.0):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = life
        self.max_life = life
        self.vy = -2 # Float up
        self.font = pygame.font.Font(None, 48) # Slightly larger font
        self.scale = 0.5 # Start small

    def update(self):
        self.y += self.vy
        self.life -= 0.02
        
        # Pop in animation
        if self.scale < 1.0:
            self.scale += 0.1
            if self.scale > 1.0: self.scale = 1.0

    def draw(self, surface):
        if self.life > 0:
            text_surf = self.font.render(self.text, True, self.color)
            
            # Apply scaling
            if self.scale != 1.0:
                w = int(text_surf.get_width() * self.scale)
                h = int(text_surf.get_height() * self.scale)
                text_surf = pygame.transform.scale(text_surf, (w, h))
            
            # Fade out effect
            alpha = int((self.life / self.max_life) * 255)
            text_surf.set_alpha(alpha)
            
            surface.blit(text_surf, (int(self.x - text_surf.get_width()//2), int(self.y)))

class VFXManager:
    """Manages all visual effects."""
    def __init__(self):
        self.particles = []
        self.popups = []
        print("✅ VFX Manager initialized!")

    def create_explosion(self, x, y, color):
        """Create a particle explosion at (x, y)."""
        for _ in range(20): # Increased particle count
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 10)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            self.particles.append(Particle(
                x, y, color, (vx, vy), life=random.uniform(0.5, 1.0), size=random.randint(4, 10)
            ))

    def create_score_popup(self, x, y, text, color):
        """Create a floating score text."""
        self.popups.append(ScorePopup(x, y, text, color))

    def update(self):
        """Update all effects."""
        for p in self.particles:
            p.update()
        for popup in self.popups:
            popup.update()
            
        # Remove dead objects
        self.particles = [p for p in self.particles if p.life > 0]
        self.popups = [p for p in self.popups if p.life > 0]

    def draw(self, surface):
        """Draw all effects."""
        for p in self.particles:
            p.draw(surface)
        for popup in self.popups:
            popup.draw(surface)

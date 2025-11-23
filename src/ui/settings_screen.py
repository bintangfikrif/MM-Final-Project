"""
Settings Screen Module

Menampilkan settings untuk volume dan difficulty.
"""

import pygame
import sys
sys.path.insert(0, 'src')

try:
    from ui.base_screen import BaseScreen
except ImportError:
    # Fallback untuk testing
    class BaseScreen:
        def __init__(self, width=1280, height=720, game_manager=None):
            self.width = width
            self.height = height
            self.game_manager = game_manager
            self.next_screen = None
            self.is_active = True
        
        def set_next_screen(self, screen_name):
            self.next_screen = screen_name
        
        def on_enter(self):
            self.is_active = True
        
        def on_exit(self):
            self.is_active = False

# ============ COLOR CONSTANTS ============
BG_DARK = (30, 30, 30)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (150, 150, 150)
TEXT_LIGHT_GRAY = (200, 200, 200)
ACCENT_ORANGE = (255, 165, 0)
BUTTON_NORMAL = (70, 130, 180)
BUTTON_HOVER = (100, 160, 220)
SLIDER_BG = (50, 50, 50)
SLIDER_FILL = (100, 180, 255)


class Slider:
    """
    Simple slider untuk mengontrol nilai (volume, difficulty, etc).
    
    Attributes:
        rect: Slider background rect
        value: Current value (0-100)
        min_val: Minimum value
        max_val: Maximum value
        label: Slider label
    """
    
    def __init__(self, x, y, width, height, label, min_val=0, max_val=100, initial_value=50):
        """
        Initialize slider.
        
        Args:
            x, y: Position
            width, height: Dimensions
            label: Slider label
            min_val: Minimum value
            max_val: Maximum value
            initial_value: Starting value
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_value
        self.is_dragging = False
        self.font = None
        
        try:
            self.font = pygame.font.Font(None, 24)
        except:
            pass
    
    def update(self, mouse_pos, is_clicking):
        """
        Update slider berdasarkan mouse position.
        
        Args:
            mouse_pos: (x, y) mouse position
            is_clicking: Is mouse button pressed?
        """
        # Check if mouse over slider
        if self.rect.collidepoint(mouse_pos):
            if is_clicking:
                self.is_dragging = True
        else:
            if not is_clicking:
                self.is_dragging = False
        
        # Update value jika dragging
        if self.is_dragging:
            relative_x = mouse_pos[0] - self.rect.x
            relative_x = max(0, min(relative_x, self.rect.width))
            
            # Calculate value dari position
            percentage = relative_x / self.rect.width
            self.value = int(self.min_val + (self.max_val - self.min_val) * percentage)
    
    def draw(self, surface):
        """
        Draw slider ke surface.
        
        Args:
            surface: Pygame surface
        """
        # Draw label
        if self.font:
            label_text = f"{self.label}: {self.value}"
            label_surface = self.font.render(label_text, True, TEXT_WHITE)
            surface.blit(label_surface, (self.rect.x, self.rect.y - 30))
        
        # Draw background
        pygame.draw.rect(surface, SLIDER_BG, self.rect)
        pygame.draw.rect(surface, TEXT_GRAY, self.rect, 2)
        
        # Draw fill (progress)
        fill_width = (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(surface, SLIDER_FILL, fill_rect)
        
        # Draw value text di sebelah kanan
        if self.font:
            value_text = f"{self.value}"
            value_surface = self.font.render(value_text, True, TEXT_LIGHT_GRAY)
            surface.blit(value_surface, (self.rect.x + self.rect.width + 20, self.rect.y + 5))


class SettingsButton:
    """Simple button untuk settings screen."""
    
    def __init__(self, x, y, width, height, text):
        """
        Initialize button.
        
        Args:
            x, y: Position
            width, height: Dimensions
            text: Button label
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        self.font = None
        
        try:
            self.font = pygame.font.Font(None, 28)
        except:
            pass
    
    def update(self, mouse_pos):
        """Update button hover state."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked_at(self, mouse_pos):
        """Check if button clicked."""
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, surface):
        """Draw button."""
        if not self.font:
            try:
                self.font = pygame.font.Font(None, 28)
            except:
                return
        
        color = BUTTON_HOVER if self.is_hovered else BUTTON_NORMAL
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, TEXT_WHITE, self.rect, 2)
        
        text_surface = self.font.render(self.text, True, TEXT_WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class SettingsScreen(BaseScreen):
    """
    Settings Screen - Mengatur volume dan difficulty.
    
    Attributes:
        sliders: Dict berisi slider objects
        buttons: Dict berisi button objects
        settings_data: Current settings data
    """
    
    def __init__(self, width=1280, height=720, game_manager=None):
        """
        Initialize Settings Screen.
        
        Args:
            width: Screen width (default 1280)
            height: Screen height (default 720)
            game_manager: Reference ke GameManager
        """
        super().__init__(width, height, game_manager)
        
        # Initialize sliders
        self.sliders = {}
        self._init_sliders()
        
        # Initialize buttons
        self.buttons = {}
        self._init_buttons()
        
        # Settings data
        self.settings_data = {
            'master_volume': 80,
            'music_volume': 70,
            'sfx_volume': 80,
            'difficulty': 3  # 1-5
        }
        
        print("✅ SettingsScreen initialized")
    
    # ============ INITIALIZATION ============
    
    def _init_sliders(self):
        """Initialize sliders untuk volume dan difficulty."""
        slider_width = 300
        slider_height = 20
        slider_x = (self.width - slider_width) // 2
        
        # Master Volume (y=150)
        self.sliders['master_volume'] = Slider(
            x=slider_x,
            y=150,
            width=slider_width,
            height=slider_height,
            label='Master Volume',
            min_val=0,
            max_val=100,
            initial_value=80
        )
        
        # Music Volume (y=250)
        self.sliders['music_volume'] = Slider(
            x=slider_x,
            y=250,
            width=slider_width,
            height=slider_height,
            label='Music Volume',
            min_val=0,
            max_val=100,
            initial_value=70
        )
        
        # SFX Volume (y=350)
        self.sliders['sfx_volume'] = Slider(
            x=slider_x,
            y=350,
            width=slider_width,
            height=slider_height,
            label='SFX Volume',
            min_val=0,
            max_val=100,
            initial_value=80
        )
        
        # Difficulty (y=450)
        self.sliders['difficulty'] = Slider(
            x=slider_x,
            y=450,
            width=slider_width,
            height=slider_height,
            label='Difficulty',
            min_val=1,
            max_val=5,
            initial_value=3
        )
    
    def _init_buttons(self):
        """Initialize buttons untuk settings screen."""
        button_width = 200
        button_height = 50
        button_y = self.height - 100
        spacing = 50
        total_width = (button_width * 2) + spacing
        start_x = (self.width - total_width) / 2
        
        # APPLY button
        self.buttons['APPLY'] = SettingsButton(
            x=int(start_x),
            y=button_y,
            width=button_width,
            height=button_height,
            text='APPLY'
        )
        
        # BACK button
        self.buttons['BACK'] = SettingsButton(
            x=int(start_x + button_width + spacing),
            y=button_y,
            width=button_width,
            height=button_height,
            text='BACK'
        )
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            str: Next screen name, atau None jika stay
        """
        # MOUSEMOTION: Update button dan slider
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            
            for button in self.buttons.values():
                button.update(mouse_pos)
        
        # MOUSEBUTTONDOWN: Check slider drag dan button click
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                
                # Start slider drag
                for slider in self.sliders.values():
                    slider.is_dragging = True
                
                # Check button clicks
                if self.buttons['APPLY'].is_clicked_at(mouse_pos):
                    print("✅ APPLY settings")
                    self._apply_settings()
                    return None
                
                if self.buttons['BACK'].is_clicked_at(mouse_pos):
                    print("🏠 BACK to menu")
                    self.set_next_screen('MENU')
                    return 'MENU'
        
        # MOUSEBUTTONUP: Stop slider drag
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                for slider in self.sliders.values():
                    slider.is_dragging = False
        
        # KEYDOWN: ESC untuk kembali
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("🏠 BACK (ESC)")
                self.set_next_screen('MENU')
                return 'MENU'
        
        return None
    
    # ============ UPDATE ============
    
    def update(self):
        """
        Update settings screen.
        """
        # Get current mouse state
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        is_clicking = mouse_buttons[0]
        
        # Update sliders
        for slider in self.sliders.values():
            slider.update(mouse_pos, is_clicking)
            # Update settings data
            self.settings_data[slider.label.lower().replace(' ', '_')] = slider.value
    
    # ============ DRAWING ============
    
    def draw(self, surface):
        """
        Draw settings screen.
        
        Args:
            surface: Pygame surface
        """
        # Fill background
        surface.fill(BG_DARK)
        
        # Draw title
        self._draw_title(surface)
        
        # Draw sliders
        self._draw_sliders(surface)
        
        # Draw buttons
        self._draw_buttons(surface)
    
    def _draw_title(self, surface):
        """Draw SETTINGS title."""
        try:
            font = pygame.font.Font(None, 80)
            title = font.render("SETTINGS", True, ACCENT_ORANGE)
            rect = title.get_rect(center=(self.width // 2, 50))
            surface.blit(title, rect)
        except:
            pass
    
    def _draw_sliders(self, surface):
        """Draw all sliders."""
        for slider in self.sliders.values():
            slider.draw(surface)
    
    def _draw_buttons(self, surface):
        """Draw APPLY dan BACK buttons."""
        for button in self.buttons.values():
            button.draw(surface)
    
    # ============ SETTINGS APPLICATION ============
    
    def _apply_settings(self):
        """Apply settings ke game."""
        print(f"🔊 Master Volume: {self.settings_data['master_volume']}%")
        print(f"🎵 Music Volume: {self.settings_data['music_volume']}%")
        print(f"🔔 SFX Volume: {self.settings_data['sfx_volume']}%")
        print(f"⚙️  Difficulty: {self.settings_data['difficulty']}/5")
        
        # TODO: Apply ke audio manager dan game manager
        if self.game_manager:
            # Pass settings ke GameManager
            pass
    
    def get_settings(self):
        """
        Get current settings.
        
        Returns:
            dict: Current settings data
        """
        return self.settings_data.copy()
    
    def set_settings(self, settings):
        """
        Set settings dari dictionary.
        
        Args:
            settings: Dict dengan settings keys
        """
        for key, value in settings.items():
            if key in self.sliders:
                self.sliders[key].value = value
            self.settings_data[key] = value
    
    # ============ LIFECYCLE ============
    
    def on_enter(self):
        """Called saat screen entered."""
        super().on_enter()
        print("⚙️  Entered SettingsScreen")
    
    def on_exit(self):
        """Called saat screen exited."""
        super().on_exit()
        print("⚙️  Exited SettingsScreen")
    
    # ============ STATUS ============
    
    def get_status(self):
        """Get screen status."""
        return {
            'screen': 'SETTINGS',
            'master_volume': self.sliders['master_volume'].value,
            'music_volume': self.sliders['music_volume'].value,
            'sfx_volume': self.sliders['sfx_volume'].value,
            'difficulty': self.sliders['difficulty'].value
        }


if __name__ == "__main__":
    """Simple visual test."""
    print("\n" + "="*60)
    print("SETTINGS SCREEN - VISUAL TEST")
    print("="*60 + "\n")
    
    pygame.init()
    
    width, height = 1280, 720
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AirBeats - Settings Screen")
    clock = pygame.time.Clock()
    
    screen = SettingsScreen(width=width, height=height)
    screen.on_enter()
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            next_screen = screen.handle_event(event)
            if next_screen:
                print(f"\n→ Transitioning to: {next_screen}")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
        
        screen.update()
        screen.draw(surface)
        pygame.display.flip()
    
    pygame.quit()
    print("\nVisual test ended.")
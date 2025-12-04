import pygame
try:
    from ui.base_screen import BaseScreen
except ImportError:
    from base_screen import BaseScreen

try:
    from components.button import Button
    from components.text import Text
    from components.color import BG_DARK, TEXT_WHITE, ACCENT_ORANGE
except ImportError:
    BG_DARK = (30, 30, 30)
    TEXT_WHITE = (255, 255, 255)
    ACCENT_ORANGE = (255, 165, 0)
    
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
            color = (100, 160, 220) if (self.is_hovered or self.is_focused) else (70, 130, 180)
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
            text_surface = self.font.render(self.text, True, TEXT_WHITE)
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)


class MenuScreen(BaseScreen):
    def __init__(self, width, height, game_manager=None):
        super().__init__(width, height, game_manager)
        
        # Fonts
        self.font_title = pygame.font.Font(None, 80)
        self.font_button = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Calculate positions
        center_x = width // 2
        title_y = 80
        buttons_start_y = 280
        button_spacing = 100
        button_width = 300
        button_height = 60
        
        # Buttons
        self.buttons = {
            'start': Button(
                center_x - button_width // 2,
                buttons_start_y,
                button_width,
                button_height,
                "START GAME",
                self.font_button,
                callback=self.on_start_clicked
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
            'exit': Button(
                center_x - button_width // 2,
                buttons_start_y + button_spacing * 2,
                button_width,
                button_height,
                "EXIT",
                self.font_button,
                callback=self.on_exit_clicked
            )
        }
        
        # Button order untuk keyboard navigation
        self.button_order = ['start', 'settings', 'exit']
        self.selected_button_index = 0
        self.buttons['start'].set_focused(True)
        
        # Title text
        self.title_text = "AirBeats"
        self.subtitle_text = "Touchless Piano Tiles"
        
        # Transition flags
        self.should_exit = False
        
        print(f"MenuScreen initialized")
    
    # EVENT HANDLING
    
    def handle_event(self, event):
        
        # KEYBOARD NAVIGATION
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
            
            # ESC (exit)
            elif event.key == pygame.K_ESCAPE:
                self.should_exit = True
                return None
        
        # MOUSE EVENTS
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
    
    # BUTTON NAVIGATION
    
    def _next_button(self):
        # Deselect current
        self.buttons[self.button_order[self.selected_button_index]].set_focused(False)
        
        # Move to next
        self.selected_button_index = (self.selected_button_index + 1) % len(self.button_order)
        
        # Select new
        self.buttons[self.button_order[self.selected_button_index]].set_focused(True)
        
        print(f"Selected: {self.button_order[self.selected_button_index]}")
    
    def _previous_button(self):
        # Deselect current
        self.buttons[self.button_order[self.selected_button_index]].set_focused(False)
        
        # Move to previous
        self.selected_button_index = (self.selected_button_index - 1) % len(self.button_order)
        
        # Select new
        self.buttons[self.button_order[self.selected_button_index]].set_focused(True)
        
        print(f"Selected: {self.button_order[self.selected_button_index]}")
    
    def _click_selected_button(self):
        button_name = self.button_order[self.selected_button_index]
        button = self.buttons[button_name]
        button._on_click()
    
    # BUTTON CALLBACKS
    
    def on_start_clicked(self):
        self.set_next_screen("GAME")
    
    def on_settings_clicked(self):
        self.set_next_screen("SETTINGS")
    
    def on_exit_clicked(self):
        self.should_exit = True
    
    def _get_transition_for_button(self, button_name):
        if button_name == 'start':
            self.set_next_screen("GAME")
            return "GAME"
        elif button_name == 'settings':
            self.set_next_screen("SETTINGS")
            return "SETTINGS"
        elif button_name == 'exit':
            self.should_exit = True
            return None
        return None
    
    # UPDATE
    
    def update(self):
        pass
    
    # RENDERING
    
    def draw(self, surface):
        # Background
        surface.fill(BG_DARK)
        
        # Title
        title_surface = self.font_title.render(self.title_text, True, ACCENT_ORANGE)
        title_rect = title_surface.get_rect(center=(self.width // 2, 80))
        surface.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle_surface = self.font_small.render(self.subtitle_text, True, TEXT_WHITE)
        subtitle_rect = subtitle_surface.get_rect(center=(self.width // 2, 170))
        surface.blit(subtitle_surface, subtitle_rect)
        
        # Buttons
        for button in self.buttons.values():
            button.draw(surface)
        
        # Version text
        version_text = self.font_small.render("v0.1 - Week 2", True, (100, 100, 100))
        surface.blit(version_text, (20, self.height - 40))
    
    # LIFECYCLE
    
    def on_enter(self):
        super().on_enter()
        print("Entered MenuScreen")
        # Reset selected button
        self.selected_button_index = 0
        self.buttons['start'].set_focused(True)
    
    def on_exit(self):
        super().on_exit()
        print("Exited MenuScreen")
    
    # STATUS
    
    def is_exiting(self):
        return self.should_exit
    
    def get_selected_button(self):
        return self.button_order[self.selected_button_index]
    
    def __str__(self):
        return f"MenuScreen (selected: {self.get_selected_button()})"
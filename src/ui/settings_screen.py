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

# COLOR CONSTANTS
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
    def __init__(self, x, y, width, height, label, min_val=0, max_val=100, initial_value=50):
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


class SettingsButton:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        self.font = None
        
        try:
            self.font = pygame.font.Font(None, 28)
        except:
            pass
    
    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked_at(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, surface):
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


class DifficultyButton:
    def __init__(self, x, y, width, height, difficulty_name, difficulty_label, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.difficulty_name = difficulty_name
        self.difficulty_label = difficulty_label
        self.color = color
        self.is_hovered = False
        self.is_selected = False
        self.font = None
        
        try:
            self.font = pygame.font.Font(None, 24)
        except:
            pass
    
    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked_at(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, surface):
        if not self.font:
            try:
                self.font = pygame.font.Font(None, 24)
            except:
                return
        
        # Determine button color based on state
        if self.is_selected:
            bg_color = self.color
            border_width = 4
        elif self.is_hovered:
            # Lighter version of color when hovered
            bg_color = tuple(min(c + 30, 255) for c in self.color)
            border_width = 3
        else:
            # Darker version when not selected
            bg_color = tuple(c // 2 for c in self.color)
            border_width = 2
        
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, TEXT_WHITE, self.rect, border_width)
        
        text_surface = self.font.render(self.difficulty_label, True, TEXT_WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class SongSelector:
    def __init__(self, x, y, width, height, songs):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.songs = songs
        self.selected_song_id = songs[0][0] if songs else None
        self.song_buttons = []
        self.font = None
        
        try:
            self.font = pygame.font.Font(None, 22)
        except:
            pass
        
        # Create button for each song
        button_height = 40
        button_spacing = 10
        for i, (song_id, song_name) in enumerate(songs):
            btn_y = y + i * (button_height + button_spacing)
            btn = {
                'rect': pygame.Rect(x, btn_y, width, button_height),
                'song_id': song_id,
                'song_name': song_name,
                'is_hovered': False
            }
            self.song_buttons.append(btn)
    
    def update(self, mouse_pos):
        for btn in self.song_buttons:
            btn['is_hovered'] = btn['rect'].collidepoint(mouse_pos)
    
    def handle_click(self, mouse_pos):
        for btn in self.song_buttons:
            if btn['rect'].collidepoint(mouse_pos):
                self.selected_song_id = btn['song_id']
                return btn['song_id']
        return None
    
    def draw(self, surface):
        if not self.font:
            try:
                self.font = pygame.font.Font(None, 22)
            except:
                return
        
        for btn in self.song_buttons:
            # Determine colors
            is_selected = (btn['song_id'] == self.selected_song_id)
            
            if is_selected:
                bg_color = ACCENT_ORANGE
                border_width = 3
            elif btn['is_hovered']:
                bg_color = BUTTON_HOVER
                border_width = 2
            else:
                bg_color = BUTTON_NORMAL
                border_width = 2
            
            # Draw button
            pygame.draw.rect(surface, bg_color, btn['rect'])
            pygame.draw.rect(surface, TEXT_WHITE, btn['rect'], border_width)
            
            # Draw text
            text_surface = self.font.render(btn['song_name'], True, TEXT_WHITE)
            text_rect = text_surface.get_rect(center=btn['rect'].center)
            surface.blit(text_surface, text_rect)


class SettingsScreen(BaseScreen):
    def __init__(self, width=1280, height=720, game_manager=None):
        super().__init__(width, height, game_manager)
        
        # Initialize sliders
        self.sliders = {}
        self._init_sliders()
        
        # Initialize difficulty buttons
        self.difficulty_buttons = []
        self._init_difficulty_buttons()
        
        # Initialize song selector
        self.song_selector = None
        self._init_song_selector()
        
        # Initialize buttons
        self.buttons = {}
        self._init_buttons()
        
        # Settings data
        self.settings_data = {
            'difficulty': 'MEDIUM',  # EASY, MEDIUM, HARD, EXPERT
            'song': 'twinkle'  # Song ID
        }
        
        # Load current settings from game_manager if available
        if self.game_manager:
            if hasattr(self.game_manager, 'difficulty_manager'):
                self.settings_data['difficulty'] = self.game_manager.difficulty_manager.get_current_difficulty()
            if hasattr(self.game_manager, 'song_manager'):
                self.settings_data['song'] = self.game_manager.song_manager.current_song
        
        # Update UI to reflect current settings
        self._update_ui_from_settings()
        
        print("SettingsScreen initialized")
    
    # INITIALIZATION 
    
    def _init_sliders(self):
        # Volume sliders removed - only difficulty and song selection needed
        pass
    
    def _init_difficulty_buttons(self):
        # Get difficulty info from game_manager if available
        difficulties = [
            ('EASY', 'Easy', (100, 255, 100)),
            ('MEDIUM', 'Medium', (255, 255, 100)),
            ('HARD', 'Hard', (255, 165, 0)),
            ('EXPERT', 'Expert', (255, 50, 50))
        ]
        
        button_width = 150
        button_height = 60
        button_spacing = 15
        total_width = (button_width * 4) + (button_spacing * 3)
        start_x = (self.width - total_width) // 2
        y_pos = 180
        
        for i, (diff_name, diff_label, color) in enumerate(difficulties):
            btn_x = start_x + i * (button_width + button_spacing)
            btn = DifficultyButton(
                x=btn_x,
                y=y_pos,
                width=button_width,
                height=button_height,
                difficulty_name=diff_name,
                difficulty_label=diff_label,
                color=color
            )
            self.difficulty_buttons.append(btn)
    
    def _init_song_selector(self):
        # Get songs from game_manager if available
        songs = []
        if self.game_manager and hasattr(self.game_manager, 'song_manager'):
            songs = self.game_manager.song_manager.get_all_songs()
        else:
            # Fallback default songs
            songs = [
                ('twinkle', 'Twinkle Twinkle Little Star'),
                ('mary', 'Mary Had a Little Lamb'),
                ('birthday', 'Happy Birthday'),
                ('jingle', 'Jingle Bells')
            ]
        
        selector_width = 500
        selector_height = 240
        selector_x = (self.width - selector_width) // 2
        selector_y = 280
        
        self.song_selector = SongSelector(
            x=selector_x,
            y=selector_y,
            width=selector_width,
            height=selector_height,
            songs=songs
        )
    
    def _init_buttons(self):
        button_width = 250
        button_height = 60
        button_y = self.height - 100
        
        # APPLY button (centered)
        self.buttons['APPLY'] = SettingsButton(
            x=(self.width - button_width) // 2,
            y=button_y,
            width=button_width,
            height=button_height,
            text='APPLY'
        )
    
    def _update_ui_from_settings(self):
        # Update difficulty button selection
        for btn in self.difficulty_buttons:
            btn.is_selected = (btn.difficulty_name == self.settings_data['difficulty'])
        
        # Update song selector
        if self.song_selector:
            self.song_selector.selected_song_id = self.settings_data['song']
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event):
        # MOUSEMOTION: Update button dan slider
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            
            for button in self.buttons.values():
                button.update(mouse_pos)
            
            for diff_btn in self.difficulty_buttons:
                diff_btn.update(mouse_pos)
            
            if self.song_selector:
                self.song_selector.update(mouse_pos)
        
        # MOUSEBUTTONDOWN: Check slider drag dan button click
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                
                # Start slider drag
                for slider in self.sliders.values():
                    slider.is_dragging = True
                
                # Check difficulty button clicks
                for diff_btn in self.difficulty_buttons:
                    if diff_btn.is_clicked_at(mouse_pos):
                        # Deselect all, select clicked
                        for btn in self.difficulty_buttons:
                            btn.is_selected = False
                        diff_btn.is_selected = True
                        self.settings_data['difficulty'] = diff_btn.difficulty_name
                        print(f"Difficulty selected: {diff_btn.difficulty_name}")
                
                # Check song selector clicks
                if self.song_selector:
                    selected_song = self.song_selector.handle_click(mouse_pos)
                    if selected_song:
                        self.settings_data['song'] = selected_song
                        print(f"Song selected: {selected_song}")
                
                # Check button clicks
                if self.buttons['APPLY'].is_clicked_at(mouse_pos):
                    print("APPLY settings")
                    self._apply_settings()
                    # Return to menu after applying
                    self.set_next_screen('MENU')
                    return 'MENU'
        

        
        # KEYDOWN: ESC untuk kembali
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("BACK (ESC)")
                self.set_next_screen('MENU')
                return 'MENU'
        
        return None
    
    # UPDATE 
    
    def update(self):
        # Get current mouse state
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        is_clicking = mouse_buttons[0]
        
        # Update sliders
        for slider in self.sliders.values():
            slider.update(mouse_pos, is_clicking)
            # Update settings data
            self.settings_data[slider.label.lower().replace(' ', '_')] = slider.value
    
    # DRAWING 
    
    def draw(self, surface):
        # Fill background
        surface.fill(BG_DARK)
        
        # Draw title
        self._draw_title(surface)
        
        # Draw sliders
        
        # Draw difficulty section
        self._draw_difficulty_section(surface)
        
        # Draw song section
        self._draw_song_section(surface)
        
        # Draw buttons
        self._draw_buttons(surface)
    
    def _draw_title(self, surface):
        try:
            font = pygame.font.Font(None, 80)
            title = font.render("SETTINGS", True, ACCENT_ORANGE)
            rect = title.get_rect(center=(self.width // 2, 50))
            surface.blit(title, rect)
        except:
            pass
    
    def _draw_difficulty_section(self, surface):
        try:
            # Draw section label
            font = pygame.font.Font(None, 42)
            label = font.render("Difficulty", True, TEXT_WHITE)
            rect = label.get_rect(center=(self.width // 2, 140))
            surface.blit(label, rect)
            
            # Draw difficulty buttons
            for btn in self.difficulty_buttons:
                btn.draw(surface)
        except:
            pass
    
    def _draw_song_section(self, surface):
        try:
            # Draw section label
            font = pygame.font.Font(None, 42)
            label = font.render("Song Selection", True, TEXT_WHITE)
            rect = label.get_rect(center=(self.width // 2, 260))
            surface.blit(label, rect)
            
            # Draw song selector
            if self.song_selector:
                self.song_selector.draw(surface)
        except:
            pass
    
    def _draw_buttons(self, surface):
        for button in self.buttons.values():
            button.draw(surface)
    
    # SETTINGS APPLICATION
    
    def _apply_settings(self):
        print(f"Difficulty: {self.settings_data['difficulty']}")
        print(f"Song: {self.settings_data['song']}")
        
        # Apply to game manager
        if self.game_manager:
            # Apply difficulty
            if hasattr(self.game_manager, 'difficulty_manager'):
                self.game_manager.difficulty_manager.set_difficulty(self.settings_data['difficulty'])
            
            # Apply song
            if hasattr(self.game_manager, 'song_manager'):
                self.game_manager.song_manager.set_current_song(self.settings_data['song'])
            
            # TODO: Apply volume settings to audio manager
    
    def get_settings(self):
        return self.settings_data.copy()
    
    def set_settings(self, settings):
        for key, value in settings.items():
            if key in self.sliders:
                self.sliders[key].value = value
            self.settings_data[key] = value
        
        self._update_ui_from_settings()
    
    # LIFECYCLE 
    
    def on_enter(self):
        super().on_enter()
        
        # Reload settings from game_manager
        if self.game_manager:
            if hasattr(self.game_manager, 'difficulty_manager'):
                self.settings_data['difficulty'] = self.game_manager.difficulty_manager.get_current_difficulty()
            if hasattr(self.game_manager, 'song_manager'):
                self.settings_data['song'] = self.game_manager.song_manager.current_song
        
        self._update_ui_from_settings()
        print("Entered SettingsScreen")
    
    def on_exit(self):
        super().on_exit()
        print("Exited SettingsScreen")
    
    # STATUS 
    
    def get_status(self):  
        return {
            'screen': 'SETTINGS',
            'master_volume': self.sliders['master_volume'].value,
            'music_volume': self.sliders['music_volume'].value,
            'sfx_volume': self.sliders['sfx_volume'].value,
            'difficulty': self.settings_data['difficulty'],
            'song': self.settings_data['song']
        }


if __name__ == "__main__":
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
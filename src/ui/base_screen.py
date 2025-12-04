import pygame
try:
    from components.color import BG_DARK
except ImportError:
    # Fallback jika components belum di-setup
    BG_DARK = (30, 30, 30)


class BaseScreen:
    
    def __init__(self, width, height, game_manager=None):
        self.width = width
        self.height = height
        self.game_manager = game_manager
        
        # Screen management
        self.next_screen = None  
        self.is_active = True    
        self.fade_out = False    
        
        print(f"BaseScreen initialized: {self.__class__.__name__} ({width}x{height})")
    
    # EVENT HANDLING
    
    def handle_event(self, event):
        return None
    
    # UPDATE
    
    def update(self):
        pass
    
    # RENDERING
    
    def draw(self, surface):
        # Default: draw background
        surface.fill(BG_DARK)
    
    # SCREEN TRANSITION
    
    def get_next_screen(self):
        return self.next_screen
    
    def set_next_screen(self, screen_name):
        self.next_screen = screen_name
        print(f"{self.__class__.__name__} → {screen_name}")
    
    # STATE MANAGEMENT    
    
    def activate(self):
        self.is_active = True
        print(f"{self.__class__.__name__} activated")
    
    def deactivate(self):
        self.is_active = False
        print(f"{self.__class__.__name__} deactivated")
    
    def on_enter(self):
        self.activate()
    
    def on_exit(self):
        self.deactivate()
    
    # UTILITY
    
    def get_screen_name(self):
        return self.__class__.__name__
    
    def get_dimensions(self):
        return (self.width, self.height)
    
    def get_center(self):
        return (self.width // 2, self.height // 2)
    
    def __str__(self):
        return f"{self.get_screen_name()} ({self.width}x{self.height})"
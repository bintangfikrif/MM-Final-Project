"""
Base Screen Module
Parent class untuk semua UI screens

Author: Rafki Haykhal Alif
ITERA - IF25-40305 Sistem Teknologi Multimedia
"""

import pygame
try:
    from components.color import BG_DARK
except ImportError:
    # Fallback jika components belum di-setup
    BG_DARK = (30, 30, 30)


class BaseScreen:
    """
    Abstract base class untuk semua screens di game.
    
    Setiap screen (Menu, Game, Pause, GameOver, Settings) harus inherit
    dari class ini dan implement methods: handle_event, update, draw
    
    Screen lifecycle:
    1. handle_event() - Process user input
    2. update() - Update internal state
    3. draw() - Render ke pygame surface
    4. get_next_screen() - Check if should transition ke screen lain
    """
    
    def __init__(self, width, height, game_manager=None):
        """
        Initialize base screen.
        
        Args:
            width (int): Screen width
            height (int): Screen height
            game_manager (GameManager): Reference ke GameManager (opsional)
        """
        self.width = width
        self.height = height
        self.game_manager = game_manager
        
        # Screen management
        self.next_screen = None  # Next screen name, None jika tidak ada transition
        self.is_active = True    # Screen sedang active?
        self.fade_out = False    # Fade out animation?
        
        print(f"✅ BaseScreen initialized: {self.__class__.__name__} ({width}x{height})")
    
    # ============ EVENT HANDLING ============
    
    def handle_event(self, event):
        """
        Handle pygame event.
        
        OVERRIDE INI DI CHILD CLASSES!
        
        Args:
            event (pygame.event.Event): Event untuk di-handle
            
        Returns:
            str: Next screen name jika ada transition, None jika tidak
        """
        # Default: tidak ada action
        return None
    
    # ============ UPDATE ============
    
    def update(self):
        """
        Update screen logic.
        
        OVERRIDE INI DI CHILD CLASSES!
        
        Called setiap frame untuk:
        - Update animations
        - Check state changes
        - Update UI elements
        """
        pass
    
    # ============ RENDERING ============
    
    def draw(self, surface):
        """
        Draw screen ke pygame surface.
        
        OVERRIDE INI DI CHILD CLASSES!
        
        Args:
            surface (pygame.Surface): Target surface untuk render
        """
        # Default: draw background
        surface.fill(BG_DARK)
    
    # ============ SCREEN TRANSITION ============
    
    def get_next_screen(self):
        """
        Get next screen name untuk transition.
        
        Returns:
            str: Next screen name, atau None jika tidak ada transition
        """
        return self.next_screen
    
    def set_next_screen(self, screen_name):
        """
        Set next screen untuk transition.
        
        Args:
            screen_name (str): Target screen name
        """
        self.next_screen = screen_name
        print(f"🔄 {self.__class__.__name__} → {screen_name}")
    
    # ============ STATE MANAGEMENT ============
    
    def activate(self):
        """Activate screen saat dimasuki."""
        self.is_active = True
        print(f"▶️  {self.__class__.__name__} activated")
    
    def deactivate(self):
        """Deactivate screen saat ditinggalkan."""
        self.is_active = False
        print(f"⏸️  {self.__class__.__name__} deactivated")
    
    def on_enter(self):
        """
        Called saat screen di-enter.
        
        OVERRIDE INI DI CHILD CLASSES jika perlu initialization!
        """
        self.activate()
    
    def on_exit(self):
        """
        Called saat screen di-exit.
        
        OVERRIDE INI DI CHILD CLASSES jika perlu cleanup!
        """
        self.deactivate()
    
    # ============ UTILITY ============
    
    def get_screen_name(self):
        """Get class name sebagai screen identifier."""
        return self.__class__.__name__
    
    def get_dimensions(self):
        """Get screen dimensions."""
        return (self.width, self.height)
    
    def get_center(self):
        """Get center point dari screen."""
        return (self.width // 2, self.height // 2)
    
    def __str__(self):
        """String representation."""
        return f"{self.get_screen_name()} ({self.width}x{self.height})"
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
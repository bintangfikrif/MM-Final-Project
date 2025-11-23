import pygame
import random
import time

class Tile:
    """
    Representasi satu tile piano yang jatuh.
    """
    def __init__(self, lane, width, speed, window_height):
        self.lane = lane
        self.width = width
        self.height = 150  # Tinggi tile
        self.x = lane * width
        self.y = -self.height  # Mulai di atas layar
        self.speed = speed
        self.window_height = window_height
        self.active = True
        self.color = self.get_color(lane)
        self.is_hit = False
        self.missed_reported = False
        
    def get_color(self, lane):
        # Warna berbeda untuk setiap lane (RGB)
        colors = [
            (100, 100, 255),  # Lane 0: Blue-ish
            (100, 255, 100),  # Lane 1: Green-ish
            (255, 100, 100),  # Lane 2: Red-ish
            (255, 255, 100)   # Lane 3: Yellow-ish
        ]
        return colors[lane % 4]

    def update(self):
        self.y += self.speed
        
    def draw(self, surface):
        if not self.active:
            return
            
        # Draw tile rectangle
        rect = pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))
        pygame.draw.rect(surface, self.color, rect)
        
        # Draw border
        pygame.draw.rect(surface, (255, 255, 255), rect, 2)

class TileManager:
    """
    Mengatur spawning dan lifecycle tiles.
    """
    def __init__(self, window_width=640, window_height=480):
        self.window_width = window_width
        self.window_height = window_height
        self.lane_width = window_width // 4
        self.tiles = []
        self.speed = 5
        self.spawn_timer = 0
        self.spawn_rate = 60  # Frames between spawns (approx)
        
        # Hit zone definition (area di bawah layar dimana tap valid)
        self.hit_zone_y = window_height - 150
        self.hit_zone_height = 100
        
        print("✅ TileManager initialized (Pygame)")

    def update(self):
        # Spawn new tiles
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_tile()
            self.spawn_timer = 0
            
        # Update existing tiles
        for tile in self.tiles:
            tile.update()
            
        # Remove off-screen tiles
        self.tiles = [t for t in self.tiles if t.y < self.window_height + 100]

    def spawn_tile(self):
        lane = random.randint(0, 3)
        new_tile = Tile(lane, self.lane_width, self.speed, self.window_height)
        self.tiles.append(new_tile)

    def draw(self, surface):
        # Draw hit zone marker
        pygame.draw.line(surface, (0, 255, 255), (0, self.hit_zone_y), (self.window_width, self.hit_zone_y), 2)
        pygame.draw.line(surface, (0, 255, 255), (0, self.hit_zone_y + self.hit_zone_height), (self.window_width, self.hit_zone_y + self.hit_zone_height), 2)
        
        # Draw lane separators
        for i in range(1, 4):
            x = i * self.lane_width
            pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, self.window_height), 1)

        # Draw tiles
        for tile in self.tiles:
            tile.draw(surface)

    def check_hit(self, lane):
        """
        Check if there is a tile in the hit zone for the given lane.
        Returns the tile if hit, None otherwise.
        """
        for tile in self.tiles:
            if tile.lane == lane and not tile.is_hit:
                # Check intersection with hit zone
                tile_bottom = tile.y + tile.height
                tile_top = tile.y
                
                # Simple overlap check
                if (tile_bottom > self.hit_zone_y and 
                    tile_top < self.hit_zone_y + self.hit_zone_height):
                    
                    tile.is_hit = True
                    tile.active = False # Hide it immediately
                    return tile
                    
        return None

    def check_misses(self):
        """
        Return list of tiles that passed the hit zone without being hit.
        """
        missed = []
        for tile in self.tiles:
            if not tile.is_hit and tile.active:
                if tile.y > self.hit_zone_y + self.hit_zone_height:
                    tile.active = False 
                    if not tile.missed_reported:
                        tile.missed_reported = True
                        missed.append(tile)
        return missed


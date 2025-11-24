"""
Tile Manager Module
"""

import pygame
import time

class Tile:
    """Tile piano yang jatuh"""
    def __init__(self, lane, width, speed, window_height, note='C'):
        self.lane = lane
        self.width = width
        self.height = 150
        self.x = lane * width
        self.y = -self.height
        self.speed = speed
        self.window_height = window_height
        self.active = True
        self.color = self.get_color(lane)
        self.is_hit = False
        self.missed_reported = False
        self.note = note
        
    def get_color(self, lane):
        colors = [
            (100, 100, 255),
            (100, 255, 100),
            (255, 100, 100),
            (255, 255, 100)
        ]
        return colors[lane % 4]

    def update(self):
        self.y += self.speed
        
    def draw(self, surface):
        if not self.active:
            return
        rect = pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2)


class TileManager:
    """Manages tiles"""
    def __init__(self, window_width=640, window_height=480, speed=5):
        self.window_width = window_width
        self.window_height = window_height
        self.lane_width = window_width // 4
        self.tiles = []
        self.speed = speed
        self.hit_zone_y = window_height - 150
        self.hit_zone_height = 100
        self.song_tiles = []
        self.game_start_time = None
        self.song_mode = False
        print(f"✅ TileManager initialized (speed={speed})")
    
    def load_song_tiles(self, tiles_data):
        """Load song tiles"""
        self.song_tiles = tiles_data.copy()
        self.song_mode = True
        self.game_start_time = None
        print(f"🎵 Loaded {len(self.song_tiles)} tiles")
    
    def start_song(self):
        """Start song"""
        self.game_start_time = time.time()
        print("▶️  Song started!")
    
    def update(self):
        """Update tiles"""
        if self.song_mode and self.game_start_time:
            current_time = time.time() - self.game_start_time
            
            for tile_data in self.song_tiles[:]:
                tile_height = 150
                travel_distance = self.hit_zone_y + tile_height
                pixels_per_second = self.speed * 60
                travel_time_seconds = travel_distance / pixels_per_second
                spawn_time = tile_data['time'] - travel_time_seconds
                
                if current_time >= spawn_time - 0.1:
                    new_tile = Tile(
                        lane=tile_data['lane'],
                        width=self.lane_width,
                        speed=self.speed,
                        window_height=self.window_height,
                        note=tile_data['note']
                    )
                    self.tiles.append(new_tile)
                    self.song_tiles.remove(tile_data)
        
        for tile in self.tiles:
            tile.update()
        
        self.tiles = [t for t in self.tiles if t.y < self.window_height + 100]

    def draw(self, surface):
        """Draw tiles and hit zone"""
        pygame.draw.line(surface, (0, 255, 255), 
                        (0, self.hit_zone_y), 
                        (self.window_width, self.hit_zone_y), 2)
        pygame.draw.line(surface, (0, 255, 255), 
                        (0, self.hit_zone_y + self.hit_zone_height), 
                        (self.window_width, self.hit_zone_y + self.hit_zone_height), 2)
        
        for i in range(1, 4):
            x = i * self.lane_width
            pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, self.window_height), 1)

        for tile in self.tiles:
            tile.draw(surface)

    def check_hit(self, lane):
        """Check hit"""
        for tile in self.tiles:
            if tile.lane == lane and not tile.is_hit:
                tile_bottom = tile.y + tile.height
                tile_top = tile.y
                
                if (tile_bottom > self.hit_zone_y and 
                    tile_top < self.hit_zone_y + self.hit_zone_height):
                    tile.is_hit = True
                    tile.active = False
                    return tile
        return None

    def check_misses(self):
        """Check misses"""
        missed = []
        for tile in self.tiles:
            if not tile.is_hit and tile.active:
                if tile.y > self.hit_zone_y + self.hit_zone_height:
                    tile.active = False 
                    if not tile.missed_reported:
                        tile.missed_reported = True
                        missed.append(tile)
        return missed
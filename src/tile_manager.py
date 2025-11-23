import cv2
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
        
    def get_color(self, lane):
        # Warna berbeda untuk setiap lane (BGR)
        colors = [
            (255, 100, 100),  # Lane 0: Blue-ish
            (100, 255, 100),  # Lane 1: Green-ish
            (100, 100, 255),  # Lane 2: Red-ish
            (255, 255, 100)   # Lane 3: Cyan-ish
        ]
        return colors[lane % 4]

    def update(self):
        self.y += self.speed
        
    def draw(self, frame):
        if not self.active:
            return
            
        # Draw tile rectangle
        cv2.rectangle(frame, 
                     (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)), 
                     self.color, 
                     cv2.FILLED)
        
        # Draw border
        cv2.rectangle(frame, 
                     (int(self.x), int(self.y)), 
                     (int(self.x + self.width), int(self.y + self.height)), 
                     (255, 255, 255), 
                     2)

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
        
        print("✅ TileManager initialized")

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
        # Note: Miss detection should be handled by checking if tile passes hit zone
        # Here we just clean up tiles that are way off screen
        self.tiles = [t for t in self.tiles if t.y < self.window_height + 100]

    def spawn_tile(self):
        lane = random.randint(0, 3)
        new_tile = Tile(lane, self.lane_width, self.speed, self.window_height)
        self.tiles.append(new_tile)

    def draw(self, frame):
        # Draw hit zone marker
        cv2.line(frame, (0, self.hit_zone_y), (self.window_width, self.hit_zone_y), (0, 255, 255), 2)
        cv2.line(frame, (0, self.hit_zone_y + self.hit_zone_height), (self.window_width, self.hit_zone_y + self.hit_zone_height), (0, 255, 255), 2)
        
        # Draw lane separators
        for i in range(1, 4):
            x = i * self.lane_width
            cv2.line(frame, (x, 0), (x, self.window_height), (50, 50, 50), 1)

        # Draw tiles
        for tile in self.tiles:
            tile.draw(frame)

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
                # Tile overlaps hit zone if:
                # Tile bottom > Hit Zone Top AND Tile Top < Hit Zone Bottom
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
                    tile.active = False # Mark as processed (visual only, logic handled by caller)
                    # But we keep it in list for a bit to fall off screen? 
                    # Actually for game logic, we just need to report it once.
                    # Let's add a flag 'processed_miss' to Tile if needed, 
                    # or just return it and let caller handle.
                    # For simplicity: if it passed and wasn't hit, it's a miss.
                    # To avoid double counting, we can mark it inactive or have a 'missed' flag.
                    if not getattr(tile, 'missed_reported', False):
                        tile.missed_reported = True
                        missed.append(tile)
        return missed

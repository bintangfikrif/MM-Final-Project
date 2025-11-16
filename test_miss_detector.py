"""
Unit Tests untuk Miss Detector

"""

import sys
sys.path.insert(0, 'src')
from miss_detector import MissDetector


def test_initialization():
    """Test initialization miss detector"""
    print("\n=== TEST 1: Initialization ===")
    
    detector = MissDetector()
    
    assert detector.total_tiles_spawned == 0
    assert detector.total_misses == 0
    assert len(detector.tracked_tiles) == 0
    assert len(detector.missed_tiles) == 0
    
    print("✅ MissDetector initialized correctly")


def test_spawn_tile():
    """Test spawn tile"""
    print("\n=== TEST 2: Spawn Tile ===")
    
    detector = MissDetector()
    
    # Spawn tile 1
    tile1 = detector.spawn_tile(lane=0, base_points=10)
    assert tile1['id'] == 1
    assert tile1['lane'] == 0
    assert tile1['state'] == MissDetector.SPAWNED
    print(f"✅ Tile 1 spawned: {tile1}")
    
    # Spawn tile 2 (auto-id)
    tile2 = detector.spawn_tile(lane=1, base_points=10)
    assert tile2['id'] == 2
    print(f"✅ Tile 2 spawned: {tile2}")
    
    # Spawn tile dengan custom id
    tile3 = detector.spawn_tile(tile_id=10, lane=2, base_points=15)
    assert tile3['id'] == 10
    print(f"✅ Tile 10 spawned (custom id): {tile3}")


def test_tile_lifecycle_hit():
    """Test tile lifecycle: spawn -> enter hit zone -> hit"""
    print("\n=== TEST 3: Tile Lifecycle - HIT ===")
    
    detector = MissDetector()
    
    # Spawn tile
    tile = detector.spawn_tile(lane=0)
    tile_id = tile['id']
    assert tile['state'] == MissDetector.SPAWNED
    print(f"1. Tile {tile_id} spawned")
    
    # Enter hit zone
    result = detector.enter_hit_zone(tile_id)
    assert result == True
    assert detector.get_tile_info(tile_id)['state'] == MissDetector.ACTIVE
    print(f"2. Tile {tile_id} entered hit zone")
    
    # Hit tile
    result = detector.on_tile_hit(tile_id)
    assert result == True
    assert detector.get_tile_info(tile_id)['state'] == MissDetector.HIT
    print(f"3. Tile {tile_id} HIT!")
    
    # Destroy tile
    result = detector.destroy_tile(tile_id)
    assert result == True
    assert tile_id not in detector.tracked_tiles
    print(f"4. Tile {tile_id} destroyed")
    
    print("✅ Hit lifecycle complete")


def test_tile_lifecycle_missed():
    """Test tile lifecycle: spawn -> enter hit zone -> missed"""
    print("\n=== TEST 4: Tile Lifecycle - MISSED ===")
    
    detector = MissDetector()
    
    # Spawn tile
    tile = detector.spawn_tile(lane=1)
    tile_id = tile['id']
    print(f"1. Tile {tile_id} spawned")
    
    # Enter hit zone
    detector.enter_hit_zone(tile_id)
    print(f"2. Tile {tile_id} entered hit zone")
    
    # Exit hit zone tanpa di-hit
    miss_info = detector.exit_hit_zone(tile_id)
    assert miss_info is not None
    assert miss_info['is_miss'] == True
    assert detector.get_tile_info(tile_id)['state'] == MissDetector.MISSED
    assert tile_id in detector.missed_tiles
    assert detector.total_misses == 1
    print(f"3. Tile {tile_id} MISSED!")
    
    print("✅ Missed lifecycle complete")


def test_tile_passed_not_in_hit_zone():
    """Test tile yang lewat tapi tidak pernah masuk hit zone (bukan miss)"""
    print("\n=== TEST 5: Tile Passed (Not in Hit Zone) ===")
    
    detector = MissDetector()
    
    # Spawn tile
    tile = detector.spawn_tile(lane=2)
    tile_id = tile['id']
    print(f"1. Tile {tile_id} spawned")
    
    # Exit hit zone tanpa pernah masuk
    miss_info = detector.exit_hit_zone(tile_id)
    assert miss_info is None  # Tidak dihitung miss
    assert detector.total_misses == 0
    print(f"2. Tile {tile_id} passed (tidak masuk hit zone, bukan miss)")
    
    print("✅ Passed tile handled correctly")


def test_multiple_tiles_mix():
    """Test multiple tiles dengan mix hit dan missed"""
    print("\n=== TEST 6: Multiple Tiles Mix ===")
    
    detector = MissDetector()
    
    # Tile 1: hit
    tile1 = detector.spawn_tile(lane=0)
    detector.enter_hit_zone(tile1['id'])
    detector.on_tile_hit(tile1['id'])
    detector.destroy_tile(tile1['id'])
    print("✅ Tile 1: HIT")
    
    # Tile 2: missed
    tile2 = detector.spawn_tile(lane=1)
    detector.enter_hit_zone(tile2['id'])
    detector.exit_hit_zone(tile2['id'])
    print("✅ Tile 2: MISSED")
    
    # Tile 3: hit
    tile3 = detector.spawn_tile(lane=2)
    detector.enter_hit_zone(tile3['id'])
    detector.on_tile_hit(tile3['id'])
    detector.destroy_tile(tile3['id'])
    print("✅ Tile 3: HIT")
    
    # Tile 4: missed
    tile4 = detector.spawn_tile(lane=3)
    detector.enter_hit_zone(tile4['id'])
    detector.exit_hit_zone(tile4['id'])
    print("✅ Tile 4: MISSED")
    
    # Check status
    status = detector.get_status()
    assert status['total_spawned'] == 4
    assert status['total_missed'] == 2
    assert status['accuracy'] == 50.0  # 2 hit, 2 missed
    assert status['miss_rate'] == 50.0
    print(f"✅ Status: {status['total_spawned']} spawned, accuracy={status['accuracy']:.1f}%")


def test_get_active_tiles():
    """Test get active tiles"""
    print("\n=== TEST 7: Get Active Tiles ===")
    
    detector = MissDetector()
    
    # Spawn 3 tiles
    tile1 = detector.spawn_tile(lane=0)
    tile2 = detector.spawn_tile(lane=1)
    tile3 = detector.spawn_tile(lane=2)
    
    # Enter hit zone untuk 2 tiles
    detector.enter_hit_zone(tile1['id'])
    detector.enter_hit_zone(tile2['id'])
    
    active = detector.get_active_tiles()
    assert len(active) == 2
    assert tile1['id'] in active
    assert tile2['id'] in active
    assert tile3['id'] not in active
    
    print(f"✅ Active tiles: {active}")


def test_get_tile_info():
    """Test get tile info"""
    print("\n=== TEST 8: Get Tile Info ===")
    
    detector = MissDetector()
    
    tile = detector.spawn_tile(tile_id=5, lane=1, base_points=15)
    
    info = detector.get_tile_info(5)
    assert info is not None
    assert info['id'] == 5
    assert info['lane'] == 1
    assert info['base_points'] == 15
    assert info['state'] == MissDetector.SPAWNED
    
    print(f"✅ Tile info: {info}")


def test_get_status():
    """Test get status"""
    print("\n=== TEST 9: Get Status ===")
    
    detector = MissDetector()
    
    # Create gameplay scenario
    for i in range(3):
        tile = detector.spawn_tile(lane=i)
        detector.enter_hit_zone(tile['id'])
        if i < 2:
            detector.on_tile_hit(tile['id'])
            detector.destroy_tile(tile['id'])
        else:
            detector.exit_hit_zone(tile['id'])
    
    status = detector.get_status()
    
    assert status['total_spawned'] == 3
    assert status['total_missed'] == 1
    assert status['total_completed'] == 3
    assert status['accuracy'] == 66.67 or abs(status['accuracy'] - 66.67) < 0.1
    
    print(f"✅ Status correct:")
    print(f"   Total spawned: {status['total_spawned']}")
    print(f"   Total missed: {status['total_missed']}")
    print(f"   Accuracy: {status['accuracy']:.1f}%")


def test_get_missed_tiles():
    """Test get list of missed tiles"""
    print("\n=== TEST 10: Get Missed Tiles ===")
    
    detector = MissDetector()
    
    # Spawn dan miss 3 tiles
    missed_ids = []
    for i in range(3):
        tile = detector.spawn_tile(lane=i)
        detector.enter_hit_zone(tile['id'])
        detector.exit_hit_zone(tile['id'])
        missed_ids.append(tile['id'])
    
    missed = detector.get_missed_tiles()
    
    assert len(missed) == 3
    assert set(missed) == set(missed_ids)
    
    print(f"✅ Missed tiles: {missed}")


def test_reset():
    """Test reset detector"""
    print("\n=== TEST 11: Reset ===")
    
    detector = MissDetector()
    
    # Create some activity
    for i in range(3):
        tile = detector.spawn_tile(lane=i)
        detector.enter_hit_zone(tile['id'])
        detector.exit_hit_zone(tile['id'])
    
    assert detector.total_tiles_spawned == 3
    assert detector.total_misses == 3
    print(f"Before reset: spawned={detector.total_tiles_spawned}, misses={detector.total_misses}")
    
    # Reset
    detector.reset()
    
    assert detector.total_tiles_spawned == 0
    assert detector.total_misses == 0
    assert len(detector.tracked_tiles) == 0
    assert len(detector.missed_tiles) == 0
    print(f"✅ After reset: all cleared")


def test_cannot_hit_already_missed_tile():
    """Test tidak bisa hit tile yang sudah missed"""
    print("\n=== TEST 12: Cannot Hit Missed Tile ===")
    
    detector = MissDetector()
    
    tile = detector.spawn_tile(lane=0)
    tile_id = tile['id']
    
    # Enter dan exit (missed)
    detector.enter_hit_zone(tile_id)
    detector.exit_hit_zone(tile_id)
    
    # Try to hit (seharusnya gagal)
    result = detector.on_tile_hit(tile_id)
    assert result == False
    print("✅ Cannot hit missed tile (rejected)")


def test_cannot_hit_nonexistent_tile():
    """Test tidak bisa hit tile yang tidak ada"""
    print("\n=== TEST 13: Cannot Hit Nonexistent Tile ===")
    
    detector = MissDetector()
    
    # Try to hit tile yang tidak ada
    result = detector.on_tile_hit(999)
    assert result == False
    print("✅ Cannot hit nonexistent tile (rejected)")


def test_full_game_simulation():
    """Test simulasi full game dengan multiple tiles"""
    print("\n=== TEST 14: Full Game Simulation ===")
    
    detector = MissDetector()
    
    print("\nSimulating game flow...")
    
    # Wave 1: 5 tiles, 3 hit, 2 missed
    for i in range(3):
        tile = detector.spawn_tile(lane=i % 4)
        detector.enter_hit_zone(tile['id'])
        detector.on_tile_hit(tile['id'])
        detector.destroy_tile(tile['id'])
    print("✅ Wave 1: 3 hits")
    
    for i in range(2):
        tile = detector.spawn_tile(lane=i % 4)
        detector.enter_hit_zone(tile['id'])
        detector.exit_hit_zone(tile['id'])
    print("✅ Wave 1: 2 misses")
    
    # Wave 2: 4 tiles, 4 hit, 0 missed
    for i in range(4):
        tile = detector.spawn_tile(lane=i % 4)
        detector.enter_hit_zone(tile['id'])
        detector.on_tile_hit(tile['id'])
        detector.destroy_tile(tile['id'])
    print("✅ Wave 2: 4 hits")
    
    # Check final status
    status = detector.get_status()
    assert status['total_spawned'] == 9
    assert status['total_missed'] == 2
    assert abs(status['accuracy'] - 77.78) < 0.1  # 7 hit, 2 missed
    
    print(f"\n✅ Game simulation complete:")
    print(f"   Total spawned: {status['total_spawned']}")
    print(f"   Total missed: {status['total_missed']}")
    print(f"   Accuracy: {status['accuracy']:.2f}%")


def run_all_tests():
    """Jalankan semua tests"""
    print("\n" + "="*60)
    print("MISS DETECTOR - UNIT TESTS")
    print("="*60)
    
    try:
        test_initialization()
        test_spawn_tile()
        test_tile_lifecycle_hit()
        test_tile_lifecycle_missed()
        test_tile_passed_not_in_hit_zone()
        test_multiple_tiles_mix()
        test_get_active_tiles()
        test_get_tile_info()
        test_get_status()
        test_get_missed_tiles()
        test_reset()
        test_cannot_hit_already_missed_tile()
        test_cannot_hit_nonexistent_tile()
        test_full_game_simulation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

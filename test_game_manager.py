"""
Unit Tests untuk Game Manager

"""

import sys
import time
sys.path.insert(0, 'src')
from game_manager import GameManager


def test_initialization():
    """Test inisialisasi game manager"""
    print("\n=== TEST 1: Initialization ===")
    
    gm = GameManager(game_duration=60, max_misses=3)
    
    assert gm.state_machine.is_menu() == True
    assert gm.score_manager.total_score == 0
    assert gm.combo_counter.current_combo == 0
    assert gm.timer.is_running == False
    
    print("✅ GameManager initialized correctly")
    print(f"   State: {gm.state_machine.get_state()}")
    print(f"   Score: {gm.score_manager.total_score}")


def test_start_game():
    """Test start game"""
    print("\n=== TEST 2: Start Game ===")
    
    gm = GameManager(game_duration=30, max_misses=3)
    
    # Should start at MENU
    assert gm.state_machine.is_menu() == True
    
    # Start game
    result = gm.start_game()
    
    assert result == True
    assert gm.state_machine.is_playing() == True
    assert gm.timer.is_running == True
    assert gm.game_session_active == True
    
    print("✅ Game started successfully")
    print(f"   State: {gm.state_machine.get_state()}")
    print(f"   Timer running: {gm.timer.is_running}")


def test_pause_resume():
    """Test pause dan resume game"""
    print("\n=== TEST 3: Pause & Resume ===")
    
    gm = GameManager()
    gm.start_game()
    
    # Pause
    result = gm.pause_game()
    assert result == True
    assert gm.state_machine.is_paused() == True
    assert gm.timer.is_paused == True
    print("✅ Game paused")
    
    # Resume
    result = gm.resume_game()
    assert result == True
    assert gm.state_machine.is_playing() == True
    assert gm.timer.is_paused == False
    print("✅ Game resumed")


def test_tile_hit():
    """Test hit tile"""
    print("\n=== TEST 4: Tile Hit ===")
    
    gm = GameManager()
    gm.start_game()
    
    # Hit tile 1
    result = gm.on_tile_hit(base_points=10)
    
    assert result is not None
    assert result['combo'] == 1
    assert result['multiplier'] == 1.0
    assert result['total_score'] == 10
    print(f"✅ Hit 1: score={result['total_score']}, combo={result['combo']}x")
    
    # Hit tile 2
    result = gm.on_tile_hit(base_points=10)
    
    assert result['combo'] == 2
    assert result['multiplier'] == 1.1
    assert result['total_score'] == 21
    print(f"✅ Hit 2: score={result['total_score']}, combo={result['combo']}x")
    
    # Hit tile 5 (untuk milestone)
    for i in range(3):
        result = gm.on_tile_hit(base_points=10)
    
    assert result['combo'] == 5
    assert result['is_milestone'] == True
    print(f"✅ Hit 5: milestone=True, combo={result['combo']}x")


def test_tile_miss():
    """Test miss tile"""
    print("\n=== TEST 5: Tile Miss ===")
    
    gm = GameManager(max_misses=3)
    gm.start_game()
    
    # Hit 3x
    for i in range(3):
        gm.on_tile_hit(10)
    
    assert gm.combo_counter.current_combo == 3
    print(f"Before miss: combo={gm.combo_counter.current_combo}x")
    
    # Miss
    result = gm.on_tile_miss()
    
    assert result is not None
    assert result['combo_broken'] == True
    assert result['miss_count'] == 1
    assert result['game_over'] == False
    assert gm.combo_counter.current_combo == 0
    print(f"✅ After miss 1: combo reset, miss_count={result['miss_count']}")


def test_game_over_by_misses():
    """Test game over saat miss limit tercapai"""
    print("\n=== TEST 6: Game Over by Misses ===")
    
    gm = GameManager(max_misses=3)  # Change to 3 misses
    gm.start_game()
    
    # Miss 1
    result = gm.on_tile_miss()
    assert result['game_over'] == False
    assert gm.state_machine.is_playing() == True
    print("✅ Miss 1: game continues")
    
    # Miss 2
    result = gm.on_tile_miss()
    assert result['game_over'] == False
    assert gm.state_machine.is_playing() == True
    print("✅ Miss 2: game continues")
    
    # Miss 3 (over limit)
    result = gm.on_tile_miss()
    assert result['game_over'] == True
    assert gm.state_machine.is_game_over() == True
    print("✅ Miss 3: GAME OVER triggered!")


def test_end_game():
    """Test end game dan return to menu"""
    print("\n=== TEST 7: End Game & Return to Menu ===")
    
    gm = GameManager()
    gm.start_game()
    
    # Hit beberapa tile
    for i in range(5):
        gm.on_tile_hit(10)
    
    score_before = gm.score_manager.total_score
    
    # End game
    result = gm.end_game()
    
    assert result == True
    assert gm.state_machine.is_game_over() == True
    assert gm.timer.is_running == False
    assert gm.final_score == score_before
    print(f"✅ Game ended, final score: {gm.final_score}")
    
    # Return to menu
    result = gm.return_to_menu()
    
    assert result == True
    assert gm.state_machine.is_menu() == True
    print("✅ Returned to menu")


def test_game_status():
    """Test get game status"""
    print("\n=== TEST 8: Game Status ===")
    
    gm = GameManager()
    gm.start_game()
    
    # Hit 3x
    for i in range(3):
        gm.on_tile_hit(10)
    
    # Miss 1x
    gm.on_tile_miss()
    
    status = gm.get_game_status()
    
    assert status['state'] == 'PLAYING'
    assert status['score'] == 33  # 10 + 11 + 12
    assert status['combo'] == 0  # Reset setelah miss
    assert status['max_combo'] == 3
    assert status['misses'] == 1
    assert status['total_hits'] == 3
    assert abs(status['accuracy'] - 75.0) < 0.1  # 3/(3+1) = 75%
    
    print("✅ Status correct:")
    print(f"   Score: {status['score']}")
    print(f"   Combo: {status['combo']}x (max: {status['max_combo']}x)")
    print(f"   Misses: {status['misses']}")
    print(f"   Accuracy: {status['accuracy']:.1f}%")


def test_final_stats():
    """Test final stats"""
    print("\n=== TEST 9: Final Stats ===")
    
    gm = GameManager()
    gm.start_game()
    
    # Simulate gameplay
    for i in range(5):
        gm.on_tile_hit(10)
    
    gm.on_tile_miss()
    
    for i in range(3):
        gm.on_tile_hit(10)
    
    # End game
    gm.end_game()
    
    stats = gm.get_final_stats()
    
    assert stats['final_score'] > 0
    assert stats['max_combo'] > 0
    assert stats['total_hits'] > 0
    assert stats['total_misses'] == 1
    
    print("✅ Final stats:")
    print(f"   Final Score: {stats['final_score']}")
    print(f"   Max Combo: {stats['max_combo']}x")
    print(f"   Total Hits: {stats['total_hits']}")
    print(f"   Total Misses: {stats['total_misses']}")


def test_full_game_flow():
    """Test full game flow: MENU -> PLAYING -> PAUSED -> PLAYING -> GAME_OVER -> MENU"""
    print("\n=== TEST 10: Full Game Flow ===")
    
    gm = GameManager(game_duration=20, max_misses=3)
    
    # Start at MENU
    assert gm.state_machine.is_menu()
    print("1. State: MENU")
    
    # Start game
    gm.start_game()
    assert gm.state_machine.is_playing()
    print("2. State: PLAYING (game started)")
    
    # Hit some tiles
    for i in range(3):
        gm.on_tile_hit(10)
    print(f"3. Hit 3 tiles: score={gm.score_manager.total_score}, combo={gm.combo_counter.current_combo}x")
    
    # Pause
    gm.pause_game()
    assert gm.state_machine.is_paused()
    print("4. State: PAUSED")
    
    # Resume
    gm.resume_game()
    assert gm.state_machine.is_playing()
    print("5. State: PLAYING (resumed)")
    
    # Miss
    gm.on_tile_miss()
    print(f"6. Miss: combo reset, misses={gm.score_manager.miss_count}")
    
    # Hit more
    for i in range(2):
        gm.on_tile_hit(10)
    print(f"7. Hit 2 more tiles: score={gm.score_manager.total_score}")
    
    # End game
    gm.end_game()
    assert gm.state_machine.is_game_over()
    print("8. State: GAME_OVER")
    
    # Return to menu
    gm.return_to_menu()
    assert gm.state_machine.is_menu()
    print("9. State: MENU")
    
    print("✅ Full game flow completed successfully!")


def test_cannot_hit_outside_playing():
    """Test tidak bisa hit tile saat tidak PLAYING"""
    print("\n=== TEST 11: Cannot Hit Outside PLAYING ===")
    
    gm = GameManager()
    
    # Try hit at MENU
    result = gm.on_tile_hit(10)
    assert result is None
    print("✅ Hit rejected at MENU")
    
    # Start game
    gm.start_game()
    
    # Hit should work
    result = gm.on_tile_hit(10)
    assert result is not None
    print("✅ Hit accepted at PLAYING")
    
    # Pause
    gm.pause_game()
    
    # Try hit at PAUSED
    result = gm.on_tile_hit(10)
    assert result is None
    print("✅ Hit rejected at PAUSED")


def test_state_transitions():
    """Test valid state transitions"""
    print("\n=== TEST 12: State Transitions ===")
    
    gm = GameManager()
    
    # MENU -> PLAYING
    assert gm.start_game() == True
    assert gm.state_machine.is_playing()
    print("✅ MENU -> PLAYING: OK")
    
    # PLAYING -> PAUSED
    assert gm.pause_game() == True
    assert gm.state_machine.is_paused()
    print("✅ PLAYING -> PAUSED: OK")
    
    # PAUSED -> PLAYING
    assert gm.resume_game() == True
    assert gm.state_machine.is_playing()
    print("✅ PAUSED -> PLAYING: OK")
    
    # PLAYING -> GAME_OVER
    assert gm.end_game() == True
    assert gm.state_machine.is_game_over()
    print("✅ PLAYING -> GAME_OVER: OK")
    
    # GAME_OVER -> MENU
    assert gm.return_to_menu() == True
    assert gm.state_machine.is_menu()
    print("✅ GAME_OVER -> MENU: OK")


def run_all_tests():
    """Jalankan semua tests"""
    print("\n" + "="*60)
    print("GAME MANAGER - UNIT TESTS")
    print("="*60)
    
    try:
        test_initialization()
        test_start_game()
        test_pause_resume()
        test_tile_hit()
        test_tile_miss()
        test_game_over_by_misses()
        test_end_game()
        test_game_status()
        test_final_stats()
        test_full_game_flow()
        test_cannot_hit_outside_playing()
        test_state_transitions()
        
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

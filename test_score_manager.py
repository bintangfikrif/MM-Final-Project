"""
Unit Tests for Score Manager

"""

import sys
sys.path.insert(0, 'src')

from score_manager import ScoreManager


def test_basic_scoring():
    """Test basic hit scoring"""
    print("\n=== TEST 1: Basic Scoring ===")
    score_mgr = ScoreManager()
    
    # First hit: 10 points, 1x multiplier
    points1 = score_mgr.add_hit(10)
    assert points1 == 10, f"Expected 10, got {points1}"
    assert score_mgr.total_score == 10
    assert score_mgr.current_combo == 1
    print("✅ First hit: 10 points")
    
    # Second hit: 10 points × 1.1 = 11 points
    points2 = score_mgr.add_hit(10)
    assert points2 == 11, f"Expected 11, got {points2}"
    assert score_mgr.total_score == 21, f"Expected 21, got {score_mgr.total_score}"
    assert score_mgr.current_combo == 2
    print("✅ Second hit: 11 points (1.1x multiplier)")
    
    # Third hit: 10 points × 1.2 = 12 points
    points3 = score_mgr.add_hit(10)
    assert points3 == 12, f"Expected 12, got {points3}"
    assert score_mgr.total_score == 33, f"Expected 33, got {score_mgr.total_score}"
    assert score_mgr.current_combo == 3
    print("✅ Third hit: 12 points (1.2x multiplier)")


def test_combo_system():
    """Test combo multiplier system"""
    print("\n=== TEST 2: Combo System ===")
    score_mgr = ScoreManager()
    
    # Build a combo of 5 hits
    total_score = 0
    for i in range(5):
        points = score_mgr.add_hit(10)
        total_score += points
        multiplier = score_mgr.get_combo_multiplier()
        print(f"  Hit {i+1}: +{points} points | Multiplier: {multiplier:.1f}x")
    
    assert score_mgr.current_combo == 5
    assert score_mgr.max_combo == 5
    print(f"✅ Combo reached 5x | Total score: {total_score}")


def test_combo_reset_on_miss():
    """Test that combo resets when missing"""
    print("\n=== TEST 3: Combo Reset on Miss ===")
    score_mgr = ScoreManager()
    
    # Hit 3 times
    score_mgr.add_hit(10)
    score_mgr.add_hit(10)
    score_mgr.add_hit(10)
    
    assert score_mgr.current_combo == 3
    print(f"  Before miss: combo = {score_mgr.current_combo}")
    
    # Miss
    score_mgr.add_miss()
    
    assert score_mgr.current_combo == 0, f"Combo should be 0, got {score_mgr.current_combo}"
    print(f"✅ After miss: combo reset to {score_mgr.current_combo}")
    
    # Next hit should be 1x multiplier again
    points = score_mgr.add_hit(10)
    assert points == 10, f"Expected 10, got {points}"
    print(f"✅ Next hit: 10 points (1.0x multiplier)")


def test_miss_tracking():
    """Test miss tracking and game over"""
    print("\n=== TEST 4: Miss Tracking & Game Over ===")
    score_mgr = ScoreManager(max_misses=3)
    
    # Add 2 misses
    game_over = score_mgr.add_miss()
    assert not game_over, "Should not be game over after 1 miss"
    print(f"  After miss 1: {score_mgr.miss_count}/{score_mgr.max_misses}")
    
    game_over = score_mgr.add_miss()
    assert not game_over, "Should not be game over after 2 misses"
    print(f"  After miss 2: {score_mgr.miss_count}/{score_mgr.max_misses}")
    
    # Third miss = game over
    game_over = score_mgr.add_miss()
    assert game_over, "Should be game over after 3 misses"
    assert score_mgr.is_game_over(), "is_game_over() should return True"
    print(f"  After miss 3: {score_mgr.miss_count}/{score_mgr.max_misses}")
    print("✅ Game over triggered correctly!")


def test_game_summary():
    """Test getting game summary statistics"""
    print("\n=== TEST 5: Game Summary ===")
    score_mgr = ScoreManager()
    
    # Play a short game
    score_mgr.add_hit(10)  # 10
    score_mgr.add_hit(10)  # 11
    score_mgr.add_miss()   # reset combo
    score_mgr.add_hit(10)  # 10
    score_mgr.add_hit(10)  # 11
    
    status = score_mgr.get_status()
    
    print(f"  Final Score: {status['score']}")
    print(f"  Max Combo: {status['max_combo']}")
    print(f"  Total Hits: {status['hits_total']}")
    print(f"  Total Misses: {status['misses']}")
    print(f"  Accuracy: {status['accuracy']:.1f}%")
    
    assert status['score'] == 42, f"Expected score 42, got {status['score']}"
    assert status['max_combo'] == 2, f"Expected max_combo 2, got {status['max_combo']}"
    assert status['hits_total'] == 4, f"Expected 4 hits, got {status['hits_total']}"
    assert status['misses'] == 1, f"Expected 1 miss, got {status['misses']}"
    
    print("✅ Game summary correct!")


def test_reset():
    """Test resetting score manager for new game"""
    print("\n=== TEST 6: Reset ===")
    score_mgr = ScoreManager()
    
    # Play and accumulate stats
    score_mgr.add_hit(10)
    score_mgr.add_hit(10)
    score_mgr.add_miss()
    
    assert score_mgr.total_score > 0
    assert score_mgr.miss_count > 0
    print(f"  Before reset: score={score_mgr.total_score}, misses={score_mgr.miss_count}")
    
    # Reset
    score_mgr.reset()
    
    assert score_mgr.total_score == 0
    assert score_mgr.current_combo == 0
    assert score_mgr.miss_count == 0
    assert score_mgr.max_combo == 0
    print(f"  After reset: score={score_mgr.total_score}, misses={score_mgr.miss_count}")
    print("✅ Reset successful!")


def test_custom_max_misses():
    """Test custom max misses value"""
    print("\n=== TEST 7: Custom Max Misses ===")
    
    # Create with 5 max misses
    score_mgr = ScoreManager(max_misses=5)
    
    # Add 4 misses
    for i in range(4):
        game_over = score_mgr.add_miss()
        assert not game_over, f"Should not be game over at {i+1} misses"
    
    # 5th miss should trigger game over
    game_over = score_mgr.add_miss()
    assert game_over, "Should be game over at 5 misses"
    assert score_mgr.is_game_over()
    print(f"✅ Game over with 5 misses (custom max)")


def test_combo_multiplier():
    """Test combo multiplier calculation"""
    print("\n=== TEST 8: Combo Multiplier ===")
    score_mgr = ScoreManager()
    
    # Test multiplier at different combo counts
    test_cases = [
        (0, 1.0),   # combo 0 → 1.0x
        (5, 1.5),   # combo 5 → 1.5x
        (10, 2.0),  # combo 10 → 2.0x
    ]
    
    for combo_value, expected_multiplier in test_cases:
        score_mgr.current_combo = combo_value
        multiplier = score_mgr.get_combo_multiplier()
        assert multiplier == expected_multiplier, f"For combo {combo_value}, expected {expected_multiplier}, got {multiplier}"
        print(f"  Combo {combo_value}: multiplier = {multiplier:.1f}x ✅")


def test_full_game_flow():
    """Test complete game flow with hits and misses"""
    print("\n=== TEST 9: Full Game Flow ===")
    score_mgr = ScoreManager(max_misses=3)
    
    print("  Simulating game flow...")
    
    # Sequence: Hit, Hit, Miss, Hit, Hit, Hit, Miss, Miss, Miss
    actions = ['hit', 'hit', 'miss', 'hit', 'hit', 'hit', 'miss', 'miss', 'miss']
    
    for i, action in enumerate(actions):
        if action == 'hit':
            score_mgr.add_hit(10)
        else:
            game_over = score_mgr.add_miss()
            if game_over:
                print(f"  Game over at action {i+1}")
                break
    
    status = score_mgr.get_status()
    print(f"  Final Status:")
    print(f"    Score: {status['score']}")
    print(f"    Combo: {status['combo']}")
    print(f"    Max Combo: {status['max_combo']}")
    print(f"    Misses: {status['misses']}")
    print(f"    Hits: {status['hits_total']}")
    print(f"    Accuracy: {status['accuracy']:.1f}%")
    
    assert score_mgr.is_game_over(), "Game should be over"
    print("✅ Full game flow completed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SCORE MANAGER - UNIT TESTS")
    print("="*60)
    
    try:
        test_basic_scoring()
        test_combo_system()
        test_combo_reset_on_miss()
        test_miss_tracking()
        test_game_summary()
        test_reset()
        test_custom_max_misses()
        test_combo_multiplier()
        test_full_game_flow()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
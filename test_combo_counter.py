"""
Unit Tests untuk Combo Counter

"""

import sys
sys.path.insert(0, 'src')
from combo_counter import ComboCounter


def test_initial_state():
    """Test initial state combo counter"""
    print("\n=== TEST 1: Initial State ===")
    
    combo = ComboCounter()
    
    assert combo.current_combo == 0
    assert combo.max_combo == 0
    assert combo.get_multiplier() == 1.0
    assert combo.is_combo_active() == False
    
    print("✅ Initial state correct")
    print(f"   Current: {combo.current_combo}x")
    print(f"   Multiplier: {combo.get_multiplier():.1f}x")


def test_add_hit():
    """Test menambah hit"""
    print("\n=== TEST 2: Add Hit ===")
    
    combo = ComboCounter()
    
    # Hit 1: combo=1, multiplier dari combo 0 = 1.0x
    result = combo.add_hit()
    assert result == 1
    assert combo.get_multiplier() == 1.0  # Multiplier dari combo 1-1=0
    print(f"✅ Hit 1: combo={result}, mult={combo.get_multiplier():.1f}x")
    
    # Hit 2: combo=2, multiplier dari combo 1 = 1.1x
    result = combo.add_hit()
    assert result == 2
    assert combo.get_multiplier() == 1.1  # Multiplier dari combo 2-1=1
    print(f"✅ Hit 2: combo={result}, mult={combo.get_multiplier():.1f}x")
    
    # Hit 3: combo=3, multiplier dari combo 2 = 1.2x
    result = combo.add_hit()
    assert result == 3
    assert combo.get_multiplier() == 1.2  # Multiplier dari combo 3-1=2
    print(f"✅ Hit 3: combo={result}, mult={combo.get_multiplier():.1f}x")


def test_add_miss():
    """Test miss dan combo reset"""
    print("\n=== TEST 3: Add Miss (Combo Reset) ===")
    
    combo = ComboCounter()
    
    # Build combo to 5
    for i in range(5):
        combo.add_hit()
    
    assert combo.current_combo == 5
    assert combo.max_combo == 5
    print(f"Before miss: combo={combo.current_combo}x")
    
    # Miss
    result = combo.add_miss()
    
    assert result == 0
    assert combo.current_combo == 0
    assert combo.max_combo == 5  # Max tetap 5
    assert combo.get_multiplier() == 1.0
    
    print(f"✅ After miss: combo={combo.current_combo}x (reset)")
    print(f"✅ Max combo tetap: {combo.max_combo}x")


def test_multiplier_calculation():
    """Test perhitungan multiplier dengan berbagai combo"""
    print("\n=== TEST 4: Multiplier Calculation ===")
    
    combo = ComboCounter()
    
    # Test cases: (combo_input, expected_mult)
    # Karena 0-based: combo 0 -> mult 1.0, combo 1 -> mult 1.1, dst
    test_cases = [
        (0, 1.0),   # 0*0.1 + 1.0 = 1.0
        (1, 1.1),   # 1*0.1 + 1.0 = 1.1
        (4, 1.4),   # 4*0.1 + 1.0 = 1.4 (combo=5 di game)
        (9, 1.9),   # 9*0.1 + 1.0 = 1.9 (combo=10 di game)
        (14, 2.4),  # 14*0.1 + 1.0 = 2.4 (combo=15 di game)
        (19, 2.9),  # 19*0.1 + 1.0 = 2.9 (combo=20 di game)
    ]
    
    for combo_val, expected_mult in test_cases:
        actual_mult = combo.get_multiplier(combo_val)
        # Use approximate comparison untuk float
        assert abs(actual_mult - expected_mult) < 0.0001, f"Expected {expected_mult}, got {actual_mult}"
        print(f"✅ Combo input {combo_val}: {actual_mult:.1f}x")


def test_max_combo_tracking():
    """Test tracking max combo"""
    print("\n=== TEST 5: Max Combo Tracking ===")
    
    combo = ComboCounter()
    
    # Sequence: 3 hits, miss, 4 hits, miss, 5 hits
    
    # First sequence: 3 hits
    for i in range(3):
        combo.add_hit()
    assert combo.max_combo == 3
    print(f"After 3 hits: current={combo.current_combo}, max={combo.max_combo}")
    
    # Miss
    combo.add_miss()
    assert combo.max_combo == 3  # Max tetap
    print(f"After miss: current={combo.current_combo}, max={combo.max_combo}")
    
    # Second sequence: 4 hits
    for i in range(4):
        combo.add_hit()
    assert combo.max_combo == 4  # Max update
    print(f"After 4 hits: current={combo.current_combo}, max={combo.max_combo}")
    
    # Miss
    combo.add_miss()
    
    # Third sequence: 5 hits
    for i in range(5):
        combo.add_hit()
    assert combo.max_combo == 5  # Max update
    print(f"After 5 hits: current={combo.current_combo}, max={combo.max_combo}")
    
    print(f"✅ Final: current={combo.current_combo}x, max={combo.max_combo}x")


def test_is_combo_active():
    """Test combo active check"""
    print("\n=== TEST 6: Is Combo Active ===")
    
    combo = ComboCounter()
    
    assert combo.is_combo_active() == False
    print("✅ Not active at start")
    
    combo.add_hit()
    assert combo.is_combo_active() == True
    print("✅ Active after 1 hit")
    
    combo.add_miss()
    assert combo.is_combo_active() == False
    print("✅ Not active after miss")


def test_milestone_detection():
    """Test milestone detection (5, 10, 15, dst)"""
    print("\n=== TEST 7: Milestone Detection ===")
    
    combo = ComboCounter()
    
    # Hit sampai 5
    for i in range(5):
        combo.add_hit()
    
    assert combo.get_combo_milestone() == 5
    assert combo.should_show_milestone_popup() == True
    print(f"✅ At combo 5: milestone={combo.get_combo_milestone()}, show_popup=True")
    
    # Hit sampai 7
    for i in range(2):
        combo.add_hit()
    
    assert combo.get_combo_milestone() == 5
    assert combo.should_show_milestone_popup() == False
    print(f"✅ At combo 7: milestone={combo.get_combo_milestone()}, show_popup=False")
    
    # Hit sampai 10
    for i in range(3):
        combo.add_hit()
    
    assert combo.get_combo_milestone() == 10
    assert combo.should_show_milestone_popup() == True
    print(f"✅ At combo 10: milestone={combo.get_combo_milestone()}, show_popup=True")


def test_reset():
    """Test reset combo counter"""
    print("\n=== TEST 8: Reset ===")
    
    combo = ComboCounter()
    
    # Build combo
    for i in range(8):
        combo.add_hit()
    
    assert combo.current_combo == 8
    assert combo.max_combo == 8
    print(f"Before reset: current={combo.current_combo}, max={combo.max_combo}")
    
    # Reset
    combo.reset()
    
    assert combo.current_combo == 0
    assert combo.max_combo == 0
    assert len(combo.combo_history) == 0
    
    print(f"✅ After reset: current={combo.current_combo}, max={combo.max_combo}")


def test_get_status():
    """Test get_status method"""
    print("\n=== TEST 9: Get Status ===")
    
    combo = ComboCounter()
    
    # Build combo to 7
    for i in range(7):
        combo.add_hit()
    
    status = combo.get_status()
    
    assert status['current'] == 7
    assert status['max'] == 7
    assert status['multiplier'] == 1.6  # 7-1=6, 6*0.1 + 1.0 = 1.6
    assert status['history_count'] == 7
    
    print(f"✅ Status correct:")
    print(f"   Current: {status['current']}x")
    print(f"   Max: {status['max']}x")
    print(f"   Multiplier: {status['multiplier']:.1f}x")
    print(f"   Events: {status['history_count']}")


def test_history_tracking():
    """Test combo history tracking"""
    print("\n=== TEST 10: History Tracking ===")
    
    combo = ComboCounter()
    
    # Sequence: hit, hit, miss, hit, hit, hit
    combo.add_hit()
    combo.add_hit()
    combo.add_miss()
    combo.add_hit()
    combo.add_hit()
    combo.add_hit()
    
    history = combo.get_history()
    
    assert len(history) == 6
    assert history[0]['type'] == 'hit'
    assert history[2]['type'] == 'miss'
    
    print(f"✅ History tracked: {len(history)} events")
    combo.print_history()


def test_full_game_simulation():
    """Test simulasi game penuh"""
    print("\n=== TEST 11: Full Game Simulation ===")
    
    combo = ComboCounter()
    
    # Simulasi: hit 3x, miss, hit 5x, miss, hit 2x
    print("\nSimulating gameplay...")
    
    # Round 1: 3 hits
    for i in range(3):
        combo.add_hit()
    print(f"Round 1: combo={combo.current_combo}x")
    
    # Miss
    combo.add_miss()
    print(f"Miss: combo={combo.current_combo}x, max={combo.max_combo}x")
    
    # Round 2: 5 hits
    for i in range(5):
        combo.add_hit()
    print(f"Round 2: combo={combo.current_combo}x, max={combo.max_combo}x")
    
    # Miss
    combo.add_miss()
    print(f"Miss: combo={combo.current_combo}x, max={combo.max_combo}x")
    
    # Round 3: 2 hits
    for i in range(2):
        combo.add_hit()
    print(f"Round 3: combo={combo.current_combo}x, max={combo.max_combo}x")
    
    # Verify
    assert combo.current_combo == 2
    assert combo.max_combo == 5
    print(f"\n✅ Final: current={combo.current_combo}x, max={combo.max_combo}x")


def run_all_tests():
    """Jalankan semua tests"""
    print("\n" + "="*60)
    print("COMBO COUNTER - UNIT TESTS")
    print("="*60)
    
    try:
        test_initial_state()
        test_add_hit()
        test_add_miss()
        test_multiplier_calculation()
        test_max_combo_tracking()
        test_is_combo_active()
        test_milestone_detection()
        test_reset()
        test_get_status()
        test_history_tracking()
        test_full_game_simulation()
        
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

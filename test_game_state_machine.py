"""
Unit Tests for Game State Machine

"""

import sys
sys.path.insert(0, 'src')
from game_state_machine import GameState


def test_initial_state():
    """Test bahwa initial state adalah MENU"""
    print("\n=== TEST 1: Initial State ===")
    
    state = GameState()
    
    assert state.current_state == GameState.MENU
    assert state.is_menu() == True
    assert state.is_playing() == False
    
    print("✅ Initial state is MENU")
    print(f"   current_state: {state.current_state}")
    print(f"   is_menu(): {state.is_menu()}")


def test_valid_transition_menu_to_playing():
    """Test valid transition: MENU → PLAYING"""
    print("\n=== TEST 2: Valid Transition MENU → PLAYING ===")
    
    state = GameState()
    
    result = state.transition_to(GameState.PLAYING)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.PLAYING
    assert state.previous_state == GameState.MENU
    assert state.is_playing() == True
    assert state.is_menu() == False
    
    print("✅ Transition MENU → PLAYING successful")
    print(f"   current_state: {state.current_state}")
    print(f"   previous_state: {state.previous_state}")


def test_valid_transition_playing_to_paused():
    """Test valid transition: PLAYING → PAUSED"""
    print("\n=== TEST 3: Valid Transition PLAYING → PAUSED ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    
    result = state.transition_to(GameState.PAUSED)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.PAUSED
    assert state.previous_state == GameState.PLAYING
    assert state.is_paused() == True
    
    print("✅ Transition PLAYING → PAUSED successful")
    print(f"   current_state: {state.current_state}")
    print(f"   previous_state: {state.previous_state}")


def test_valid_transition_paused_to_playing():
    """Test valid transition: PAUSED → PLAYING"""
    print("\n=== TEST 4: Valid Transition PAUSED → PLAYING ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.PAUSED)
    
    result = state.transition_to(GameState.PLAYING)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.PLAYING
    assert state.previous_state == GameState.PAUSED
    
    print("✅ Transition PAUSED → PLAYING successful")


def test_valid_transition_playing_to_gameover():
    """Test valid transition: PLAYING → GAME_OVER"""
    print("\n=== TEST 5: Valid Transition PLAYING → GAME_OVER ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    
    result = state.transition_to(GameState.GAME_OVER)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.GAME_OVER
    assert state.is_game_over() == True
    
    print("✅ Transition PLAYING → GAME_OVER successful")


def test_valid_transition_paused_to_gameover():
    """Test valid transition: PAUSED → GAME_OVER"""
    print("\n=== TEST 6: Valid Transition PAUSED → GAME_OVER ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.PAUSED)
    
    result = state.transition_to(GameState.GAME_OVER)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.GAME_OVER
    
    print("✅ Transition PAUSED → GAME_OVER successful")


def test_valid_transition_gameover_to_menu():
    """Test valid transition: GAME_OVER → MENU"""
    print("\n=== TEST 7: Valid Transition GAME_OVER → MENU ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.GAME_OVER)
    
    result = state.transition_to(GameState.MENU)
    
    assert result == True, "Transition should succeed"
    assert state.current_state == GameState.MENU
    assert state.is_menu() == True
    
    print("✅ Transition GAME_OVER → MENU successful")


def test_invalid_transition_menu_to_paused():
    """Test INVALID transition: MENU → PAUSED (should be rejected)"""
    print("\n=== TEST 8: INVALID Transition MENU → PAUSED ===")
    
    state = GameState()
    
    # MENU hanya bisa ke PLAYING, bukan ke PAUSED
    result = state.transition_to(GameState.PAUSED)
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.MENU  # State tetap MENU
    
    print("✅ INVALID transition correctly rejected")
    print(f"   State tetap di: {state.current_state}")


def test_invalid_transition_menu_to_gameover():
    """Test INVALID transition: MENU → GAME_OVER"""
    print("\n=== TEST 9: INVALID Transition MENU → GAME_OVER ===")
    
    state = GameState()
    
    result = state.transition_to(GameState.GAME_OVER)
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.MENU
    
    print("✅ INVALID transition correctly rejected")


def test_invalid_transition_playing_to_menu():
    """Test INVALID transition: PLAYING → MENU (harus via GAME_OVER)"""
    print("\n=== TEST 10: INVALID Transition PLAYING → MENU ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    
    result = state.transition_to(GameState.MENU)
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.PLAYING
    
    print("✅ INVALID transition correctly rejected")


def test_invalid_transition_paused_to_menu():
    """Test INVALID transition: PAUSED → MENU (harus via PLAYING dulu)"""
    print("\n=== TEST 11: INVALID Transition PAUSED → MENU ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.PAUSED)
    
    result = state.transition_to(GameState.MENU)
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.PAUSED
    
    print("✅ INVALID transition correctly rejected")


def test_invalid_transition_gameover_to_playing():
    """Test INVALID transition: GAME_OVER → PLAYING (harus via MENU)"""
    print("\n=== TEST 12: INVALID Transition GAME_OVER → PLAYING ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.GAME_OVER)
    
    result = state.transition_to(GameState.PLAYING)
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.GAME_OVER
    
    print("✅ INVALID transition correctly rejected")


def test_invalid_state_name():
    """Test transition dengan state name yang tidak valid"""
    print("\n=== TEST 13: Invalid State Name ===")
    
    state = GameState()
    
    result = state.transition_to("INVALID_STATE")
    
    assert result == False, "Transition should fail"
    assert state.current_state == GameState.MENU
    
    print("✅ Invalid state name correctly rejected")


def test_helper_methods():
    """Test semua helper methods"""
    print("\n=== TEST 14: Helper Methods ===")
    
    state = GameState()
    
    # Test at MENU
    assert state.is_menu() == True
    assert state.is_playing() == False
    assert state.is_paused() == False
    assert state.is_game_over() == False
    print("✅ MENU state helpers correct")
    
    # Test at PLAYING
    state.transition_to(GameState.PLAYING)
    assert state.is_menu() == False
    assert state.is_playing() == True
    assert state.is_paused() == False
    assert state.is_game_over() == False
    print("✅ PLAYING state helpers correct")
    
    # Test at PAUSED
    state.transition_to(GameState.PAUSED)
    assert state.is_menu() == False
    assert state.is_playing() == False
    assert state.is_paused() == True
    assert state.is_game_over() == False
    print("✅ PAUSED state helpers correct")
    
    # Test at GAME_OVER
    state.transition_to(GameState.GAME_OVER)
    assert state.is_menu() == False
    assert state.is_playing() == False
    assert state.is_paused() == False
    assert state.is_game_over() == True
    print("✅ GAME_OVER state helpers correct")


def test_get_state():
    """Test get_state() method"""
    print("\n=== TEST 15: get_state() Method ===")
    
    state = GameState()
    
    assert state.get_state() == GameState.MENU
    print(f"✅ get_state() returns: {state.get_state()}")
    
    state.transition_to(GameState.PLAYING)
    assert state.get_state() == GameState.PLAYING
    print(f"✅ get_state() returns: {state.get_state()}")


def test_get_status():
    """Test get_status() method"""
    print("\n=== TEST 16: get_status() Method ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.PAUSED)
    
    status = state.get_status()
    
    assert status['current'] == GameState.PAUSED
    assert status['previous'] == GameState.PLAYING
    assert status['valid_next_states'] == [GameState.PLAYING, GameState.GAME_OVER]
    
    print(f"✅ get_status() returns correct info:")
    print(f"   current: {status['current']}")
    print(f"   previous: {status['previous']}")
    print(f"   valid_next_states: {status['valid_next_states']}")


def test_reset():
    """Test reset() method"""
    print("\n=== TEST 17: reset() Method ===")
    
    state = GameState()
    state.transition_to(GameState.PLAYING)
    state.transition_to(GameState.GAME_OVER)
    
    assert state.current_state == GameState.GAME_OVER
    print(f"Before reset: {state.current_state}")
    
    state.reset()
    
    assert state.current_state == GameState.MENU
    print(f"After reset: {state.current_state}")
    print("✅ reset() successfully returns to MENU")


def test_full_game_flow():
    """Test full game flow: MENU → PLAYING → PAUSED → PLAYING → GAME_OVER → MENU"""
    print("\n=== TEST 18: Full Game Flow ===")
    
    state = GameState()
    
    # Start
    assert state.is_menu()
    print("1. State: MENU")
    
    # Start game
    state.transition_to(GameState.PLAYING)
    assert state.is_playing()
    print("2. State: PLAYING (game started)")
    
    # Pause
    state.transition_to(GameState.PAUSED)
    assert state.is_paused()
    print("3. State: PAUSED (player paused)")
    
    # Resume
    state.transition_to(GameState.PLAYING)
    assert state.is_playing()
    print("4. State: PLAYING (resumed)")
    
    # Game Over
    state.transition_to(GameState.GAME_OVER)
    assert state.is_game_over()
    print("5. State: GAME_OVER (time/misses)")
    
    # Back to Menu
    state.transition_to(GameState.MENU)
    assert state.is_menu()
    print("6. State: MENU")
    
    print("✅ Full game flow completed successfully")


def run_all_tests():
    """Run semua tests"""
    print("\n" + "="*60)
    print("GAME STATE MACHINE - UNIT TESTS")
    print("="*60)
    
    try:
        # Test initial state
        test_initial_state()
        
        # Test valid transitions
        test_valid_transition_menu_to_playing()
        test_valid_transition_playing_to_paused()
        test_valid_transition_paused_to_playing()
        test_valid_transition_playing_to_gameover()
        test_valid_transition_paused_to_gameover()
        test_valid_transition_gameover_to_menu()
        
        # Test invalid transitions
        test_invalid_transition_menu_to_paused()
        test_invalid_transition_menu_to_gameover()
        test_invalid_transition_playing_to_menu()
        test_invalid_transition_paused_to_menu()
        test_invalid_transition_gameover_to_playing()
        test_invalid_state_name()
        
        # Test helper methods
        test_helper_methods()
        test_get_state()
        test_get_status()
        test_reset()
        
        # Test full flow
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

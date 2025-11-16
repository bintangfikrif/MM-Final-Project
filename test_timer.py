"""
Unit Tests untuk Timer

"""

import sys
import time
sys.path.insert(0, 'src')
from timer import Timer


def test_initial_state():
    """Test initial state timer"""
    print("\n=== TEST 1: Initial State ===")
    
    timer = Timer(duration=60, mode=Timer.STOPWATCH)
    
    assert timer.is_running == False
    assert timer.is_paused == False
    assert timer.is_finished == False
    assert timer.get_elapsed_time() == 0.0
    
    print("✅ Initial state correct")
    print(f"   Running: {timer.is_running}")
    print(f"   Elapsed: {timer.get_elapsed_time():.1f}s")


def test_timer_start_stop():
    """Test start dan stop timer"""
    print("\n=== TEST 2: Start & Stop ===")
    
    timer = Timer(duration=5, mode=Timer.STOPWATCH)
    
    # Start
    result = timer.start()
    assert result == True
    assert timer.is_running == True
    print("✅ Timer started")
    
    # Tunggu sebentar
    time.sleep(1)
    
    # Check elapsed
    elapsed = timer.get_elapsed_time()
    assert elapsed > 0.9  # Lebih dari 0.9s (tolerance)
    print(f"✅ After 1s: elapsed={elapsed:.1f}s")
    
    # Stop
    timer.stop()
    assert timer.is_running == False
    print("✅ Timer stopped")


def test_timer_pause_resume():
    """Test pause dan resume timer"""
    print("\n=== TEST 3: Pause & Resume ===")
    
    timer = Timer(duration=10, mode=Timer.STOPWATCH)
    
    # Start
    timer.start()
    time.sleep(0.5)
    
    # Pause
    result = timer.pause()
    assert result == True
    assert timer.is_paused == True
    elapsed_at_pause = timer.get_elapsed_time()
    print(f"✅ Paused at {elapsed_at_pause:.1f}s")
    
    # Tunggu (tapi timer paused jadi elapsed tidak berubah)
    time.sleep(0.5)
    elapsed_after_wait = timer.get_elapsed_time()
    assert abs(elapsed_at_pause - elapsed_after_wait) < 0.1
    print(f"✅ After wait (paused): elapsed={elapsed_after_wait:.1f}s (tidak berubah)")
    
    # Resume
    result = timer.start()  # start() saat paused = resume
    assert result == True
    assert timer.is_paused == False
    print("✅ Timer resumed")
    
    # Tunggu
    time.sleep(0.5)
    elapsed_after_resume = timer.get_elapsed_time()
    assert elapsed_after_resume > elapsed_at_pause
    print(f"✅ After resume+wait: elapsed={elapsed_after_resume:.1f}s (bertambah)")


def test_countdown_mode():
    """Test countdown mode"""
    print("\n=== TEST 4: Countdown Mode ===")
    
    timer = Timer(duration=5, mode=Timer.COUNTDOWN)
    
    timer.start()
    
    # Initial (gunakan approximate comparison)
    remaining = timer.get_remaining_time()
    assert 4.99 < remaining <= 5.0
    print(f"✅ Initial remaining: {remaining:.1f}s")
    
    # Tunggu 1 detik
    time.sleep(1)
    
    remaining = timer.get_remaining_time()
    assert 3.9 < remaining < 4.1  # Sekitar 4s
    print(f"✅ After 1s: remaining={remaining:.1f}s")
    
    # Check not time up
    assert timer.is_time_up() == False
    print("✅ Time not up yet")
    
    timer.stop()


def test_countdown_time_up():
    """Test countdown sampai time up"""
    print("\n=== TEST 5: Countdown Time Up ===")
    
    timer = Timer(duration=2, mode=Timer.COUNTDOWN)
    
    timer.start()
    print("✅ Countdown started (2s)")
    
    # Tunggu sampai habis
    time.sleep(2.1)
    
    remaining = timer.get_remaining_time()
    assert remaining == 0.0
    assert timer.is_time_up() == True
    print(f"✅ Time up! Remaining: {remaining:.1f}s")
    
    timer.stop()


def test_stopwatch_mode():
    """Test stopwatch mode (hitung naik)"""
    print("\n=== TEST 6: Stopwatch Mode ===")
    
    timer = Timer(duration=10, mode=Timer.STOPWATCH)
    
    timer.start()
    
    # Initial
    elapsed = timer.get_elapsed_time()
    assert elapsed < 0.1
    print(f"✅ Initial elapsed: {elapsed:.1f}s")
    
    # Tunggu 1 detik
    time.sleep(1)
    
    elapsed = timer.get_elapsed_time()
    assert 0.9 < elapsed < 1.1
    print(f"✅ After 1s: elapsed={elapsed:.1f}s")
    
    timer.stop()


def test_format_time():
    """Test format time MM:SS"""
    print("\n=== TEST 7: Format Time ===")
    
    timer = Timer(duration=120, mode=Timer.STOPWATCH)
    
    # Test format_time dengan berbagai nilai
    test_cases = [
        (0, "00:00"),
        (30, "00:30"),
        (60, "01:00"),
        (90, "01:30"),
        (125, "02:05"),
    ]
    
    for seconds, expected in test_cases:
        result = timer.format_time(seconds)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"✅ {seconds}s → {result}")


def test_display_time():
    """Test display time untuk UI"""
    print("\n=== TEST 8: Display Time ===")
    
    # Stopwatch
    timer_sw = Timer(duration=120, mode=Timer.STOPWATCH)
    timer_sw.start()
    time.sleep(0.5)
    display = timer_sw.get_display_time()
    print(f"✅ Stopwatch display: {display}")
    assert display != ""  # Just check not empty
    
    # Countdown
    timer_cd = Timer(duration=60, mode=Timer.COUNTDOWN)
    timer_cd.start()
    display = timer_cd.get_display_time()
    print(f"✅ Countdown display: {display}")
    # Extract minutes dan seconds
    parts = display.split(":")
    assert len(parts) == 2
    minutes = int(parts[0])
    seconds = int(parts[1])
    # Should be close to 00:59 atau 01:00 (tolerance untuk execution time)
    assert (minutes == 0 and seconds >= 58) or (minutes == 1 and seconds == 0)


def test_time_percentage():
    """Test progress percentage"""
    print("\n=== TEST 9: Time Percentage ===")
    
    # Stopwatch: 50% ketika elapsed = 5s dari 10s duration
    timer_sw = Timer(duration=10, mode=Timer.STOPWATCH)
    timer_sw.start()
    time.sleep(0.5)
    
    percentage = timer_sw.get_time_percentage()
    # Should be roughly 5%
    print(f"✅ Stopwatch (1s elapsed): {percentage:.1f}%")
    
    # Countdown: 50% ketika remaining = 5s dari 10s duration
    timer_cd = Timer(duration=10, mode=Timer.COUNTDOWN)
    timer_cd.start()
    time.sleep(5)
    
    percentage = timer_cd.get_time_percentage()
    # Should be roughly 50%
    print(f"✅ Countdown (5s elapsed): {percentage:.1f}%")


def test_get_status():
    """Test get_status method"""
    print("\n=== TEST 10: Get Status ===")
    
    timer = Timer(duration=30, mode=Timer.COUNTDOWN)
    timer.start()
    time.sleep(1)
    
    status = timer.get_status()
    
    assert status['mode'] == Timer.COUNTDOWN
    assert status['is_running'] == True
    assert status['is_paused'] == False
    assert status['elapsed'] > 0.9
    assert status['remaining'] is not None
    assert status['display_time'] != ""
    
    print(f"✅ Status correct:")
    print(f"   Mode: {status['mode']}")
    print(f"   Running: {status['is_running']}")
    print(f"   Elapsed: {status['elapsed']:.1f}s")
    print(f"   Remaining: {status['remaining']:.1f}s")
    print(f"   Display: {status['display_time']}")


def test_reset():
    """Test reset timer"""
    print("\n=== TEST 11: Reset ===")
    
    timer = Timer(duration=30, mode=Timer.STOPWATCH)
    
    timer.start()
    time.sleep(0.5)
    assert timer.get_elapsed_time() > 0.4
    print(f"Before reset: elapsed={timer.get_elapsed_time():.1f}s")
    
    # Reset
    timer.reset()
    
    assert timer.is_running == False
    assert timer.is_paused == False
    assert timer.get_elapsed_time() == 0.0
    print(f"✅ After reset: elapsed={timer.get_elapsed_time():.1f}s")


def test_multiple_pause_resume():
    """Test pause dan resume multiple times"""
    print("\n=== TEST 12: Multiple Pause/Resume ===")
    
    timer = Timer(duration=20, mode=Timer.STOPWATCH)
    
    # Start
    timer.start()
    time.sleep(0.3)
    
    # Pause 1
    timer.pause()
    elapsed1 = timer.get_elapsed_time()
    print(f"After pause 1: {elapsed1:.1f}s")
    
    time.sleep(0.3)
    
    # Resume 1
    timer.start()
    time.sleep(0.3)
    
    elapsed2 = timer.get_elapsed_time()
    assert elapsed2 > elapsed1
    print(f"After resume 1: {elapsed2:.1f}s")
    
    # Pause 2
    timer.pause()
    elapsed3 = timer.get_elapsed_time()
    
    time.sleep(0.3)
    
    # Resume 2
    timer.start()
    time.sleep(0.3)
    
    elapsed4 = timer.get_elapsed_time()
    assert elapsed4 > elapsed3
    print(f"After resume 2: {elapsed4:.1f}s")
    
    print("✅ Multiple pause/resume works correctly")


def test_cannot_start_twice():
    """Test tidak bisa start timer yang sudah running"""
    print("\n=== TEST 13: Cannot Start Twice ===")
    
    timer = Timer(duration=10, mode=Timer.STOPWATCH)
    
    timer.start()
    result = timer.start()
    
    assert result == False
    print("✅ Second start rejected")


def test_cannot_pause_stopped_timer():
    """Test tidak bisa pause timer yang tidak running"""
    print("\n=== TEST 14: Cannot Pause Stopped Timer ===")
    
    timer = Timer(duration=10, mode=Timer.STOPWATCH)
    
    result = timer.pause()
    
    assert result == False
    print("✅ Pause rejected (timer not running)")


def test_countdown_negative_remaining():
    """Test countdown tidak bisa negative"""
    print("\n=== TEST 15: No Negative Remaining ===")
    
    timer = Timer(duration=1, mode=Timer.COUNTDOWN)
    
    timer.start()
    time.sleep(1.5)
    
    remaining = timer.get_remaining_time()
    assert remaining >= 0.0
    assert remaining == 0.0
    print(f"✅ Remaining clamped to 0: {remaining:.1f}s")


def run_all_tests():
    """Jalankan semua tests"""
    print("\n" + "="*60)
    print("TIMER - UNIT TESTS")
    print("="*60)
    
    try:
        test_initial_state()
        test_timer_start_stop()
        test_timer_pause_resume()
        test_countdown_mode()
        test_countdown_time_up()
        test_stopwatch_mode()
        test_format_time()
        test_display_time()
        test_time_percentage()
        test_get_status()
        test_reset()
        test_multiple_pause_resume()
        test_cannot_start_twice()
        test_cannot_pause_stopped_timer()
        test_countdown_negative_remaining()
        
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
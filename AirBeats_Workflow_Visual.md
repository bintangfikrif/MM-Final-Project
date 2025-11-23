# 🎮 AirBeats Project Workflow - Visual Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🎯 FINAL GOAL: PLAYABLE GAME                        │
│                     ✅ Real-time gesture → 🎹 Piano tiles                   │
│                     ✅ Video recording with audio output                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 4-Week Sprint Overview

```
WEEK 1          WEEK 2            WEEK 3              WEEK 4
[✅ DONE]      [🔨 BUILD]       [🔗 INTEGRATE]      [🎬 POLISH]
   
Input          Core Game         Full Game           Output
System         Mechanics         Experience          System
   
🤚 Gesture     🎯 Tiles          🎮 Menu→Game        📹 Video
Detection      🎵 Audio          🎵 Collision        📝 Docs
               📊 Scoring        💎 UI/UX            🐛 Testing
```

---

## 📅 WEEK-BY-WEEK BREAKDOWN

---

### ✅ MINGGU 1 (SELESAI) - Foundation

```
┌──────────────────────────────────────────────────────┐
│  🎯 MILESTONE: Gesture Detection System Ready        │
└──────────────────────────────────────────────────────┘

✅ Webcam capture working
✅ MediaPipe hand tracking active  
✅ 4 fingers identified (index, middle, ring, pinky)
✅ Tap gesture detection (downward movement)
✅ Position smoothing implemented

OUTPUT: 
┌─────────────────────────┐
│  👆 Finger tracking     │
│  🔽 "TAP!" detection    │
│  📍 Coordinates display │
└─────────────────────────┘
```

---

### 🔨 MINGGU 2 - Core Game Build

```
┌──────────────────────────────────────────────────────────────────┐
│  🎯 MILESTONE: Playable Prototype (Tiles + Audio + Score)       │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  BINTANG        │  │  FAKHRI         │  │  RAFKI          │
│  🎯 Tile System │  │  🎵 Audio       │  │  📊 Game State  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
   
  📦 Tile Class        🎹 4 Piano Notes    🎮 State Machine
  ⬇️  Spawning         🎼 Background Music  💯 Score System
  🚀 Movement          🔊 SFX Manager      🔥 Combo Counter
  🎨 4 Lanes           🔇 Volume Control   ⏱️  Timer
  ❌ Destruction       🎚️  Audio Mixing    ❌ Miss Detection


        ┌─────────────────────────────────┐
        │   END OF WEEK 2 DEMO:           │
        │                                 │
        │   ┌───┐ ┌───┐ ┌───┐ ┌───┐     │
        │   │ 🟥│ │ 🟦│ │ 🟩│ │ 🟨│  ← Tiles falling  │
        │   └─⬇─┘ └─⬇─┘ └─⬇─┘ └─⬇─┘     │
        │      ▼     ▼     ▼     ▼       │
        │   ┌────────────────────────┐    │
        │   │     HIT ZONE HERE      │    │
        │   └────────────────────────┘    │
        │                                 │
        │   👆 Tap → 🎹 Sound → 💯 +10   │
        │   Score: 150  Combo: x5        │
        └─────────────────────────────────┘

✅ CHECK: Tiles jatuh smooth dengan kecepatan konstan
✅ CHECK: Setiap gesture tap menghasilkan piano sound
✅ CHECK: Score bertambah saat action performed
```

**⚠️ CRITICAL PATH:**
```
Day 8-9:   Setup systems (classes, managers)
Day 10-11: Core mechanics working independently  
Day 12-13: Light integration testing
Day 14:    Fix bugs + dokumentasi
```

---

### 🔗 MINGGU 3 - Integration & Polish

```
┌──────────────────────────────────────────────────────────────────┐
│  🎯 MILESTONE: Complete Game Loop (Menu → Gameplay → Game Over) │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  BINTANG        │  │  FAKHRI         │  │  RAFKI          │
│  🎯 Collision   │  │  🎮 Game Modes  │  │  🖼️  UI/UX      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
   
  🤝 Gesture↔Tile      📊 Difficulty      📱 Main Menu
  ⏱️  Hit Timing       📈 Progressive      ⏸️  Pause System
  💥 Visual Feedback   🏆 High Score      🏁 Game Over
  ✨ Particle FX       🎵 Song Select     ⚙️  Settings


        GAME FLOW VISUALIZATION:
        
        ┌─────────────┐
        │ MAIN MENU   │
        │ ► START     │
        │   SETTINGS  │
        │   EXIT      │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  GAMEPLAY   │◄────┐
        │  👆🎹💯     │     │ RETRY
        │  Tiles fall │     │
        │  Score: 500 │     │
        └──────┬──────┘     │
               │            │
        [ESC]  │  [MISS 3x] │
               ▼            │
        ┌─────────────┐     │
        │ GAME OVER   │─────┘
        │ Score: 500  │
        │ Best: 850   │
        │ ► RETRY     │
        └─────────────┘


✅ CHECK: Bisa bermain dari start sampai game over
✅ CHECK: Collision detection akurat (hit/miss tepat)
✅ CHECK: Menu navigation smooth
✅ CHECK: Pause/resume works perfectly
```

**🔥 INTEGRATION POINTS:**
```
Day 15-16: Connect all systems together
Day 17-18: Collision + timing calibration  
Day 19-20: UI polish + gameplay balancing
Day 21:    Full playthrough testing
```

---

### 🎬 MINGGU 4 - Output & Documentation

```
┌──────────────────────────────────────────────────────────────────┐
│  🎯 MILESTONE: Shippable Product (Game + Video + Documentation)  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  BINTANG        │  │  FAKHRI         │  │  RAFKI          │
│  📹 Recording   │  │  🐛 Testing     │  │  📝 Docs        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
   
  📹 Video Capture     🧪 Edge Cases      📄 README.md
  🎤 Audio Record      🐞 Bug Fixing      📋 User Manual
  🔄 Sync Video+Audio  ⚡ Performance     📊 Report.pdf
  💾 Save .mp4         ✅ Final Testing   📚 Docstrings


        VIDEO OUTPUT SYSTEM:
        
        ┌───────────────────────────────────────┐
        │  🎮 GAMEPLAY SCREEN                   │
        │  ┌───────────────────────────────┐   │
        │  │  👆 Hand Tracking            │   │
        │  │  🎹 Tiles Falling            │   │
        │  │  💯 Score: 1250              │   │
        │  └───────────────────────────────┘   │
        │                                       │
        │  📹 REC ●  [Recording...]            │
        └───────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  💾 OUTPUT FILE                       │
        │  📁 gameplay_20250604_143022.mp4     │
        │  📊 Resolution: 1280x720              │
        │  🎵 Audio: Included & Synced         │
        │  ⏱️  Duration: 02:15                  │
        └───────────────────────────────────────┘


✅ CHECK: Video recording berfungsi sempurna
✅ CHECK: Audio tersinkronisasi di output video
✅ CHECK: Dokumentasi lengkap dan rapi
✅ CHECK: Ready untuk demo & pengumpulan
```

**📝 DOCUMENTATION CHECKLIST:**
```
□ README.md lengkap (instalasi, cara main, fitur)
□ Logbook minggu 1-4 updated
□ Report.pdf (template IF ITERA)
□ Code comments & docstrings
□ requirements.txt accurate
```

---

## 🎯 MILESTONE CHECKLIST

```
WEEK 1: ✅ [DONE] Gesture input system working
            └─ Can detect 4 fingers + tap gesture

WEEK 2: 🔨 [TODO] Core gameplay mechanics
            ├─ Tiles spawn and fall
            ├─ Audio plays on tap
            └─ Score system tracks points

WEEK 3: 🔗 [TODO] Complete game integration  
            ├─ Menu → Game → Game Over loop
            ├─ Collision detection accurate
            └─ UI/UX polished

WEEK 4: 🎬 [TODO] Production ready
            ├─ Video recording implemented
            ├─ All bugs fixed
            └─ Documentation complete

FINAL:  🚀 [TODO] Demo & Submission
            └─ May 10, 2025 deadline
```

---

## ⚡ CRITICAL SUCCESS FACTORS

```
┌────────────────────────────────────────────────────────────┐
│  🎯 TOP 3 PRIORITIES                                       │
├────────────────────────────────────────────────────────────┤
│  1. AUDIO-VISUAL SYNC                                      │
│     └─ Tiles harus sampai hit zone PAS saat beat         │
│                                                            │
│  2. RESPONSIVE GESTURE                                     │
│     └─ Tap detection harus instant (<100ms latency)      │
│                                                            │
│  3. VIDEO OUTPUT QUALITY                                   │
│     └─ Recording must be smooth 30+ FPS with audio       │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 WORKLOAD DISTRIBUTION

```
PERSON       WEEK 2          WEEK 3          WEEK 4
────────────────────────────────────────────────────────
BINTANG      🎯 Tiles        🎯 Collision    📹 Video
             (35% work)      (40% work)      (25% work)

FAKHRI       🎵 Audio        🎮 Modes        🐛 Testing
             (35% work)      (35% work)      (30% work)

RAFKI        📊 Scoring      🖼️  UI/UX       📝 Docs
             (30% work)      (40% work)      (30% work)
```

---

## 🚨 RISK MITIGATION

```
POTENTIAL ISSUE              SOLUTION                    OWNER
─────────────────────────────────────────────────────────────────
🔴 Audio sync drift          → Use pygame.time.get_ticks()  Fakhri
🔴 Gesture lag               → Lower MediaPipe confidence   Bintang  
🔴 Video recording slow      → Record at lower FPS first    Bintang
🔴 Integration conflicts     → Daily merge + test           ALL
🔴 Time shortage             → Cut bonus features first     ALL
```

---

## 🎓 LEARNING OUTCOMES (Aligned with CPMK)

```
✅ CPMK 1: Konsep multimedia (video + audio processing)
✅ CPMK 2: Aplikasi real-world (interactive music game)
✅ CPMK 3: Implementasi program (Python + libraries)
✅ CPMK 4: Sistem terintegrasi (input→processing→output)
```

---

## 📅 WEEKLY SYNC MEETINGS

```
EVERY MONDAY:    Review progress, adjust plan
EVERY THURSDAY:  Integration testing
EVERY SUNDAY:    Weekly milestone check

Duration: 30-45 minutes
Platform: Google Meet / Discord
```

---

## 🏆 DEFINITION OF DONE

```
Project is DONE when:

✅ User can play game from menu to game over
✅ Gesture controls work reliably (>80% accuracy)
✅ Audio and visual are synchronized (<50ms drift)
✅ Game can record and save video with audio
✅ Code is documented and clean
✅ README + Report are complete
✅ GitHub commits are consistent across all members
✅ Demo runs smoothly without crashes
```

---

## 🎯 QUICK REFERENCE - DAILY TARGETS

```
DAY         FOCUS                           OUTPUT
─────────────────────────────────────────────────────────────
Week 2:
Day 8       Setup classes & managers        Empty classes ready
Day 9       Basic mechanics working         Tiles fall, audio plays
Day 10      Integration attempt #1          Systems talk to each other
Day 11      Feature completion              All Week 2 features done
Day 12-13   Bug fixing & polish            Smooth gameplay
Day 14      Documentation                   Code comments + README

Week 3:
Day 15      Start integration               Connect all systems
Day 16      Collision system                Hit detection working
Day 17      UI implementation               Menu screens done
Day 18      Gameplay polish                 Smooth experience
Day 19      Balancing                       Difficulty feels right
Day 20      Final touches                   All features complete
Day 21      Testing                         No critical bugs

Week 4:
Day 22      Video system setup              Recording infrastructure
Day 23      Video capture working           Can record screen
Day 24      Audio capture working           Can record audio
Day 25      Sync video + audio              Output is synchronized
Day 26      Testing & fixing                Video quality good
Day 27      Documentation sprint            All docs complete
Day 28      Final polish & backup           Ready to submit
```

---

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              🎮 AIRBEATS - TOUCHLESS PIANO TILES               │
│                                                                 │
│  Dari gesture detection → menjadi game musik yang complete!    │
│                                                                 │
│              Timeline: 4 Weeks | Team: 3 People                │
│                                                                 │
│                    Target: May 10, 2025 ✅                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 FINAL TIPS

```
1. 📱 Daily Standups (15 min):
   - What did I do yesterday?
   - What will I do today?  
   - Any blockers?

2. 🔧 Git Workflow:
   - Branch: feature/tile-system
   - Commit: "feat: add tile spawning logic"
   - Merge: End of day after testing

3. 🧪 Test Early, Test Often:
   - Don't wait until Week 4
   - Integration test every 2 days

4. 📞 Communication:
   - Over-communicate progress
   - Ask for help early
   - Share blockers immediately

5. 🎯 Focus on MVP:
   - Make it work first
   - Make it pretty later
   - Don't over-engineer
```

---

**Good luck team! 🚀 Let's build an amazing AirBeats! 🎹**

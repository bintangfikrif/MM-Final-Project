# Tugas Besar Mata Kuliah Sistem Teknologi Multimedia (IF25-40305)
## Dosen Pengampu: Martin Clinton Tosima Manullang, S.T.,M.T.

## Anggota Kelompok
| Nama                        | NIM       | ID GitHub                                         |
|-----------------------------|-----------|---------------------------------------------------|
| Bintang Fikri Fauzan        | 122140008 |[@bintangfikrif](https://github.com/bintangfikrif) |
| M. Fakhri Nur               | 122140034 |[@Sovenable](https://github.com/Sovenable)     |
| Rafki Haykhal Alif          | 122140035 |[@RafkiHaykhalAlif](https://github.com/RafkiHaykhalAlif)       |

## Deskripsi Aplikasi
**AirBeats: Touchless Piano Tiles** adalah permainan ritme interaktif yang dimainkan tanpa sentuhan fisik. Aplikasi ini memanfaatkan teknologi *Computer Vision* (OpenCV) dan *Hand Tracking* (MediaPipe) untuk mendeteksi gerakan jari pemain secara real-time melalui webcam. Pemain berinteraksi dengan permainan dengan mengetukkan jari mereka di udara, seolah-olah menekan tuts piano virtual yang jatuh pada layar, menciptakan pengalaman bermain yang imersif dan futuristik.

## Demo Aplikasi
https://drive.google.com/drive/folders/1ihKPCHknQireDc-MPuYUoWwTxccaUs2k

## Fitur Aplikasi
- **Touchless Gameplay**: Menggunakan deteksi gestur tangan untuk bermain tanpa menyentuh keyboard atau mouse.
- **Hand Tracking 4 Jari**: Mendeteksi gerakan jari telunjuk, tengah, manis, dan kelingking secara independen sebagai input.
- **Rhythm Game Mechanics**: Tiles jatuh sesuai dengan irama musik.
- **Scoring System**: Penilaian presisi (Perfect, Good, Bad) dan sistem Combo.
- **Visual Feedback (VFX)**: Efek visual menarik saat tiles ditekan atau terlewat.
- **Game States**: Mendukung Menu Utama, Gameplay, Pause, dan Game Over Screen.
- **Difficulty Modes**: Tingkat kesulitan yang dapat disesuaikan (kecepatan tiles).
- **Audio Integration**: Sinkronisasi musik dan efek suara (SFX) saat tile ditekan.

## Library yang Digunakan
1. **OpenCV (`opencv-python`)**: Untuk memproses frame dari webcam dan visualisasi dasar debug.
2. **MediaPipe (`mediapipe`)**: Untuk mendeteksi landmark tangan (hand tracking) secara akurat dan cepat.
3. **Pygame (`pygame`)**: Sebagai engine utama untuk rendering grafis (UI/UX), manajemen window, dan sistem audio.
4. **NumPy (`numpy`)**: Untuk operasi matematika array dan koordinat yang efisien.

## Teknologi/Tools yang Digunakan
- **Python 3.10**: Bahasa pemrograman utama.
- **Visual Studio Code (VS Code)**: Code Editor untuk pengembangan.
- **Git & GitHub**: Untuk version control dan kolaborasi tim.

## Cara Menjalankan
1. **Clone Repository** (atau download source code).
2. **Install Dependencies**:
   Pastikan Python sudah terinstall, lalu jalankan perintah berikut di terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan Aplikasi**:
   Arahkan terminal ke folder root project, lalu jalankan:
   ```bash
   python src/main.py
   ```
4. **Cara Bermain**:
   - Pastikan webcam aktif dan ruangan cukup cahaya.
   - Arahkan tangan ke depan kamera hingga terdeteksi (muncul landmark garis-garis).
   - Gerakkan jari (telunjuk, tengah, manis, kelingking) ke bawah seolah menekan tombol ("Tap").
   - Tekan tiles tepat saat menyentuh garis batas bawah untuk mendapatkan poin!

| Minggu ke-    | Progress| 
|-------------------------|-------------------------|
|| Melakukan brainstorming dan pembagian peran pengerjaan  |
| 1 | Melakukan setup project dan struktur folder |
|| Membuat gesture tracking pada tangan & tap gesture |
||Merancang tile system sederhana|
|2|Memasukkan audio notes dan mengintegrasikannya dengan tile system|
||Membuat sistem scoring dan combo sederhana|
||Menambahkan musik menyesuaikan dari audio notes yang diimport|
|3|Menambahkan difficulty mode dan timer|
||Membuat menu dan game over screen|
|4|Memperbaiki bug|
||Membuat laporan|
||Pembuatan video demo|


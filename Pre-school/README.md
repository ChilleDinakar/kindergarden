# 🎓 KinderLearn — EdgeAI Kindergarten Application

A fully offline-capable EdgeAI application for kindergarten and pre-school learning.
Uses MediaPipe for real-time AI (finger detection + motion tracking), pyttsx3 for offline
text-to-speech, and Flask as the web server.

---

## 📁 File Structure

```
kindergarten/
├── app.py                          # Main Flask application (all routes + AI logic)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── school.db                   # SQLite database (auto-created on first run)
├── static/
│   └── css/
│       └── style.css               # Master stylesheet (all pages)
└── templates/
    ├── login.html                  # Login & Register page
    ├── home.html                   # Home dashboard with sidebar
    ├── rhymes.html                 # Trending rhymes (YouTube)
    ├── singalong.html              # Sing Along hub (3 sub-options)
    ├── alphabets.html              # A-Z alphabet flashcards
    ├── words.html                  # Easy word flashcards
    ├── karaoke.html                # Karaoke / read-along lyrics
    ├── numbers.html                # Numbers finger-detection game
    └── motion.html                 # Group motion maths game
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies

```bash
cd kindergarten
pip install -r requirements.txt
```

> On Linux you may need:
> ```bash
> sudo apt-get install espeak espeak-ng   # for pyttsx3 TTS
> ```

> On macOS pyttsx3 uses the built-in speech engine (no extra install needed).
> On Windows pyttsx3 uses SAPI5 (built-in, no extra install needed).

### 2. Run the application

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## 🔐 Login

- Register your school with **Gmail or mobile number + password**
- The app remembers previous logins (Quick Login chips)
- All credentials stored locally in `data/school.db` (SQLite)

---

## 🌐 Internet Requirements

| Feature | Internet Needed? |
|---|---|
| Login / Register | ❌ No |
| Alphabets | ❌ No |
| Words | ❌ No |
| Karaoke Lyrics | ❌ No |
| TTS (voices) | ❌ No |
| Finger Detection (Numbers) | ❌ No |
| Motion Games | ❌ No |
| **Rhymes (YouTube)** | ✅ Yes |
| Google Fonts (styling) | Recommended (falls back gracefully) |

---

## 🧠 EdgeAI Features

### Numbers Game (MediaPipe Hands)
- Camera opens and captures frames every 2 seconds
- Frame is sent to `/api/detect_fingers`
- MediaPipe Hands detects finger landmarks and counts extended fingers
- pyttsx3 speaks the number aloud (offline TTS)

### Motion Maths Game (OpenCV)
- Classroom camera tracks how many people/kids stand in frame
- Uses Gaussian blur + contour detection (lightweight, no GPU needed)
- Maths problem is shown; kids arrange themselves to match the answer
- Teacher clicks "Next" to advance questions

---

## ⚙️ Configuration

All hardcoded data (rhyme links, alphabets, words, karaoke lyrics) is in `app.py`
under clearly labeled sections. You can freely add/edit entries.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Camera not working | Allow browser camera permission; use http://localhost (not 127.0.0.1) |
| TTS silent | Install `espeak` on Linux: `sudo apt install espeak` |
| MediaPipe import error | `pip install mediapipe --upgrade` |
| Port in use | Change `port=5000` in `app.py` to another port |
| Google Fonts not loading | App still works; uses system fallback fonts |

---

## 📝 Extending the App

- **Add more rhymes**: Edit `RHYMES` list in `app.py`
- **Add more words**: Edit `WORDS` list in `app.py`
- **Add more karaoke songs**: Edit `KARAOKE_RHYMES` list in `app.py`
- **Improve person detection**: Replace contour heuristic in `count_people_and_pose()` with a proper MediaPipe Holistic / YOLO model for production use

---

Built with ❤️ for children's education. EdgeAI — runs on your local machine, no cloud needed.
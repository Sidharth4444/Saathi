# 🤝 SAATHI – AI Assistive Communication System

An AI-powered assistive system designed to bridge communication gaps between deaf and blind individuals using Computer Vision, Speech Processing, and Deep Learning.

---

## 🌟 Overview

SAATHI enables:

- 👁 Blind Assist Mode  
  Detects facial emotions + sign language and converts them into speech.

- 👂 Deaf Assist Mode  
  Converts spoken language into text in real-time.

This system promotes inclusive communication using AI.

---

## 🧠 Features

### 👁 Blind Assist Mode
- Facial Emotion Recognition (CNN model)
- Indian Sign Language Recognition (MediaPipe + Deep Learning)
- Sentence Formation
- Text-to-Speech Output

### 👂 Deaf Assist Mode
- Real-time Audio Recording
- Speech-to-Text using Faster-Whisper
- Display recognized text

---

## 🛠 Tech Stack

- Python 3.10
- TensorFlow 2.13
- OpenCV
- MediaPipe
- Faster-Whisper
- Coqui TTS
- NumPy
- Joblib

---

## 📂 Project Structure
SAATHI/
│
├── models/
│   ├── emotion_model.h5
│   ├── isl_landmark_model.h5
│   └── label_encoder.save
│
├── output/
│   └── output.wav
│
├── main.py
├── requirements.txt
├── README.md

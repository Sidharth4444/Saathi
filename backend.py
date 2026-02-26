import cv2
import numpy as np
import mediapipe as mp
import joblib
import tempfile
import soundfile as sf
import uuid
import os

from tensorflow.keras.models import load_model
from faster_whisper import WhisperModel
from TTS.api import TTS

# ==============================
# LOAD MODELS
# ==============================

emotion_model = load_model("emotion_model.h5")
sign_model = load_model("isl_landmark_model.h5")
encoder = joblib.load("label_encoder.save")

speech_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

tts = TTS(
    "tts_models/en/ljspeech/tacotron2-DDC",
    gpu=False
)

emotion_labels = [
    "Angry","Disgust","Fear",
    "Happy","Sad","Surprise","Neutral"
]

emotion_phrases = {
    "Happy": "I am feeling very happy.",
    "Sad": "I feel sad.",
    "Angry": "I am angry.",
    "Fear": "I feel scared.",
    "Surprise": "I am surprised.",
    "Neutral": "I feel calm."
}

# ==============================
# OLD MEDIAPIPE SYSTEM
# ==============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

emotion_buffer = []
sign_buffer = []

# ==============================
# OLD FACE + SIGN PIPELINE
# ==============================

def process_frame(frame):

    global emotion_buffer, sign_buffer

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    emotion_text = ""
    sign_text = ""

    # -------- FACE EMOTION --------
    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        pred = emotion_model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(pred)]

        emotion_buffer.append(emotion)

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        emotion_text = emotion

    # -------- HAND SIGN --------
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            landmark_array = np.array(
                landmark_list
            ).reshape(1, -1)

            pred = sign_model.predict(
                landmark_array,
                verbose=0
            )

            class_id = np.argmax(pred)
            confidence = np.max(pred)

            sign = encoder.inverse_transform(
                [class_id]
            )[0]

            if confidence > 0.8:
                sign_buffer.append(sign)

            cv2.putText(
                frame,
                f"{sign} ({confidence:.2f})",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

            sign_text = sign

    return frame, emotion_text, sign_text


# ==============================
# BLIND OUTPUT + TTS
# ==============================

def generate_blind_output():

    global emotion_buffer, sign_buffer

    final_emotion = (
        max(set(emotion_buffer),
        key=emotion_buffer.count)
        if emotion_buffer else "Neutral"
    )

    words = []
    last = None

    for w in sign_buffer:
        if w != last:
            words.append(w)
            last = w

    sign_sentence = (
        " ".join(words)
        if words else
        "I am signaling."
    )

    final_text = (
        emotion_phrases[final_emotion]
        + " "
        + sign_sentence
    )

    filename = os.path.join(
        tempfile.gettempdir(),
        f"speech_{uuid.uuid4().hex}.wav"
    )

    tts.tts_to_file(
        text=final_text,
        file_path=filename
    )

    emotion_buffer.clear()
    sign_buffer.clear()

    return final_text, filename


# ==============================
# DEAF SYSTEM (UNCHANGED)
# ==============================

def deaf_assist(audio):

    if audio is None:
        return "No audio detected."

    sr, audio_np = audio

    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype(np.float32) / 32768.0

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as f:

        sf.write(f.name, audio_np, sr)

        segments, _ = speech_model.transcribe(f.name)

        text = "".join([s.text for s in segments])

    return text if text.strip() else "Could not understand audio."
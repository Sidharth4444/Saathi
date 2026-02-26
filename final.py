import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

from tensorflow.keras.models import load_model
from TTS.api import TTS

# ==============================
# LOAD MODELS
# ==============================

emotion_model = load_model("emotion_model.h5")
sign_model = load_model("isl_landmark_model.h5")
encoder = joblib.load("label_encoder.save")

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

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
# MEDIAPIPE + FACE
# ==============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==============================
# GLOBAL BUFFERS
# ==============================

emotion_buffer = []
sign_buffer = []
start_time = None

# ==============================
# FRAME PROCESSING (LIVE)
# ==============================

def process_frame(frame):

    global emotion_buffer, sign_buffer, start_time

    if start_time is None:
        start_time = time.time()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    emotion_text = ""
    sign_text = ""

    # ------------------ EMOTION ------------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = face/255.0
        face = np.reshape(face,(1,48,48,1))

        pred = emotion_model.predict(face,verbose=0)
        emotion = emotion_labels[np.argmax(pred)]

        emotion_buffer.append(emotion)
        emotion_text = emotion

        cv2.putText(frame, emotion,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    # ------------------ SIGN ------------------
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_list=[]
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x,lm.y,lm.z])

            landmark_array=np.array(landmark_list).reshape(1,-1)

            pred=sign_model.predict(landmark_array)
            class_id=np.argmax(pred)
            sign=encoder.inverse_transform([class_id])[0]
            confidence=np.max(pred)

            if confidence>0.8:
                sign_buffer.append(sign)
                sign_text = sign

            cv2.putText(frame,
                        f"{sign} ({confidence:.2f})",
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,0,0),2)

    return frame, emotion_text, sign_text

# ==============================
# FINAL SENTENCE + TTS
# ==============================

def generate_output():

    global emotion_buffer, sign_buffer, start_time

    if len(emotion_buffer)==0:
        final_emotion="Neutral"
    else:
        final_emotion=max(
            set(emotion_buffer),
            key=emotion_buffer.count
        )

    words=[]
    last=None

    for w in sign_buffer:
        if w!=last:
            words.append(w)
            last=w

    sign_text=" ".join(words)

    if sign_text.strip()=="":
        sign_text="I am trying to communicate."

    final_text=emotion_phrases[final_emotion]+" "+sign_text

    output_path="output.wav"

    tts.tts_to_file(
        text=final_text,
        file_path=output_path
    )

    # Reset buffers
    emotion_buffer=[]
    sign_buffer=[]
    start_time=None

    return final_text, output_path
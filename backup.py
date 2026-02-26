import cv2
import numpy as np
import time
import os
import sounddevice as sd
import mediapipe as mp
import joblib

from tensorflow.keras.models import load_model
from TTS.api import TTS
from faster_whisper import WhisperModel

# ==============================
# LOAD MODELS
# ==============================

emotion_model = load_model("emotion_model.h5")
sign_model = load_model("isl_landmark_model.h5")
encoder = joblib.load("label_encoder.save")

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
speech_model = WhisperModel("base")

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
# BLIND ASSIST MODE
# ==============================

def blind_assist():

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Saathi Capture", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Saathi Capture",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # Countdown
    for i in range(3,0,-1):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.putText(frame,f"Starting in {i}",
                    (400,300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,(0,0,255),4)
        cv2.imshow("Saathi Capture",frame)
        cv2.waitKey(1000)

    emotion_buffer=[]
    sign_buffer=[]

    start_time=time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # TIMER
        elapsed=int(time.time()-start_time)
        remaining=30-elapsed

        cv2.putText(frame,
                    f"Time Left: {remaining}s",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,255),3)

        # -------- EMOTION --------
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,(48,48))
            face=face/255.0
            face=np.reshape(face,(1,48,48,1))

            pred=emotion_model.predict(face,verbose=0)
            emotion=emotion_labels[np.argmax(pred)]
            emotion_buffer.append(emotion)

            cv2.putText(frame,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        # -------- SIGN --------
        results=hands.process(rgb)

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

                landmark_array=np.array(
                    landmark_list
                ).reshape(1,-1)

                pred=sign_model.predict(landmark_array)
                class_id=np.argmax(pred)

                sign=encoder.inverse_transform(
                    [class_id]
                )[0]

                confidence=np.max(pred)

                if confidence>0.8:
                    sign_buffer.append(sign)

                cv2.putText(frame,
                            f"{sign} ({confidence:.2f})",
                            (10,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255,0,0),2)

        cv2.imshow("Saathi Capture",frame)

        if elapsed>=30:
            break

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # FINAL EMOTION
    final_emotion=max(set(emotion_buffer),
                      key=emotion_buffer.count)

    # BUILD SENTENCE
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

    print("\nSpeaking:",final_text)

    tts.tts_to_file(
        text=final_text,
        file_path="output.wav"
    )

    os.system("start output.wav")

# ==============================
# DEAF ASSIST MODE
# ==============================

def deaf_assist():

    print("\nSpeak now...")

    duration=5
    fs=16000

    audio=sd.rec(int(duration*fs),
                 samplerate=fs,
                 channels=1,
                 dtype='int16')
    sd.wait()

    audio_np=audio.flatten().astype(np.float32)/32768.0

    segments,_=speech_model.transcribe(audio_np)

    text=""

    for seg in segments:
        text+=seg.text

    print("\nRecognized:",text)

# ==============================
# MAIN MENU
# ==============================

def main():

    while True:

        print("\n===== SAATHI SYSTEM =====")
        print("1 → Blind Assist")
        print("2 → Deaf Assist")
        print("3 → Exit")

        choice=input("Select Mode: ")

        if choice=="1":
            blind_assist()

        elif choice=="2":
            deaf_assist()

        elif choice=="3":
            break

        else:
            print("Invalid choice.")

if __name__=="__main__":
    main()
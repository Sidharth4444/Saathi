import gradio as gr
import cv2
from backend import process_frame, generate_blind_output, deaf_assist

def stream_wrapper(img):

    if img is None:
        return None, "", ""

    processed, emo, sign = process_frame(img)

    return processed, emo, sign


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧠 SAATHI: Assistive Communication")

    # ================= BLIND =================
    with gr.Tab("👁️ Blind Assist"):

        with gr.Row():
            input_feed = gr.Image(
                sources=["webcam"],
                streaming=True
            )
            output_feed = gr.Image()

        emotion_display = gr.Textbox(
            label="Detected Emotion"
        )

        sign_display = gr.Textbox(
            label="Current Sign"
        )

        input_feed.stream(
            stream_wrapper,
            inputs=input_feed,
            outputs=[
                output_feed,
                emotion_display,
                sign_display
            ],
            show_progress="hidden"
        )

        speak_btn = gr.Button(
            "🔊 Construct & Speak Sentence"
        )

        final_txt = gr.Textbox(
            label="Interpreted Text"
        )

        final_audio = gr.Audio(
            label="Voice Output",
            autoplay=True
        )

        speak_btn.click(
            generate_blind_output,
            outputs=[final_txt, final_audio]
        )

    # ================= DEAF =================
    with gr.Tab("🧏 Deaf Assist"):

        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy"
        )

        transcribe_btn = gr.Button(
            "Convert Speech to Text"
        )

        transcription_out = gr.Textbox(
            label="Transcription"
        )

        transcribe_btn.click(
            deaf_assist,
            inputs=mic_input,
            outputs=transcription_out
        )

if __name__ == "__main__":
    demo.launch()
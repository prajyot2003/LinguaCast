import gradio as gr
from transformers import pipeline
from langdetect import detect
from gtts import gTTS
import os

# 🔥 Load Hugging Face translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

def translate_text(text, target_lang, speech_output):
    try:
        # Detect source language
        source_lang = detect(text)

        # Translate text
        translation = translator(text, tgt_lang=target_lang)
        translated_text = translation[0]['translation_text']

        # Generate speech if enabled
        audio_file = None
        if speech_output:
            tts = gTTS(translated_text, lang=target_lang)
            audio_file = "output.mp3"
            tts.save(audio_file)

        return f"Detected Language: {source_lang}\n\nTranslation: {translated_text}", audio_file

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None

# 🎨 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 LinguaCast\nAI-Powered Multilingual Translator with Speech Output")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Enter your text", placeholder="Type here...")
            target_lang = gr.Dropdown(
                ["en", "es", "fr", "de", "it", "nl", "ru", "zh"], 
                label="Select Target Language", 
                value="en"
            )
            speech_output = gr.Checkbox(label="Enable Speech Output", value=False)
            submit_btn = gr.Button("Translate")

        with gr.Column():
            result_output = gr.Textbox(label="Translation Result")
            audio_output = gr.Audio(label="Speech Output", type="filepath")

    submit_btn.click(
        translate_text,
        inputs=[text_input, target_lang, speech_output],
        outputs=[result_output, audio_output]
    )

# Run the app
if __name__ == "__main__":
    demo.launch()

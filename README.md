# 🌍 LinguaCast

LinguaCast is an AI-powered translation app with:
- Automatic language detection
- Direct any-to-any translation (Meta NLLB-200)
- Speech input (OpenAI Whisper, runs locally, no API key)
- Speech output (gTTS)
- Interactive Gradio UI

## 🚀 Features
- Detects the source language automatically
- Translates text into 54 languages
- Speak or upload audio to fill the text box
- Option to listen to translated speech
- ⇄ Swap button to translate the result back
- Translation history, saved in your browser
- Document translation: .txt, .md, .srt, .docx, .pdf
- Deployable on Hugging Face Spaces

## 🛠 Installation
```bash
git clone https://github.com/prajyot2003/LinguaCast.git
cd LinguaCast
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open http://127.0.0.1:7860. The first run downloads about 3.4 GB of models (NLLB-200 and Whisper-small).

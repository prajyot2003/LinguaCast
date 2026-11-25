import gradio as gr
from transformers import pipeline
from langdetect import detect
from gtts import gTTS
from gtts.lang import tts_langs
import numpy as np
import re
import tempfile
from pathlib import Path
from docx import Document
from pypdf import PdfReader

# 🔥 Load Hugging Face translation model (NLLB-200: direct any-to-any translation)
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

# 🎤 Local speech recognition (Whisper, no API key needed)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30)

def transcribe(audio):
    if audio is None:
        return gr.update()
    sr, data = audio
    data = data.astype(np.float32)
    if np.issubdtype(audio[1].dtype, np.integer):
        data /= np.iinfo(audio[1].dtype).max  # int PCM -> [-1, 1]
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo -> mono
    if sr != 16000:  # Whisper expects 16 kHz; FFT resample drops frequencies above 8 kHz so they don't alias
        n = int(len(data) * 16000 / sr)
        data = (np.fft.irfft(np.fft.rfft(data)[: n // 2 + 1], n) * n / len(data)).astype(np.float32)
    result = transcriber({"raw": data, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"})
    return result["text"].strip()

# langdetect / dropdown codes -> NLLB codes (covers every language langdetect can detect)
NLLB = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn",
    "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "kn": "kan_Knda",
    "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym",
    "mr": "mar_Deva", "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "so": "som_Latn", "sq": "als_Latn", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn",
    "zh": "zho_Hans", "zh-cn": "zho_Hans", "zh-tw": "zho_Hant",
}

# Target languages shown in the dropdown
LANGUAGES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam",
    "mr": "Marathi", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Chinese (Simplified)",
}

HISTORY_HEADERS = ["Original", "From", "To", "Translation"]
HISTORY_LIMIT = 50

def translate_text(text, target_lang, speech_output, history):
    try:
        # Detect source language
        source_lang = detect(text)

        # Translate text
        if source_lang == target_lang:
            translated_text = text
        else:
            translation = translator(text, src_lang=NLLB[source_lang], tgt_lang=NLLB[target_lang])
            translated_text = translation[0]['translation_text']

        # Generate speech if enabled
        audio_file = None
        note = ""
        if speech_output:
            tts_lang = {"zh": "zh-CN", "he": "iw"}.get(target_lang, target_lang)  # gTTS's own codes
            if tts_lang in tts_langs():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:  # one file per request
                    audio_file = f.name
                gTTS(translated_text, lang=tts_lang).save(audio_file)
            else:
                note = f"\n\n(Speech output isn't available for {LANGUAGES[target_lang]}.)"

        source_name = LANGUAGES.get(source_lang.split("-")[0], source_lang)
        history = ([[text, source_name, LANGUAGES[target_lang], translated_text]] + (history or []))[:HISTORY_LIMIT]
        last = [source_lang.split("-")[0], translated_text]  # for the Swap button
        return f"Detected Language: {source_lang}\n\nTranslation: {translated_text}{note}", audio_file, history, history, last

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, history, history, None

def swap_languages(last):
    if not last:
        raise gr.Error("Translate something first, then swap.")
    source_lang, translated_text = last
    return translated_text, source_lang

# 📄 Document translation
SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")
MARKDOWN_PREFIX = re.compile(r"^(\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|>\s)?)")
BATCH = 16

def translate_many(texts, src, tgt, progress):
    """Translate a list of strings sentence by sentence (NLLB truncates long inputs); blank strings pass through."""
    pieces, owners = [], []
    for i, t in enumerate(texts):
        for sentence in SENTENCE_END.split(t.strip()) if t.strip() else []:
            pieces.append(sentence)
            owners.append(i)
    results = [[] for _ in texts]
    for start in progress.tqdm(range(0, len(pieces), BATCH), desc="Translating"):
        batch = pieces[start:start + BATCH]
        for j, r in enumerate(translator(batch, src_lang=src, tgt_lang=tgt, batch_size=BATCH)):
            results[owners[start + j]].append(r["translation_text"])
    return [" ".join(r) if texts[i].strip() else texts[i] for i, r in enumerate(results)]

def translate_lines(lines, src, tgt, progress, srt=False):
    """Translate text lines, keeping SRT cue numbers/timestamps and Markdown markers as they are."""
    keep = [srt and (l.strip().isdigit() or "-->" in l) for l in lines]
    prefixes = [MARKDOWN_PREFIX.match(l).group(1) for l in lines]
    bodies = [l[len(p):] for l, p in zip(lines, prefixes)]
    todo = [i for i, k in enumerate(keep) if not k]
    done = translate_many([bodies[i] for i in todo], src, tgt, progress)
    out = list(lines)
    for i, t in zip(todo, done):
        out[i] = prefixes[i] + t
    return out

def translate_document(file, target_lang, progress=gr.Progress()):
    if file is None:
        return None, "⚠️ Please upload a document first."
    try:
        path = Path(file)
        ext = path.suffix.lower()
        if ext == ".docx":
            doc = Document(path)
            paras = list(doc.paragraphs) + [p for t in doc.tables for row in t.rows for cell in row.cells for p in cell.paragraphs]
            texts = [p.text for p in paras]
        elif ext == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            texts = [line.rstrip() for line in text.strip().splitlines()]  # drop layout padding
        elif ext in (".txt", ".md", ".srt"):
            texts = path.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            return None, "⚠️ Supported formats: .txt, .md, .srt, .docx, .pdf"

        sample = " ".join(t for t in texts if t.strip() and "-->" not in t and not t.strip().isdigit())[:5000]
        if not sample:
            return None, "⚠️ No text found in this document (scanned PDFs need OCR first)."
        source_lang = detect(sample)
        src, tgt = NLLB[source_lang], NLLB[target_lang]

        out_dir = Path(tempfile.mkdtemp())
        if ext == ".docx":
            for p, t in zip(paras, translate_many(texts, src, tgt, progress)):
                if p.runs and p.text.strip():
                    p.runs[0].text = t  # keep the paragraph's style; inline formatting collapses to the first run's
                    for run in p.runs[1:]:
                        run.text = ""
            out_path = out_dir / f"{path.stem}_{target_lang}.docx"
            doc.save(out_path)
        else:
            out_ext = ".txt" if ext == ".pdf" else ext
            out_path = out_dir / f"{path.stem}_{target_lang}{out_ext}"
            out_path.write_text("\n".join(translate_lines(texts, src, tgt, progress, srt=ext == ".srt")), encoding="utf-8")

        source_name = LANGUAGES.get(source_lang.split("-")[0], source_lang)
        return str(out_path), f"✅ Translated from {source_name} to {LANGUAGES[target_lang]}: {out_path.name}"

    except Exception as e:
        return None, f"⚠️ Error: {str(e)}"

# 🎨 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 LinguaCast\nAI-Powered Multilingual Translator with Speech Output")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="🎤 Speak or upload audio (fills the text box)")
            text_input = gr.Textbox(label="Enter your text", placeholder="Type here...")
            with gr.Row(equal_height=True):
                target_lang = gr.Dropdown(
                    sorted(((name, code) for code, name in LANGUAGES.items())),
                    label="Select Target Language",
                    value="en",
                    scale=4
                )
                swap_btn = gr.Button("⇄ Swap", scale=1, min_width=80, interactive=False)  # enabled after a translation
            speech_output = gr.Checkbox(label="Enable Speech Output", value=False)
            submit_btn = gr.Button("Translate")

        with gr.Column():
            result_output = gr.Textbox(label="Translation Result")
            audio_output = gr.Audio(label="Speech Output", type="filepath")

    # 📄 Document translation (uses the target language selected above)
    gr.Markdown("### 📄 Document Translation\nUpload a .txt, .md, .srt, .docx or .pdf file. It's translated into the target language selected above.")
    with gr.Row():
        doc_input = gr.File(label="Document", file_types=[".txt", ".md", ".srt", ".docx", ".pdf"], type="filepath")
        doc_output = gr.File(label="Translated document")
    doc_status = gr.Markdown()
    doc_btn = gr.Button("Translate Document")
    doc_btn.click(translate_document, inputs=[doc_input, target_lang], outputs=[doc_output, doc_status])

    # 🕘 Translation history, saved in each visitor's browser
    history = gr.BrowserState([], storage_key="linguacast_history")
    gr.Markdown("### 🕘 Translation History")
    history_table = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, wrap=True)
    clear_btn = gr.Button("Clear History", size="sm")

    demo.load(lambda h: h or [], inputs=history, outputs=history_table)
    audio_input.change(transcribe, inputs=audio_input, outputs=text_input)

    last_translation = gr.State(None)
    translate_args = dict(
        fn=translate_text,
        inputs=[text_input, target_lang, speech_output, history],
        outputs=[result_output, audio_output, history, history_table, last_translation]
    )
    enable_swap = dict(fn=lambda last: gr.update(interactive=bool(last)), inputs=last_translation, outputs=swap_btn)
    submit_btn.click(**translate_args).then(**enable_swap)
    # Swap: translated text becomes the input, detected source becomes the target, then translate back
    swap_btn.click(swap_languages, inputs=last_translation, outputs=[text_input, target_lang]).success(**translate_args).then(**enable_swap)
    clear_btn.click(lambda: ([], []), outputs=[history, history_table])

# Run the app
if __name__ == "__main__":
    demo.launch()

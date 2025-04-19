import streamlit as st

# ------------------------ STREAMLIT UI NAVIGATION ------------------------
st.set_page_config(page_title="CrickVision AI", layout="wide")

# Sidebar for navigation
st.sidebar.title("📂 Navigation")
app_mode = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "🎥 Real-time Commentary", "📘 About", "❓ FAQ", "🔒 Privacy Policy", "© Copyright"]
)

# ------------------------ 🏠 HOME PAGE ------------------------
if app_mode == "🏠 Home":
    st.title("🏏 Welcome to CrickVision AI")
    st.markdown("""
    **CrickVision AI** is an intelligent system that generates real-time multilingual cricket commentary from videos.

    ### Features:
    - 🎯 Frame-based captioning using BLIP
    - 🗣️ Commentary styled by Google Gemini
    - 🎧 Audio commentary using Edge TTS
    - 🎞️ Synchronized with video playback

    Select the **Real-time Commentary** tab from the sidebar to begin!
    """)

# ------------------------ 🎥 MAIN FEATURE ------------------------
elif app_mode == "🎥 Real-time Commentary":
    st.title("🎥 Real-time AI Cricket Commentary")

    import tempfile
    import os
    import cv2
    import torch
    import asyncio
    import numpy as np
    import edge_tts
    import sounddevice as sd
    import soundfile as sf
    import time
    import logging
    from concurrent.futures import ThreadPoolExecutor
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    from google.generativeai import configure, GenerativeModel

    # Logging
    logging.basicConfig(filename='logs.txt', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Language and style maps
    LANGUAGE_OPTIONS = {
        "Hindi": "hi-IN-MadhurNeural",
        "Marathi": "mr-IN-AarohiNeural",
        "Gujarati": "gu-IN-NiranjanNeural",
        "Tamil": "ta-IN-PallaviNeural",
        "Telugu": "te-IN-MohanNeural",
    }

    COMMENTARY_STYLES = {
        "Akash Chorda Style": "as Akash Chopra, deliver thrilling and witty cricket commentary.",
        "Bollywood Style": "in a dramatic Bollywood-style tone, add flair and emotions to the commentary.",
        "Shayari Style": "in poetic Shayari form, describe the incident creatively.",
        "Funny Style": "use humorous and light-hearted tone to comment on the incident.",
    }

    # Configs
    cv2.setNumThreads(0)
    configure(api_key="AIzaSyBRZ9cEMHE6Ys8jt27EuACWP7TEM7C2Yac")
    model_gemini = GenerativeModel("gemini-2.0-flash")
    blip_saved_model_path = "blip_saved_model4"
    processor = BlipProcessor.from_pretrained(blip_saved_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_blip = BlipForConditionalGeneration.from_pretrained(blip_saved_model_path).to(device)
    os.makedirs("data", exist_ok=True)

    def process_frame(video_path, frame_number, lang, style):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        inputs = processor(img, return_tensors="pt").to(device)
        outputs = model_blip.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        commentary = generate_commentary(caption, lang, style)
        audio_path = f"data/commentary_{frame_number}.mp3"
        asyncio.run(text_to_speech(commentary, audio_path, lang))
        return frame_number, commentary, audio_path

    def extract_and_process_frames(video_path, lang, style, frame_rate=5):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = fps * frame_rate
        frame_numbers = list(range(0, total_frames, frame_interval))
        cap.release()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda f: process_frame(video_path, f, lang, style), frame_numbers))
        return results, fps

    def generate_commentary(incident, lang, style):
        style_prompt = COMMENTARY_STYLES.get(style, "")
        prompt = (
            f"You are a cricket commentator. Generate commentary in {lang} language, {style_prompt}\n"
            f"🎯 Focus on the incident only.\n"
            f"🚫 No team names, match results, or other details.\n"
            f"🗣️ Output only the commentary text. No nonsense texts or pretexts.\n\n"
            f"Incident: \"{incident}\""
        )
        try:
            response = model_gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating commentary: {e}"

    async def text_to_speech(text, output_file, lang):
        try:
            voice = LANGUAGE_OPTIONS.get(lang, "hi-IN-MadhurNeural")
            tts = edge_tts.Communicate(text, voice=voice, rate="+15%", pitch="+5Hz")
            await tts.save(output_file)
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")

    # UI
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.header("Video & Preferences")
        video_file = st.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi", "mkv"])
        selected_lang = st.selectbox("Choose Commentary Language", list(LANGUAGE_OPTIONS.keys()))
        selected_style = st.selectbox("Choose Commentary Style", list(COMMENTARY_STYLES.keys()))
        generate_btn = st.button("▶️ Continue and Generate Commentary")

        if video_file and generate_btn:
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, video_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(video_file.read())

            st.success("✅ Video uploaded successfully!")
            st.write("⏳ Processing video...")
            results, fps = extract_and_process_frames(temp_file_path, selected_lang, selected_style, frame_rate=5)
            cap = cv2.VideoCapture(temp_file_path)
            frame_index = 0
            audio_index = 0
            start_time = time.time()
            audio_playing = False
            video_element = st.empty()

            if results and os.path.exists(results[0][2]):
                frame_num, commentary, audio_file = results[audio_index]
                data, samplerate = sf.read(audio_file)
                sd.play(data, samplerate)
                audio_index += 1
                audio_playing = True

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                current_time = time.time() - start_time
                expected_frame_index = int(current_time * fps)
                if expected_frame_index > frame_index:
                    frame_index = expected_frame_index
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    video_element.image(img, use_container_width=True)
                    if audio_index < len(results) and not audio_playing:
                        frame_num, commentary, audio_file = results[audio_index]
                        if os.path.exists(audio_file):
                            data, samplerate = sf.read(audio_file)
                            sd.play(data, samplerate)
                            audio_index += 1
                            audio_playing = True
                if audio_playing and not sd.get_stream().active:
                    audio_playing = False
                time.sleep(1 / fps)

            cap.release()
            st.write("🎬 Video playback completed.")

    with col2:
        st.header("🗣️ Commentary Log")
        if video_file and generate_btn:
            for frame_num, commentary, _ in results:
                st.markdown(f"**Frame {frame_num}**: {commentary}")

# ------------------------ 📘 ABOUT PAGE ------------------------
elif app_mode == "📘 About":
    st.title("📘 About CrickVision AI")
    st.markdown("""
CrickVision AI is an AI-powered application that generates real-time commentary from cricket videos.

### Features:
- 🧠 AI-generated cricket insights using BLIP + Gemini
- 🌐 Multilingual support
- 🎙️ Customizable commentary styles
- 🔊 Real-time audio sync
    """)

# ------------------------ ❓ FAQ PAGE ------------------------
elif app_mode == "❓ FAQ":
    st.title("❓ Frequently Asked Questions")
    st.markdown("""
**Q: What kind of videos can I upload?**  
A: Short cricket video clips in MP4, MOV, AVI, or MKV format.

**Q: How does the commentary work?**  
A: Each frame is captioned using BLIP and interpreted by Gemini to generate commentary.

**Q: Is it real-time?**  
A: Yes, commentary is generated and played shortly after frame extraction.

**Q: Can I download the audio?**  
A: Not yet, but the feature is coming soon.
    """)

# ------------------------ 🔒 PRIVACY POLICY PAGE ------------------------
elif app_mode == "🔒 Privacy Policy":
    st.title("🔒 Privacy Policy")
    st.markdown("""
We respect your privacy:
- Videos are processed locally and not stored.
- No data is sent to external servers.
- Logging is for debugging only and does not include personal data.
    """)

# ------------------------ © COPYRIGHT PAGE ------------------------
elif app_mode == "© Copyright":
    st.title("© Copyright")
    st.markdown("""
All rights to CrickVision AI are reserved by Pratik Vairat.

This application is for educational and demo purposes only.  
BLIP and Gemini are used under their respective licenses.
    """)

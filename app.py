import os
import streamlit as st
import tempfile
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

# ------------------------ STREAMLIT UI NAVIGATION ------------------------
st.set_page_config(page_title="CrickVision AI", layout="wide")

# Sidebar for navigation
import streamlit as st
from PIL import Image

# Custom Sidebar Design with Icons
def custom_sidebar():
    st.sidebar.image('img/logo.png', width=200)  # Optional: Add project logo
    

    app_mode = st.sidebar.radio(
        "", 
        ["🏠 Home", "🎥 Real-time Commentary", "📘 About", "❓ FAQ", "🔒 Privacy Policy", "© Copyright"],
        index=0,  # Default to Home
        format_func=lambda x: f"**{x}**"  # Bold the selected page name
    )

    return app_mode

# Calling custom sidebar design function
app_mode = custom_sidebar()

# ------------------------ 🏠 HOME PAGE ------------------------
# ------------------------ 🏠 HOME PAGE ------------------------
if app_mode == "🏠 Home":
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: var(--text-color);
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: var(--text-color);
            opacity: 0.8;
            margin-bottom: 40px;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
            font-size: 1rem;
            color: var(--text-color);
            transition: all 0.3s ease-in-out;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
        }
        .cta-button {
            text-align: center;
            margin-top: 40px;
        }
        .footer-note {
            text-align: center;
            margin-top: 60px;
            color: var(--text-color);
            opacity: 0.6;
            font-size: 0.9rem;
        }
        .cta-button {
            text-align: center;
            margin-top: 50px;
        }

        .launch-button {
            background: linear-gradient(135deg, #4a90e2, #007aff);
            color: var(--text-color) !important;
            padding: 14px 30px;
            font-size: 1.2rem;
            font-weight: 900;
            border: none;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
            text-decoration: none !important;
            display: inline-block;
        }

        .launch-button:hover {
            background: linear-gradient(135deg, #3b7dd8, #0061c0);
            transform: scale(1.05);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🏏 CrickVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time multilingual cricket commentary powered by state-of-the-art AI</div>', unsafe_allow_html=True)

    st.subheader("✨ Why Choose CrickVision AI?")
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.markdown('<div class="feature-card">🎯 <b>Frame-based Captioning</b><br>BLIP intelligently captures the match moment-by-moment.</div>', unsafe_allow_html=True)
    with row1_col2:
        st.markdown('<div class="feature-card">🧠 <b>Context-aware Commentary</b><br>Google Gemini crafts natural, relevant commentary in real-time.</div>', unsafe_allow_html=True)
    with row1_col3:
        st.markdown('<div class="feature-card">🔊 <b>Natural Voice Playback</b><br>Edge TTS converts text into human-like speech instantly.</div>', unsafe_allow_html=True)

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown('<div class="feature-card">🌐 <b>Multilingual Support</b><br>Enjoy commentary in Hindi, Marathi, Gujarati, Tamil, Telugu & more.</div>', unsafe_allow_html=True)
    with row2_col2:
        st.markdown('<div class="feature-card">🎙️ <b>Style + Voice</b><br>Choose your favorite style and voice (male/female) for an immersive experience.</div>', unsafe_allow_html=True)
    with row2_col3:
        st.markdown('<div class="feature-card">📘 <b>Final Year Project</b><br>Built as an academic project with cutting-edge AI: BLIP, Gemini, Streamlit, TTS.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-button">
        <a class="launch-button" href="#">🎥 Launch Real-time Commentary</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer-note">© 2025 CrickVision AI | Built with ❤️ by Pratik Vairat</div>', unsafe_allow_html=True)

# ------------------------ 🎥 MAIN FEATURE ------------------------
elif app_mode == "🎥 Real-time Commentary":
    st.title("🎥 Real-time AI Cricket Commentary")

    VOICE_MAP = {
        "Hindi": {
            "Male": "hi-IN-MadhurNeural",
            "Female": "hi-IN-SwaraNeural",
        },
        "Marathi": {
            "Male": "mr-IN-SameerNeural",
            "Female": "mr-IN-AarohiNeural",
        },
        "Gujarati": {
            "Male": "gu-IN-NiranjanNeural",
            "Female": "gu-IN-DhwaniNeural",
        },
        "Tamil": {
            "Male": "ta-IN-ValluvarNeural",
            "Female": "ta-IN-PallaviNeural",
        },
        "Telugu": {
            "Male": "te-IN-MohanNeural",
            "Female": "te-IN-ShrutiNeural",
        },
    }

    COMMENTARY_STYLES = {
        "Akash Chorda Style": "as Akash Chopra, deliver thrilling and witty cricket commentary.",
        "Bollywood Style": "in a dramatic Bollywood-style tone, add flair and emotions to the commentary.",
        "Shayari Style": "in poetic Shayari form, describe the incident creatively.",
        "Funny Style": "use humorous and light-hearted tone to comment on the incident.",
    }

    configure(api_key="AIzaSyBRZ9cEMHE6Ys8jt27EuACWP7TEM7C2Yac")
    model_gemini = GenerativeModel("gemini-2.0-flash")
    processor = BlipProcessor.from_pretrained("blip_saved_model4")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_blip = BlipForConditionalGeneration.from_pretrained("blip_saved_model4").to(device)
    os.makedirs("data", exist_ok=True)

    def process_frame(video_path, frame_number, lang, gender, style):
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
        asyncio.run(text_to_speech(commentary, audio_path, lang, gender))
        return frame_number, commentary, audio_path

    def extract_and_process_frames(video_path, lang, gender, style, frame_rate=5):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = fps * frame_rate
        frame_numbers = list(range(0, total_frames, frame_interval))
        cap.release()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda f: process_frame(video_path, f, lang, gender, style), frame_numbers))
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

    async def text_to_speech(text, output_file, lang, gender):
        try:
            voice = VOICE_MAP.get(lang, {}).get(gender, "hi-IN-MadhurNeural")
            tts = edge_tts.Communicate(text, voice=voice, rate="+15%", pitch="+5Hz")
            await tts.save(output_file)
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.header("Video & Preferences")
        video_file = st.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi", "mkv"])
        selected_lang = st.selectbox("Choose Language", list(VOICE_MAP.keys()))
        selected_gender = st.radio("Select Voice", ["Male", "Female"], horizontal=True)
        selected_style = st.selectbox("Choose Commentary Style", list(COMMENTARY_STYLES.keys()))
        generate_btn = st.button("▶️ Continue and Generate Commentary")

        if video_file and generate_btn:
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, video_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(video_file.read())

            st.success("✅ Video uploaded successfully!")
            st.write("⏳ Processing video...")
            results, fps = extract_and_process_frames(temp_file_path, selected_lang, selected_gender, selected_style, frame_rate=5)

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
                if not ret:
                    break
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


# ------------------------ 📘 ABOUT PAGE ------------------------
elif app_mode == "📘 About":
    st.title("ℹ️ About CrickVision AI")

    st.markdown("### 🏏 Real-time AI Cricket Commentary Engine")

    # Two-column layout for Highlights and Tech Stack
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Key Features")
        st.markdown("""
        - 🎥 **Frame-based captioning** with BLIP  
        - 🧠 **Styled commentary** using Gemini 1.5  
        - 🗣️ **Edge TTS** for realistic voice (Male/Female)  
        - 🌐 Supports **Hindi, Marathi, Gujarati, Tamil, Telugu**  
        - 🎭 Choose styles: Bollywood, Shayari, Funny, Akash Chorda  
        - 🔄 **Real-time sync** of audio and video  
        - ⚙️ Efficient with `ThreadPoolExecutor` + `asyncio`  
        """)

    with col2:
        st.subheader("🧰 Tech Stack")
        st.markdown("""
        - 🧠 **BLIP** for vision-language modeling  
        - ✨ **Gemini API** for natural language generation  
        - 🔊 **Edge TTS** for voice synthesis  
        - 🎞️ `cv2`, `PIL`, `torch`, `transformers`  
        - 🧵 `ThreadPoolExecutor`, `sounddevice`, `asyncio`  
        - 🖥️ Built with **Streamlit**  
        """)

    st.markdown("---")

    # Architecture Expander
    with st.expander("📊 Architecture & Workflow"):
        st.markdown("""
        1. **Video Upload** ➡️ stored in temp directory  
        2. **Frame Sampling** ➡️ one every 5 seconds using OpenCV  
        3. **Captioning** ➡️ BLIP generates visual description  
        4. **Commentary Generation** ➡️ Gemini interprets the scene with selected style  
        5. **TTS Conversion** ➡️ Voice created using Edge TTS  
        6. **Video + Audio Sync** ➡️ Played in real-time using Python concurrency  
        """)

    # Vision Expander
    with st.expander("🎯 Vision & Future Scope"):
        st.markdown("""
        We aim to revolutionize sports content by enabling **AI-powered multilingual commentary**.

        **Potential Extensions:**
        - 🎙️ Live webcam commentary  
        - 🧵 Auto subtitle overlays  
        - 📺 Streaming integration  
        - 🕶️ VR/AR cricket experience  
        """)

    # Developer Info
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By")
    dev_col1, dev_col2 = st.columns([1, 4])
    with dev_col1:
        st.markdown(
        """
        <style>
            .round-image {
                border-radius: 50%;
                width: 120px;
                height: 120px;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
        </style>
        <img src="https://avatars.githubusercontent.com/u/84719258?v=4" class="round-image"/>
        """, unsafe_allow_html=True
        )
    with dev_col2:
        st.markdown("""
        **Pratik Vairat**  
        Final Year IT Student – Trinity Academy of Engineering, Pune  
        Full-stack Developer | AI Enthusiast | Cricket Buff  
        🔗 [GitHub](https://github.com/pratikvairat) | [LinkedIn](https://linkedin.com/in/pratik-vairat)
        """)

    st.markdown("---")


# ------------------------ ❓ FAQ PAGE ------------------------
elif app_mode == "❓ FAQ":
    st.title("❓ Frequently Asked Questions")
    st.markdown("Browse through common questions related to CrickVision AI.")

    with st.expander("📁 What kind of videos can I upload?"):
        st.markdown("""
        You can upload **short cricket video clips** in the following formats:
        - MP4
        - MOV
        - AVI
        - MKV
        """)

    with st.expander("🧠 How does the commentary system work?"):
        st.markdown("""
        Here's how it works:
        1. **BLIP** analyzes each selected frame from the video.
        2. The **Gemini API** interprets the visual caption into commentary text.
        3. **Edge TTS** generates realistic male or female voice commentary.
        4. Audio is synced and played alongside video playback in real time.
        """)

    with st.expander("🌐 Which languages and voices are supported?"):
        st.markdown("""
        Currently supported languages:
        - Hindi (Male & Female)
        - Marathi (Male & Female)
        - Gujarati (Male & Female)
        - Tamil (Male & Female)
        - Telugu (Male & Female)

        You can choose **male or female** voice style before generating commentary.
        """)

    with st.expander("🎭 What commentary styles can I select?"):
        st.markdown("""
        You can style your commentary in 4 fun ways:
        - **Akash Chorda Style** (realistic + witty)
        - **Bollywood Style** (dramatic + emotional)
        - **Shayari Style** (poetic + creative)
        - **Funny Style** (humorous + light-hearted)
        """)

    with st.expander("⏱️ Is the commentary generation real-time?"):
        st.markdown("""
        Yes! While it's not *live streaming*, it’s **near real-time**.  
        Commentary is generated **frame-by-frame**, just a few seconds after each frame is extracted.

        Audio and video are **played together** for a seamless experience.
        """)

    with st.expander("🔊 Can I download the audio commentary?"):
        st.markdown("""
        Not yet — but this feature is on our roadmap!  
        In future updates, users will be able to **download and share** generated audio clips.
        """)

    with st.expander("🛠️ Can I fine-tune or train it on custom cricket data?"):
        st.markdown("""
        This feature is under consideration.  
        Future versions may allow:
        - **Custom datasets** (like player-specific commentary)
        - Integration with **live match feeds**
        - Personalized commentary based on team/player preferences
        """)

# ------------------------ 🔒 PRIVACY POLICY PAGE ------------------------
elif app_mode == "🔒 Privacy Policy":
    st.title("🔒 Privacy Policy")
    st.markdown("""
    ### CrickVision AI – Privacy Policy

    **Effective Date:** April 2025  
    **Project Type:** BE Final Year Project (Information Technology)  
    **Institution:** Trinity Academy of Engineering, Pune University

    ---

    #### 🔍 Data Collection
    - We **do not collect, store, or transmit any personal information** from users.
    - Uploaded videos are processed **locally** in memory and are not saved or shared externally.
    - No data is sent to external servers except for requests made to AI models (e.g., Google Gemini) for commentary generation.

    #### 📁 Uploaded Files
    - All uploaded video files are stored temporarily in a local folder (`temp/`).
    - Files are used **only for processing** and are deleted automatically after use.
    - We recommend using only non-personal, demo cricket clips for testing.

    #### 🔊 Audio Generation
    - Voice synthesis is powered by Microsoft's **Edge TTS** engine.
    - Commentary text may be processed via **Google Gemini API** under its respective API policy.

    #### 🧪 Academic Nature
    - This application is created **solely for academic research and demonstration**.
    - It is **not a commercial product** and is not intended for production use or public deployment.
    - All AI model usage is aligned with their academic/open-source licenses.

    #### 🔐 Security
    - This app runs on **Streamlit**, a secure and sandboxed Python environment.
    - There is **no user login or authentication system**, and no user profiles are created.

    #### ❗ Disclaimer
    - CrickVision AI is an experimental prototype.
    - Users are responsible for any data they upload.
    - We are not liable for misuse of the app or the output generated.

    ---

    If you have any concerns or questions regarding this Privacy Policy, feel free to contact the project author.

    **Project Author:** Pratik Vairat  
    **Email:** [Add your email or GitHub here if needed]  
    **GitHub:** [github.com/your-profile](https://github.com/your-profile)
    """)

# ------------------------ © COPYRIGHT PAGE ------------------------
elif app_mode == "© Copyright":
    st.title("© Copyright")
    st.markdown("""
    All rights to **CrickVision AI** are reserved by **Pratik Vairat**.

    ### Copyright Information:
    > This project is a **Final Year BE project** developed by Pratik Vairat at **Trinity Academy of Engineering, Pune**.  
    > The copyright for this application is officially filed under the **Copyright Act of India**.  
    > Unauthorized use, reproduction, or distribution of any part of this application is prohibited.

    **Copyright Filing Reference**:  
    Official copyright filed with the **Indian Copyright Office** under application number: [Insert Application Number Here]  
    (You can add the official number here once you have it).

    This project is intended for **educational and demo purposes only**. For collaboration, licensing, or other inquiries, please contact:  
    - 📧 Email: [Your Email Here]
    - 🔗 [LinkedIn Profile](https://linkedin.com/in/pratik-vairat)
    
    ### Legal Disclaimer:
    This project utilizes publicly available technologies such as **BLIP**, **Google Gemini**, and **Microsoft Edge TTS** under their respective licenses.
    """)

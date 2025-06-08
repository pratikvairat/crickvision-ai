# AI Cricket Commentary Generator

## 📌 Project Title

**Integrating Vision and Textual Models for AI Cricket Commentary: Attention and BLIP Approaches**

## 🧠 Overview

This project presents an innovative system that automatically generates real-time cricket commentary by combining vision and language models. By utilizing BLIP for image captioning, Gemini for language generation, and a TTS engine for audio synthesis, the system delivers human-like commentary for live or recorded cricket footage.

## 🚀 Key Features

- ⚡ Frame-by-frame video processing for real-time analysis
- 🧾 Visual event understanding using BLIP (Bootstrapped Language-Image Pretraining)
- 💬 Text generation using Gemini/LLM for contextual commentary
- 🔊 Text-to-speech (TTS) conversion for synchronized audio output
- 🎥 Output video with overlaid commentary

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Video frame processing
- **BLIP** – Vision-language pre-trained model
- **Gemini / GPT** – Text generation
- **gTTS / pyttsx3** – Text-to-speech synthesis
- **FFmpeg** – Video and audio merging

## ⚙️ How It Works

1. **Video Input**: A cricket video is fed into the system.
2. **Frame Extraction**: Frames are extracted at fixed intervals (e.g., every 125 ms).
3. **Visual Captioning**: BLIP model generates a descriptive caption for each selected frame.
4. **Text Generation**: A language model processes the caption and generates human-like commentary.
5. **Speech Synthesis**: The text is converted to speech and saved as `.mp3` files (e.g., 0.mp3, 125.mp3...).
6. **Merging**: Commentary is synchronized with the video to generate the final output.

## 🧪 Evaluation

- Precision, Recall, and F1 Score used to evaluate relevance and fluency of generated commentary.
- Human evaluation for context and naturalness.

## 🧩 Use Cases

- Sports broadcasting automation
- Assistive tools for the visually impaired
- Real-time game analysis

# Step 1: Clone the repository

git clone https://github.com/your-username/ai-cricket-commentary.git
cd ai-cricket-commentary

# Step 2: Install dependencies

pip install -r requirements.txt

# Step 3: Launch the Streamlit app

streamlit run src/app.py

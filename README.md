# 🤟 Sign Language Detection System

An AI-powered real-time sign language detection system that recognizes 7 common hand gestures using computer vision and deep learning. Works with any standard webcam.

📋 Overview

This system bridges the communication gap between deaf and hearing communities by recognizing sign language gestures in real-time. It detects 7 gestures: **Hello, I Love You, No, Okay, Please, Thank You, and Yes**.

Features
- ✅ Real-time detection (30+ FPS)
- ✅ Works with standard webcam
- ✅ No specialized hardware needed
- ✅ 85-95% accuracy
- ✅ Lightweight and fast

---

## 🛠️ Tech Stack

- **Python 3.8+** - Programming language
- **OpenCV 4.12.0** - Image processing & webcam capture
- **MediaPipe 0.10.11** - Hand detection & tracking
- **TensorFlow 2.15.0** - Deep learning framework
- **CVZone 1.6.1** - Simplified hand detection
- **NumPy 1.24.4** - Numerical computations

---

📥 Installation

Prerequisites
- Python 3.8 or higher
- Webcam (built-in or external)
- 4GB RAM minimum

Setup

1. **Clone the repository**
```bash
git clone https://github.com/karansinghgogadev/sign-language-detection.git
cd sign-language-detection
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv38
venv38\Scripts\activate

# macOS/Linux
python3 -m venv venv38
source venv38/bin/activate
```

3. **Install dependencies**
```bash
pip install tensorflow opencv-python cvzone mediapipe numpy
```

---

## 🚀 Usage

### Quick Start (Use Pre-trained Model)

1.**Download**: [Google Drive Link](https://drive.google.com/drive/folders/1S6lvktsOTPbfRn4AdDv1_sq9OHlREYAT?usp=drive_link)
2. **Extract** and place in project root:
   ```
   Model/
   ├── keras_model.h5
   └── labels.txt
   ```
3. **Run detection**
   ```bash
   python detection.py
   ```
4. Show gestures to webcam, press **'Q'** to quit

### Train Your Own Model

**Step 1: Collect Data**
```bash
python datacollection.py
```
- Press **'S'** to save images (300-500 per gesture)
- Change `folder = "Data/Hello"` for different gestures
- Press **'Q'** to quit

**Step 2: Train Model**
```bash
python test.py
```
Wait 15-30 minutes for training to complete

**Step 3: Run Detection**
```bash
python detection.py
```

---

## 📊 Dataset

**Download**: [Google Drive Link](https://drive.google.com/drive/folders/1S6lvktsOTPbfRn4AdDv1_sq9OHlREYAT?usp=drive_link)

### Structure
```
Data/
├── Hello/          (300-500 images)
├── I love you/
├── No/
├── Okay/
├── Please/
├── Thankyou/
└── Yes/
```

---

## 📸 Output Screenshots



---

## 📁 Project Structure

```
sign-language-detection/
├── Data/                  # Training dataset
├── Model/                 # Trained model files
│   ├── keras_model.h5
│   └── labels.txt
├── screenshots/           # Output screenshots
├── datacollection.py     # Data collection script
├── test.py               # Model training script
├── detection.py          # Real-time detection
└── README.md
```

---



<div align="center">

**Made with ❤️ for the Deaf Community**

⭐ Star this repo if you find it helpful!

</div>

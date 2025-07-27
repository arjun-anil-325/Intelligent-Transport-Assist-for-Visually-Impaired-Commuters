# Intelligent Transport Assist for Visually Impaired Commuters

This project presents a real-time AI-based assistive system designed to help visually impaired individuals and non-native language users identify public buses using computer vision and speech synthesis technologies.

## 👁️‍🗨️ Project Overview

The system detects and reads Malayalam bus name boards from live video input, and provides clear audible announcements using a text-to-speech engine.

### 🔍 Features
- **Bus Nameboard Detection** using YOLOv8
- **OCR (Optical Character Recognition)** for Malayalam using Tesseract
- **Text Categorization** into primary (main place) and secondary (route details)
- **Fuzzy Matching** to handle OCR errors
- **Audio Announcements** using VITS TTS model in Malayalam
- **Real-time processing** on Jetson Orin Nano hardware

---

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- Tesseract OCR (with `mal.traineddata`)
- VITS TTS (Text-to-Speech)
- OpenCV, NumPy, Pillow, Sounddevice
- Jetson Orin Nano + JetPack SDK
- FuzzyWuzzy for string matching

---

## ⚙️ System Workflow
1. **Video Input** from camera
2. **YOLOv8 detects** the bus nameboard
3. **Image Preprocessing** (grayscale, denoise, threshold)
4. **Text Extraction** using Tesseract OCR (Malayalam)
5. **Categorization** into main destination and sub-routes
6. **Text Matching** with route database using fuzzy matching
7. **Speech Output** generated using VITS TTS
8. **Repeat detection** avoided using frequency-based filtering

---

## 📊 Results
- Achieved **Precision**: 82.6% and **Recall**: 92.1% during validation
- Real-time inference and audio announcement tested on live video
- Handles most common OCR errors using fuzzy match + filtering

---

## 🔗 Resources

- **YOLOv8 Weights**: [Click here](https://drive.google.com/drive/folders/1ncUbd2AaAfa111S2aGlBSKzdvnSr3SWk?usp=sharing)
- **Location Directory (text files)**: [Click here](https://drive.google.com/drive/folders/1-G078eFwcFvi8ET-vwtu-xX6h12eEQMl?usp=sharing)
- **Malayalam OCR Model**: [Click here](https://drive.google.com/drive/folders/1RkUU2PwVr7yU9cwRWfyWQI0_9ijmvYH6?usp=sharing)
- **Test Video (Zoomed)**: [Click here](https://drive.google.com/drive/folders/1Jda5ANy8cwcdVvzQRpAfg1MN6D30m7dK?usp=sharing)
- **Test Video (Non-Zoomed)**: [Click here](https://drive.google.com/drive/folders/19qvpAa2J7ptpSXs-M_Hqjuzuj2FcNpdE?usp=sharing)

---

## 🚀 Future Improvements
- Improve robustness under low-light conditions using IR cameras or image enhancement
- Add support for scrolling LED bus nameboards
- Train a deep learning-based Malayalam OCR for higher accuracy
- Add multilingual support (Hindi, Tamil, etc.)
- Deploy as a mobile application or cloud-integrated service
- Use GPS + live database to update route files dynamically


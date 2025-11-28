# Object Detector for Blind Users
An assistive Python tool that detects objects such as people, cars, buses, and more using computer vision.  
The goal of this project is to help blind and visually-impaired users understand their surroundings.

---

## ✨ Features
- Detects common objects (person, car, bus, dog, etc.)
- Works with images or real-time webcam
- Optional audio feedback using text-to-speech
- Simple and accessible interface
- Built using Python and modern machine learning models

---

## 📦 Requirements
Install dependencies with pip:

```bash
pip install opencv-python
pip install numpy
pip install torch torchvision
pip install pillow
# optional for audio output
pip install pyttsx3

```
## ▶️ How to Use
Run object detection on an image
```
python detect.py --image example.jpg
```
Run real-time webcam detection
```
python detect.py --webcam
```
Run webcam detection with audio
```
python detect.py --webcam --audio

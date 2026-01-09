# Face Recognition (MediaPipe + OpenCV)

A lightweight facial recognition system built using **MediaPipe**, **OpenCV**, and **NumPy**.  
The system performs real-time face detection from a webcam feed and recognizes identities using stored face embeddings — no deep learning model training required.

---

## 🚀 Features

- 🔍 **Real-time face detection** using MediaPipe Face Detection / Face Mesh
- 👤 **Face recognition** via embedding extraction + distance matching
- 🎥 **Webcam + video stream support**
- 💾 **Embedding persistence** (saved as `.npy` arrays)
- 🧠 **No model training required**
- 🎯 **Fast & lightweight** — runs on CPU
- 🧩 **Beginner-friendly codebase**
- 📝 Modular design for experimentation

---

## 🛠 Tech Stack

- **Python 3.x**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- _(Optional)_ **scikit-learn** for improved similarity metrics

---

## 🧱 System Overview

The recognition pipeline works as follows:

1. MediaPipe detects and tracks face landmarks
2. A feature embedding is generated for each detected face
3. Embeddings are compared with saved vectors
4. Identity is predicted using distance thresholds

---

## 📁 Project Structure


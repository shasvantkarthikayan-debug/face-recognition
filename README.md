# FacePass — Browser-based Face Recognition (face-api.js + Flask)

FacePass is a hybrid real-time facial recognition system.  
The browser handles face detection & embedding extraction using **face-api.js**, while a lightweight **Flask** backend performs identity matching using stored embeddings.

This approach requires **no GPU**, works in real-time on standard hardware, and does not require training a deep learning model locally.

---

## 🚀 Features

- 🔍 Real-time face detection (client-side)
- 👤 128-D face embeddings extracted in browser via face-api.js
- 🌐 WebRTC webcam streaming
- 🧠 Identity matching via Flask backend
- 💾 Embedding storage as `.npy` or JSON for persistence
- 🎯 Runs on CPU (no GPU needed)
- 🧩 Lightweight codebase — easy to understand & modify
- 🔐 No cloud APIs, all inference handled locally

---

## 🛠 Tech Stack

**Client**
- face-api.js (TensorFlow.js)
- WebRTC
- HTML/CSS/JS

**Backend**
- Flask (Python)
- NumPy
- Scikit-Learn (optional for metrics)

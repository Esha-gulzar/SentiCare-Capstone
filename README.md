<!-- SentiCare README -->
<div align="center">

# 🧠 SentiCare  
### Voice-Based AI for Mental Health Support

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

> A compassionate AI that listens to your voice and supports your mental health journey.

[Key Features](#key-features) •
[Tech Stack](#tech-stack) •
[Installation](#installation) •
[Usage](#usage) •
[Architecture](#architecture) •
[Team & Credits](#team--credits)

</div>

---

## 📌 About
SentiCare is a capstone project developed at the **University of Sargodha**. It combines **voice emotion detection** with a **conversational AI chatbot** to provide mental health support. The system classifies anxiety, stress, and depression by fusing vocal cues with self‑reported questionnaire answers, then offers empathetic, evidence‑based responses.

---

## ✨ Key Features
- 🎤 **Voice Check‑In** – Record up to 30 seconds of speech; emotions are extracted from pitch, tone, and rhythm.
- 🧠 **ML‑Powered Assessment** – Combines voice features with user‑filled surveys to predict anxiety, stress, and depression levels.
- 💬 **Therapeutic Chatbot** – Supportive AI companion that remembers your progress and suggests coping exercises.
- 📈 **Emotion Tracking** – Visual charts of your emotional trends over time.
- 🔉 **Text‑to‑Speech** – Bot responses played aloud (bilingual: English & Urdu).
- 🔒 **Privacy First** – Data stays in your session; no personal information is stored.

---

## 🧰 Tech Stack

### 🎨 Frontend
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![recharts](https://img.shields.io/badge/recharts-22b5bf?style=flat-square&logo=recharts&logoColor=white)

### ⚙️ Backend
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Edge-TTS](https://img.shields.io/badge/Edge--TTS-0078D7?style=flat-square&logo=microsoft-edge&logoColor=white)

### 🧪 Machine Learning
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

### 📦 Deployment & Tools
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

## 📂 Repository Structure

```text
.
├── Datasets/                 # CSV datasets for anxiety, stress, depression
├── Design Document/          # Detailed design doc (Word + PDF)
├── Presentation/             # Final project presentation
├── SentiCare Diagrams/       # UML diagrams (Class, ER, Sequence, etc.)
├── SentiCare SRS/            # Software Requirements Specification
├── senticare-frontend/       # React application
│   ├── src/
│   │   ├── components/       # VoiceCheckIn, TherapyCards, etc.
│   │   └── App.jsx
│   └── package.json
├── artifacts/                # Trained model pipelines (.joblib)
├── app.py                    # Flask API entry point
├── Dockerfile                # Container configuration
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend Setup
```
git clone https://github.com/sheikh-zain786/SentiCare-Capstone.git
cd SentiCare-Capstone
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py # Runs on http://localhost:5000
```
### Frontend Setup
```
cd senticare-frontend
npm install
npm start # Runs on http://localhost:3000
```

Make sure the backend is running before using the frontend.


## 💡 Usage
1. Open the frontend in your browser.
2. Allow microphone access when prompted.
3. Click **Start Voice Check‑In** and speak for up to 30 seconds.
4. Answer the follow‑up questionnaire about your mood, sleep, etc.
5. The AI fuses your voice emotion with your answers and gives a preliminary assessment.
6. Continue the conversation in the chat – the bot will remember your data and offer personalised support.


## 🖼️ Architecture (High‑Level)
```
User Browser
    │
    ├── Voice Recording (MediaRecorder)
    └── Questionnaire Answers
            │
            ▼
     ┌──────────────┐
     │  React App   │
     └──────┬───────┘
            │ HTTP/REST
            ▼
     ┌──────────────┐
     │ Flask API    │
     │ (app.py)     │
     └──────┬───────┘
            │
            ├── Load Preprocessor & Models (joblib)
            ├── Process voice features (extracted by frontend)
            ├── Predict Anxiety / Stress / Depression
            └── Generate empathetic response
                   │
                   ▼
            Text‑to‑Speech (Edge‑TTS)
                   │
                   ▼
            Audio sent back to browser

```

## 👥 Team & Credits

### Project Manager
- **Dr. Illyas** – *University of Sargodha*

### Project Supervisor
- **Dr. Saad Razaq** – *University of Sargodha*

### Developers
| Name | 
|------|
| **Esha Gulzar** | 
| **Wajeeha Ijaz**  |
| **Sheikh Zain**  |   


## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements
- Mental health datasets: [Student Mental Health Survey](https://www.kaggle.com/datasets/sonia22222/students-mental-health-assessments)
- Edge TTS for high‑quality, multilingual speech synthesis
- Hugging Face Spaces for cloud deployment inspiration

---

<div align="center">
Made with ❤️ by the SentiCare Team – University of Sargodha, 2025–2026
</div>

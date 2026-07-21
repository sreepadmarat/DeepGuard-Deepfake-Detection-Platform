# DeepGuard: Deepfake Detection & Forensic Analysis Platform

DeepGuard is a comprehensive, full-stack Flask application designed for deepfake detection and detailed forensic analysis of images and videos. The platform utilizes advanced deep learning architectures (EfficientViT-B0) for model inference and integrates with the Google Gemini API to provide interactive, expert forensic explanations of the model's classifications.

## 🚀 Key Features

- **Multi-Modal Detection**: Supports both image and video uploads (`.png`, `.jpg`, `.jpeg`, `.mp4`, `.avi`, `.mov`, `.mkv`).
- **Explainable AI (XAI)**:
  - **GradCAM Visualizations**: Dynamically generates heatmaps highlighting features/regions the model focused on (e.g., boundary blending, facial anomalies).
  - **RGB Channel Spectral Analysis**: Generates channel frequency distribution histograms to identify unnatural color gradients or generator patterns.
- **DeepGuard AI Chat Assistant**: An interactive chatbot powered by Gemini that references forensic evidence (confidence scores, GradCAM activations, RGB patterns) to answer user questions in real-time.
- **PDF Report Generation**: Automatically compiles findings into a downloadable, professional forensic report.
- **Security & Alerts**: Sends automated email alerts to users when a uploaded file is classified as a deepfake.
- **User History Dashboard**: View previous detection runs, review metrics, view visualizations, and download past reports.

---

## 🛠️ Technology Stack

- **Backend**: Flask, Flask-Session, SQLite3
- **Deep Learning**: PyTorch, Timm (EfficientViT-B0), Albumentations
- **Explainable AI**: PyTorch-Grad-CAM, Matplotlib
- **Forensic Report**: ReportLab
- **AI Explainer**: Google Generative AI (Gemini 2.5 Flash Lite)

---

## 📋 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sreepadmarat/deepfake-detection.git
cd deepfake-detection
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your details:
- **SECRET_KEY**: A custom secret key for session signing.
- **GEMINI_API_KEYS**: Your Gemini API key(s) (comma-separated if using multiple for rotation/quota fallback).
- **SMTP_EMAIL** and **SMTP_PASSWORD**: Gmail/SMTP credentials for sending alerts.

### 5. Model Weights Setup
Download the model weights and place them in the correct directory:
- Image classification model weights: Save to `models/efficientvit_b0_final.pth`
- Video classification model weights: Save to `models/final_video_models/efficientvit_b0_video_final.pth`

---

## 💻 Running the Application

Start the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📂 Project Structure

```
├── app.py                     # Application entrypoint
├── config.py                  # Dynamic configuration loader (via dotenv)
├── database.py                # Database helper functions & schema migrations
├── requirements.txt           # Main python dependencies
├── .env.example               # Template environment configuration
├── routes/
│   ├── auth_routes.py         # Login, logout, signup routes
│   ├── chat_routes.py         # File upload, model predictions, chat & reports
│   └── history_routes.py      # History log and report retrieval
├── services/
│   ├── email_service.py       # SMTP email notification dispatch
│   ├── gemini_service.py      # Prompt construction and Gemini API connection
│   ├── image_service.py       # Image inference, GradCAM & RGB generation
│   ├── report_service.py      # PDF Report construction
│   └── video_service.py       # Frame extraction, video inference & GradCAM grid
├── templates/                 # Frontend HTML pages
├── static/
│   ├── temp/                  # Auto-generated reports and visual heatmaps
│   └── CSS/JS static assets
└── utils/
    └── file_utils.py          # File validation and cleanup utilities
```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This platform is developed for research and educational purposes. While the underlying models achieve high accuracy on benchmark datasets, no AI-driven system is flawless. Results should be verified using additional forensic methods when used in critical settings.

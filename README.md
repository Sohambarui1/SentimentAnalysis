# Sentiment Analysis & Mental Health Detection System

This project is a machine learning–based web application designed to analyze user text and detect sentiment with a focus on mental health indicators. It uses a transformer-based deep learning model and provides results through a simple web interface.

---

## 📌 Project Overview

The system accepts textual input from users and analyzes it to identify sentiment patterns related to mental health conditions such as stress, anxiety, or emotional polarity. The backend is implemented using Python and Flask, while the frontend is built with HTML, CSS, and JavaScript.

---

## 🧠 Technologies Used

### Programming Language
- Python

### Machine Learning & NLP
- Transformer-based model (BERT/BART)
- PyTorch
- Hugging Face Transformers

### Web Framework
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Version Control
- Git & GitHub
- Git Large File Storage (Git LFS) for large model files

---
## How to Run the Project

```bash
git lfs install
git clone https://github.com/Subham-Singha/SentimentAnalysis.git
cd SentimentAnalysis
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python app.py

```

Open your browser and go to:
http://127.0.0.1:5000

Note
If the model file does not download automatically, run:
git lfs pull

## 📂 Project Structure


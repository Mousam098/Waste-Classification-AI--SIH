# ♻️ Waste Classification AI

A professional and lightweight **Waste Classification Web App** that classifies waste images into two categories:

- **Biodegradable**
- **Non-Biodegradable**

This project is built using **Python**, **Streamlit**, and **Pillow**. It is designed to run smoothly on **Python 3.13** without TensorFlow, NumPy, or compiler-related installation issues.

---

## 📌 Project Overview

Waste segregation is an important step for better recycling, composting, and environmental management. This application allows users to upload an image of a waste item and predicts whether the item is biodegradable or non-biodegradable.

The application provides:

- Clean Streamlit-based web interface
- Image upload system
- Waste category prediction
- Confidence score
- Disposal suggestion
- Built-in default model
- Custom model training using your own images
- GitHub-ready project structure

---

## 🚀 Features

- Upload waste images in **JPG, PNG, WEBP, or BMP** format
- Classify waste as **Biodegradable** or **Non-Biodegradable**
- Display prediction confidence score
- Provide basic waste disposal guidance
- Works even without training using a built-in default model
- Allows custom training with your own dataset
- Lightweight and easy to run on Windows

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pillow
- JSON-based lightweight model
- Image feature extraction

---

## 📁 Project Structure

```text
waste_classification_WORKING_Python313/
│
├── app/
│   └── app.py
│
├── src/
│   ├── features.py
│   ├── predict.py
│   └── train_model.py
│
├── dataset/
│   ├── train/
│   │   ├── biodegradable/
│   │   └── non_biodegradable/
│   │
│   └── validation/
│       ├── biodegradable/
│       └── non_biodegradable/
│
├── models/
│   └── waste_model.json
│
├── requirements.txt
├── run_app.bat
├── train_model.bat
├── .gitignore
└── README.md

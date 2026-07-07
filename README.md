# Waste Classification App

A ready-to-run waste classification project for Python 3.13.

This version does not use TensorFlow, NumPy, or any heavy compiled dependency.  
It works smoothly on Windows with Python 3.13.

The app classifies waste as:

- Biodegradable
- Non-Biodegradable

It includes:

- Streamlit web interface
- Lightweight image feature extraction
- Trainable JSON model
- Default built-in model fallback
- Dataset folder structure
- GitHub-ready files

---

## 1. Install

Open CMD inside the project folder:

```bash
pip install -r requirements.txt
```

---

## 2. Run Directly

You can run the app even before training:

```bash
streamlit run app/app.py
```

The app will use a default built-in model.

---

## 3. Add Your Dataset

Put images here:

```text
dataset/train/biodegradable
dataset/train/non_biodegradable
dataset/validation/biodegradable
dataset/validation/non_biodegradable
```

Examples:

### Biodegradable

- banana peel
- leaves
- vegetable waste
- food waste
- paper
- cardboard

### Non-Biodegradable

- plastic bottle
- chips packet
- glass bottle
- metal can
- e-waste
- wrappers

---

## 4. Train the Model

After adding images:

```bash
python src/train_model.py
```

It will save:

```text
models/waste_model.json
```

---

## 5. Run the App

```bash
streamlit run app/app.py
```

---

## 6. GitHub Upload

```bash
git init
git add .
git commit -m "Initial waste classification project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

---

## Important Note

This is a lightweight educational image classifier.  
For industrial-level accuracy, use a large image dataset and deep learning models such as MobileNetV2, EfficientNet, or ResNet.  
This project is designed to run easily on your current Python 3.13 setup.

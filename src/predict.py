import os
import json
import math
from features import extract_features, feature_distance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_model.json")


DEFAULT_MODEL = {
    "class_names": {
        "biodegradable": "Biodegradable",
        "non_biodegradable": "Non-Biodegradable"
    },
    "prototypes": {
        "biodegradable": {
            "avg_r": 118.0,
            "avg_g": 106.0,
            "avg_b": 72.0,
            "avg_brightness": 98.0,
            "green_ratio": 0.18,
            "brown_ratio": 0.32,
            "white_ratio": 0.05,
            "dark_ratio": 0.16,
            "bright_ratio": 0.08,
            "plastic_like_ratio": 0.05,
            "texture_score": 55.0,
            "roughness": 65.0
        },
        "non_biodegradable": {
            "avg_r": 145.0,
            "avg_g": 145.0,
            "avg_b": 150.0,
            "avg_brightness": 146.0,
            "green_ratio": 0.04,
            "brown_ratio": 0.04,
            "white_ratio": 0.18,
            "dark_ratio": 0.08,
            "bright_ratio": 0.27,
            "plastic_like_ratio": 0.20,
            "texture_score": 38.0,
            "roughness": 42.0
        }
    },
    "training_counts": {
        "biodegradable": 0,
        "non_biodegradable": 0
    }
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        return DEFAULT_MODEL, False

    with open(MODEL_PATH, "r", encoding="utf-8") as file:
        return json.load(file), True


def confidence_from_distances(best_distance, other_distance):
    total = best_distance + other_distance

    if total <= 0:
        return 50.0

    separation = abs(other_distance - best_distance) / total
    confidence = 55 + separation * 45

    return round(min(max(confidence, 55.0), 99.0), 2)


def predict_waste(uploaded_file):
    model, is_trained = load_model()
    features = extract_features(uploaded_file)

    biodegradable_proto = model["prototypes"]["biodegradable"]
    non_biodegradable_proto = model["prototypes"]["non_biodegradable"]

    bio_distance = feature_distance(features, biodegradable_proto)
    non_bio_distance = feature_distance(features, non_biodegradable_proto)

    if bio_distance <= non_bio_distance:
        class_key = "biodegradable"
        confidence = confidence_from_distances(bio_distance, non_bio_distance)
        suggestion = (
            "This item looks suitable for composting or organic waste processing, "
            "if it is not contaminated with plastic, metal, or chemicals."
        )
    else:
        class_key = "non_biodegradable"
        confidence = confidence_from_distances(non_bio_distance, bio_distance)
        suggestion = (
            "This item should be separated for recycling or safe disposal. "
            "Do not mix it with organic kitchen waste."
        )

    return {
        "label": model["class_names"][class_key],
        "class_key": class_key,
        "confidence": confidence,
        "suggestion": suggestion,
        "is_trained": is_trained,
        "bio_distance": round(bio_distance, 4),
        "non_bio_distance": round(non_bio_distance, 4),
        "features": features,
        "training_counts": model.get("training_counts", {})
    }

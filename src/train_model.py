import os
import json
from features import extract_features, average_features, IMAGE_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "waste_model.json")

CLASSES = ["biodegradable", "non_biodegradable"]


def collect_images(folder):
    image_paths = []

    if not os.path.exists(folder):
        return image_paths

    for name in os.listdir(folder):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(os.path.join(folder, name))

    return image_paths


def main():
    model = {
        "class_names": {
            "biodegradable": "Biodegradable",
            "non_biodegradable": "Non-Biodegradable"
        },
        "prototypes": {},
        "training_counts": {}
    }

    for class_name in CLASSES:
        folder = os.path.join(TRAIN_DIR, class_name)
        image_paths = collect_images(folder)

        if not image_paths:
            print(f"No images found in {folder}")
            continue

        print(f"Processing {len(image_paths)} images for {class_name}...")

        features = []
        for path in image_paths:
            try:
                features.append(extract_features(path))
            except Exception as error:
                print(f"Skipped {path}: {error}")

        if features:
            model["prototypes"][class_name] = average_features(features)
            model["training_counts"][class_name] = len(features)

    if len(model["prototypes"]) < 2:
        raise RuntimeError(
            "Training failed. Add images in both folders: "
            "dataset/train/biodegradable and dataset/train/non_biodegradable"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=4)

    print("Model trained successfully.")
    print(f"Saved model at: {MODEL_PATH}")
    print("Training counts:", model["training_counts"])


if __name__ == "__main__":
    main()

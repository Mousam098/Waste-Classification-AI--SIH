from PIL import Image, ImageStat
import math


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_image(path_or_file, size=(160, 160)):
    image = Image.open(path_or_file).convert("RGB")
    image = image.resize(size)
    return image


def extract_features(path_or_file):
    """
    Extract simple visual features from an image.

    This keeps the project lightweight and Python 3.13 compatible.
    No TensorFlow, NumPy, or compiler is required.
    """
    image = load_image(path_or_file)
    width, height = image.size
    pixels = list(image.getdata())
    total = len(pixels)

    r_sum = 0
    g_sum = 0
    b_sum = 0

    green_pixels = 0
    brown_pixels = 0
    white_pixels = 0
    dark_pixels = 0
    bright_pixels = 0
    plastic_like_pixels = 0

    for r, g, b in pixels:
        r_sum += r
        g_sum += g
        b_sum += b

        brightness = (r + g + b) / 3

        if g > r * 1.08 and g > b * 1.08:
            green_pixels += 1

        if r > 70 and g > 45 and b < 90 and r >= g:
            brown_pixels += 1

        if r > 190 and g > 190 and b > 190:
            white_pixels += 1

        if brightness < 65:
            dark_pixels += 1

        if brightness > 185:
            bright_pixels += 1

        # Smooth, bright, artificial-looking color regions
        if (r > 120 or g > 120 or b > 120) and abs(r - g) + abs(g - b) + abs(r - b) > 110:
            plastic_like_pixels += 1

    avg_r = r_sum / total
    avg_g = g_sum / total
    avg_b = b_sum / total
    avg_brightness = (avg_r + avg_g + avg_b) / 3

    # Color variation / texture approximation
    stat = ImageStat.Stat(image)
    std_r, std_g, std_b = stat.stddev
    texture_score = (std_r + std_g + std_b) / 3

    # Edge-like roughness approximation using neighbor difference
    edge_total = 0
    edge_count = 0
    sample_step = 4
    px = image.load()

    for y in range(0, height - sample_step, sample_step):
        for x in range(0, width - sample_step, sample_step):
            r1, g1, b1 = px[x, y]
            r2, g2, b2 = px[x + sample_step, y + sample_step]
            edge_total += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            edge_count += 1

    roughness = edge_total / max(edge_count, 1)

    features = {
        "avg_r": avg_r,
        "avg_g": avg_g,
        "avg_b": avg_b,
        "avg_brightness": avg_brightness,
        "green_ratio": green_pixels / total,
        "brown_ratio": brown_pixels / total,
        "white_ratio": white_pixels / total,
        "dark_ratio": dark_pixels / total,
        "bright_ratio": bright_pixels / total,
        "plastic_like_ratio": plastic_like_pixels / total,
        "texture_score": texture_score,
        "roughness": roughness,
    }

    return features


def feature_distance(features, prototype):
    """
    Weighted distance between image features and a class prototype.
    Lower distance means higher similarity.
    """
    weights = {
        "avg_r": 0.01,
        "avg_g": 0.01,
        "avg_b": 0.01,
        "avg_brightness": 0.015,
        "green_ratio": 3.0,
        "brown_ratio": 3.0,
        "white_ratio": 1.2,
        "dark_ratio": 1.0,
        "bright_ratio": 1.0,
        "plastic_like_ratio": 3.5,
        "texture_score": 0.04,
        "roughness": 0.025,
    }

    distance = 0.0
    for key, weight in weights.items():
        distance += abs(features.get(key, 0) - prototype.get(key, 0)) * weight

    return distance


def average_features(feature_list):
    if not feature_list:
        return {}

    keys = feature_list[0].keys()
    averaged = {}

    for key in keys:
        averaged[key] = sum(item[key] for item in feature_list) / len(feature_list)

    return averaged

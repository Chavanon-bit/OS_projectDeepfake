# ================================
# Deepfake Detection (FINAL - PART1 ONLY)
# ================================

import json
import os
import cv2
from multiprocessing import Pool, Lock, Manager

# ================================
# CONFIG
# ================================
JSON_PATH = "data/annotations/Test-Dev_poly.json"
IMAGE_FOLDER = "data"
OUTPUT_DIR = "output"

OUTPUT_TXT = os.path.join(OUTPUT_DIR, "result.txt")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "result.json")

NUM_WORKERS = 4
BATCH_SIZE = 10

# ================================
# FIX PATH
# ================================
def fix_path(path):
    return path.replace("Images", "images")

# ================================
# LOAD ONLY AVAILABLE IMAGES
# ================================
def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    image_map = {}

    for img in data["images"]:
        path = fix_path(img["file_name"])
        full_path = os.path.join(IMAGE_FOLDER, path)

        if os.path.exists(full_path):
            image_map[img["id"]] = img["file_name"]

    print(f"✅ Found {len(image_map)} available images")

    return image_map

# ================================
# BATCH LOADER (MEMORY OPTIMIZED)
# ================================
def batch_loader(image_items, batch_size):
    items = list(image_items.items())
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

# ================================
# DETECTION FUNCTION
# ================================
def detect_image(args):
    img_id, img_path = args

    img_path_fixed = fix_path(img_path)
    full_path = os.path.join(IMAGE_FOLDER, img_path_fixed)

    try:
        img = cv2.imread(full_path)

        if img is None:
            return None

        score = img.mean()
        result = "Fake" if score < 100 else "Real"

        return (img_path, result)

    except:
        return None

# ================================
# WRITE OUTPUT
# ================================
def write_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock = Lock()

    results = [r for r in results if r is not None]

    with lock:
        with open(OUTPUT_TXT, "w") as f:
            for img, res in results:
                f.write(f"{img} -> {res}\n")

    with lock:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(
                [{"image": img, "result": res} for img, res in results],
                f,
                indent=4
            )

# ================================
# SCHEDULER (FCFS + MULTIPROCESS)
# ================================
def run_scheduler(image_map):
    manager = Manager()
    results = manager.list()

    total = len(image_map)
    processed = 0

    with Pool(processes=NUM_WORKERS) as pool:
        for batch in batch_loader(image_map, BATCH_SIZE):
            batch_results = pool.map(detect_image, batch)

            for res in batch_results:
                if res:
                    results.append(res)

            processed += len(batch)
            print(f"⏳ Progress: {processed}/{total}")

    return list(results)

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("🔄 Loading dataset...")

    image_map = load_data(JSON_PATH)

    print("🚀 Running detection (multiprocessing)...")
    results = run_scheduler(image_map)

    print("💾 Saving results...")
    write_results(results)

    print(f"✅ Done! Total processed: {len(results)} images")
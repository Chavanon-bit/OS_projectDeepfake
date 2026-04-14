
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
# LOAD JSON DATA
# ================================
def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    image_map = {img["id"]: img["file_name"] for img in data["images"]}

    return image_map

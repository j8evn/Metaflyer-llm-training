import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "models/merged/preprocessor_config.json")

def patch_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found!")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 학습 시 설정했던 값으로 강제 주입
    # train_pixels = 256 * 256 = 65536
    # min_pixels = 28 * 28 = 784
    
    print(f"Old max_pixels: {config.get('max_pixels')}")
    print(f"Old min_pixels: {config.get('min_pixels')}")

    config["max_pixels"] = 65536
    config["min_pixels"] = 784
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"New max_pixels: {config.get('max_pixels')}")
    print(f"New min_pixels: {config.get('min_pixels')}")
    print(f"Successfully patched {CONFIG_PATH}")

if __name__ == "__main__":
    patch_config()

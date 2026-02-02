import json
import os

# Path to the merged model configuration
MODEL_PATH = "/dataset/cep/llm-training/person-test/models/merged"
CONFIG_PATH = os.path.join(MODEL_PATH, "preprocessor_config.json")

def patch_resolution():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Current Setting
    old_pixels = config.get("max_pixels", "Not Set")
    
    # Target Setting: 512 * 512 = 262144
    NEW_PIXELS = 512 * 512 

    print(f"Current max_pixels: {old_pixels}")
    
    if old_pixels == NEW_PIXELS:
        print(f"Already set to {NEW_PIXELS} (512x512). No changes needed.")
        return

    # Update
    config["max_pixels"] = NEW_PIXELS
    
    # Save
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    patch_resolution()

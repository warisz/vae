from datasets import load_dataset
import os

DATASET_NAME = "maurice-fp/stanford-dogs" 
SAVE_DIR = "data/images"

def main():
    print(f"Downloading '{DATASET_NAME}' from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)
    
    print(f"Saving data locally to '{SAVE_DIR}'...")
    dataset.save_to_disk(SAVE_DIR)
    
    print(f"Download complete! Data saved to {SAVE_DIR}/")

if __name__ == "__main__":
    main()
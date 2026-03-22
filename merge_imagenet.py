import os
import shutil

# 1. Change this to wherever your extracted 17GB Kaggle dataset currently is
raw_dataset_path = "C:/Users/Sumer Singh/Desktop/assignment_2/data/ImageNet-100"

# 2. This matches what data_loader.py expects
target_dir = "C:/Users/Sumer Singh/Desktop/assignment_2/data/ImageNet100"

def merge_split(split_name):
    print(f"\nMerging {split_name} folders...")
    target_split_dir = os.path.join(target_dir, split_name)
    os.makedirs(target_split_dir, exist_ok=True)
    
    # Find all chunked folders (e.g., train.X1, train.X2 or val.X)
    chunks = [d for d in os.listdir(raw_dataset_path) if d.startswith(f"{split_name}.X")]
    
    for chunk in chunks:
        chunk_path = os.path.join(raw_dataset_path, chunk)
        print(f"  -> Moving contents of {chunk}...")
        
        # Loop through each class folder (the n0... folders)
        for class_name in os.listdir(chunk_path):
            source_class_path = os.path.join(chunk_path, class_name)
            target_class_path = os.path.join(target_split_dir, class_name)
            
            if os.path.isdir(source_class_path):
                # Ensure the class folder exists in our final target directory
                os.makedirs(target_class_path, exist_ok=True)
                
                # Move all images from the chunk into the unified class folder
                for img_file in os.listdir(source_class_path):
                    src_file = os.path.join(source_class_path, img_file)
                    dst_file = os.path.join(target_class_path, img_file)
                    
                    # Using shutil.move is instantly fast if on the same hard drive
                    shutil.move(src_file, dst_file)

if __name__ == "__main__":
    if not os.path.exists(raw_dataset_path):
        print(f"[ERROR] Could not find the raw dataset at: {raw_dataset_path}")
        print("Please update the 'raw_dataset_path' variable in this script.")
    else:
        merge_split("train")
        merge_split("val")
        print("\n Dataset successfully merged into ./data/imagenet100/")
        print("You can now safely delete the empty raw kaggle folders to save space!")
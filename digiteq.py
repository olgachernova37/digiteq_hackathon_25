import numpy as np
import pandas as pd
import os
from tqdm import tqdm
def find_emoji_in_image(main_image, emoji_img):
    """
    Robust emoji detection using multiple verification methods
    Args:
        main_image: The main image (numpy array)
        emoji_img: The emoji image (numpy array)
    Returns:
        tuple: (x, y) coordinates or None if not found
    """
    # Convert to grayscale
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    emoji_gray = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2GRAY)
    # Method 1: Template Matching with high threshold
    res = cv2.matchTemplate(main_gray, emoji_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.9:
        # Verify with pixel-wise comparison
        x, y = max_loc
        roi = main_gray[y:y+emoji_gray.shape[0], x:x+emoji_gray.shape[1]]
        # Ensure ROI matches emoji size
        if roi.shape == emoji_gray.shape:
            # Count exact pixel matches
            exact_matches = np.sum(roi == emoji_gray)
            match_ratio = exact_matches / (emoji_gray.size)
            if match_ratio > 0.85:  # 85% of pixels must match exactly
                return (x, y)
    return None
def test_emoji_detection(dataset_path, labels_csv_path):
    """Testing function with proper dimension handling"""
    labels_df = pd.read_csv(labels_csv_path, sep=';')
    # Load all emojis first with proper dimension handling
    emoji_data = []
    for _, row in labels_df.iterrows():
        emoji_path = os.path.join(dataset_path, row['file_name'])
        emoji = cv2.imread(emoji_path)
        if emoji is not None:
            h, w = emoji.shape[:2]  # Correct way to get dimensions
            x = int(row['x_s'].strip("[]"))
            y = int(row['y_s'].strip("[]"))
            emoji_data.append({
                'path': emoji_path,
                'x': x,
                'y': y,
                'emoji': emoji,
                'width': w,
                'height': h
            })
    if not emoji_data:
        print("No valid emoji images found!")
        return
    # Determine main image size
    max_x = max(item['x'] + item['width'] for item in emoji_data)
    max_y = max(item['y'] + item['height'] for item in emoji_data)
    main_image = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    correct = 0
    total = len(emoji_data)
    print(f"Testing {total} emojis with robust verification...")
    for item in tqdm(emoji_data, total=total):
        # Place emoji at correct location
        temp_img = main_image.copy()
        x_truth, y_truth = item['x'], item['y']
        h, w = item['height'], item['width']
        temp_img[y_truth:y_truth+h, x_truth:x_truth+w] = item['emoji']
        # Detect with verification
        coords = find_emoji_in_image(temp_img, item['emoji'])
        if coords:
            dx, dy = coords
            if abs(dx - x_truth) <= 1 and abs(dy - y_truth) <= 1:
                correct += 1
            else:
                print(f"Mismatch: {os.path.basename(item['path'])} detected at {coords}, should be ({x_truth}, {y_truth})")
        else:
            print(f"Not detected: {os.path.basename(item['path'])} at ({x_truth}, {y_truth})")
    accuracy = (correct / total) * 100
    print(f"\nFinal Results:")
    print(f"Correct detections: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
if __name__ == "__main__":
    DATASET_PATH = "./data/basic/dataset"
    LABELS_CSV_PATH = "./data/basic/labels.csv"
    # Verify paths
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path does not exist: {DATASET_PATH}")
        exit(1)
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"Labels CSV path does not exist: {LABELS_CSV_PATH}")
        exit(1)
    test_emoji_detection(DATASET_PATH, LABELS_CSV_PATH)
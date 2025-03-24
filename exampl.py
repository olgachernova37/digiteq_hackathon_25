import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
def find_emoji_position(main_image, emoji_img):
    """
    Finds emoji position using explicit sliding window approach
    that strictly follows all requirements
    """
    # Get dimensions
    img_h, img_w = main_image.shape[:2]
    emoji_h, emoji_w = emoji_img.shape[:2]
    # Convert to grayscale for simpler comparison
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    emoji_gray = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2GRAY)
    # Slide window across the image
    for y in range(img_h - emoji_h + 1):
        for x in range(img_w - emoji_w + 1):
            # Extract window
            window = main_gray[y:y+emoji_h, x:x+emoji_w]
            # Compare pixel values
            if window.shape == emoji_gray.shape:
                # Exact match check
                if np.array_equal(window, emoji_gray):
                    return (x, y)
    return None
def test_emoji_detection(dataset_path, labels_csv_path):
    """Testing function that follows all requirements exactly"""
    # Read labels
    labels_df = pd.read_csv(labels_csv_path, sep=';')
    # Load all emojis
    emoji_data = []
    for _, row in labels_df.iterrows():
        emoji_path = os.path.join(dataset_path, row['file_name'])
        emoji = cv2.imread(emoji_path)
        if emoji is not None:
            x = int(row['x_s'].strip("[]"))
            y = int(row['y_s'].strip("[]"))
            emoji_data.append({
                'path': emoji_path,
                'x': x, 'y': y,
                'emoji': emoji,
                'width': emoji.shape[1],
                'height': emoji.shape[0]
            })
    # Create main image
    max_x = max(item['x'] + item['width'] for item in emoji_data)
    max_y = max(item['y'] + item['height'] for item in emoji_data)
    main_image = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    correct = 0
    total = len(emoji_data)
    print(f"Testing {total} emojis with explicit sliding window...")
    for item in tqdm(emoji_data, total=total):
        # Place emoji in main image
        temp_img = main_image.copy()
        x_truth, y_truth = item['x'], item['y']
        h, w = item['height'], item['width']
        temp_img[y_truth:y_truth+h, x_truth:x_truth+w] = item['emoji']
        # Find using sliding window
        coords = find_emoji_position(temp_img, item['emoji'])
        if coords:
            dx, dy = coords
            if dx == x_truth and dy == y_truth:
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
    # handle for path
    test_emoji_detection(DATASET_PATH, LABELS_CSV_PATH)

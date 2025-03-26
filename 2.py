#!/usr/bin/python3

import cv2
import numpy as np
import os

# def ft_files_count(im_path)
#     file_count = 0
#     for item in os.listdir(im_path)
#     item_path = os.path.join(im_path, item)
#     if os.path.isfile(item_path):
#         file_count +=1

def ft_green_window_and_store(x, y):
    cv2.rectangle(display_img, (x, y), (x+window_size, y+window_size), (0, 255, 0), 2)
    print(f"Found emoji at (x={x}, y={y})")

def move_window_and_find(filepath, emoji_path):

    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    emoji = cv2.imread(emoji_path, cv2.IMREAD_COLOR)
    
    if image is None or emoji is None:
        print("Error: Could not load images")
        return None
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    emoji_gray = cv2.cvtColor(emoji, cv2.COLOR_RGB2GRAY)
    
    img_height, img_width = img_gray.shape
    emoji_height, emoji_width = emoji_gray.shape
    
    window_size = emoji_height
    
    
    for y in range(0, img_height - window_size + 1, window_size):
        for x in range(0, img_width - window_size + 1, window_size):
            window = img_gray[y:y+window_size, x:x+window_size]

            # print("i am working here")
            # display_img = image.copy()
            # cv2.rectangle(display_img, (x, y), (x+window_size, y+window_size), (0, 0, 255), 2)
            # print(f"we at (x={x}, y={y})")
            # cv2.imshow('Found Emoji', display_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if np.any(window < 250):
                for y2 in range(max(0, y-5), min(y+5, img_height - window_size + 1), 1):
                    for x2 in range(max(0, x-5), min(x+5, img_width - window_size +1), 1):
                        precise_window = img_gray[y2:y2+window_size, x2:x2+window_size]
                        # print("i am working here")
                        # print(f"Found wind = {precise_window},\n emoji={emoji_gray})")
                        # print(f"windov ={precise_window.astype(int)},\n  emoji ={emoji_gray.astype(int)})")
                        res = cv2.matchTemplate(precise_window, emoji_gray, cv2.TM_CCOEFF_NORMED)
                        # if res.max() > 0.1:  # Adjust threshold as needed
                        print("Match found with confidence:", res)
                            
                        
                        if (np.max(cv2.absdiff(precise_window, emoji_gray)) < 150):
                            print("i am working here")
                            display_img = image.copy()
                            cv2.rectangle(display_img, (x2, y2), (x2+window_size, y2+window_size), (0, 255, 0), 2)
                            print(f"Found emoji at (x={x2}, y={y2})")
                            cv2.imshow('Found Emoji', display_img)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            return (x2, y2)
    
    print("Emoji not found")
    return None

# If loop completes without finding
    print("Emoji not found")

                            # ft_green_window_and store(x, y)
                            # print(f"Found emoji at (x={x}, y={y})")
                    
        


def loop_in_dir(image_path, emoji):
    for filename in os.listdir(image_path):
        filepath = os.path.join(image_path, filename)
        if os.path.isfile(filepath):
            move_window_and_find(filepath, emoji)




if __name__ == "__main__":
    DATASET_PATH = "./data/basic/dataset"
    LABELS_CSV_PATH = "./data/basic/my_labels.csv"
    EMOJI = "./emoji_original.jpg"

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path does not exist: {DATASET_PATH}")
        exit(1)
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"Labels CSV path does not exist: {LABELS_CSV_PATH}")
        exit(1)
    if not os.path.exists(EMOJI):
        print(f"Labels CSV path does not exist: {LABELS_CSV_PATH}")
        exit(1)
    loop_in_dir(DATASET_PATH, EMOJI)


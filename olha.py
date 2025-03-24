#!/usr/bin/python3

import cv2
import numpy as np






#     MY_LABELS_CSV_PATH = "./data/basic/labels.csv" # where i will write coord.

#         if not os.path.exists(DATASET_PATH):
#         print(f"Dataset path does not exist: {DATASET_PATH}")
#         exit(1)
#         if not os.path.exists(EMOJI_ORIG_PATH):
#         print(f"Labels CSV path does not exist: {EMOJI_ORIG_PATH}")
#         exit(1)
#         if not os.path.exists(MY_LABELS_CSV_PATH):
#         print(f"Labels CSV path does not exist: {MY_LABELS_CSV_PATH}")
#         exit(1)
#    # test_emoji_detection(DATASET_PATH, LABELS_CSV_PATH)
#         else


# Завантаження зображень
# image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# emoji = cv2.imread(EMOJI_ORIG_PATH, cv2.IMREAD_COLOR)
EMOJI_ORIG_PATH = "emoji_0.jpg"
IMAGE_PATH = "emoji_original.jpg"
# image_path = "./data/basic/dataset/emoji_0.jpg"
# emoji_path = "path_to_emoji.png"  # Replace with your emoji path
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
emoji = cv2.imread(EMOJI_ORIG_PATH, cv2.IMREAD_COLOR)

# if image is None or emoji is None:
#     print("Error: Could not load images")
#     return None

img_height, img_width = image.shape[:2]   #understand the height and width
emoji_height, emoji_width = emoji.shape[:2]
    
    # Slide window across the image
for y in range(0, img_height - emoji_height + 1):
    for x in range(0, img_width - emoji_width + 1):
        # Extract the current window
        window = image[y:y+emoji_height, x:x+emoji_width]
        
        # Compare the window with the emoji
        if np.array_equal(window, emoji):
            print(f"Emoji found at coordinates: (x={x}, y={y})")
            # return (x, y)


print("Emoji not found in the image")




# DATASET_PATH = "./data/basic/dataset"
# EMOJI_ORIG_PATH = "emoji_original.jpg"
# IMAGE_PATH = "emoji_0.jpg"
# image_path = "./data/basic/dataset/emoji_0.jpg"
# emoji_path = "path_to_emoji.png"  # Replace with your emoji path




















# # Отримання пікселя у певній координаті (наприклад, 50,50)
# pixel = image[50, 50]  # Це список [B, G, R]
# print(f"Pixel at (50,50): {pixel}")

# # Всі пікселі зображення збережені у `image` як NumPy масив

# image = cv2.imread("./data/basic/dataset/emoji_0.jpg")

# # Display the image
# cv2.imshow("Image", image)

# # Wait for the user to press a key
# cv2.waitKey(0)

# # Close all windows
# cv2.destroyAllWindows()
#!/usr/bin/python3

import cv2
import numpy as np

EMOJI_H = 50 #EMOJI
EMOJI_W = 50 #EMOJI

def find_emoji(image_path, emoji_path, success=0.8):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    emoji = cv2.imread(emoji_path, cv2.IMREAD_COLOR)
    
    if image is None or emoji is None:
        print("Error: Could not load images")
        return None
    image_h, image_w = image.shape[:2]
    emoji_h, emoji_w = image.shape[:2]
    

    res = cv2.matchTemplate(image, emoji, cv2.TM_CCOEFF_NORMED) # matching
    
    # Find locations where correlation exceeds threshold
    loc = np.where(res >= success)
    confidence = res[loc]
    print(confidence)
    if len(loc[0]) > 0:
        # Get all matches (there might be multiple)
        matches = list(zip(*loc[::-1]))  # Swap x,y coordinates
        
        # For this example, just return the first match
        x, y = matches[0]
        print(f"Found emoji at (x={x}, y={y})")
        
        # Visualization (optional)
        cv2.rectangle(image, (x, y), (x+EMOJI_W, y+EMOJI_H), (0,255,0), 2)
        cv2.imshow('Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return (x, y)
    else:
        print("Emoji not found")
        return None




find_emoji("emoji_0.jpg", "emoji_original.jpg", success=0.9)

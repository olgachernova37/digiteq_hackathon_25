#!/usr/bin/python3

import cv2
import numpy as np
import time
import os
import csv
import pandas as pd
import signal
import sys

EMOJI_H = 50 
EMOJI_W = 50  
IMAGE_H = 600 
IMAGE_W = 800 

# Global variable to track if program is interrupted
interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt gracefully"""
    global interrupted
    print("\nProgram interrupted. Saving results and exiting...")
    interrupted = True
    # Allow loop to continue to next iteration to save current results

# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def linear_scan_for_emoji(image, emoji, success_threshold=0.9, display=True):
    """Find emoji using a linear scan with visualization"""
    global interrupted
    if display:
        cv2.namedWindow('Emoji Scan', cv2.WINDOW_NORMAL)
    
    res = cv2.matchTemplate(image, emoji, cv2.TM_CCOEFF_NORMED)
    
    step_size = 50
    
    y = 0
    while y <= IMAGE_H - EMOJI_H and not interrupted:
        x = 0
        row_has_content = False 
        
        while x <= IMAGE_W - EMOJI_W and not interrupted:
            current_corr = res[y, x] if y < res.shape[0] and x < res.shape[1] else 0
            
            region = image[y:y+EMOJI_H, x:x+EMOJI_W]
            has_content = not np.all(region > 240)
            
            if has_content:
                row_has_content = True
            
            if has_content:
                current_step = 1
            else:
                current_step = step_size
            
            if display:
                vis_img = image.copy()
                
                cv2.rectangle(vis_img, (x, y), (x + EMOJI_W, y + EMOJI_H), (0, 0, 255), 2)
                
                cv2.putText(vis_img, f"Scanning: ({x}, {y})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(vis_img, f"Correlation: {current_corr:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis_img, f"Step size X: {current_step}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis_img, f"Content detected: {'Yes' if has_content else 'No'}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis_img, f"Row has content: {'Yes' if row_has_content else 'No'}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Emoji Scan', vis_img)
                
                key = cv2.waitKey(1)
                
                if key == 27:  
                    cv2.destroyAllWindows()
                    return None
            
            if current_corr >= success_threshold:
                print(f"Found emoji at ({x}, {y}) with confidence {current_corr:.2f}")
                
                if display:
                    final_img = image.copy()
                    cv2.rectangle(final_img, (x, y), (x + EMOJI_W, y + EMOJI_H), (0, 255, 0), 3)
                    cv2.putText(final_img, f"Found! ({x}, {y})", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(final_img, f"Correlation: {current_corr:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Emoji Scan', final_img)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                
                return (x, y)
            
            x += current_step
        
        if row_has_content:
            y += 1 
            next_step = 1
        else:
            y += step_size  
            next_step = step_size
            
        if display and not interrupted:
            vis_img = image.copy()
            cv2.putText(vis_img, f"Moving to next row...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis_img, f"Vertical step: {next_step}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow('Emoji Scan', vis_img)
            cv2.waitKey(50) 
    
    if not interrupted:
        print("Emoji not found")
    if display:
        cv2.destroyAllWindows()
    return None

def find_emoji(image_path, emoji_path, success=0.9, display=True):
    """Main function to find emoji in image"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    emoji = cv2.imread(emoji_path, cv2.IMREAD_COLOR)
    
    if image is None or emoji is None:
        print("Error: Could not load images")
        return None
    
    global IMAGE_H, IMAGE_W
    IMAGE_H, IMAGE_W = image.shape[:2]
    
    return linear_scan_for_emoji(image, emoji, success_threshold=success, display=display)

def move_window_and_find(image_path, emoji_path, success=0.9, display=True):
    """Process a single image and find emoji"""
    print(f"Processing {image_path}...")
    coords = find_emoji(image_path, emoji_path, success, display)
    
    base_filename = os.path.basename(image_path)
    
    if coords:
        x, y = coords
        print(f"File: {base_filename}, Emoji found at ({x}, {y})")
        return base_filename, x, y
    else:
        print(f"File: {base_filename}, Emoji not found")
        return base_filename, -1, -1

def update_csv_with_result(result, csv_path):
    """Update the CSV file with a single result"""
    filename, x, y = result
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)
    
    # Get the next index
    next_index = 0
    if file_exists:
        try:
            df = pd.read_csv(csv_path, sep=';')
            next_index = len(df)
        except:
            # If file exists but can't be read, assume it's empty
            file_exists = False
    
    # Create a single-row DataFrame
    new_row = pd.DataFrame({
        'file_name': [filename],
        'moods': [['happy']],
        'x_s': [[x]],
        'y_s': [[y]]
    }, index=[next_index])
    
    # Write to CSV
    if not file_exists:
        # Create new file with header
        new_row.to_csv(csv_path, sep=';', index=True)
    else:
        # Append without header
        new_row.to_csv(csv_path, sep=';', index=True, mode='a', header=False)
    
    print(f"Result for {filename} saved to {csv_path}")

def loop_in_dir(image_path, emoji_path, csv_path, display=True):
    """Process all images in directory and save results to CSV immediately after each image"""
    global interrupted
    
    emoji = cv2.imread(emoji_path, cv2.IMREAD_COLOR)
    
    if emoji is None:
        print(f"Error: Could not load emoji from {emoji_path}")
        return
    
    # Create or clear the CSV file
    # Create an empty DataFrame with the required structure
    empty_df = pd.DataFrame(columns=['file_name', 'moods', 'x_s', 'y_s'])
    empty_df.to_csv(csv_path, sep=';', index=True)
    
    try:
        for filename in sorted(os.listdir(image_path)):
            if interrupted:
                break
                
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(image_path, filename)
                if os.path.isfile(filepath):
                    # Process image
                    result = move_window_and_find(filepath, emoji_path, display=display)
                    if result:
                        # Immediately update CSV with this result
                        update_csv_with_result(result, csv_path)
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        print("Processing complete or interrupted. Results saved to CSV.")
        if interrupted:
            sys.exit(0)

if __name__ == "__main__":
    DATASET_PATH = "./data/basic/dataset2"
    LABELS_CSV_PATH = "./data/basic/my_labels.csv"
    EMOJI = "./emoji_original.jpg"

    os.makedirs(os.path.dirname(LABELS_CSV_PATH), exist_ok=True)
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path does not exist: {DATASET_PATH}")
        exit(1)
    if not os.path.exists(EMOJI):
        print(f"Emoji file does not exist: {EMOJI}")
        exit(1)
    
    print("Starting emoji detection. Press Ctrl+C to interrupt and save progress.")
    loop_in_dir(DATASET_PATH, EMOJI, LABELS_CSV_PATH, display=True)
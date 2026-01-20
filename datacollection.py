import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters - MUST MATCH YOUR MODEL (224x224)
offset = 20
imgSize = 224
counter = 0

# Folder setup - CHANGE THIS for different gestures
folder = "Data/Hello"

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Created folder: {folder}")

print("="*50)
print("SIGN LANGUAGE DATA COLLECTION")
print("="*50)
print(f"Image Size: {imgSize}x{imgSize}")
print(f"Saving to: {folder}")
print("Press 'S' to save image")
print("Press 'Q' to quit")
print("="*50)

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image from camera")
        break
    
    # Flip image for mirror effect (more intuitive)
    img = cv2.flip(img, 1)
    
    # Detect hands
    hands, img = detector.findHands(img, draw=True, flipType=False)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Handle boundary issues
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
        # Check if crop is valid
        if imgCrop.size == 0:
            cv2.imshow('Image', img)
            continue
        
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        
        try:
            if aspectRatio > 1:
                # Height is greater
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > imgSize:
                    wCal = imgSize
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                
            else:
                # Width is greater
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > imgSize:
                    hCal = imgSize
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # Display windows
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    # Add counter display on main image
    cv2.putText(img, f'Images Saved: {counter}', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Size: {imgSize}x{imgSize}', (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, 'Press S to Save | Q to Quit', (10, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Image', img)
    
    key = cv2.waitKey(1)
    
    # Save image when 's' is pressed
    if key == ord("s") or key == ord("S"):
        if hands:
            counter += 1
            filename = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"✓ Saved: Image_{counter} (Total: {counter})")
        else:
            print("✗ No hand detected! Cannot save.")
    
    # Quit when 'q' is pressed
    elif key == ord("q") or key == ord("Q"):
        print(f"\n{'='*50}")
        print(f"Session Complete!")
        print(f"Total images saved: {counter}")
        print(f"Location: {folder}")
        print(f"{'='*50}")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
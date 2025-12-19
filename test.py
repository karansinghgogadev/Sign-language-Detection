import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Try different import methods
try:
    from tensorflow.keras.models import load_model
except:
    try:
        from keras.models import load_model
    except:
        print("ERROR: TensorFlow/Keras not installed!")
        print("Run: pip install tensorflow")
        exit()

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters (MUST match your training!)
offset = 20
imgSize = 224  # Changed from 300 to 224 to match your model

# Load your trained model
model = load_model("Model/keras_model.h5")

# Load labels from file
labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        labels.append(line.strip().split()[1])  # Extract label name

print(f"Loaded {len(labels)} gestures: {labels}")

print("="*50)
print("SIGN LANGUAGE DETECTION - READY")
print("="*50)
print("Press 'Q' to quit")
print("="*50)

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image")
        break
    
    # Flip for mirror effect
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    
    # Detect hands
    hands, img = detector.findHands(img)
    
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
            cv2.imshow('Image', imgOutput)
            continue
        
        aspectRatio = h / w
        
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # Prepare image for prediction
            imgWhite = cv2.resize(imgWhite, (224, 224))  # Changed to 224x224
            imgWhite = np.array(imgWhite, dtype='float32')
            imgWhite = imgWhite / 255.0  # Normalize
            imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(imgWhite, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            # Get predicted label
            predicted_label = labels[index]
            
            # Display results on image
            # Draw bounding box
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                         (x - offset + 400, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, predicted_label, (x, y - 30), 
                       cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.putText(imgOutput, f'{confidence*100:.1f}%', (x, y - 5), 
                       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Draw hand bounding box
            cv2.rectangle(imgOutput, (x - offset, y - offset), 
                         (x + w + offset, y + h + offset), (0, 255, 0), 4)
            
            # Show processed image
            cv2.imshow('ImageWhite', cv2.resize(imgWhite[0], (224, 224)))
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
    
    else:
        # No hand detected
        cv2.putText(imgOutput, 'No Hand Detected', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display main image
    cv2.imshow('Image', imgOutput)
    
    key = cv2.waitKey(1)
    if key == ord("q") or key == ord("Q"):
        print("\nExiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
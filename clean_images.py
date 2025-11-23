import os
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOWNLOAD_DIR = r"d:\images_extractor"

# Load MobileNetSSD model
net = None
try:
    proto_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    if os.path.exists(proto_path) and os.path.exists(model_path):
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        logging.info("MobileNetSSD model loaded successfully.")
    else:
        logging.warning("MobileNetSSD model files not found. Car detection will be skipped.")
except Exception as e:
    logging.error(f"Failed to load MobileNetSSD model: {e}")

def dhash(image, hash_size=8):
    try:
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = gray[:, 1:] > gray[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except:
        return 0

def is_car(image):
    if net is None: return True # Skip check if model not loaded
    try:
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx == 7: # Class 7 is 'car' in VOC
                    return True
        return False
    except:
        return True # Assume valid on error to be safe

# Load EAST text detector
net_east = None
try:
    east_path = "frozen_east_text_detection.pb"
    if os.path.exists(east_path):
        net_east = cv2.dnn.readNet(east_path)
        logging.info("EAST text detector loaded successfully.")
    else:
        logging.warning("EAST model not found. Watermark removal will be skipped.")
except Exception as e:
    logging.error(f"Failed to load EAST model: {e}")

def process_image(image):
    if net_east is None: return image, False
    
    H, W = image.shape[:2]
    target_H = int(W / 2)
    
    if H <= target_H:
        # Image is already wider than 2:1 or exact. 
        # If we want to strictly enforce 2:1, we might need to crop width, but let's assume we only crop height to avoid cutting cars.
        # Or we can just return if it's close.
        return image, False
        
    pixels_to_remove = H - target_H
    
    # Detect text to guide cropping
    orig = image.copy()
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net_east.setInput(blob)
    (scores, geometry) = net_east.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            if scoresData[x] < 0.5: continue
            
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
    
    suggested_bottom = 0
    suggested_top = 0
    
    if len(boxes) > 0:
        for i in boxes.flatten():
            (startX, startY, endX, endY) = rects[i]
            startY = int(startY * rH)
            endY = int(endY * rH)
            
            if startY > H * 0.7: # Text at bottom
                suggested_bottom = max(suggested_bottom, H - startY)
            elif endY < H * 0.3: # Text at top
                suggested_top = max(suggested_top, endY)
    
    # Calculate actual crop
    crop_top = 0
    crop_bottom = 0
    
    if suggested_bottom > suggested_top:
        # Bias towards bottom
        crop_bottom = min(pixels_to_remove, suggested_bottom + 10) # Add padding
        crop_top = pixels_to_remove - crop_bottom
    elif suggested_top > suggested_bottom:
        # Bias towards top
        crop_top = min(pixels_to_remove, suggested_top + 10)
        crop_bottom = pixels_to_remove - crop_top
    else:
        # Center crop
        crop_top = pixels_to_remove // 2
        crop_bottom = pixels_to_remove - crop_top
        
    # Final safety check
    if crop_top < 0: crop_top = 0
    if crop_bottom < 0: crop_bottom = 0
    
    # Perform crop
    cropped = image[crop_top:H-crop_bottom, :]
    return cropped, True

def clean_images():
    logging.info(f"Scanning {DOWNLOAD_DIR} for cleanup and 2:1 cropping...")
    seen_hashes = set()
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    removed_count = 0
    cropped_count = 0
    
    for filename in files:
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                logging.warning(f"Could not read {filename}. Deleting.")
                os.remove(file_path)
                removed_count += 1
                continue

            # Car Detection
            if not is_car(img):
                logging.info(f"Removing {filename}: Not a car")
                os.remove(file_path)
                removed_count += 1
                continue

            # Similarity Check
            img_hash = dhash(img)
            if img_hash in seen_hashes:
                logging.info(f"Removing {filename}: Duplicate image")
                os.remove(file_path)
                removed_count += 1
                continue
            
            seen_hashes.add(img_hash)
            
            # 2:1 Cropping & Watermark Removal
            cleaned_img, cropped = process_image(img)
            if cropped:
                logging.info(f"Cropped {filename} to 2:1")
                cv2.imwrite(file_path, cleaned_img)
                cropped_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    logging.info(f"Cleanup complete. Removed {removed_count} files. Cropped {cropped_count} files to 2:1.")

if __name__ == "__main__":
    clean_images()

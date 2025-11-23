import os
import shutil
import logging
import cv2
import numpy as np
import json
import html
import re
from datetime import datetime
from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler
from icrawler.builtin.bing import BingParser
from clean_images import is_car, dhash
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama library not installed. Falling back to manual filename sanitization.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DOWNLOAD_DIR = r"d:\images_extractor"
TEMP_DIR = os.path.join(DOWNLOAD_DIR, "temp")
MIN_WIDTH = 1000
MIN_FILE_SIZE_KB = 50
BACKGROUND_VARIANCE_THRESHOLD = 500

def setup_directories():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def sanitize_filename(name):
    # Remove invalid characters and excessive length
    clean = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
    return clean[:100] # Limit length

def generate_filename_with_ollama(original_filename, max_retries=2):
    """
    Use Ollama's LLaMA 3.1 8B model to generate a cleaned filename.
    
    Args:
        original_filename: The original filename/title from the image metadata
        max_retries: Number of retry attempts if API call fails
    
    Returns:
        A cleaned filename string suitable for filesystem use
    """
    if not OLLAMA_AVAILABLE:
        logging.debug(f"Ollama not available, using manual sanitization for: {original_filename}")
        return sanitize_filename(original_filename)
    
    if not original_filename or original_filename.strip() == "":
        return "Unknown_Car"
    
    # Construct the prompt
    prompt = f"""Suggest a title for this image removing any helper and unnecessary texts, remove the owner name and give the cleaned file name.

Original filename: {original_filename}

Provide ONLY the cleaned filename without any extension, explanations, or additional text. The filename should be concise, descriptive, and filesystem-safe (use only letters, numbers, spaces, hyphens, and underscores)."""
    
    for attempt in range(max_retries):
        try:
            response = ollama.generate(
                model='llama3.1:8b',
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent output
                    'num_predict': 50,   # Limit response length
                }
            )
            
            # Extract the response text
            cleaned_name = response.get('response', '').strip()
            
            # Remove any quotes, periods, or other unwanted characters that might be in the response
            cleaned_name = cleaned_name.strip('"\'.,')
            
            # Additional sanitization to ensure filesystem compatibility
            cleaned_name = sanitize_filename(cleaned_name)
            
            if cleaned_name:
                logging.info(f"AI-generated filename: '{original_filename}' -> '{cleaned_name}'")
                return cleaned_name
            else:
                logging.warning(f"Empty response from Ollama for: {original_filename}")
                
        except Exception as e:
            logging.warning(f"Ollama API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
    
    # Fallback to manual sanitization if all retries fail
    logging.info(f"Falling back to manual sanitization for: {original_filename}")
    return sanitize_filename(original_filename)

def get_dominant_color(image_path, k=3):
    try:
        img = cv2.imread(image_path)
        if img is None: return "Unknown"
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to a list of pixels
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # K-Means Clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Count labels to find the most common cluster
        _, counts = np.unique(labels, return_counts=True)
        dominant_index = np.argmax(counts)
        dominant_color = centers[dominant_index]
        
        # Map RGB to Color Name
        r, g, b = dominant_color
        
        colors = {
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
            "Gray": (128, 128, 128),
            "Orange": (255, 165, 0),
            "Purple": (128, 0, 128),
            "Brown": (165, 42, 42),
            "Pink": (255, 192, 203)
        }
        
        min_dist = float("inf")
        closest_color = "Unknown"
        
        for name, (cr, cg, cb) in colors.items():
            dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            if dist < min_dist:
                min_dist = dist
                closest_color = name
                
        return closest_color
    except Exception as e:
        logging.error(f"Error detecting color: {e}")
        return "Unknown"

def is_valid_image(image_path, min_width=MIN_WIDTH, min_file_size_kb=MIN_FILE_SIZE_KB, variance_threshold=BACKGROUND_VARIANCE_THRESHOLD):
    try:
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb < min_file_size_kb:
            logging.info(f"Rejected {os.path.basename(image_path)}: Size {file_size_kb:.1f}KB < {min_file_size_kb}KB")
            return False

        img = cv2.imread(image_path)
        if img is None:
            logging.info(f"Rejected {os.path.basename(image_path)}: cv2.imread failed")
            return False

        height, width = img.shape[:2]
        if width < min_width:
            logging.info(f"Rejected {os.path.basename(image_path)}: Width {width}px < {min_width}px")
            return False

        # Skip variance check if threshold is negative
        if variance_threshold < 0:
            return True

        h_border = int(height * 0.1)
        w_border = int(width * 0.1)
        regions = [img[0:h_border, :], img[height-h_border:, :], img[:, 0:w_border], img[:, width-w_border:]]
        
        variances = []
        for region in regions:
            if region.size == 0: continue
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            variances.append(np.var(gray))
        
        avg_variance = np.mean(variances)
        if avg_variance > variance_threshold:
            logging.info(f"Rejected {os.path.basename(image_path)}: Background variance {avg_variance:.1f} > {variance_threshold}")
            return False

        return True
    except Exception as e:
        logging.error(f"Error validating {os.path.basename(image_path)}: {e}")
        return False

class CustomBingParser(BingParser):
    def parse(self, response):
        # Override to extract title ('t') from 'm' attribute
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content.decode("utf-8", "ignore"), "lxml")
        image_divs = soup.find_all("div", class_="imgpt")
        
        for div in image_divs:
            try:
                if div.a and "m" in div.a.attrs:
                    m_attr = html.unescape(div.a["m"])
                    meta = json.loads(m_attr)
                    img_url = meta.get("murl")
                    title = meta.get("t")
                    
                    if img_url:
                        yield dict(file_url=img_url, meta_title=title)
            except Exception as e:
                continue

class CustomImageDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        # We save with a temporary name first, renaming happens in the main loop
        # But we can try to incorporate the title here if we wanted.
        # For now, let's stick to default hashing to avoid filesystem issues, 
        # and pass the title through to the post-processing.
        # Wait, icrawler doesn't easily pass metadata to the saved file unless we write a metadata file.
        # Strategy: Save the title in a sidecar map or just use the hash and rely on the main loop?
        # The main loop iterates over files in TEMP_DIR. It doesn't know which file corresponds to which title.
        
        # Better approach: Rename HERE in the downloader if possible, or save a mapping.
        # I'll save a mapping file.
        
        filename = super().get_filename(task, default_ext)
        
        # Save metadata mapping
        meta_path = os.path.join(self.storage.root_dir, filename + ".meta")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(task.get("meta_title", ""))
            
        return filename

class CustomBingCrawler(BingImageCrawler):
    pass

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
        crop_bottom = min(pixels_to_remove, suggested_bottom + 10)
        crop_top = pixels_to_remove - crop_bottom
    elif suggested_top > suggested_bottom:
        crop_top = min(pixels_to_remove, suggested_top + 10)
        crop_bottom = pixels_to_remove - crop_top
    else:
        crop_top = pixels_to_remove // 2
        crop_bottom = pixels_to_remove - crop_top
        
    if crop_top < 0: crop_top = 0
    if crop_bottom < 0: crop_bottom = 0
    
    cropped = image[crop_top:H-crop_bottom, :]
    return cropped, True

class LogQueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

def download_images(queries, min_width=MIN_WIDTH, min_file_size_kb=MIN_FILE_SIZE_KB, search_suffix="car studio background", validate_car=True, variance_threshold=BACKGROUND_VARIANCE_THRESHOLD, stop_event=None, log_queue=None):
    # Setup logging for web app if queue provided
    if log_queue:
        queue_handler = LogQueueHandler(log_queue)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(queue_handler)

    setup_directories()
    
    # Create a unique subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DOWNLOAD_DIR, f"run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    logging.info(f"Saving images to: {run_dir}")
    
    # Load existing hashes to avoid duplicates across runs
    seen_hashes = set()
    
    for query in queries:
        if stop_event and stop_event.is_set():
            logging.info("Download stopped by user.")
            break

        logging.info(f"Searching for: {query}")
        
        crawler = CustomBingCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            downloader_cls=CustomImageDownloader,
            parser_cls=CustomBingParser,
            storage={'root_dir': TEMP_DIR}
        )
        
        keyword = f"{query} {search_suffix}".strip()
        crawler.crawl(keyword=keyword, max_num=1000, filters=dict(size='large'))
        
        # Clean up crawler threads
        try:
            if hasattr(crawler, 'feeder_cls'):
                crawler.signal.set(crawler.signal.SignalType.STOP)
        except:
            pass
        
        # Process downloaded images
        for filename in os.listdir(TEMP_DIR):
            if stop_event and stop_event.is_set():
                logging.info("Processing stopped by user.")
                break

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            temp_path = os.path.join(TEMP_DIR, filename)
            meta_path = temp_path + ".meta"
            
            # Read title from meta file
            # Use query as fallback title instead of "Unknown_Car"
            title = f"Unknown_{query}"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    title = f.read().strip()
                os.remove(meta_path) # Cleanup
            
            if is_valid_image(temp_path, min_width, min_file_size_kb, variance_threshold):
                # Advanced Validation
                img = cv2.imread(temp_path)
                
                # Car Detection
                if validate_car and not is_car(img):
                    logging.info(f"Skipping {filename}: Not a car")
                    os.remove(temp_path)
                    continue
                
                # Similarity Check
                img_hash = dhash(img)
                if img_hash in seen_hashes:
                    logging.info(f"Skipping {filename}: Duplicate image")
                    os.remove(temp_path)
                    continue
                seen_hashes.add(img_hash)

                # 2:1 Cropping & Watermark Removal
                cleaned_img, cropped = process_image(img)
                if cropped:
                    logging.info(f"Cropped {filename} to 2:1")
                    cv2.imwrite(temp_path, cleaned_img)

                detected_color = get_dominant_color(temp_path)
                
                # Construct new filename from title using AI
                clean_title = generate_filename_with_ollama(title)
                
                # Remove common junk words as additional cleanup
                clean_title = clean_title.replace("Wallpapers", "").replace("Background", "").replace("Images", "").strip()
                
                # If the cleaned title is empty after removing junk words, use a fallback
                if not clean_title:
                    clean_title = sanitize_filename(title)
                
                new_filename = f"{clean_title}_{detected_color}.jpg"
                dest_path = os.path.join(run_dir, new_filename)
                
                # Handle duplicates
                counter = 0
                while os.path.exists(dest_path):
                    counter += 1
                    dest_path = os.path.join(run_dir, f"{clean_title}_{detected_color}_{counter}.jpg")

                shutil.move(temp_path, dest_path)
                logging.info(f"Saved: {dest_path}")
            else:
                os.remove(temp_path)

if __name__ == "__main__":
    print("--- Car Image Downloader ---")
    
    # Get search queries
    user_query = input("Enter car names to search (comma-separated), or press Enter for default list: ").strip()
    if user_query:
        queries = [q.strip() for q in user_query.split(",") if q.strip()]
    else:
        # Comprehensive list of popular car queries
        queries = [
            # Hypercars & Supercars
            "Bugatti Chiron", "Bugatti Veyron", "Koenigsegg Jesko", "Koenigsegg Agera RS",
            "Pagani Huayra", "Pagani Zonda", "McLaren P1", "McLaren 720S", "McLaren Senna",
            "Ferrari LaFerrari", "Ferrari SF90 Stradale", "Ferrari 488 Pista", "Ferrari F40",
            "Lamborghini Aventador SVJ", "Lamborghini Huracan Performante", "Lamborghini Revuelto",
            "Porsche 918 Spyder", "Porsche Carrera GT", "Aston Martin Valkyrie",
            
            # Sports Cars
            "Porsche 911 GT3 RS", "Porsche 911 Turbo S", "Chevrolet Corvette Z06",
            "Nissan GT-R Nismo", "Audi R8 V10", "Mercedes-AMG GT Black Series",
            "BMW M4 Competition", "BMW M3 Touring", "Toyota Supra MK5", "Lotus Emira",
            
            # JDM Legends
            "Nissan Skyline GT-R R34", "Toyota Supra MK4", "Mazda RX-7 FD",
            "Honda NSX Type R", "Mitsubishi Lancer Evolution IX", "Subaru Impreza WRX STI 22B",
            
            # Muscle Cars
            "Ford Mustang Shelby GT500", "Chevrolet Camaro ZL1", "Dodge Challenger SRT Demon",
            "Dodge Charger Hellcat", "Shelby Cobra 427",
            
            # Luxury & SUVs
            "Rolls-Royce Phantom", "Rolls-Royce Cullinan", "Bentley Continental GT",
            "Bentley Bentayga", "Mercedes-Maybach S-Class", "Mercedes-Benz G-Class G63 AMG",
            "Lamborghini Urus", "Range Rover SVAutobiography", "Cadillac Escalade V",
            
            # Electric & Concepts
            "Rimac Nevera", "Tesla Roadster 2.0", "Porsche Taycan Turbo S",
            "Lucid Air Sapphire", "Pininfarina Battista"
        ]

    # Get image quality parameters
    try:
        width_input = input(f"Enter minimum image width (pixels) [default {MIN_WIDTH}]: ").strip()
        min_width = int(width_input) if width_input else MIN_WIDTH
        
        size_input = input(f"Enter minimum file size (KB) [default {MIN_FILE_SIZE_KB}]: ").strip()
        min_size = int(size_input) if size_input else MIN_FILE_SIZE_KB
    except ValueError:
        print("Invalid input. Using defaults.")
        min_width = MIN_WIDTH
        min_size = MIN_FILE_SIZE_KB

    # Get search suffix
    search_suffix = input("Enter search suffix (e.g. 'car studio background') [default 'car studio background']: ").strip()
    if not search_suffix:
        search_suffix = "car studio background"
        
    # Get validation preference
    validate_input = input("Enable car validation (y/n)? [default y]: ").strip().lower()
    validate_car = validate_input != 'n'

    # Get variance threshold
    try:
        var_input = input(f"Enter background variance threshold (lower=stricter, -1 to disable) [default {BACKGROUND_VARIANCE_THRESHOLD}]: ").strip()
        variance_threshold = int(var_input) if var_input else BACKGROUND_VARIANCE_THRESHOLD
    except ValueError:
        print("Invalid input. Using default.")
        variance_threshold = BACKGROUND_VARIANCE_THRESHOLD

    print(f"\nStarting download with:")
    print(f"- Queries: {len(queries)} items")
    print(f"- Min Width: {min_width}px")
    print(f"- Min Size: {min_size}KB")
    print(f"- Suffix: '{search_suffix}'")
    print(f"- Validate Car: {validate_car}")
    print(f"- Variance Threshold: {variance_threshold}")
    print("-" * 30)
    
    download_images(queries, min_width, min_size, search_suffix, validate_car, variance_threshold)

import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    files = {
        "MobileNetSSD_deploy.prototxt": [
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/923b3128f25262b5010cef67e4fb9e4b6728ae7b/MobileNetSSD_deploy.prototxt",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/923b3128f25262b5010cef67e4fb9e4b6728ae7b/voc/MobileNetSSD_deploy.prototxt"
        ],
        "MobileNetSSD_deploy.caffemodel": [
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
            "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel",
            "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffe/MobileNetSSD_deploy.caffemodel",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.caffemodel" 
        ],
        "frozen_east_text_detection.pb": [
            "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb",
            "https://github.com/oyyd/frozen_east_text_detection.pb/blob/master/frozen_east_text_detection.pb?raw=true"
        ]
    }
    
    for filename, urls in files.items():
        if os.path.exists(filename):
            print(f"{filename} already exists.")
            continue
            
        success = False
        for url in urls:
            print(f"Trying to download {filename} from {url}...")
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded {filename}")
                    success = True
                    break
                else:
                    print(f"Failed with status {response.status_code}")
            except Exception as e:
                print(f"Failed with error: {e}")
        
        if not success:
            print(f"Could not download {filename} from any source.")

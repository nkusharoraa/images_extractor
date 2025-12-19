import os
import requests
import logging
import time
import urllib.parse


class WallHavenDownloader:
    """
    Downloader for WallHaven (wallhaven.cc) - a popular wallpaper site with API support
    Note: WallHaven has a free API that doesn't require authentication for basic searches
    """
    
    def __init__(self, temp_dir, delay=1.0, api_key=None):
        """
        Initialize WallHaven downloader
        
        Args:
            temp_dir: Directory to save downloaded images
            delay: Delay between requests in seconds (default 1.0)
            api_key: Optional WallHaven API key for higher rate limits
        """
        self.temp_dir = temp_dir
        self.delay = delay
        self.api_key = api_key
        self.api_base = "https://wallhaven.cc/api/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    def search_and_download(self, query, max_images=100, min_resolution="1920x1080", categories="111"):
        """
        Search and download images from WallHaven
        
        Args:
            query: Search query (e.g., "Ferrari 488")
            max_images: Maximum number of images to download
            min_resolution: Minimum resolution (e.g., "1920x1080")
            categories: Categories to search (1=General, 1=Anime, 1=People, default="111" for all)
        
        Returns:
            List of tuples (file_path, metadata_dict)
        """
        downloaded_files = []
        
        logging.info(f"Searching WallHaven for: {query}")
        
        try:
            page = 1
            images_downloaded = 0
            
            while images_downloaded < max_images:
                # Build API request
                params = {
                    'q': query,
                    'page': page,
                    'categories': categories,
                    'purity': '100',  # 1=SFW, 0=Sketchy, 0=NSFW
                    'sorting': 'relevance',
                    'atleast': min_resolution
                }
                
                url = f"{self.api_base}/search"
                logging.info(f"Fetching page {page}: {url}")
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    logging.warning(f"Failed to fetch page {page}: HTTP {response.status_code}")
                    break
                
                data = response.json()
                wallpapers = data.get('data', [])
                
                if not wallpapers:
                    logging.info(f"No more images found on page {page}")
                    break
                
                logging.info(f"Found {len(wallpapers)} images on page {page}")
                
                # Download each image
                for wallpaper in wallpapers:
                    if images_downloaded >= max_images:
                        break
                    
                    img_url = wallpaper.get('path')
                    wallpaper_id = wallpaper.get('id', 'unknown')
                    resolution = f"{wallpaper.get('dimension_x', 0)}x{wallpaper.get('dimension_y', 0)}"
                    colors = wallpaper.get('colors', [])
                    
                    # Create title from query and metadata
                    title = f"{query}_{wallpaper_id}_{resolution}"
                    
                    # Download the image
                    file_path = self._download_image(img_url, title)
                    
                    if file_path:
                        metadata = {
                            'title': title,
                            'source': 'wallhaven',
                            'url': img_url,
                            'query': query,
                            'id': wallpaper_id,
                            'resolution': resolution,
                            'colors': colors
                        }
                        downloaded_files.append((file_path, metadata))
                        images_downloaded += 1
                        logging.info(f"Downloaded {images_downloaded}/{max_images}: {title}")
                    
                    # Rate limiting
                    time.sleep(self.delay)
                
                # Check if there are more pages
                meta = data.get('meta', {})
                current_page = meta.get('current_page', page)
                last_page = meta.get('last_page', page)
                
                if current_page >= last_page:
                    logging.info("Reached last page of results")
                    break
                
                page += 1
                time.sleep(self.delay)
                
        except Exception as e:
            logging.error(f"Error searching WallHaven: {e}")
        
        logging.info(f"Total images downloaded from WallHaven: {len(downloaded_files)}")
        return downloaded_files
    
    def _download_image(self, url, title):
        """
        Download a single image from URL
        
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            response = self.session.get(url, timeout=15, stream=True)
            
            if response.status_code != 200:
                logging.warning(f"Failed to download {url}: HTTP {response.status_code}")
                return None
            
            # Determine file extension from URL
            ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]
            if not ext or ext not in ['.jpg', '.jpeg', '.png']:
                ext = '.jpg'
            
            # Create filename
            safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')])[:50]
            if not safe_title:
                safe_title = f"wallhaven_{int(time.time() * 1000)}"
            
            filename = f"{safe_title}{ext}"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Handle duplicates
            counter = 1
            while os.path.exists(filepath):
                filename = f"{safe_title}_{counter}{ext}"
                filepath = os.path.join(self.temp_dir, filename)
                counter += 1
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Save metadata
            meta_path = filepath + ".meta"
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(title)
            
            return filepath
            
        except Exception as e:
            logging.error(f"Error downloading image from {url}: {e}")
            return None


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_wallhaven")
    downloader = WallHavenDownloader(temp_dir)
    
    # Test with a car query
    results = downloader.search_and_download("Ferrari 488", max_images=5)
    
    print(f"\nDownloaded {len(results)} images:")
    for filepath, metadata in results:
        print(f"  - {filepath}")
        print(f"    Resolution: {metadata['resolution']}")
        print(f"    URL: {metadata['url']}")

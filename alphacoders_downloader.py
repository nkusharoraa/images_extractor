import os
import logging
from bs4 import BeautifulSoup
import time
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    import requests
    CLOUDSCRAPER_AVAILABLE = False
    logging.warning("cloudscraper not installed. AlphaCoders may block requests. Install with: pip install cloudscraper")
import urllib.parse

class AlphaCodersDownloader:
    """
    Custom downloader for AlphaCoders (wall.alphacoders.com)
    Note: AlphaCoders has different sections:
    - wall.alphacoders.com (general wallpapers)
    - images.alphacoders.com (car images)
    - art.alphacoders.com (digital art)
    """
    
    def __init__(self, temp_dir, delay=1.0):
        """
        Initialize AlphaCoders downloader
        
        Args:
            temp_dir: Directory to save downloaded images
            delay: Delay between requests in seconds (default 1.0)
        """
        self.temp_dir = temp_dir
        self.delay = delay
        
        # Use cloudscraper if available (bypasses Cloudflare), otherwise use requests
        if CLOUDSCRAPER_AVAILABLE:
            self.session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'desktop': True
                }
            )
            logging.info("Using cloudscraper for AlphaCoders (Cloudflare bypass enabled)")
        else:
            self.session = requests.Session()
            # Enhanced headers to appear as a real browser
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            })
            logging.warning("Using requests without cloudscraper - may encounter blocking")
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    def search_and_download(self, query, max_images=100, min_width=1920, category="cars"):
        """
        Search and download images from AlphaCoders
        
        Args:
            query: Search query (e.g., "Ferrari 488")
            max_images: Maximum number of images to download
            min_width: Minimum image width filter
            category: Category to search in ("cars", "general", "art")
        
        Returns:
            List of tuples (file_path, metadata_dict)
        """
        downloaded_files = []
        
        # Use wall.alphacoders.com as it may have less strict restrictions
        # Note: images.alphacoders.com often returns 403 for automated requests
        base_url = "https://wall.alphacoders.com"
        
        # For cars, we can still search with "car" keyword on wall.alphacoders
        if category == "cars":
            # Append "car" to the query for better results
            enhanced_query = f"{query} car"
            search_url = f"{base_url}/search.php?search={urllib.parse.quote(enhanced_query)}"
        elif category == "art":
            search_url = f"{base_url}/search.php?search={urllib.parse.quote(query)}"
        else:
            search_url = f"{base_url}/search.php?search={urllib.parse.quote(query)}"
        
        logging.info(f"Searching AlphaCoders (wall.alphacoders.com) for: {query} (category: {category})")
        logging.info(f"URL: {search_url}")
        
        try:
            # Get search results
            page = 1
            images_downloaded = 0
            
            while images_downloaded < max_images:
                # Construct paginated URL
                if category == "cars":
                    page_url = f"{search_url}&page={page}"
                else:
                    page_url = f"{search_url}&page={page}"
                
                logging.info(f"Fetching page {page}: {page_url}")
                
                # Add Referer header for this request
                headers = {'Referer': base_url + '/'}
                response = self.session.get(page_url, timeout=10, headers=headers)
                
                if response.status_code != 200:
                    logging.warning(f"Failed to fetch page {page}: HTTP {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find image links (AlphaCoders uses different HTML structures)
                image_links = self._extract_image_links(soup, category, base_url)
                
                if not image_links:
                    logging.info(f"No more images found on page {page}")
                    break
                
                # Download each image
                for link_data in image_links:
                    if images_downloaded >= max_images:
                        break
                    
                    img_url = link_data['url']
                    title = link_data.get('title', query)
                    
                    # Download the image
                    file_path = self._download_image(img_url, title)
                    
                    if file_path:
                        metadata = {
                            'title': title,
                            'source': 'alphacoders',
                            'url': img_url,
                            'query': query
                        }
                        downloaded_files.append((file_path, metadata))
                        images_downloaded += 1
                        logging.info(f"Downloaded {images_downloaded}/{max_images}: {title}")
                    
                    # Rate limiting
                    time.sleep(self.delay)
                
                page += 1
                time.sleep(self.delay)
                
        except Exception as e:
            logging.error(f"Error searching AlphaCoders: {e}")
        
        logging.info(f"Total images downloaded from AlphaCoders: {len(downloaded_files)}")
        return downloaded_files
    
    def _extract_image_links(self, soup, category, base_url):
        """
        Extract image links from AlphaCoders search results
        Using wall.alphacoders.com structure for all categories
        """
        image_links = []
        
        try:
            # wall.alphacoders.com uses consistent structure
            # Look for thumbnail containers
            thumbs = soup.find_all('div', class_='thumb-container-big')
            
            if not thumbs:
                # Fallback to alternative class names
                thumbs = soup.find_all('div', class_='thumb-container')
            
            if not thumbs:
                # Another fallback - look for image links directly
                thumbs = soup.find_all('a', href=re.compile(r'/wallpaper/'))
            
            logging.info(f"Found {len(thumbs)} thumbnail containers")
            
            for thumb in thumbs:
                try:
                    # Get detail page URL
                    if thumb.name == 'a':
                        a_tag = thumb
                    else:
                        a_tag = thumb.find('a')
                    
                    if not a_tag or 'href' not in a_tag.attrs:
                        continue
                    
                    detail_url = a_tag['href']
                    if not detail_url.startswith('http'):
                        detail_url = base_url + detail_url
                    
                    # Get title from image alt or title
                    img_tag = thumb.find('img') if thumb.name != 'img' else thumb
                    title = ""
                    if img_tag:
                        title = img_tag.get('alt', '') or img_tag.get('title', '')
                    
                    # For wall.alphacoders.com, we can try to get direct image URL
                    # or visit detail page
                    img_url = self._get_fullsize_image_url(detail_url)
                    
                    if img_url:
                        image_links.append({
                            'url': img_url,
                            'title': title
                        })
                except Exception as e:
                    logging.debug(f"Error parsing thumbnail: {e}")
                    continue
            
            # If we didn't find images with the above method, try generic approach
            if not image_links:
                # Generic fallback: find all image links
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    if 'thumb' not in src and any(ext in src for ext in ['.jpg', '.jpeg', '.png']):
                        # Try to get full resolution
                        full_url = src.replace('_thumb', '').replace('/thumbs/', '/images/')
                        image_links.append({
                            'url': full_url,
                            'title': img.get('alt', img.get('title', ''))
                        })
        
        except Exception as e:
            logging.error(f"Error extracting image links: {e}")
        
        return image_links
    
    def _get_fullsize_image_url(self, detail_page_url):
        """
        Visit the detail page to extract the full-size image URL
        """
        try:
            response = self.session.get(detail_page_url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # AlphaCoders typically has a download button or full-size image link
            # Look for common patterns
            
            # Method 1: Look for download link
            download_link = soup.find('a', class_='button download')
            if download_link and 'href' in download_link.attrs:
                return download_link['href']
            
            # Method 2: Look for full-size image in img tag with specific ID
            full_img = soup.find('img', id='main-image')
            if full_img and 'src' in full_img.attrs:
                return full_img['src']
            
            # Method 3: Look for image in center-block
            center_img = soup.find('div', class_='center-block').find('img') if soup.find('div', class_='center-block') else None
            if center_img and 'src' in center_img.attrs:
                return center_img['src']
            
            # Method 4: Look for any large image
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if any(ext in src for ext in ['.jpg', '.jpeg', '.png']) and 'thumb' not in src:
                    return src
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting full-size image from {detail_page_url}: {e}")
            return None
    
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
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            else:
                # Try to get from URL
                ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]
                if not ext or ext not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
            
            # Create filename
            safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')])[:50]
            if not safe_title:
                safe_title = f"alphacoders_{int(time.time() * 1000)}"
            
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
    
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_alphacoders")
    downloader = AlphaCodersDownloader(temp_dir)
    
    # Test with a car query
    results = downloader.search_and_download("Ferrari 488", max_images=5, category="cars")
    
    print(f"\nDownloaded {len(results)} images:")
    for filepath, metadata in results:
        print(f"  - {filepath}")
        print(f"    Title: {metadata['title']}")
        print(f"    URL: {metadata['url']}")

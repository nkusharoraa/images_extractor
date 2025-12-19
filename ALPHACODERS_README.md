# AlphaCoders Image Source Guide

## Overview
I've added **AlphaCoders** as an alternative image source to your car image downloader. AlphaCoders (images.alphacoders.com) is a high-quality wallpaper site that specializes in automotive photography.

## What Was Changed

### New Files
- **`alphacoders_downloader.py`**: Custom downloader module for scraping AlphaCoders

### Modified Files
- **`download_images.py`**: 
  - Added `source` parameter to choose between "bing" and "alphacoders"
  - Added interactive menu to select image source at startup
  - Modified output directory naming to include source (e.g., `run_20251219_095000_alphacoders`)

- **`requirements.txt`**: 
  - Added `beautifulsoup4`, `lxml`, and `requests` for web scraping

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python download_images.py
```

### 3. Select Image Source
When prompted, choose your image source:
```
Available image sources:
1. Bing (default) - Uses Bing Image Search
2. AlphaCoders - High-quality wallpapers from images.alphacoders.com
Choose image source (1 or 2) [default 1]:
```

- **Option 1**: Bing (original functionality)
- **Option 2**: AlphaCoders (new feature)

### 4. Continue with Normal Configuration
The rest of the prompts remain the same:
- Enter car names to search
- Set minimum image width
- Set minimum file size
- Enable/disable car validation
- Set background variance threshold

## Key Features

### AlphaCoders Downloader
- **Category Support**: Automatically searches in the "cars" category
- **Pagination**: Fetches multiple pages of results (up to 100 images per query)
- **Rate Limiting**: Built-in 1-second delay between requests to be respectful to the server
- **Multiple Methods**: Uses multiple fallback methods to extract full-resolution images
- **Metadata Preservation**: Saves image titles in `.meta` files for AI-powered renaming

### How It Works
1. **Search**: Queries images.alphacoders.com with your search term
2. **Parse**: Extracts thumbnail links from search results
3. **Fetch Full-Size**: Visits each detail page to get the full-resolution download URL
4. **Download**: Downloads the high-quality image to the temp directory
5. **Process**: Same validation, cropping, and AI renaming as Bing images

## Comparison: Bing vs AlphaCoders

| Feature | Bing | AlphaCoders |
|---------|------|-------------|
| **Image Quality** | Good | Excellent (wallpaper-focused) |
| **Search Suffix** | Used ("car studio background") | Ignored (category-based) |
| **Results Volume** | High (1000+ per query) | Medium (100 per query) |
| **Rate Limiting** | Built into icrawler | 1s delay between requests |
| **Image Type** | Mixed (includes stock photos) | Professional wallpapers |
| **API Required** | No | No (web scraping) |

## Code Example

### Using AlphaCoders Programmatically
```python
from alphacoders_downloader import AlphaCodersDownloader

# Initialize downloader
downloader = AlphaCodersDownloader(temp_dir="./temp", delay=1.0)

# Download images
results = downloader.search_and_download(
    query="Ferrari 488",
    max_images=50,
    category="cars"
)

# Process results
for filepath, metadata in results:
    print(f"Downloaded: {filepath}")
    print(f"Title: {metadata['title']}")
    print(f"URL: {metadata['url']}")
```

### Using in Main Script
```python
# Call with source parameter
download_images(
    queries=["Lamborghini Aventador", "Porsche 911"],
    source="alphacoders",  # or "bing"
    min_width=1920,
    min_file_size_kb=100,
    validate_car=True
)
```

## Categories Available
The AlphaCoders downloader supports different categories:
- **cars**: High-quality automotive wallpapers (default)
- **general**: General wallpapers from wall.alphacoders.com
- **art**: Digital art from art.alphacoders.com

To change category, modify the `category` parameter in `alphacoders_downloader.py` or the function call.

## Tips for Best Results

### For AlphaCoders:
- Use specific car model names (e.g., "Ferrari 488 Pista" instead of just "Ferrari")
- Results are typically higher quality but fewer in number
- The site specializes in wallpapers, so aspect ratios may vary
- All images go through the same validation pipeline (car detection, quality checks, etc.)

### General Tips:
- Both sources benefit from AI-powered filename cleaning (requires Ollama)
- All images are processed for watermark removal and 2:1 cropping
- Duplicate detection works across both sources
- Output folders are named by source for easy organization

## Troubleshooting

### AlphaCoders Not Working?
- **Check Internet Connection**: The scraper needs to access images.alphacoders.com
- **Rate Limiting**: If you see HTTP 429 errors, increase the delay parameter
- **Site Structure Changes**: AlphaCoders may update their HTML structure; the parser may need updates
- **No Results Found**: Try different search terms or check if the site is accessible

### Still Seeing Errors?
- Check the logs - they'll show exactly what's happening
- Try with Bing source first to verify your setup is working
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Future Enhancements
Potential improvements:
- Add more wallpaper sites (WallpaperHub, HDQwalls, etc.)
- Implement async/parallel downloads for AlphaCoders
- Add resolution filtering (4K, 8K, etc.)
- Support for custom categories
- API integration if AlphaCoders provides one

## License & Ethics
- **Respect robots.txt**: The downloader includes rate limiting
- **Personal Use**: Downloaded images should be for personal use only
- **Copyright**: Images belong to their respective photographers/owners
- **Terms of Service**: Review AlphaCoders' ToS before heavy usage

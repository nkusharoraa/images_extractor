"""
Quick test script for AlphaCoders downloader
This will download a few images to verify the integration works
"""

import os
import sys
import logging
from alphacoders_downloader import AlphaCodersDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_alphacoders():
    """Test AlphaCoders downloader with a simple query"""
    
    print("=" * 60)
    print("AlphaCoders Downloader Test")
    print("=" * 60)
    
    # Create temp directory
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_test_alphacoders")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    print(f"\nTemp directory: {temp_dir}")
    print(f"Testing with query: 'Porsche 911'\n")
    
    # Initialize downloader
    downloader = AlphaCodersDownloader(temp_dir, delay=1.0)
    
    # Download a few images
    results = downloader.search_and_download(
        query="Porsche 911",
        max_images=3,  # Just download 3 for testing
        category="cars"
    )
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    if results:
        print(f"\n✓ Successfully downloaded {len(results)} images:\n")
        for i, (filepath, metadata) in enumerate(results, 1):
            print(f"{i}. {os.path.basename(filepath)}")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')[:60]}...")
            
            # Check file size
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   Size: {size_kb:.1f} KB\n")
    else:
        print("\n✗ No images downloaded. Check logs above for errors.")
        print("\nPossible issues:")
        print("  - Network connectivity")
        print("  - AlphaCoders site structure changed")
        print("  - Rate limiting")
        print("  - Search query returned no results")
    
    print("\n" + "=" * 60)
    print("Note: Downloaded images are saved in:", temp_dir)
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_alphacoders()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

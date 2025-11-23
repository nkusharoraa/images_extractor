"""
Test script to verify Ollama integration for filename generation
"""
import logging
import sys
import os

# Add the parent directory to the path to import from download_images
sys.path.insert(0, os.path.dirname(__file__))

from download_images import generate_filename_with_ollama, OLLAMA_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ollama_integration():
    """Test the Ollama filename generation with sample inputs"""
    
    print("=" * 60)
    print("Testing Ollama LLaMA 3.1 8B Integration")
    print("=" * 60)
    print()
    
    if not OLLAMA_AVAILABLE:
        print("❌ ERROR: Ollama library is not available!")
        print("Please install it with: pip install ollama")
        return False
    
    print("✓ Ollama library is available")
    print()
    
    # Test cases with various types of filenames
    test_cases = [
        "2023 Mercedes-Benz S-Class wallpaper HD - JohnDoe Photography",
        "Lamborghini Aventador SVJ by AutoMaxImages Studio Background",
        "Ferrari F8 Tributo - Red Sports Car - High Resolution Image",
        "BMW M4 Competition | Car Wallpapers | 4K Background",
        "Porsche 911 GT3 RS - Owner: Mike Smith - Studio Shot",
        "Tesla Model S Plaid Electric Car Wallpaper HD Images",
    ]
    
    print("Testing with sample filenames:")
    print("-" * 60)
    
    success_count = 0
    for i, original in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Original: {original}")
        
        try:
            cleaned = generate_filename_with_ollama(original)
            print(f"  Cleaned:  {cleaned}")
            
            if cleaned and cleaned != "Unknown_Car":
                success_count += 1
                print("  Status:   ✓ Success")
            else:
                print("  Status:   ⚠ Warning - Empty or fallback result")
        except Exception as e:
            print(f"  Status:   ❌ Error - {e}")
    
    print()
    print("=" * 60)
    print(f"Results: {success_count}/{len(test_cases)} successful")
    print("=" * 60)
    
    return success_count == len(test_cases)

if __name__ == "__main__":
    print("\nNote: Make sure Ollama is running locally with llama3.1:8b model installed.")
    print("You can verify with: ollama list")
    print()
    
    input("Press Enter to start the test...")
    print()
    
    success = test_ollama_integration()
    
    if success:
        print("\n✓ All tests passed! The integration is working correctly.")
    else:
        print("\n⚠ Some tests failed. Please check the logs above.")

import requests
from bs4 import BeautifulSoup
import html

def inspect_bing(keyword):
    url = f"https://www.bing.com/images/async?q={keyword}&first=0&count=10"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "lxml")
    
    image_divs = soup.find_all("div", class_="imgpt")
    print(f"Found {len(image_divs)} images.")
    
    for i, div in enumerate(image_divs[:3]):
        if div.a and "m" in div.a.attrs:
            m_attr = html.unescape(div.a["m"])
            print(f"\nImage {i+1} 'm' attribute:")
            print(m_attr)

if __name__ == "__main__":
    inspect_bing("Porsche 911 GT3")

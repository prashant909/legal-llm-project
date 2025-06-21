import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://legislative.gov.in" 
SAVE_DIR = "data/legislative_documents"

os.makedirs(SAVE_DIR, exist_ok=True)

def scrape_document_pages():
    for page_num in range(1, 50):  # Adjust based on number of pages
        url = f"{BASE_URL}/documents/page/{page_num}/"
        print(f"📄 Fetching page: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                if href.endswith(".pdf"):
                    pdf_url = urljoin(BASE_URL, href)
                    file_name = os.path.join(SAVE_DIR, os.path.basename(href))
                    download_pdf(pdf_url, file_name)
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")

def download_pdf(pdf_url, file_name):
    if not os.path.exists(file_name):
        print(f"📥 Downloading: {pdf_url}")
        try:
            pdf_data = requests.get(pdf_url, timeout=20)
            pdf_data.raise_for_status()
            with open(file_name, "wb") as f:
                f.write(pdf_data.content)
        except Exception as e:
            print(f"❌ Failed to download {pdf_url}: {e}")
    else:
        print(f"✅ Already exists: {file_name}")

def main():
    scrape_document_pages()

if __name__ == "__main__":
    main()
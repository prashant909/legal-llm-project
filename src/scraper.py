import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://legislative.gov.in"
SAVE_DIR = "data/legislative_documents"

os.makedirs(SAVE_DIR, exist_ok=True)

def scrape_document_pages(force_download=False):
    """Scrape PDF documents from legislative.gov.in only if not already downloaded."""
    existing_files = set(os.listdir(SAVE_DIR))

    for page_num in range(1, 50):  # You can adjust page range as needed
        url = f"{BASE_URL}/documents/page/{page_num}/"
        print(f"\n📄 Fetching page: {url}")
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

                    if not force_download and os.path.basename(href) in existing_files:
                        print(f"✅ Skipping already downloaded: {file_name}")
                        continue

                    download_pdf(pdf_url, file_name)
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")

def download_pdf(pdf_url, file_name):
    print(f"📥 Downloading: {pdf_url}")
    try:
        pdf_data = requests.get(pdf_url, timeout=20)
        pdf_data.raise_for_status()
        with open(file_name, "wb") as f:
            f.write(pdf_data.content)
        print(f"✅ Saved to: {file_name}")
    except Exception as e:
        print(f"❌ Failed to download {pdf_url}: {e}")

def main(force_download=False):
    if os.listdir(SAVE_DIR) and not force_download:
        print(f"📁 PDF folder already populated in '{SAVE_DIR}'. Skipping scraping.")
    else:
        scrape_document_pages(force_download=force_download)

if __name__ == "__main__":
    # Change to True to force re-download all files
    main(force_download=False)

from playwright.async_api import async_playwright
import asyncio
import logging
import re
import requests
import csv
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from tqdm.asyncio import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

concurrent_requests = 5
retry_limit = 3
delay_between_batches = 2  # Delay between batches in seconds

# Read robots.txt to enforce politeness
def read_robots_txt(domain):
    robots_url = urljoin(domain, "robots.txt")
    response = requests.get(robots_url)
    disallowed_paths = []
    if response.status_code == 200:
        rules = response.text.split("\n")
        for rule in rules:
            if rule.lower().startswith("disallow"):
                path = rule.split(":")[1].strip()
                disallowed_paths.append(path)
    return disallowed_paths

# Data cleaning function
def clean_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text.strip()

async def fetch_page(playwright, url, visited_urls, to_visit_urls, writer, pbar, disallowed_paths, domain):
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    for attempt in range(retry_limit):
        try:
            await page.goto(url, timeout=60000)  # Set a generous timeout for loading
            await page.wait_for_load_state('networkidle')
            
            visited_urls.add(url)

            # Extract links from the page
            links = await page.query_selector_all('a')
            for link in links:
                href = await link.get_attribute('href')
                if href:
                    href = urljoin(domain, href)
                    parsed_href = urlparse(href)
                    if parsed_href.netloc == urlparse(domain).netloc and href not in visited_urls and href not in to_visit_urls:
                        if not any(parsed_href.path.startswith(disallowed) for disallowed in disallowed_paths):
                            to_visit_urls.append(href)

            # Extract content from the page
            title = await page.title()
            content = await page.content()

            # Clean the content
            clean_content = clean_text(content)

            # Save to CSV
            writer.writerow([url, title, clean_content])
            logging.info(f'Scraped {url}')
            pbar.update(1)
            break
        except Exception as e:
            logging.error(f'Error scraping {url} on attempt {attempt + 1}: {e}')
            if attempt < retry_limit - 1:
                backoff_delay = (2 ** attempt) * delay_between_batches
                logging.info(f'Waiting for {backoff_delay} seconds before retrying...')
                await asyncio.sleep(backoff_delay)
            else:
                logging.error(f'Failed to scrape {url} after {retry_limit} attempts')
        finally:
            await browser.close()

async def main():
    start_url = input("Enter the starting URL: ")
    domain = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
    
    disallowed_paths = read_robots_txt(domain)
    logging.info(f'Disallowed paths from robots.txt: {disallowed_paths}')

    async with async_playwright() as playwright:
        visited_urls = set()
        to_visit_urls = [start_url]
        
        output_file = input("Enter the output CSV file name (with .csv extension): ")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['url', 'title', 'content'])

            total_urls = len(to_visit_urls)
            with tqdm(total=total_urls) as pbar:
                while to_visit_urls:
                    current_batch = [to_visit_urls.pop(0) for _ in range(min(concurrent_requests, len(to_visit_urls)))]
                    tasks = [fetch_page(playwright, url, visited_urls, to_visit_urls, writer, pbar, disallowed_paths, domain) for url in current_batch]
                    await asyncio.gather(*tasks)

                    # Rate limiting
                    logging.info(f'Waiting for {delay_between_batches} seconds before next batch...')
                    await asyncio.sleep(delay_between_batches)

asyncio.run(main())

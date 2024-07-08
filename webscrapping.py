from playwright.async_api import async_playwright
import asyncio
import logging
import time
import re
import requests
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

concurrent_requests = 5
retry_limit = 3
delay_between_batches = 2  # Delay between batches in seconds

# Read robots.txt to enforce politeness
def read_robots_txt():
    robots_url = "https://www.wichita.edu/robots.txt"
    response = requests.get(robots_url)
    if response.status_code == 200:
        rules = response.text.split("\n")
        for rule in rules:
            if rule.lower().startswith("crawl-delay"):
                delay = int(re.findall(r'\d+', rule)[0])
                return delay
    return 2

crawl_delay = read_robots_txt()
logging.info(f'Using a crawl delay of {crawl_delay} seconds based on robots.txt')

async def fetch_page(playwright, url, visited_urls, to_visit_urls, writer):
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
                if href and 'wichita.edu' in href and href not in visited_urls and href not in to_visit_urls:
                    to_visit_urls.append(href)

            # Extract content from the page
            title = await page.title()
            content = await page.content()

            # Save to CSV
            writer.writerow([url, title, content])
            logging.info(f'Scraped {url}')
            break
        except Exception as e:
            logging.error(f'Error scraping {url} on attempt {attempt + 1}: {e}')
            if attempt < retry_limit - 1:
                backoff_delay = (2 ** attempt) * crawl_delay
                logging.info(f'Waiting for {backoff_delay} seconds before retrying...')
                await asyncio.sleep(backoff_delay)
            else:
                logging.error(f'Failed to scrape {url} after {retry_limit} attempts')
        finally:
            await browser.close()

async def main():
    async with async_playwright() as playwright:
        visited_urls = set()
        to_visit_urls = ['https://www.wichita.edu']
        
        with open('wichita_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['url', 'title', 'content'])

            while to_visit_urls:
                current_batch = [to_visit_urls.pop(0) for _ in range(min(concurrent_requests, len(to_visit_urls)))]
                tasks = [fetch_page(playwright, url, visited_urls, to_visit_urls, writer) for url in current_batch]
                await asyncio.gather(*tasks)

                # Rate limiting
                logging.info(f'Waiting for {delay_between_batches} seconds before next batch...')
                await asyncio.sleep(delay_between_batches)

asyncio.run(main())

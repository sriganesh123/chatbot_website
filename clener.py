import csv
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import sys
from dask import dataframe as dd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Increase CSV field size limit to a large value
csv.field_size_limit(10**7)

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def clean_html(raw_html):
    """Remove HTML tags and extract text."""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        logging.error(f"Error cleaning HTML: {e}")
        return ""

def normalize_text(text):
    """Normalize text by removing punctuation, stopwords, and converting to lowercase."""
    try:
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Convert to lowercase
        tokens = [word.lower() for word in tokens]
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        return ""

def process_chunk(chunk):
    """Process a chunk of data: clean and normalize."""
    chunk['cleaned_content'] = chunk['content'].apply(clean_html)
    chunk['normalized_content'] = chunk['cleaned_content'].apply(normalize_text)
    return chunk

def main():
    # Read the scraped data in chunks and process each chunk
    chunk_size = 1000000  # Adjust chunk size as needed
    dtypes = {'url': str, 'title': str, 'content': str}

    try:
        # Sample a few rows to infer the schema
        sample_df = pd.read_csv('wichita_data.csv', dtype=dtypes, nrows=10)
        logging.info("Sample data loaded successfully.")

        # Initialize a Dask DataFrame with inferred schema
        ddf = dd.read_csv('wichita_data.csv', dtype=dtypes, blocksize=chunk_size, assume_missing=True)

        # Process each chunk and compute the result
        ddf = ddf.map_partitions(process_chunk)

        # Save the cleaned data to a new CSV file
        ddf.to_csv('cleaned_wichita_data-*.csv', single_file=True)
        logging.info("Data cleaning complete. Cleaned data saved to 'cleaned_wichita_data.csv'")

        # Load the cleaned data into a DataFrame to verify
        df = pd.read_csv('cleaned_wichita_data.csv')
        logging.info("Sample of cleaned data:")
        logging.info(df.head())
    except pd.errors.EmptyDataError:
        logging.error("No data found in the CSV file.")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()

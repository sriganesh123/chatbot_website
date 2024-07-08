import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
import chromadb
import os
import uuid
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Install necessary spaCy model
import subprocess
import sys

def install_spacy_model():
    try:
        import en_core_web_sm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

install_spacy_model()
nlp = spacy.load("en_core_web_sm")

# Load the data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit()

# Data cleaning function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.lower().strip()

def perform_analysis(data):
    # Clean content
    data['cleaned_content'] = data['content'].apply(clean_text)

    # Add a new column for content length
    data['content_length'] = data['cleaned_content'].apply(len)

    # Function to tokenize text
    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    # Get all tokens from the content
    all_tokens = data['cleaned_content'].apply(tokenize).sum()

    # Get the frequency distribution of tokens
    token_counts = Counter(all_tokens)

    # Function to get sentiment polarity
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    # Add a new column for sentiment polarity
    data['sentiment'] = data['cleaned_content'].apply(get_sentiment)

    # Function to extract named entities
    def extract_entities(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    # Add a new column for named entities
    data['entities'] = data['cleaned_content'].apply(extract_entities)

    return data, token_counts

def visualize_data(data, token_counts):
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))  # Adjust the grid size as needed

    # Plot the distribution of content length
    sns.histplot(data['content_length'], bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Content Length')
    axes[0, 0].set_xlabel('Content Length')
    axes[0, 0].set_ylabel('Frequency')

    # Plot the most common words
    common_words = token_counts.most_common(20)
    words, counts = zip(*common_words)
    sns.barplot(x=list(words), y=list(counts), ax=axes[0, 1])
    axes[0, 1].set_title('Most Common Words')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_xticklabels(words, rotation=45)
    axes[0, 1].set_ylabel('Frequency')

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(token_counts)
    axes[1, 0].imshow(wordcloud, interpolation='bilinear')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Word Cloud')

    # Plot the distribution of sentiment polarity
    sns.histplot(data['sentiment'], bins=20, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Sentiment Polarity')
    axes[1, 1].set_xlabel('Sentiment Polarity')
    axes[1, 1].set_ylabel('Frequency')

    # Analyze sentiment by category (e.g., positive, neutral, negative)
    def categorize_sentiment(polarity):
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    data['sentiment_category'] = data['sentiment'].apply(categorize_sentiment)
    sentiment_counts = data['sentiment_category'].value_counts()

    # Plot sentiment distribution
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[2, 0])
    axes[2, 0].set_title('Sentiment Distribution')
    axes[2, 0].set_xlabel('Sentiment')
    axes[2, 0].set_ylabel('Frequency')

    # Flatten the list of entities for analysis
    all_entities = [entity for entities in data['entities'] for entity in entities]
    entity_counts = Counter(all_entities)

    # Display the most common entities
    common_entities = entity_counts.most_common(20)
    entities, counts = zip(*common_entities)
    entity_texts, entity_labels = zip(*entities)
    sns.barplot(x=list(entity_texts), y=list(counts), hue=list(entity_labels), ax=axes[2, 1])
    axes[2, 1].set_title('Most Common Entities')
    axes[2, 1].set_xlabel('Entities')
    axes[2, 1].set_xticklabels(entity_texts, rotation=45)
    axes[2, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def save_analyzed_data(data, output_file):
    try:
        data.to_csv(output_file, index=False)
        print(f"Analyzed data saved to {output_file}")
    except Exception as e:
        print(f"Failed to save analyzed data: {e}")

def store_in_chromadb(data, collection_name, persist_directory):
    # Trim any leading or trailing spaces from the persist directory
    persist_directory = persist_directory.strip()

    # Initialize ChromaDB client with persistence
    logging.info(f"Initializing ChromaDB client with persist directory: {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Ensure collection is created or accessed
    try:
        collection = client.get_collection(name=collection_name)
        logging.info(f"Collection {collection_name} accessed successfully.")
    except ValueError:
        collection = client.create_collection(name=collection_name)
        logging.info(f"Collection {collection_name} created successfully.")
    
    # Load the pre-trained model for text embedding
    logging.info("Loading the pre-trained model for text embedding")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare data for storage in ChromaDB
    ids = []
    embeddings = []
    metadatas = []

    for idx, item in data.iterrows():
        unique_id = f"{item['title']}_{uuid.uuid4()}"  # Ensure IDs are unique using UUID
        ids.append(unique_id)
        embedding = model.encode(item['cleaned_content']).tolist()  # Convert to list
        embeddings.append(embedding)
        metadatas.append({
            'title': item['title'],
            'url': item['url'],
            'content_length': item['content_length'],
            'sentiment': item['sentiment'],
            'entities': str(item['entities'])  # Convert list to string
        })
        logging.info(f"Processed item: {unique_id}, embedding length: {len(embedding)}")

    # Save ids, embeddings, and metadatas separately for verification
    ids_path = os.path.join(persist_directory, "ids.json")
    embeddings_path = os.path.join(persist_directory, "embeddings.json")
    metadatas_path = os.path.join(persist_directory, "metadatas.json")

    with open(ids_path, 'w') as f:
        json.dump(ids, f)
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings, f)
    with open(metadatas_path, 'w') as f:
        json.dump(metadatas, f)
    
    logging.info(f"IDs saved to {ids_path}")
    logging.info(f"Embeddings saved to {embeddings_path}")
    logging.info(f"Metadatas saved to {metadatas_path}")

    # Storing data in ChromaDB
    logging.info(f"Storing data in ChromaDB with the following details:\nIDs: {ids}\nEmbeddings: {embeddings}\nMetadatas: {metadatas}")
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    logging.info(f"Data stored in ChromaDB at {persist_directory}.")

    # Verify the persistence directory
    if not os.path.exists(persist_directory):
        logging.error(f"Persistence directory does not exist: {persist_directory}")
    else:
        logging.info("Listing contents of the ChromaDB persistence directory:")
        for root, dirs, files in os.walk(persist_directory):
            for file in files:
                logging.info(os.path.join(root, file))

    # Attempt to retrieve the first item to verify storage
    if ids:
        try:
            # Query by using the vector search
            query_result = collection.query(query_embeddings=[embeddings[0]], n_results=1)
            logging.info(f"Retrieved item to verify storage: {query_result}")
        except Exception as e:
            logging.error(f"Error retrieving item: {e}")

def main():
    # Get file path from user
    file_path = input("Enter the path to the CSV file containing the scraped data: ")
    data = load_data(file_path)
    
    # Perform data analysis
    data, token_counts = perform_analysis(data)
    
    # Visualize the data
    visualize_data(data, token_counts)
    
    # Save the analyzed data
    output_file = input("Enter the output CSV file name to save the analyzed data: ")
    save_analyzed_data(data, output_file)
    
    # Store analyzed data in ChromaDB
    collection_name = input("Enter the name of the collection to store the data in ChromaDB: ")
    persist_directory = input("Enter the directory to store the ChromaDB data: ")
    store_in_chromadb(data, collection_name, persist_directory)

if __name__ == "__main__":
    main()

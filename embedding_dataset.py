from datasets import Dataset, load_dataset
from datasets import Dataset, Features, Value, Array2D
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
# Load your data
data_path = 'book_analysis.csv'  # Path to your analyzed data CSV file
data = pd.read_csv(data_path)

# Load the pre-trained model for text embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare the dataset
dataset = pd.DataFrame({
    'title': data['title'],
    'text': data['cleaned_content'],
    'embeddings': data['cleaned_content'].apply(lambda x: model.encode(x).tolist())
})

# Define dataset features
features = Features({
    'title': Value('string'),
    'text': Value('string'),
    'embeddings': Array2D(shape=(384,), dtype='float32')  # Assuming 'all-MiniLM-L6-v2' produces 384-dimensional vectors
})

# Create a HuggingFace dataset
hf_dataset = Dataset.from_pandas(dataset, features=features)

# Save the dataset to disk
dataset_path = 'correct_dataset_with_embeddings'
index_path = 'correct_embeddings_index'

hf_dataset.save_to_disk(dataset_path)
hf_dataset.add_faiss_index(column='embeddings')
hf_dataset.get_index('embeddings').save(index_path)

print("Dataset and index saved successfully.")

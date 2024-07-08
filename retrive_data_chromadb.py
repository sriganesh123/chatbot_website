import chromadb
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_from_chromadb(persist_directory, collection_name, query_text):
    # Initialize ChromaDB client with persistence
    logging.info(f"Initializing ChromaDB client with persist directory: {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)
    
    # List available collections
    collections = client.list_collections()
    logging.info(f"Available collections: {collections}")

    # Access the specified collection
    try:
        collection = client.get_collection(name=collection_name)
        logging.info(f"Collection {collection_name} accessed successfully.")
    except ValueError:
        logging.error(f"Collection {collection_name} does not exist.")
        sys.exit()

    # Query the collection
    try:
        query_result = collection.query(query_texts=[query_text], n_results=5)
        logging.info(f"Query results: {query_result}")
        
        # Extract and return metadata
        results = []
        for metadata in query_result['metadatas'][0]:
            results.append({
                'title': metadata['title'],
                'url': metadata['url'],
                'content_length': metadata['content_length'],
                'sentiment': metadata['sentiment'],
                'entities': metadata['entities']
            })
        return results
    except Exception as e:
        logging.error(f"Error querying ChromaDB: {e}")
        sys.exit()

def main():
    # Get the directory and collection name from the user
    persist_directory = input("Enter the directory where ChromaDB data is stored: ").strip()
    collection_name = input("Enter the name of the collection to query in ChromaDB: ").strip()
    query_text = input("Enter the query text: ").strip()
    
    # Retrieve data from ChromaDB
    results = retrieve_from_chromadb(persist_directory, collection_name, query_text)
    print("Query results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

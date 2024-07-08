import os
from dotenv import load_dotenv
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Custom embedding function using SentenceTransformers model
class LocalEmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        embeddings = self.model.encode(input, convert_to_tensor=True)
        return [embedding.cpu().tolist() for embedding in embeddings]

# Initialize ChromaDB client
def initialize_chromadb(persist_directory, collection_name):
    client = chromadb.PersistentClient(path=persist_directory)
    local_ef = LocalEmbeddingFunction(model_name="sentence-transformers/LaBSE")

    # Attempt to access the collection
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        # Create the collection if it does not exist
        collection = client.create_collection(name=collection_name, embedding_function=local_ef)
    
    return client, collection

# Retrieve data from ChromaDB
def retrieve_from_chromadb(collection, query_text):
    try:
        query_result = collection.query(query_texts=[query_text], n_results=5)
        return query_result.get('metadatas', [])[0] if 'metadatas' in query_result else []
    except Exception:
        return []

# Generate a response using Ollama model
def generate_response(prompt):
    llm = Ollama(model="llama3")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.invoke({'question': prompt})
    return response

# Main function to run the chatbot
def main():
    st.title("Chatbot Test with Ollama")
    
    # Get inputs from user
    persist_directory = st.text_input("Enter the directory where ChromaDB data is stored: ").strip()
    collection_name = st.text_input("Enter the name of the collection to query in ChromaDB (e.g., 'books_collection'): ").strip()
    
    if st.button("Initialize ChromaDB"):
        client, collection = initialize_chromadb(persist_directory, collection_name)
        st.session_state['collection'] = collection
        st.write("ChromaDB Initialized!")

    input_text = st.text_input("Enter your question")

    if input_text and 'collection' in st.session_state:
        collection = st.session_state['collection']
        
        # Retrieve relevant data from ChromaDB
        results = retrieve_from_chromadb(collection, input_text)
        if results:
            context = "\n".join([f"- {result.get('title', 'Unknown')} ({result.get('url', 'No URL')})" for result in results])
        else:
            context = "Sorry, I couldn't find any relevant information."
        
        # Create the prompt
        formatted_context_question = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and friendly customer service representative."),
                ("user", f"Context: {context}\n\nQuestion: {input_text}\n\nOnly return the helpful answer below and nothing else.\nHelpful answer:")
            ]
        )

        # Generate a response using Ollama model
        try:
            bot_response = generate_response(formatted_context_question)
            st.write(bot_response)
        except Exception as e:
            st.error(f"Error running model: {e}")

if __name__ == "__main__":
    main()

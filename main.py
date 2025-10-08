# Import necessary libraries
# WebBaseLoader: To load documents from a URL
# load_dotenv: To load environment variables (like API keys) from a .env file
# RecursiveCharacterTextSplitter: To split large documents into smaller chunks
# OpenAIEmbeddings: To create embedding vectors for each chunk of data
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()


# --- 1. LOAD THE DOCUMENT ---

# Initialize the WebBaseLoader with the target URL
loader = WebBaseLoader(
    "https://python.langchain.com/docs/integrations/document_loaders/web_base/"
)
# Optional: disable SSL certificate verification
loader.requests_kwargs = {"verify": False}

# Execute the load operation from the target URL
docs = loader.load()

# --- 2. SPLIT THE DOCUMENT INTO CHUNKS

# Initialize the Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Execute the split operation
splits = text_splitter.split_documents(docs)

# --- 3. EMBEDDING VECTOR FOR EACH CHUNK ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

# Initialize ChromaDB
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("Successfully created vector store.")

# INSPECT THE RESULTS ---

# print(f"Split the document into {len(splits)} chunks")
# print(splits[1].page_content)

# Create a sample query to test the vector store
query = "how to use WebBaseLoader"

# Perform a similarity search.
retrieved_docs = vectorstore.similarity_search(query)

# Print the number of documents retrieved (by default, it's 4)
print(f"\nRetrieved {len(retrieved_docs)} documents for the query: '{query}'")

# Print the content of the most relevant document chunk
print("\n--- Most Relevant Chunk ---")
print(retrieved_docs[0].page_content)

# Import necessary libraries
# WebBaseLoader: To load documents from a URL
# load_dotenv: To load environment variables (like API keys) from a .env file
# RecursiveCharacterTextSplitter: To split large documents into smaller chunks
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()


# --- 1. LOAD THE DOCUMENT ---

# Initialize the WebBaseLoader with the target URL
loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/web_base/")
# Optional: disable SSL certificate verification
loader.requests_kwargs = {'verify':False}

# Execute the load operation from the target URL
docs = loader.load()

# --- 2. SPLIT THE DOCUMENT INTO CHUNKS

# Initialize the Splitter
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)

# Execute the split operation
splits = text_splitter.split_documents(docs)

# --- 3. INSPECT THE RESULTS ---

print(f"Split the document into {len(splits)} chunks")
print(splits[1].page_content)


from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# 1. Charger tes documents d'exemple (ici des fichiers .txt dans un dossier)
loader = TextLoader("data/examples.txt", encoding="utf-8")  # ou autre dossier
documents = loader.load()

# 2. Splitter en morceaux
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Embedding + Chroma
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
vectordb.persist()
print("✅ Base Chroma initialisée")

from langchain.retrievers import ParentDocumentRetriever
import pandas as pd
from langchain.schema import Document
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from GCStore import GCSFileStore
from MongoDBStore import MongoDBFileStore
from RegexSplitter import RegexTextSplitter
load_dotenv()

uri = os.getenv("MONGO_URI")

path = os.path.join(os.getcwd(), "YOUR_KEY.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

index_name = "test-db"


df = pd.read_csv("test.csv", encoding='utf-8', low_memory=False)
documents = []
id = []

for _, row in df.iterrows():
    document_content = "\n".join([f"{k}: {v}" for k, v in row.items()])
    documents.append(Document(page_content=document_content))
    if 'ID' in row:
        id.append(row['ID'])


pattern = r'(\w+): ([^\n]+)'
child_splitter = RegexTextSplitter(pattern=pattern)
result = child_splitter.split_documents(documents)
print(result)

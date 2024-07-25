import pandas as pd
from langchain.schema import Document
from RegexSplitter import RegexTextSplitter

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

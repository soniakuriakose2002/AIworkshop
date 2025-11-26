!pip install -q sentence-transformers faiss-cpu pandas

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV
data = pd.read_csv('/content/flipkart_style_qa.csv')

# Encode FAQ questions
faq_embeddings = model.encode(data['question'].tolist())

# Create FAISS index
faq_index = faiss.IndexFlatL2(faq_embeddings.shape[1])
faq_index.add(faq_embeddings)

# Query example
query = "Is the shirt true to size"
query_embedding = model.encode([query])

D, I = faq_index.search(np.array(query_embedding), k=1)

print("Query:", query)
for i in I[0]:
    print("Answer:", data.iloc[i]['answer'])

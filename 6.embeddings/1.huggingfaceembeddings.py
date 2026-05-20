from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  
import os
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

# checking similarity based on the huggingface embeddings. we will use the sentence-transformers/all-MiniLM-L6-v2 model for this purpose. it is a small model but it is good for our purpose. it is also very fast.
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India"

documents=["Delhi is the capital of India",
           "Kolkata is the capital of the west bengal",
           "paris is the capital of france",
           "I love pizza"
           ]

result=embeddings.embed_query(text)

result_doc=embeddings.embed_documents(documents)

similarity_score=cosine_similarity([result], result_doc)
print(similarity_score)
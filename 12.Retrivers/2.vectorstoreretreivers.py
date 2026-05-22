
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# the documents are stored in the chroma db and the query is matched to the documents 
# and related ones are retrieved.
documents=[
    Document(page_content="Langchain helps developers build applications powered by language models."),
    Document(page_content="Chroma is a vector database that allows you to store and query vectors efficiently."),
    Document(page_content="Embeddings are numerical representations of text that capture semantic meaning."),
    Document(page_content="HuggingFace provides a wide range of pre-trained models for natural language processing tasks.")
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="sample",
)

retriever=vector_store.as_retriever(search_kwargs={"k":2})

query="what does chroma do ?"

result = retriever.invoke(query)

print(result)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


doc1 = Document(
        page_content="Virat Kohli is one of the most successful batsmen in cricket history. He has scored over 70 international centuries.",
        metadata={"team": "Royal Challengers Bangalore"}
    )

doc2 =  Document(
        page_content="MS Dhoni is a legendary Indian cricketer and former captain. He is known for his captaincy and finishing skills.",
        metadata={"team": "Chennai Super Kings"}
    )

doc3 = Document(
        page_content="Rohit Sharma is an opening batsman known for his elegant batting style and led India to T20 World Cup victory.",
        metadata={"team": "Mumbai Indians"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in modern cricket with a unique bowling action.",
        metadata={"team": "Mumbai Indians"}
)

docs=[doc1, doc2, doc3, doc4]

vector_store=Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="sample"
)

vector_store.add_documents(docs)

vector_store.get(include=["embeddings", "documents", "metadatas"])

query="Who among these are bowlers ?"

result=vector_store.similarity_search(query=query, k=3)

print(result)


# Import ChatOpenAI for LLM and OpenAIEmbeddings for converting text to vectors
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Import load_dotenv to read environment variables from .env file
from dotenv import load_dotenv
# Import TextLoader to load documents from text files
from langchain_community.document_loaders import TextLoader
# Import RecursiveCharacterTextSplitter to split long documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import FAISS for vector database to store and retrieve embeddings efficiently
from langchain_community.vectorstores import FAISS
# Import ChatPromptTemplate to create structured prompts for the model
from langchain_core.prompts import ChatPromptTemplate
# Import StrOutputParser to parse model output as a string
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file (API keys, etc.)
load_dotenv()

# Create a TextLoader to read the docs.txt file from parent directory
loader = TextLoader("../docs.txt")
# Load the documents into memory
documents = loader.load()

# Initialize text splitter with 500 character chunks and 50 character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Split documents into smaller chunks for better retrieval and processing
docs = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings model to convert text to numerical vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Create FAISS vector store from document chunks and embeddings
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever from the vector store to find similar documents
retriever = vectorstore.as_retriever()
# Initialize the GPT-4 Turbo model for answering questions
model = ChatOpenAI(model="gpt-4-turbo")

# Create a prompt template with placeholders for context and input question
prompt = ChatPromptTemplate.from_template("""Answer based on context:
{context}

Question: {input}""")

# Create a chain: template → model → output parser (converts to string)
chain = prompt | model | StrOutputParser()

# Define the user's question
query = "What are the main challenges in the field of AI?"
# Retrieve relevant document chunks similar to the query
relevant_docs = retriever.invoke(query)
# Combine retrieved documents into a single context string with newlines
context = "\n".join([doc.page_content for doc in relevant_docs])

# Invoke the chain with context and query to get the answer
result = chain.invoke({"context": context, "input": query})
# Print the model's response
print(result)
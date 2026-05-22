from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

# video_id = "QiYKOYLbjVk"
video_id="cInHoM5fTIo"

try:
    # Fetch transcript
    # transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    
    yt_api = YouTubeTranscriptApi()

    transcript_list = yt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video")
    exit()

# Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(transcript)

print("Number of chunks:", len(chunks))

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

# Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Prompt template
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# User question
question = "Which song is this?"

# Retrieve relevant chunks
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

# Create final prompt
final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

# LLM
model = ChatOpenAI(model="gpt-4-turbo")

answer = model.invoke(final_prompt)

print("\nAnswer:")
print(answer.content)
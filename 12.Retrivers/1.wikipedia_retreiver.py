from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Initialize Wikipedia API wrapper
api_wrapper = WikipediaAPIWrapper()

# Create retriever
retriever = WikipediaQueryRun(api_wrapper=api_wrapper)

query = "Indian premier league"

docs = retriever.invoke(query)

print(docs)
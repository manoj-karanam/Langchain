from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chains import LLMChain



# LLMChain is dereciated. ignore this exercise. we can use chain = promtpt | model ...... 
load_dotenv()

model=ChatOpenAI(model="gpt-4-turbo")

prompt=PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

chain=LLMChain(prompt=prompt, llm=model)

topic="Artificial Intelligence"
response=chain.invoke({"topic":topic})

print(response)
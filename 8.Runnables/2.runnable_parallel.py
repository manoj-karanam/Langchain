from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel   


load_dotenv()
model=ChatOpenAI(model="gpt-4-turbo")

prompt1=PromptTemplate(template="generate a tweet about {topic}", input_variables=["topic"])

prompt2=PromptTemplate(template="generate a linkedin post about {topic}", input_variables=["topic"])


model=ChatOpenAI(model="gpt-4-turbo")

parser=StrOutputParser()

chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser

# runnable sequnece is depreciated so i am using chains
parallel_chain=RunnableParallel(
    {
        "tweet": chain1,
        "linkedin_post": chain2
    }
)

result=parallel_chain.invoke({"topic":"Runnable Parallel in Langchain"})

print(result)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   



# runnbalesequene is depreciated

load_dotenv()
model=ChatOpenAI(model="gpt-4-turbo")

prompt1=PromptTemplate(template="Write a joke about {topic}")

prompt2=PromptTemplate(template="Explain the following joke \n {joke}", input_variables=["joke"])


parser=StrOutputParser()

# chain=RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# result = chain.invoke({"topic":"monkey"})

# print(result)

# Chain 1: topic → prompt1 → model → parser (produces joke)
chain1 = prompt1 | model | parser

# Chain 2: joke → prompt2 → model → parser (explains joke)
chain2 = prompt2 | model | parser

# Run first chain to get a joke
joke = chain1.invoke({"topic": "monkey"})
print("Joke:", joke)

# Run second chain to explain the joke
explanation = chain2.invoke({"joke": joke})
print("Explanation:", explanation)